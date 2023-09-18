// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/MdotOpenEdgeAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_algorithms/MdotAlgDriver.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpReduceUtils.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
MdotOpenEdgeAlg<BcAlgTraits>::MdotOpenEdgeAlg(
  Realm& realm, stk::mesh::Part* part, MdotAlgDriver& mdotDriver)
  : Algorithm(realm, part),
    mdotDriver_(mdotDriver),
    elemData_(realm.meta_data()),
    faceData_(realm.meta_data()),
    coordinates_(
      get_field_ordinal(realm.meta_data(), realm.get_coordinates_name())),
    velocityRTM_(get_field_ordinal(
      realm.meta_data(), realm.does_mesh_move() ? "velocity_rtm" : "velocity")),
    pressure_(get_field_ordinal(realm.meta_data(), "pressure")),
    pressureBC_(get_field_ordinal(
      realm.meta_data(),
      realm_.solutionOptions_->activateOpenMdotCorrection_ ? "pressure"
                                                           : "pressure_bc")),
    Gpdx_(get_field_ordinal(realm.meta_data(), "dpdx")),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    openMassFlowRate_(get_field_ordinal(
      realm_.meta_data(),
      "open_mass_flow_rate",
      realm_.meta_data().side_rank())),
    Udiag_(get_field_ordinal(realm.meta_data(), "momentum_diag")),
    dynPress_(get_field_ordinal(
      realm_.meta_data(), "dynamic_pressure", realm_.meta_data().side_rank())),
    meFC_(MasterElementRepo::get_surface_master_element_on_dev(
      BcAlgTraits::FaceTraits::topo_)),
    meSCS_(MasterElementRepo::get_surface_master_element_on_dev(
      BcAlgTraits::ElemTraits::topo_))
{
  faceData_.add_cvfem_face_me(meFC_);
  elemData_.add_cvfem_surface_me(meSCS_);

  faceData_.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData_.add_face_field(dynPress_, BcAlgTraits::numFaceIp_);

  faceData_.add_gathered_nodal_field(velocityRTM_, BcAlgTraits::nDim_);
  faceData_.add_gathered_nodal_field(density_, 1);
  faceData_.add_gathered_nodal_field(pressureBC_, 1);
  faceData_.add_gathered_nodal_field(Udiag_, 1);
  faceData_.add_gathered_nodal_field(Gpdx_, BcAlgTraits::nDim_);

  elemData_.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemData_.add_gathered_nodal_field(pressure_, 1);
}

template <typename BcAlgTraits>
void
MdotOpenEdgeAlg<BcAlgTraits>::execute()
{
  using SimdDataType = nalu_ngp::FaceElemSimdData<stk::mesh::NgpMesh>;
  const auto& meta = realm_.meta_data();
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto openMdot = fieldMgr.template get_field<double>(openMassFlowRate_);
  const auto mdotOps =
    nalu_ngp::simd_face_elem_field_updater(ngpMesh, openMdot);

  const stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  const std::string algName = "MdotOpenEdgeAlg_" +
                              std::to_string(BcAlgTraits::faceTopo_) + "_" +
                              std::to_string(BcAlgTraits::elemTopo_);

  const std::string dofName = "pressure";
  const DoubleType nocFac = realm_.get_noc_usage(dofName) ? 1.0 : 0.0;
  const DoubleType pstabFac =
    realm_.solutionOptions_->activateOpenMdotCorrection_ ? 0.0 : 1.0;
  const unsigned coordsID = coordinates_;
  const unsigned velocityID = velocityRTM_;
  const unsigned pressureID = pressure_;
  const unsigned pbcID = pressureBC_;
  const unsigned densityID = density_;
  const unsigned GpdxID = Gpdx_;
  const unsigned udiagID = Udiag_;
  const unsigned areavecID = exposedAreaVec_;
  const unsigned dynPressID = dynPress_;

  MasterElement* meSCS = meSCS_;

  double mdotOpen = 0.0;
  Kokkos::Sum<double> mdotOpenReducer(mdotOpen);
  nalu_ngp::run_face_elem_par_reduce(
    algName, meshInfo, faceData_, elemData_, sel,
    KOKKOS_LAMBDA(SimdDataType & fdata, double& pSum) {
      auto& simdElemView = fdata.simdElemView;
      auto& simdFaceView = fdata.simdFaceView;

      const auto& v_coord = simdElemView.get_scratch_view_2D(coordsID);
      const auto& v_pressure = simdElemView.get_scratch_view_1D(pressureID);

      const auto& v_pbc = simdFaceView.get_scratch_view_1D(pbcID);
      const auto& v_rho = simdFaceView.get_scratch_view_1D(densityID);
      const auto& v_vel = simdFaceView.get_scratch_view_2D(velocityID);
      const auto& v_udiag = simdFaceView.get_scratch_view_1D(udiagID);
      const auto& v_area = simdFaceView.get_scratch_view_2D(areavecID);
      const auto& v_Gpdx = simdFaceView.get_scratch_view_2D(GpdxID);
      const auto& v_dyn_press = simdFaceView.get_scratch_view_1D(dynPressID);

      for (int ip = 0; ip < BcAlgTraits::nodesPerFace_; ++ip) {
        const int nodeR = meSCS->side_node_ordinals(fdata.faceOrd)[ip];
        const int nodeL = meSCS->opposingNodes(fdata.faceOrd, ip);

        const auto projTimeScale = pstabFac / v_udiag(ip);

        DoubleType asq = 0.0;
        DoubleType axdx = 0.0;
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          const DoubleType coordIp =
            0.5 * (v_coord(nodeR, d) + v_coord(nodeL, d));
          const DoubleType dxj = v_coord(nodeR, d) - coordIp;
          asq += v_area(ip, d) * v_area(ip, d);
          axdx += v_area(ip, d) * dxj;
        }
        const DoubleType inv_axdx = 1.0 / axdx;

        DoubleType pbc = v_pbc(ip) - v_dyn_press(ip);
        DoubleType pressureIp = 0.5 * (v_pressure(nodeL) + v_pressure(nodeR));
        DoubleType tmdot = -projTimeScale * (pbc - pressureIp) * asq * inv_axdx;

        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          const DoubleType coordIp =
            0.5 * (v_coord(nodeR, d) + v_coord(nodeL, d));
          const DoubleType axj = v_area(ip, d);
          const DoubleType dxj = v_coord(nodeR, d) - coordIp;
          const DoubleType kxj = axj - asq * inv_axdx * dxj;
          const DoubleType Gjp = v_Gpdx(ip, d);

          tmdot += ((v_rho(ip) * v_vel(ip, d) + projTimeScale * Gjp) * axj -
                   projTimeScale * kxj * Gjp * nocFac);
        }

        mdotOps(fdata, ip) = tmdot;
        nalu_ngp::simd_reduce_sum(pSum, tmdot, fdata.numSimdElems);
      }
    },
    mdotOpenReducer);

  mdotDriver_.add_open_mdot(mdotOpen);
  openMdot.modify_on_device();
}

INSTANTIATE_KERNEL_FACE_ELEMENT(MdotOpenEdgeAlg)

} // namespace nalu
} // namespace sierra
