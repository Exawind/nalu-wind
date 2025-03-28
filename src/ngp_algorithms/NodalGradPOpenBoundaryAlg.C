// Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include "ngp_algorithms/NodalGradPOpenBoundaryAlg.h"
#include "Algorithm.h"

#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// NodalGradPOpenBoundary - adds in boundary contribution
//                          for elem/edge proj nodal gradient
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename AlgTraits>
NodalGradPOpenBoundary<AlgTraits>::NodalGradPOpenBoundary(
  Realm& realm, stk::mesh::Part* part, const bool useShifted)
  : Algorithm(realm, part),
    useShifted_(useShifted),
    zeroGrad_(realm_.solutionOptions_->explicitlyZeroOpenPressureGradient_),
    massCorr_(realm_.solutionOptions_->activateOpenMdotCorrection_),
    exposedAreaVec_(get_field_ordinal(
      realm_.meta_data(),
      "exposed_area_vector",
      realm.meta_data().side_rank())),
    dualNodalVol_(get_field_ordinal(realm_.meta_data(), "dual_nodal_volume")),
    exposedPressureField_(get_field_ordinal(
      realm_.meta_data(), (massCorr_ ? "pressure" : "pressure_bc"))),
    pressureField_(get_field_ordinal(realm_.meta_data(), "pressure")),
    gradP_(get_field_ordinal(realm_.meta_data(), "dpdx")),
    coordinates_(
      get_field_ordinal(realm_.meta_data(), realm.get_coordinates_name())),
    dynPress_(get_field_ordinal(
      realm_.meta_data(), "dynamic_pressure", realm.meta_data().side_rank())),
    meFC_(MasterElementRepo::get_surface_master_element_on_dev(
      AlgTraits::FaceTraits::topo_)),
    meSCS_(MasterElementRepo::get_surface_master_element_on_dev(
      AlgTraits::ElemTraits::topo_)),
    faceData_(realm.meta_data()),
    elemData_(realm.meta_data())
{
  faceData_.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  elemData_.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  faceData_.add_cvfem_face_me(meFC_);
  elemData_.add_cvfem_surface_me(meSCS_);
  elemData_.add_gathered_nodal_field(pressureField_, 1);

  faceData_.add_face_field(
    exposedAreaVec_, AlgTraits::numFaceIp_, AlgTraits::nDim_);
  faceData_.add_face_field(dynPress_, AlgTraits::numFaceIp_);
  faceData_.add_gathered_nodal_field(dualNodalVol_, 1);
  faceData_.add_gathered_nodal_field(exposedPressureField_, 1);
  faceData_.add_gathered_nodal_field(gradP_, AlgTraits::nDim_);

  const ELEM_DATA_NEEDED fc_shape_fcn =
    useShifted_ ? FC_SHIFTED_SHAPE_FCN : FC_SHAPE_FCN;
  const ELEM_DATA_NEEDED scs_shape_fcn =
    useShifted_ ? SCS_SHIFTED_SHAPE_FCN : SCS_SHAPE_FCN;
  faceData_.add_master_element_call(fc_shape_fcn, CURRENT_COORDINATES);
  elemData_.add_master_element_call(scs_shape_fcn, CURRENT_COORDINATES);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
template <typename AlgTraits>
void
NodalGradPOpenBoundary<AlgTraits>::execute()
{
  using SimdDataType = nalu_ngp::FaceElemSimdData<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta_data = meshInfo.meta();

  const bool useShifted = useShifted_;
  const bool zeroGrad = zeroGrad_;

  const unsigned exposedAreaVec = exposedAreaVec_;
  const unsigned dualNodalVol = dualNodalVol_;
  const unsigned exposedPressureField = exposedPressureField_;
  const unsigned pressureField = pressureField_;
  const unsigned coordsID = coordinates_;
  const unsigned dynPID = dynPress_;

  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto ngpMesh = meshInfo.ngp_mesh();
  auto gradP = fieldMgr.template get_field<double>(gradP_);
  const auto gradPOps =
    nalu_ngp::simd_face_elem_nodal_field_updater(ngpMesh, gradP);

  MasterElement* meFC = meFC_;
  MasterElement* meSCS = meSCS_;

  stk::mesh::Selector s_locally_owned_union =
    meta_data.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  const std::string algName = "NodalGradPOpenBoundary_" +
                              std::to_string(AlgTraits::faceTopo_) + "_" +
                              std::to_string(AlgTraits::elemTopo_);

  const auto pstabFac =
    realm_.solutionOptions_->activateOpenMdotCorrection_ ? 0.0 : 1.0;

  const auto f_shp = shape_fcn<typename AlgTraits::FaceTraits, QuadRank::SCV>(
    use_shifted_quad(useShifted));

  const auto e_shp = shape_fcn<typename AlgTraits::ElemTraits, QuadRank::SCS>(
    use_shifted_quad(useShifted));

  nalu_ngp::run_face_elem_algorithm(
    algName, meshInfo, faceData_, elemData_, s_locally_owned_union,
    KOKKOS_LAMBDA(SimdDataType & fdata) {
      const int* ipNodeMap = meFC->ipNodeMap();

      auto& faceView = fdata.simdFaceView;
      auto& elemView = fdata.simdElemView;
      const auto v_areav = faceView.get_scratch_view_2D(exposedAreaVec);
      const auto v_dnv = faceView.get_scratch_view_1D(dualNodalVol);
      const auto face_p_field =
        faceView.get_scratch_view_1D(exposedPressureField);
      const auto dyn_p_field = faceView.get_scratch_view_1D(dynPID);

      const auto v_coord = faceView.get_scratch_view_2D(coordsID);
      const auto elem_p_field = elemView.get_scratch_view_1D(pressureField);

      const auto meFaceViews =
        fdata.simdFaceView.get_me_views(CURRENT_COORDINATES);
      const auto meElemViews = elemView.get_me_views(CURRENT_COORDINATES);

      const int faceOrdinal = fdata.faceOrd;

      for (int ip = 0; ip < AlgTraits::numFaceIp_; ++ip) {
        DoubleType pIp = 0.0;
        if (zeroGrad) {
          // evaluate pressure at opposing face.
          const int oip = meSCS->opposingFace(faceOrdinal, ip);
          for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
            pIp += e_shp(oip, n) * elem_p_field(n);
          }
        } else {
          for (int n = 0; n < AlgTraits::nodesPerFace_; ++n) {
            pIp += f_shp(ip, n) * face_p_field(n);
          }
        }

        const int node = ipNodeMap[ip];
        const DoubleType vol = v_dnv(node);
        const DoubleType press_div_vol =
          (pIp - pstabFac * dyn_p_field(ip)) / vol;

        for (int d = 0; d < AlgTraits::nDim_; ++d) {
          const DoubleType areav = v_areav(ip, d);
          gradPOps(fdata, node, d) += areav * press_div_vol;
        }
      }
    });
  gradP.modify_on_device();
}

INSTANTIATE_KERNEL_FACE_ELEMENT(NodalGradPOpenBoundary)

} // namespace nalu
} // namespace sierra
