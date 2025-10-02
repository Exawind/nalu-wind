// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/MdotInflowAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_algorithms/MdotAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpReduceUtils.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
MdotInflowAlg<BcAlgTraits>::MdotInflowAlg(
  Realm& realm, stk::mesh::Part* part, MdotAlgDriver& mdotAlg, bool useShifted)
  : Algorithm(realm, part),
    mdotDriver_(mdotAlg),
    faceData_(realm.meta_data()),
    velocityBC_(get_field_ordinal(
      realm.meta_data(),
      realm.solutionOptions_->activateOpenMdotCorrection_ ? "velocity_bc"
                                                          : "velocity")),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    useShifted_(useShifted),
    meFC_(
      MasterElementRepo::get_surface_master_element_on_dev(BcAlgTraits::topo_))
{
  faceData_.add_cvfem_surface_me(meFC_);

  const auto coordID = get_field_ordinal(
    realm_.meta_data(), realm_.solutionOptions_->get_coordinates_name());
  faceData_.add_coordinates_field(
    coordID, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceData_.add_gathered_nodal_field(density_, 1);
  faceData_.add_gathered_nodal_field(velocityBC_, BcAlgTraits::nDim_);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
}

template <typename BcAlgTraits>
void
MdotInflowAlg<BcAlgTraits>::execute()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = realm_.meta_data();
  const auto ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  const stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  // Bring variables to local scope for GPU capture
  const auto velocityBCID = velocityBC_;
  const auto densityID = density_;
  const auto exposedAreaVecID = exposedAreaVec_;
  const auto useShifted = useShifted_;
  const DoubleType interpTogether = realm_.solutionOptions_->get_mdot_interp();
  const DoubleType om_interpTogether = (1.0 - interpTogether);

  stk::mesh::NgpField<double> edgeFaceVelMag;

  if (realm_.has_mesh_deformation()) {
    edgeFaceVelMag_ = get_field_ordinal(
      realm_.meta_data(), "edge_face_velocity_mag", stk::topology::EDGE_RANK);
    edgeFaceVelMag = fieldMgr.template get_field<double>(edgeFaceVelMag_);
  }

  DoubleType mdotInflowTotal = 0.0;
  Kokkos::Sum<DoubleType> mdotReducer(mdotInflowTotal);
  const std::string algName =
    "MdotInflowAlg_" + std::to_string(BcAlgTraits::topo_);

  const auto shp =
    shape_fcn<BcAlgTraits, QuadRank::SCV>(use_shifted_quad(useShifted));

  nalu_ngp::run_elem_par_reduce(
    algName, meshInfo, meta.side_rank(), faceData_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata, DoubleType & mdotInflow) {
       DoubleType uBip[BcAlgTraits::nDim_];
       DoubleType rhoUBip[BcAlgTraits::nDim_];

      auto& scrView = edata.simdScrView;
      const auto& v_vel = scrView.get_scratch_view_2D(velocityBCID);
      const auto& v_rho = scrView.get_scratch_view_1D(densityID);
      const auto& v_areav = scrView.get_scratch_view_2D(exposedAreaVecID);

      for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        DoubleType rhoBip = 0.0;
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          uBip[d] = 0.0;
          rhoUBip[d] = 0.0;
        }

        for (int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic) {
          const DoubleType r = shp(ip, ic);
          rhoBip += r * v_rho(ic);
          for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
            uBip[d] += r * v_vel(ic, d);
            rhoUBip[d] += r * v_rho(ic) * v_vel(ic, d);
          }
        }

        DoubleType mdot = 0.0;
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          mdot += (interpTogether * rhoUBip[d] +
                   om_interpTogether * rhoBip * uBip[d]) *
                  v_areav(ip, d);
        }

        mdotInflow += mdot;
      }
    },
    mdotReducer);

  mdotDriver_.add_inflow_mdot(mdotInflowTotal);
}

INSTANTIATE_KERNEL_FACE(MdotInflowAlg)

} // namespace nalu
} // namespace sierra
