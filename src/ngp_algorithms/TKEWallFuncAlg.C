// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/TKEWallFuncAlg.h"
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

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
TKEWallFuncAlg<BcAlgTraits>::TKEWallFuncAlg(Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    faceData_(realm.meta_data()),
    bcNodalTke_(get_field_ordinal(realm.meta_data(), "wall_model_tke_bc")),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    wallFricVel_(get_field_ordinal(
      realm.meta_data(),
      "wall_friction_velocity_bip",
      realm.meta_data().side_rank())),
    cMu_(realm.get_turb_model_constant(TM_cMu)),
    meFC_(sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      BcAlgTraits::topo_))
{
  faceData_.add_cvfem_face_me(meFC_);

  faceData_.add_coordinates_field(
    get_field_ordinal(realm.meta_data(), realm.get_coordinates_name()),
    BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData_.add_face_field(wallFricVel_, BcAlgTraits::numFaceIp_);
}

template <typename BcAlgTraits>
void
TKEWallFuncAlg<BcAlgTraits>::execute()
{
  using ElemSimdData = sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  // Bring class members into local scope for device capture
  const DoubleType sqrt_cmu = stk::math::sqrt(cMu_);
  const auto* meFC = meFC_;
  const unsigned exposedAreaID = exposedAreaVec_;
  const unsigned wallFricVelID = wallFricVel_;

  auto ngpBcNodalTke = fieldMgr.template get_field<double>(bcNodalTke_);
  const auto ngpTkeOps =
    nalu_ngp::simd_elem_nodal_field_updater(ngpMesh, ngpBcNodalTke);

  const stk::mesh::Selector sel =
    realm_.meta_data().locally_owned_part() & stk::mesh::selectUnion(partVec_);

  const std::string algName =
    "TKEWallFuncAlg_" + std::to_string(BcAlgTraits::topo_);
  nalu_ngp::run_elem_algorithm(
    algName, meshInfo, realm_.meta_data().side_rank(), faceData_, sel,
    KOKKOS_LAMBDA(ElemSimdData & edata) {
      auto& scrViews = edata.simdScrView;
      const auto& v_areav = scrViews.get_scratch_view_2D(exposedAreaID);
      const auto& v_utau = scrViews.get_scratch_view_1D(wallFricVelID);

      for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        DoubleType aMag = 0.0;
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          aMag += v_areav(ip, d) * v_areav(ip, d);
        }
        aMag = stk::math::sqrt(aMag);

        const auto nodeID = meFC->ipNodeMap()[ip];
        const DoubleType tkeBip = v_utau(ip) * v_utau(ip) / sqrt_cmu;
        ngpTkeOps(edata, nodeID, 0) += tkeBip * aMag;
      }
    });

  ngpBcNodalTke.modify_on_device();
}

INSTANTIATE_KERNEL_FACE(TKEWallFuncAlg)

} // namespace nalu
} // namespace sierra
