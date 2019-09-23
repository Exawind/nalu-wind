/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/TKEWallFuncAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
TKEWallFuncAlg<BcAlgTraits>::TKEWallFuncAlg(Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    faceData_(realm.meta_data()),
    tke_(get_field_ordinal(
      realm.meta_data(), "turbulent_ke", stk::mesh::StateNP1)),
    bctke_(get_field_ordinal(realm_.meta_data(), "tke_bc")),
    bcNodalTke_(get_field_ordinal(realm.meta_data(), "wall_model_tke_bc")),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    wallFricVel_(get_field_ordinal(
      realm.meta_data(),
      "wall_friction_velocity_bip",
      realm.meta_data().side_rank())),
    wallArea_(get_field_ordinal(realm.meta_data(), "assembled_wall_area_wf")),
    cMu_(realm.get_turb_model_constant(TM_cMu)),
    meFC_(sierra::nalu::MasterElementRepo::get_surface_master_element<
          BcAlgTraits>())
{
  faceData_.add_cvfem_face_me(meFC_);

  faceData_.add_coordinates_field(
    get_field_ordinal(realm.meta_data(), realm.get_coordinates_name()),
    BcAlgTraits::nDim_,
    CURRENT_COORDINATES);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData_.add_face_field(wallFricVel_, BcAlgTraits::numFaceIp_);
}

template<typename BcAlgTraits>
void TKEWallFuncAlg<BcAlgTraits>::execute()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<ngp::Mesh>::MeshIndex;
  using ElemSimdData = sierra::nalu::nalu_ngp::ElemSimdData<ngp::Mesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  const unsigned exposedAreaID = exposedAreaVec_;
  const unsigned wallFricVelID = wallFricVel_;

  auto ngpBcNodalTke = fieldMgr.template get_field<double>(bcNodalTke_);
  auto ngpBcTke = fieldMgr.template get_field<double>(bctke_);
  auto ngpTke = fieldMgr.template get_field<double>(tke_);
  auto ngpWallArea = fieldMgr.template get_field<double>(wallArea_);

  // Reset 'assembled' BC TKE nodal field
  ngpBcNodalTke.set_all(ngpMesh, 0.0);

  const DoubleType sqrt_cmu = stk::math::sqrt(cMu_);
  const auto* meFC = meFC_;

  const stk::mesh::Selector sel = realm_.meta_data().locally_owned_part()
    & stk::mesh::selectUnion(partVec_);

  nalu_ngp::run_elem_algorithm(
    meshInfo, realm_.meta_data().side_rank(), faceData_, sel,
    KOKKOS_LAMBDA(ElemSimdData& edata) {
      const auto ngpTkeOps = nalu_ngp::simd_nodal_field_updater(
        ngpMesh, ngpBcNodalTke, edata);

      auto& scrViews = edata.simdScrView;
      const auto& v_areav = scrViews.get_scratch_view_2D(exposedAreaID);
      const auto& v_utau = scrViews.get_scratch_view_1D(wallFricVelID);

      for (int ip=0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        DoubleType aMag = 0.0;
        for (int d=0; d < BcAlgTraits::nDim_; ++d) {
          aMag += v_areav(ip, d) * v_areav(ip, d);
        }
        aMag = stk::math::sqrt(aMag);

        const auto nodeID = meFC->ipNodeMap()[ip];
        const DoubleType tkeBip = v_utau(ip) * v_utau(ip) / sqrt_cmu;
        ngpTkeOps(nodeID, 0) += tkeBip * aMag;
      }
    });

  {
    // TODO: Replace logic with STK NGP parallel sum, handle periodic the NGP way
    ngpBcNodalTke.modify_on_device();
    ngpBcNodalTke.sync_to_host();

    stk::mesh::FieldBase* bcNodalTkeField =
      realm_.meta_data().get_fields()[bcNodalTke_];
    stk::mesh::parallel_sum(realm_.bulk_data(), {bcNodalTkeField});

    if (realm_.hasPeriodic_) {
      const unsigned nComp = 1;
      const bool bypassFieldCheck = false;
      realm_.periodic_field_update(bcNodalTkeField, nComp, bypassFieldCheck);
    }

    const stk::mesh::Selector sel =
      (realm_.meta_data().locally_owned_part() |
       realm_.meta_data().globally_shared_part()) &
      stk::mesh::selectUnion(partVec_);

    nalu_ngp::run_entity_algorithm(
      ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const MeshIndex& mi) {
        const double warea = ngpWallArea.get(mi, 0);
        const double tkeVal = ngpBcNodalTke.get(mi, 0) / warea;
        ngpBcNodalTke.get(mi, 0) = tkeVal;
        ngpBcTke.get(mi, 0) = tkeVal;
        ngpTke.get(mi, 0) = tkeVal;
      });

    ngpBcNodalTke.modify_on_device();
    ngpBcTke.modify_on_device();
    ngpTke.modify_on_device();
  }
}

INSTANTIATE_KERNEL_FACE(TKEWallFuncAlg)

}  // nalu
}  // sierra
