/*------------------------------------------------------------------------*/
/*  Copyright 2019 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "ngp_algorithms/TurbKineticEnergyWallAlg.h"


#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"

#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpMeshInfo.h"

#include "Realm.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_util/parallel/Parallel.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"
#include "stk_ngp/NgpFieldParallel.hpp"

// basic c++
#include <cmath>
#include <vector>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TurbKineticEnergyWallAlg - utau at wall bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename AlgTraits>
TurbKineticEnergyWallAlg<AlgTraits>::TurbKineticEnergyWallAlg(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    cMu_                         (realm.get_turb_model_constant(TM_cMu)),
    dataNeeded_                  (realm.meta_data()),
    turbKineticEnergy_           (get_field_ordinal(realm_.meta_data(), "turbulent_ke", stk::mesh::StateNP1)),
    bcTurbKineticEnergy_         (get_field_ordinal(realm_.meta_data(), "tke_bc")),
    bcAssembledTurbKineticEnergy_(get_field_ordinal(realm_.meta_data(), "wall_model_tke_bc")),
    assembledWallArea_           (get_field_ordinal(realm_.meta_data(), "assembled_wall_area_wf")),
    wallFrictionVelocityBip_     (get_field_ordinal(realm_.meta_data(), "wall_friction_velocity_bip",realm_.meta_data().side_rank())),
    exposedAreaVec_              (get_field_ordinal(realm_.meta_data(), "exposed_area_vector",realm_.meta_data().side_rank())),
    meFC_(MasterElementRepo::get_surface_master_element<AlgTraits>())
{
  dataNeeded_.add_cvfem_face_me(meFC_);
  dataNeeded_.add_coordinates_field(
    realm.meta_data().coordinate_field()->mesh_meta_data_ordinal(),
    AlgTraits::nDim_, CURRENT_COORDINATES);
  dataNeeded_.add_face_field(exposedAreaVec_, AlgTraits::numFaceIp_, AlgTraits::nDim_);
  dataNeeded_.add_face_field(wallFrictionVelocityBip_,  AlgTraits::numFaceIp_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
template <typename AlgTraits>
void TurbKineticEnergyWallAlg<AlgTraits>::execute()
{
  zero_nodal_fields();
  assemble_nodal_fields();
  normalize_nodal_fields();
}


//--------------------------------------------------------------------------
//-------- zero_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
template <typename AlgTraits>
void TurbKineticEnergyWallAlg<AlgTraits>::zero_nodal_fields()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto  ngpMesh  = meshInfo.ngp_mesh();

  ngp::Field<DblType> &bcAssembledTurbKineticEnergy = 
    fieldMgr.get_field<DblType>(bcAssembledTurbKineticEnergy_);

  bcAssembledTurbKineticEnergy.set_all(ngpMesh, 0.0);
}

//--------------------------------------------------------------------------
//-------- assemmble_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
template <typename AlgTraits>
void TurbKineticEnergyWallAlg<AlgTraits>::assemble_nodal_fields()
{
  using ElemSimdDataType = sierra::nalu::nalu_ngp::ElemSimdData<ngp::Mesh>;

  const auto& meshInfo = realm_.mesh_info();
  const stk::mesh::MetaData& meta_data = meshInfo.meta();
  const auto&               ngpMesh = meshInfo.ngp_mesh();
  const ngp::FieldManager& fieldMgr = meshInfo.ngp_field_manager();

  const DblType inv_sqrt_cMu = 1./std::sqrt(cMu_);

  const unsigned areaVecID = exposedAreaVec_;
  const unsigned fricVelID = wallFrictionVelocityBip_;
  auto* meFC = meFC_;

  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  ngp::Field<DblType> &bcAssembledTurbKineticEnergy = fieldMgr.get_field<DblType>(bcAssembledTurbKineticEnergy_);

  nalu_ngp::run_elem_algorithm(
    meshInfo, meta_data.side_rank(), dataNeeded_, s_locally_owned_union,
    KOKKOS_LAMBDA(ElemSimdDataType& edata) {

    const auto assTurbKenetic = nalu_ngp::simd_nodal_field_updater(
        ngpMesh, bcAssembledTurbKineticEnergy, edata);

    auto& scrViews = edata.simdScrView;
    const auto& v_areavec  = scrViews.get_scratch_view_2D(areaVecID);
    const auto& v_wallfric = scrViews.get_scratch_view_1D(fricVelID);

    const int* faceIpNodeMap = meFC->ipNodeMap();
    for ( int ip = 0; ip < AlgTraits::numFaceIp_; ++ip ) {
      DoubleType aMag = 0.0;
      for (int d = 0; d < AlgTraits::nDim_; ++d) {
        const DoubleType axj = v_areavec(ip,d);
        aMag += axj*axj;
      }
      aMag = stk::math::sqrt(aMag);

      // extract utau and compute wall value for tke
      const DoubleType utau = v_wallfric(ip);
      const DoubleType tkeBip = utau*utau*inv_sqrt_cMu;
      // assemble to nodal quantities
      const int ni = faceIpNodeMap[ip];
      assTurbKenetic(ni,0) += aMag*tkeBip;
    }
  });

  bcAssembledTurbKineticEnergy.modify_on_device();
  bcAssembledTurbKineticEnergy.sync_to_host();

  // Synchronize fields for parallel runs
  stk::mesh::FieldBase* assTurbKenetic = meta_data.get_field(
    stk::topology::NODE_RANK, "wall_model_tke_bc");
  stk::mesh::parallel_sum(realm_.bulk_data(), {assTurbKenetic});

  bcAssembledTurbKineticEnergy.modify_on_host();
  bcAssembledTurbKineticEnergy.sync_to_device();
}

//--------------------------------------------------------------------------
//-------- normalize_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
template <typename AlgTraits>
void TurbKineticEnergyWallAlg<AlgTraits>::normalize_nodal_fields()
{
  using Traits = nalu_ngp::NGPMeshTraits<ngp::Mesh>;
  const stk::mesh::MetaData& meta_data = realm_.mesh_info().meta();

  const auto& meshInfo = realm_.mesh_info();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto  ngpMesh  = meshInfo.ngp_mesh();

  // periodic assemble
  auto bcAssembledTurbKineticEnergy = fieldMgr.get_field<DblType>(bcAssembledTurbKineticEnergy_);

//stk::mesh::BulkData& bulk_data = realm_.bulk_data();
//const std::vector<ngp::Field<DblType>*> fields(1,&bcAssembledTurbKineticEnergy);
//ngp::parallel_sum(bulk_data, fields);

  ThrowRequireMsg(!realm_.hasPeriodic_, "realm_.periodic_field_update not implimented '");
  if ( realm_.hasPeriodic_) {
    //const unsigned fieldSize = 1;
    //const bool bypassFieldCheck = false; // fields are not defined at all slave/master node pairs
    //realm_.periodic_field_update(bcAssembledTurbKineticEnergy, fieldSize, bypassFieldCheck);
  }

  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(partVec_);

  auto tkeNp1              = fieldMgr.get_field<DblType>(turbKineticEnergy_);
  auto bcTurbKineticEnergy = fieldMgr.get_field<DblType>(bcTurbKineticEnergy_);
  auto assembledWallArea   = fieldMgr.get_field<DblType>(assembledWallArea_);

  nalu_ngp::run_entity_algorithm(
    ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& meshIdx) {

    const DblType ass_wall_area       = assembledWallArea.get(meshIdx,0);
    const DblType bc_ass_turb_kin_eng = bcAssembledTurbKineticEnergy.get(meshIdx,0);
    const DblType tkeBnd              = bc_ass_turb_kin_eng/ass_wall_area;

    bcAssembledTurbKineticEnergy.get(meshIdx,0) = tkeBnd;
    bcTurbKineticEnergy.get(meshIdx,0) = tkeBnd;
    // make sure that the next matrix assembly uses the proper tke value
    tkeNp1.get(meshIdx, 0) = tkeBnd;
  });
   bcAssembledTurbKineticEnergy.modify_on_device();
}

INSTANTIATE_KERNEL_FACE_3D(TurbKineticEnergyWallAlg) 
INSTANTIATE_KERNEL_FACE_2D(TurbKineticEnergyWallAlg) 
INSTANTIATE_KERNEL_FACE_2D_HO(TurbKineticEnergyWallAlg) 
INSTANTIATE_KERNEL_FACE_3D_HO(TurbKineticEnergyWallAlg) 

} // namespace nalu
} // namespace Sierra

