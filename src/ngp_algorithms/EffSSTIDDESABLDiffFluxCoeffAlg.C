/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/EffSSTIDDESABLDiffFluxCoeffAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra{
namespace nalu{

EffSSTIDDESABLDiffFluxCoeffAlg::EffSSTIDDESABLDiffFluxCoeffAlg(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *visc,
  ScalarFieldType *tvisc,
  ScalarFieldType *evisc,
  const double sigmaOne,
  const double sigmaTwo,
  const double sigmaABL)
  : Algorithm(realm, part),
    viscField_(visc),
    visc_(visc->mesh_meta_data_ordinal()),
    tvisc_(tvisc->mesh_meta_data_ordinal()),
    evisc_(evisc->mesh_meta_data_ordinal()),
    fOneBlend_(get_field_ordinal(realm.meta_data(), "sst_f_one_blending")),
    wallDist_(get_field_ordinal(realm.meta_data(), "minimum_distance_to_wall")),
    sigmaOne_(sigmaOne),
    sigmaTwo_(sigmaTwo),
    sigmaABL_(sigmaABL),
    abl_bndtw_(realm.get_turb_model_constant(TM_abl_bndtw)),
    abl_deltandtw_(realm.get_turb_model_constant(TM_abl_deltandtw))
{}

void
EffSSTIDDESABLDiffFluxCoeffAlg::execute()
{
  using Traits = nalu_ngp::NGPMeshTraits<>;

  const auto& meta = realm_.meta_data();

  stk::mesh::Selector sel
    = (meta.locally_owned_part() | meta.globally_shared_part())
    &stk::mesh::selectField(*viscField_);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto visc = fieldMgr.get_field<double>(visc_);
  const auto tvisc = fieldMgr.get_field<double>(tvisc_);
  auto evisc = fieldMgr.get_field<double>(evisc_);
  const auto fOneBlend = fieldMgr.get_field<double>(fOneBlend_);
  const auto wallDist = fieldMgr.get_field<double>(wallDist_);

  const DblType sigmaOne = sigmaOne_;
  const DblType sigmaTwo = sigmaTwo_;
  const DblType sigmaABL = sigmaABL_;
  const DblType bndtw = abl_bndtw_;
  const DblType deltandtw = abl_deltandtw_;
  
  nalu_ngp::run_entity_algorithm(
    "EffSSTIDDESABLDiffFluxCoeffAlg",
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
      const DblType blendedConstantSST = fOneBlend.get(meshIdx, 0)*sigmaOne + (1.0-fOneBlend.get(meshIdx, 0))*sigmaTwo;
      const DblType dw = wallDist.get(meshIdx, 0);
      const DblType f_des_abl = 0.5*stk::math::tanh( (bndtw - dw)/deltandtw) + 0.5;
      evisc.get(meshIdx, 0) = visc.get(meshIdx, 0) + tvisc.get(meshIdx, 0) *
          (f_des_abl * blendedConstantSST + (1.0 - f_des_abl)*sigmaABL);
    });

  // Set flag indicating that the field has been modified on device
  evisc.modify_on_device();
}

} // namespace nalu
} // namespace Sierra
