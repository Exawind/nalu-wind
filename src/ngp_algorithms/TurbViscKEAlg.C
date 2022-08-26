// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/TurbViscKEAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

TurbViscKEAlg::TurbViscKEAlg(
  Realm& realm, stk::mesh::Part* part, ScalarFieldType* tvisc)
  : Algorithm(realm, part),
    tviscField_(tvisc),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    tke_(get_field_ordinal(realm.meta_data(), "turbulent_ke")),
    tdr_(get_field_ordinal(realm.meta_data(), "total_dissipation_rate")),
    tvisc_(tvisc->mesh_meta_data_ordinal()),
    dplus_(get_field_ordinal(realm.meta_data(), "dplus_wall_function")),
    cMu_(realm.get_turb_model_constant(TM_cMu)),
    fMuExp_(realm.get_turb_model_constant(TM_fMuExp))
{
}

void
TurbViscKEAlg::execute()
{
  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  const auto& meta = realm_.meta_data();

  stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*tviscField_);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto density = fieldMgr.get_field<double>(density_);
  const auto tke = fieldMgr.get_field<double>(tke_);
  const auto tdr = fieldMgr.get_field<double>(tdr_);
  const auto dplus = fieldMgr.get_field<double>(dplus_);
  auto tvisc = fieldMgr.get_field<double>(tvisc_);

  tvisc.sync_to_device();

  const DblType cMu = cMu_;
  const DblType fMuExp = fMuExp_;

  nalu_ngp::run_entity_algorithm(
    "TurbViscKEAlg", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
      const DblType fMu = 1.0 - stk::math::exp(fMuExp * dplus.get(meshIdx, 0));

      tvisc.get(meshIdx, 0) = cMu * fMu * density.get(meshIdx, 0) *
                              tke.get(meshIdx, 0) * tke.get(meshIdx, 0) /
                              stk::math::max(tdr.get(meshIdx, 0), 1.0e-16);
    });
  tvisc.modify_on_device();
}

} // namespace nalu
} // namespace sierra
