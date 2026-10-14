// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/TurbViscSAAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace kynema_ugf {

TurbViscSAAlg::TurbViscSAAlg(
  Realm& realm, stk::mesh::Part* part, ScalarFieldType* tvisc)
  : Algorithm(realm, part),
    tviscField_(tvisc),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    viscosity_(get_field_ordinal(realm.meta_data(), "viscosity")),
    nuTilda_(get_field_ordinal(realm.meta_data(), "sa_nu_tilda")),
    tvisc_(tvisc->mesh_meta_data_ordinal()),
    Cv1_(realm.get_turb_model_constant(TM_saCV1))
{
}

void
TurbViscSAAlg::execute()
{
  using Traits = kynema_ugf_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  const auto& meta = realm_.meta_data();

  stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*tviscField_);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto density = fieldMgr.get_field<double>(density_);
  const auto visc = fieldMgr.get_field<double>(viscosity_);
  const auto nuTilda = fieldMgr.get_field<double>(nuTilda_);
  auto tvisc = fieldMgr.get_field<double>(tvisc_);

  tvisc.sync_to_device();

  const DblType Cv1 = Cv1_;

  kynema_ugf_ngp::run_entity_algorithm(
    "TurbViscSAAlg", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
      const DblType rho = density.get(meshIdx, 0);
      const DblType mu = visc.get(meshIdx, 0);
      const DblType nu = mu / rho;
      const DblType nuT = nuTilda.get(meshIdx, 0);

      // chi = nuTilda / nu
      const DblType chi = nuT / nu;
      const DblType chi3 = chi * chi * chi;
      const DblType Cv1_3 = Cv1 * Cv1 * Cv1;

      // fv1 = chi^3 / (chi^3 + Cv1^3)
      const DblType fv1 = chi3 / (chi3 + Cv1_3);

      // mu_t = rho * nuTilda * fv1
      tvisc.get(meshIdx, 0) = rho * nuT * fv1;
    });
  tvisc.modify_on_device();
}

} // namespace kynema_ugf
} // namespace sierra
