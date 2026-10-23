// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/EffDiffFluxCoeffSAAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include <stk_mesh/base/NgpMesh.hpp>

namespace sierra {
namespace kynema_ugf {

EffDiffFluxCoeffSAAlg::EffDiffFluxCoeffSAAlg(
  Realm& realm,
  stk::mesh::Part* part,
  ScalarFieldType* visc,
  ScalarFieldType* evisc,
  const double sigma)
  : Algorithm(realm, part),
    viscField_(visc),
    visc_(visc->mesh_meta_data_ordinal()),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    nuTilda_(get_field_ordinal(realm.meta_data(), "sa_nu_tilda")),
    evisc_(evisc->mesh_meta_data_ordinal()),
    invSigma_(1.0 / sigma),
    Cn1_(realm.get_turb_model_constant(TM_saCn1))
{
}

void
EffDiffFluxCoeffSAAlg::execute()
{
  using Traits = kynema_ugf_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  const auto& meta = realm_.meta_data();

  stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*viscField_);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto visc = fieldMgr.get_field<double>(visc_);
  const auto density = fieldMgr.get_field<double>(density_);
  const auto nuTilda = fieldMgr.get_field<double>(nuTilda_);
  auto evisc = fieldMgr.get_field<double>(evisc_);

  const DblType invSigma = invSigma_;
  const DblType Cn1 = Cn1_;

  // evisc = (mu + rho * nuTilda * fn) / sigma, fn = 1 for nuTilda >= 0
  kynema_ugf_ngp::run_entity_algorithm(
    "EffDiffFluxCoeffSAAlg", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
      const DblType mu = visc.get(meshIdx, 0);
      const DblType rho = density.get(meshIdx, 0);
      const DblType nuT = nuTilda.get(meshIdx, 0);
      DblType fn = 1.0;
      if (nuT < 0.0) {
        const DblType chi = rho * nuT / mu;
        const DblType chi3 = chi * chi * chi;
        fn = (Cn1 + chi3) / (Cn1 - chi3);
      }
      evisc.get(meshIdx, 0) = (mu + rho * nuT * fn) * invSigma;
    });

  evisc.modify_on_device();
}

} // namespace kynema_ugf
} // namespace sierra
