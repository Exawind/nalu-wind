// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include <ngp_algorithms/EnthalpyEffDiffFluxCoeffAlg.h>
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include <stk_mesh/base/MetaData.hpp>

namespace sierra{
namespace nalu{

EnthalpyEffDiffFluxCoeffAlg::EnthalpyEffDiffFluxCoeffAlg(
  Realm &realm,
  stk::mesh::Part* part,
  ScalarFieldType* thermalCond,
  ScalarFieldType* specHeat,
  ScalarFieldType* tvisc,
  ScalarFieldType* evisc,
  const double sigmaTurb,
  const bool isTurbulent
) : Algorithm(realm, part),
    specHeatField_(specHeat),
    thermalCond_(thermalCond->mesh_meta_data_ordinal()),
    specHeat_(specHeat->mesh_meta_data_ordinal()),
    evisc_(evisc->mesh_meta_data_ordinal()),
    invSigmaTurb_(1.0 / sigmaTurb),
    isTurbulent_(isTurbulent)
{
  // Delay getting the ordinal as tvisc could be undefined for laminar flow
  // cases
  if (isTurbulent_)
    tvisc_ = tvisc->mesh_meta_data_ordinal();
}

void
EnthalpyEffDiffFluxCoeffAlg::execute()
{
  using Traits = nalu_ngp::NGPMeshTraits<ngp::Mesh>;

  const auto& meta = realm_.meta_data();

  stk::mesh::Selector sel = (
    meta.locally_owned_part() | meta.globally_shared_part())
    & stk::mesh::selectField(*specHeatField_);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto thermalCond = fieldMgr.get_field<double>(thermalCond_);
  const auto specHeat = fieldMgr.get_field<double>(specHeat_);
  auto evisc = fieldMgr.get_field<double>(evisc_);
  const DblType invSigmaTurb = invSigmaTurb_;

  if (isTurbulent_) {
    const auto tvisc = fieldMgr.get_field<double>(tvisc_);
    nalu_ngp::run_entity_algorithm(
      "EnthalpyEffDiffFluxCoeffAlg_turbulent",
      ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
        evisc.get(meshIdx, 0) = (
          thermalCond.get(meshIdx, 0) / specHeat.get(meshIdx, 0) +
          tvisc.get(meshIdx, 0) * invSigmaTurb);
      });
  } else {
    nalu_ngp::run_entity_algorithm(
      "EnthalpyEffDiffFluxCoeffAlg_laminar",
      ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
        evisc.get(meshIdx, 0) = (
          thermalCond.get(meshIdx, 0) / specHeat.get(meshIdx, 0));
      });
  }

  evisc.modify_on_device();
}

} // namespace nalu
} // namespace Sierra
