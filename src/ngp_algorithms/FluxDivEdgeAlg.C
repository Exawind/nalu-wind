// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/FluxDivEdgeAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra::kynema_ugf {

FluxDivEdgeAlg::FluxDivEdgeAlg(
  Realm& realm,
  stk::mesh::Part* part,
  ScalarFieldType* flux,
  ScalarFieldType* div_flux)
  : Algorithm(realm, part),
    flux_(flux->mesh_meta_data_ordinal()),
    div_flux_(div_flux->mesh_meta_data_ordinal()),
    dnv_(get_field_ordinal(realm_.meta_data(), "dual_nodal_volume"))
{
}

void
FluxDivEdgeAlg::execute()
{
  using EntityInfoType = kynema_ugf_ngp::EntityInfo<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  const auto mdot = fieldMgr.template get_field<double>(flux_);
  const auto dnv = fieldMgr.template get_field<double>(dnv_);
  auto div_mdot = fieldMgr.template get_field<double>(div_flux_);
  const auto div_flux_ops =
    kynema_ugf_ngp::edge_nodal_field_updater(ngpMesh, div_mdot);

  const auto sel =
    (meta.locally_owned_part() & stk::mesh::selectUnion(partVec_)) -
    realm_.get_inactive_selector();

  div_mdot.sync_to_device();
  kynema_ugf_ngp::run_edge_algorithm(
    "div_mdot_edge", ngpMesh, sel, KOKKOS_LAMBDA(const EntityInfoType& einfo) {
      const auto edge_idx = einfo.meshIdx;
      const auto l = ngpMesh.fast_mesh_index(einfo.entityNodes[0]);
      const auto r = ngpMesh.fast_mesh_index(einfo.entityNodes[1]);
      div_flux_ops(einfo, 0, 0) += mdot(edge_idx, 0) / dnv(l, 0);
      div_flux_ops(einfo, 1, 0) -= mdot(edge_idx, 0) / dnv(r, 0);
    });
  div_mdot.modify_on_device();
}

} // namespace sierra::kynema_ugf
