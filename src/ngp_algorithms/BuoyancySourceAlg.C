// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/BuoyancySourceAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"
#include "SolutionOptions.h"

namespace sierra {
namespace nalu {

BuoyancySourceAlg::BuoyancySourceAlg(
  Realm& realm,
  stk::mesh::Part* part,
  VectorFieldType* source,
  ScalarFieldType* source_weight)
  : Algorithm(realm, part),
    source_(source->mesh_meta_data_ordinal()),
    source_weight_(source_weight->mesh_meta_data_ordinal()),
    edgeAreaVec_(get_field_ordinal(
      realm_.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    dualNodalVol_(get_field_ordinal(realm_.meta_data(), "dual_nodal_volume")),
    coordinates_(
      get_field_ordinal(realm_.meta_data(), realm.get_coordinates_name())),
    density_(
      get_field_ordinal(realm_.meta_data(), "density", stk::mesh::StateNP1))
{
}

void
BuoyancySourceAlg::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const unsigned ndim = meta.spatial_dimension();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto density = fieldMgr.template get_field<double>(density_);
  const auto edgeAreaVec = fieldMgr.template get_field<double>(edgeAreaVec_);
  auto source = fieldMgr.template get_field<double>(source_);
  auto sourceweight = fieldMgr.template get_field<double>(source_weight_);
  const auto sourceOps = nalu_ngp::edge_nodal_field_updater(ngpMesh, source);
  const auto sourceweightOps =
    nalu_ngp::edge_nodal_field_updater(ngpMesh, sourceweight);

  const stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  double gravity[3] = {0.0, 0.0, 0.0};

  if (realm_.solutionOptions_->gravity_.size() >= ndim)
    for (unsigned idim = 0; idim < ndim; ++idim)
      gravity[idim] = realm_.solutionOptions_->gravity_[idim];

  source.sync_to_device();

  const std::string algName = meta.get_fields()[source_]->name() + "_edge";
  nalu_ngp::run_edge_algorithm(
    algName, ngpMesh, sel, KOKKOS_LAMBDA(const EntityInfoType& einfo) {
       DblType av[NDimMax];

      for (unsigned d = 0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(einfo.meshIdx, d);

      const auto nodeL = ngpMesh.fast_mesh_index(einfo.entityNodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(einfo.entityNodes[1]);

      const DblType rhoIp =
        0.5 * (density.get(nodeL, 0) + density.get(nodeR, 0));

      DblType weight = 0.0;

      for (unsigned i = 0; i < ndim; ++i) {
        weight += stk::math::pow(gravity[i] * av[i], 2);
      }

      weight = stk::math::sqrt(weight);

      for (unsigned i = 0; i < ndim; ++i) {
        sourceOps(einfo, 0, i) += weight * rhoIp * gravity[i];
        sourceOps(einfo, 1, i) += weight * rhoIp * gravity[i];
      }
      sourceweightOps(einfo, 0, 0) += weight;
      sourceweightOps(einfo, 1, 0) += weight;
    });
  source.modify_on_device();
  sourceweight.modify_on_device();
}

} // namespace nalu
} // namespace sierra
