// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "ngp_algorithms/GeometryAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_ngp/NgpFieldParallel.hpp"

namespace sierra {
namespace nalu {

GeometryAlgDriver::GeometryAlgDriver(
  Realm& realm
) : NgpAlgDriver(realm)
{}

void GeometryAlgDriver::pre_work()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();

  auto ngpDualVol = nalu_ngp::get_ngp_field(meshInfo, "dual_nodal_volume");
  ngpDualVol.set_all(ngpMesh, 0.0);

  if (realm_.realmUsesEdges_) {
    auto ngpEdgeArea = nalu_ngp::get_ngp_field(
      meshInfo, "edge_area_vector", stk::topology::EDGE_RANK);
    ngpEdgeArea.set_all(ngpMesh, 0.0);
  }

  if (hasWallFunc_) {
    auto wdist = nalu_ngp::get_ngp_field(meshInfo, "assembled_wall_normal_distance");
    auto warea = nalu_ngp::get_ngp_field(meshInfo, "assembled_wall_area_wf");

    wdist.set_all(ngpMesh, 0.0);
    warea.set_all(ngpMesh, 0.0);
  }
}

void GeometryAlgDriver::post_work()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<ngp::Mesh>::MeshIndex;

  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  std::vector<NGPDoubleFieldType*> fields;

  auto ngpDualVol = nalu_ngp::get_ngp_field(meshInfo, "dual_nodal_volume");
  fields.push_back(&ngpDualVol);

  if (realm_.realmUsesEdges_) {
    auto ngpEdgeArea = nalu_ngp::get_ngp_field(
      meshInfo, "edge_area_vector", stk::topology::EDGE_RANK);
    fields.push_back(&ngpEdgeArea);
  }

  if (hasWallFunc_) {
    auto wallAreaF = nalu_ngp::get_ngp_field(meshInfo, "assembled_wall_area_wf");
    auto wallDistF = nalu_ngp::get_ngp_field(meshInfo, "assembled_wall_normal_distance");
    fields.push_back(&wallAreaF);
    fields.push_back(&wallDistF);
  }

  // Algorithms should have marked the fields as modified, but call this here to
  // ensure the next step does a sync to host
  for (auto* fld: fields) {
    fld->modify_on_device();
  }

  bool doFinalSyncToDevice = false;
  ngp::parallel_sum(realm_.bulk_data(), fields, doFinalSyncToDevice);

  if (realm_.hasPeriodic_) {
    const auto& meta = realm_.meta_data();
    const unsigned nComponents = 1;
    auto* dualVol = meta.template get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume");
    realm_.periodic_field_update(dualVol, nComponents);

    if (hasWallFunc_) {
      const bool bypassFieldCheck = false;
      stk::mesh::FieldBase* wallAreaF = meta.get_field(
        stk::topology::NODE_RANK, "assembled_wall_area_wf");
      stk::mesh::FieldBase* wallDistF = meta.get_field(
        stk::topology::NODE_RANK, "assembled_wall_normal_distance");
      realm_.periodic_field_update(wallAreaF, nComponents, bypassFieldCheck);
      realm_.periodic_field_update(wallDistF, nComponents, bypassFieldCheck);
    }
  }

  for (auto* fld: fields) {
    fld->modify_on_host();
    fld->sync_to_device();
  }

  if (hasWallFunc_) {
    stk::mesh::FieldBase* wallDistF = realm_.meta_data().get_field(
      stk::topology::NODE_RANK, "assembled_wall_normal_distance");

    const stk::mesh::Selector sel =
      (realm_.meta_data().locally_owned_part() |
       realm_.meta_data().globally_shared_part()) &
      stk::mesh::selectField(*wallDistF);

    auto wdist = nalu_ngp::get_ngp_field(meshInfo, "assembled_wall_normal_distance");
    auto warea = nalu_ngp::get_ngp_field(meshInfo, "assembled_wall_area_wf");

    sierra::nalu::nalu_ngp::run_entity_algorithm(
      "GeometryAlgDriver_wdist_normalize",
      ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const MeshIndex& mi) {
        wdist.get(mi, 0) /= warea.get(mi, 0);
      });

    wdist.modify_on_device();
    warea.modify_on_device();
  }
}

}  // nalu
}  // sierra
