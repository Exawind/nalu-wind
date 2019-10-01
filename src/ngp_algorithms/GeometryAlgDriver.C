/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/GeometryAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

GeometryAlgDriver::GeometryAlgDriver(
  Realm& realm
) : NgpAlgDriver(realm)
{}

void GeometryAlgDriver::pre_work()
{
  const auto& meta = realm_.meta_data();
  auto* dualVol = meta.template get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");

  stk::mesh::field_fill(0.0, *dualVol);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpDualVol = fieldMgr.template get_field<double>(dualVol->mesh_meta_data_ordinal());

  ngpDualVol.set_all(ngpMesh, 0.0);

  if (realm_.realmUsesEdges_) {
    auto* edgeAreaVec = meta.template get_field<VectorFieldType>(
      stk::topology::EDGE_RANK, "edge_area_vector");
    stk::mesh::field_fill(0.0, *edgeAreaVec);

    auto ngpEdgeArea = fieldMgr.template get_field<double>(
      edgeAreaVec->mesh_meta_data_ordinal());
    ngpEdgeArea.set_all(ngpMesh, 0.0);
  }

  if (hasWallFunc_) {
    const auto wallNormDist = get_field_ordinal(meta, "assembled_wall_normal_distance");
    const auto wallArea = get_field_ordinal(meta, "assembled_wall_area_wf");
    auto wdist = fieldMgr.template get_field<double>(wallNormDist);
    auto warea = fieldMgr.template get_field<double>(wallArea);

    wdist.set_all(ngpMesh, 0.0);
    warea.set_all(ngpMesh, 0.0);
  }
}

void GeometryAlgDriver::post_work()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<ngp::Mesh>::MeshIndex;

  const auto& meta = realm_.meta_data();
  const auto& bulk = realm_.bulk_data();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  // TODO: Convert to NGP version of parallel and periodic updates
  auto* dualVol = meta.template get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  std::vector<const stk::mesh::FieldBase*> fields{dualVol};

  if (realm_.realmUsesEdges_) {
    auto* edgeAreaVec = meta.template get_field<VectorFieldType>(
      stk::topology::EDGE_RANK, "edge_area_vector");
    fields.push_back(edgeAreaVec);
  }

  if (hasWallFunc_) {
    stk::mesh::FieldBase* wallAreaF = meta.get_field(
      stk::topology::NODE_RANK, "assembled_wall_area_wf");
    stk::mesh::FieldBase* wallDistF = meta.get_field(
      stk::topology::NODE_RANK, "assembled_wall_normal_distance");
    fields.push_back(wallAreaF);
    fields.push_back(wallDistF);
  }

  for (auto* fld: fields) {
    auto ngpFld = fieldMgr.get_field<double>(fld->mesh_meta_data_ordinal());
    ngpFld.modify_on_device();
    ngpFld.sync_to_host();
  }

  stk::mesh::parallel_sum(bulk, fields);

  if (realm_.hasPeriodic_) {
    const unsigned nComponents = 1;
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
    auto ngpFld = fieldMgr.get_field<double>(fld->mesh_meta_data_ordinal());
    ngpFld.modify_on_host();
    ngpFld.sync_to_device();
  }

  if (hasWallFunc_) {
    stk::mesh::FieldBase* wallDistF = meta.get_field(
      stk::topology::NODE_RANK, "assembled_wall_normal_distance");

    const stk::mesh::Selector sel =
      (realm_.meta_data().locally_owned_part() |
       realm_.meta_data().globally_shared_part()) &
      stk::mesh::selectField(*wallDistF);

    const auto wallNormDist = get_field_ordinal(meta, "assembled_wall_normal_distance");
    const auto wallArea = get_field_ordinal(meta, "assembled_wall_area_wf");
    auto wdist = fieldMgr.template get_field<double>(wallNormDist);
    auto warea = fieldMgr.template get_field<double>(wallArea);

    sierra::nalu::nalu_ngp::run_entity_algorithm(
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
