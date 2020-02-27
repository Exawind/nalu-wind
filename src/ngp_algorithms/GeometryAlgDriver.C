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
#include "ngp_utils/NgpReducers.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_ngp/NgpFieldParallel.hpp"

namespace sierra {
namespace nalu {

namespace {

template<typename MeshInfo>
void compute_volume_stats(const MeshInfo& meshInfo)
{
  using Traits = nalu_ngp::NGPMeshTraits<typename MeshInfo::NgpMeshType>;

  const auto& meta = meshInfo.meta();
  auto* dualVol = meta.template get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto& ngpDualVol =
    fieldMgr.template get_field<double>(dualVol->mesh_meta_data_ordinal());

  const stk::mesh::Selector sel = stk::mesh::selectField(*dualVol)
    & meta.locally_owned_part();

  nalu_ngp::MinMaxSumScalar<double> volStats;
  nalu_ngp::MinMaxSum<double> volReducer(volStats);

  nalu_ngp::run_entity_par_reduce(
    "GeometryAlgDriver::compute_volume_stats", ngpMesh,
    stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(
      const typename Traits::MeshIndex& mi,
      nalu_ngp::MinMaxSumScalar<double>& threadVal){
      const double dVol = ngpDualVol.get(mi, 0);

      if (dVol < threadVal.min_val) threadVal.min_val = dVol;
      if (dVol > threadVal.max_val) threadVal.max_val = dVol;
      threadVal.total_sum += dVol;
    }, volReducer);

  double lVolStats[3] = {volStats.min_val, volStats.max_val,
                         volStats.total_sum};
  double gVolStats[3] = { 0.0, 0.0, 0.0 };
  stk::all_reduce_min(
    meshInfo.bulk().parallel(), &lVolStats[0], &gVolStats[0], 1);
  stk::all_reduce_max(
    meshInfo.bulk().parallel(), &lVolStats[1], &gVolStats[1], 1);
  stk::all_reduce_sum(
    meshInfo.bulk().parallel(), &lVolStats[2], &gVolStats[2], 1);

  NaluEnv::self().naluOutputP0()
    << " DualNodalVolume min: " << gVolStats[0]
    << " max: " << gVolStats[1]
    << " total: " << gVolStats[2]
    << std::endl;
}

} // namespace

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

  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  std::vector<NGPDoubleFieldType*> fields;

  auto& ngpDualVol = nalu_ngp::get_ngp_field(meshInfo, "dual_nodal_volume");
  fields.push_back(&ngpDualVol);

  if (realm_.realmUsesEdges_) {
    auto& ngpEdgeArea = nalu_ngp::get_ngp_field(
      meshInfo, "edge_area_vector", stk::topology::EDGE_RANK);
    fields.push_back(&ngpEdgeArea);
  }

  if (hasWallFunc_) {
    auto& wallAreaF = nalu_ngp::get_ngp_field(meshInfo, "assembled_wall_area_wf");
    auto& wallDistF = nalu_ngp::get_ngp_field(meshInfo, "assembled_wall_normal_distance");
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

  // Compute volume statistics and print out
  compute_volume_stats(realm_.mesh_info());
}

}  // nalu
}  // sierra
