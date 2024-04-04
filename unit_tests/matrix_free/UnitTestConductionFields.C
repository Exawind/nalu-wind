// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionInfo.h"

#include "matrix_free/StkSimdConnectivityMap.h"

#include "gtest/gtest.h"
#include "mpi.h"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MeshBuilder.hpp"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/FEMHelpers.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/SkinBoundary.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"
#include "stk_topology/topology.hpp"
#include "stk_io/IossBridge.hpp"

#include <numeric>

namespace sierra {
namespace nalu {
namespace matrix_free {

class ConductionFieldsFixture : public ::testing::Test
{
protected:
  static constexpr int order = 1;
  ConductionFieldsFixture()
  {
    stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
    meshBuilder.set_spatial_dimension(3u);
    bulk = meshBuilder.create();
    meta = &bulk->mesh_meta_data();

    q_field = &meta->declare_field<stk::mesh::Field<double>>(
      stk::topology::NODE_RANK, conduction_info::q_name, 3);
    alpha_field = &meta->declare_field<stk::mesh::Field<double>>(
      stk::topology::NODE_RANK, conduction_info::volume_weight_name);
    lambda_field = &meta->declare_field<stk::mesh::Field<double>>(
      stk::topology::NODE_RANK, conduction_info::diffusion_weight_name);

    stk::topology topo(stk::topology::HEX_8);

    stk::mesh::Part& block_1 =
      meta->declare_part_with_topology("block_1", topo);
    stk::io::put_io_part_attribute(block_1);

    stk::mesh::PartVector allSurfaces = {
      &meta->declare_part("all_surfaces", meta->side_rank())};
    stk::io::put_io_part_attribute(*allSurfaces.front());

    stk::mesh::PartVector individualSurfaces(topo.num_sides());
    for (unsigned k = 0u; k < topo.num_sides(); ++k) {
      individualSurfaces[k] = &meta->declare_part_with_topology(
        "surface_" + std::to_string(k), topo.side_topology(k));
    }

    // set a coordinate field
    using vector_field_type = stk::mesh::Field<double, stk::mesh::Cartesian3d>;
    auto& coordField = meta->declare_field<vector_field_type>(
      stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coordField, block_1, nullptr);
    stk::mesh::put_field_on_mesh(*q_field, block_1, 1, nullptr);
    stk::mesh::put_field_on_mesh(*lambda_field, block_1, 1, nullptr);
    stk::mesh::put_field_on_mesh(*alpha_field, block_1, 1, nullptr);
    stk::mesh::put_field_on_mesh(
      coordField, stk::mesh::selectUnion(allSurfaces), nullptr);
    meta->set_coordinate_field(&coordField);
    meta->commit();

    stk::mesh::EntityIdVector nodeIds(topo.num_nodes());
    std::iota(nodeIds.begin(), nodeIds.end(), 8 * bulk->parallel_rank() + 1);

    bulk->modification_begin();

    for (auto id : nodeIds) {
      bulk->declare_entity(
        stk::topology::NODE_RANK, id, stk::mesh::PartVector{});
    }
    auto elem = stk::mesh::declare_element(
      *bulk, block_1, bulk->parallel_rank() + 1, nodeIds);
    stk::mesh::create_all_sides(*bulk, block_1, allSurfaces, false);

    bulk->modification_end();

    auto surfaceSelector = stk::mesh::selectUnion(allSurfaces);
    stk::mesh::EntityVector all_faces;
    stk::mesh::get_selected_entities(
      surfaceSelector, bulk->get_buckets(meta->side_rank(), surfaceSelector),
      all_faces);
    ThrowRequire(all_faces.size() == topo.num_sides());

    bulk->modification_begin();
    for (unsigned k = 0u; k < all_faces.size(); ++k) {
      const int ordinal = bulk->begin_element_ordinals(all_faces[k])[0];
      bulk->change_entity_parts(
        all_faces[k], {individualSurfaces[ordinal]}, stk::mesh::PartVector{});
    }
    bulk->modification_end();

    std::vector<std::vector<double>> nodeLocations = {
      {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
      {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};

    const auto* nodes = bulk->begin_nodes(elem);
    for (int j = 0; j < 8; ++j) {
      for (int d = 0; d < 3; ++d) {
        stk::mesh::field_data(coordField, nodes[j])[d] =
          nodeLocations.at(j).at(d);
      }
    }

    for (const auto* ib :
         bulk->get_buckets(stk::topology::NODE_RANK, meta->universal_part())) {
      for (auto node : *ib) {
        *stk::mesh::field_data(
          q_field->field_of_state(stk::mesh::StateNP1), node) = 1.0;
        *stk::mesh::field_data(
          q_field->field_of_state(stk::mesh::StateN), node) = 1.0;
        *stk::mesh::field_data(
          q_field->field_of_state(stk::mesh::StateNM1), node) = 1.0;
        *stk::mesh::field_data(*alpha_field, node) = 1.0;
        *stk::mesh::field_data(*lambda_field, node) = 1.0;
      }
    }
    mesh = stk::mesh::get_updated_ngp_mesh(*bulk);
  }
  stk::mesh::MetaData* meta;
  std::shared_ptr<stk::mesh::BulkData> bulk;
  stk::mesh::Field<double>* q_field;
  stk::mesh::Field<double>* alpha_field;
  stk::mesh::Field<double>* lambda_field;
  stk::mesh::NgpMesh mesh;
};

TEST_F(ConductionFieldsFixture, gather)
{
  const auto conn = stk_connectivity_map<order>(mesh, meta->universal_part());
  auto fields = gather_required_conduction_fields<order>(*meta, conn);
  const auto qp1_h = Kokkos::create_mirror_view(fields.qp1);
  Kokkos::deep_copy(qp1_h, fields.qp1);

  for (int k = 0; k < order + 1; ++k) {
    for (int j = 0; j < order + 1; ++j) {
      for (int i = 0; i < order + 1; ++i) {
        ASSERT_DOUBLE_EQ(stk::simd::get_data(qp1_h(0, k, j, i), 0), 1.0);
      }
    }
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
