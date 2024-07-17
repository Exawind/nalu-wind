// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/KokkosViewTypes.h"

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "stk_io/IossBridge.hpp"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MeshBuilder.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/FEMHelpers.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/SkinBoundary.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_simd/Simd.hpp"
#include "stk_topology/topology.hpp"

#include <numeric>
#include <string>
#include <vector>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <typename Scalar>
Scalar
qfunc(const Scalar* x)
{
  return 1.0 + 2.0 * x[0] - 1.0 * x[1] + 6.666 * x[2];
}
constexpr double max_value = 10.666;

class SimdGatherFixture : public ::testing::Test
{
protected:
  static constexpr int order = 1;

  SimdGatherFixture()
    : bulkPtr(stk::mesh::MeshBuilder(MPI_COMM_WORLD)
                .set_spatial_dimension(3u)
                .create()),
      bulk(*bulkPtr),
      meta(bulk.mesh_meta_data())
  {
    meta.use_simple_fields();
    stk::topology topo(stk::topology::HEX_8);

    auto& q_field = meta.declare_field<double>(stk::topology::NODE_RANK, "q");

    stk::mesh::Part& block_1 = meta.declare_part_with_topology("block_1", topo);
    stk::io::put_io_part_attribute(block_1);

    stk::mesh::PartVector allSurfaces = {
      &meta.declare_part("all_surfaces", meta.side_rank())};
    stk::io::put_io_part_attribute(*allSurfaces.front());

    stk::mesh::PartVector individualSurfaces(topo.num_sides());
    for (unsigned k = 0u; k < topo.num_sides(); ++k) {
      individualSurfaces[k] = &meta.declare_part_with_topology(
        "surface_" + std::to_string(k), topo.side_topology(k));
    }

    // set a coordinate field
    auto& coordField =
      meta.declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(
      coordField, block_1, meta.spatial_dimension(), nullptr);
    stk::mesh::put_field_on_mesh(q_field, block_1, 1, nullptr);
    stk::mesh::put_field_on_mesh(
      coordField, stk::mesh::selectUnion(allSurfaces), meta.spatial_dimension(),
      nullptr);
    meta.set_coordinate_field(&coordField);
    meta.commit();

    stk::mesh::EntityIdVector nodeIds(topo.num_nodes());
    std::iota(nodeIds.begin(), nodeIds.end(), 8 * bulk.parallel_rank() + 1);

    bulk.modification_begin();

    for (auto id : nodeIds) {
      bulk.declare_entity(
        stk::topology::NODE_RANK, id, stk::mesh::PartVector{});
    }
    auto elem = stk::mesh::declare_element(
      bulk, block_1, bulk.parallel_rank() + 1, nodeIds);

    bulk.modification_end();

    stk::mesh::create_all_sides(bulk, block_1, allSurfaces, false);

    std::vector<std::vector<double>> nodeLocations = {
      {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
      {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};

    const auto* nodes = bulk.begin_nodes(elem);
    for (int j = 0; j < 8; ++j) {
      for (int d = 0; d < 3; ++d) {
        stk::mesh::field_data(coordField, nodes[j])[d] =
          nodeLocations.at(j).at(d);
      }
    }

    for (const auto* ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        *stk::mesh::field_data(q_field, node) =
          qfunc(stk::mesh::field_data(coordField, node));
      }
    }
    mesh = stk::mesh::NgpMesh(bulk);
    q_field_ngp = stk::mesh::get_updated_ngp_field<double>(q_field);
    coord_field_ngp = stk::mesh::get_updated_ngp_field<double>(coordField);
  }
  std::shared_ptr<stk::mesh::BulkData> bulkPtr;
  stk::mesh::BulkData& bulk;
  stk::mesh::MetaData& meta;
  stk::mesh::NgpMesh mesh;
  stk::mesh::NgpField<double> q_field_ngp;
  stk::mesh::NgpField<double> coord_field_ngp;
};

TEST_F(SimdGatherFixture, coordinates_values_are_possible)
{
  const auto map = stk_connectivity_map<order>(mesh, meta.universal_part());
  vector_view<order> coord_view{"coord_view", 1};
  field_gather<order>(map, coord_field_ngp, coord_view);

  auto coord_view_h = Kokkos::create_mirror_view(coord_view);
  Kokkos::deep_copy(coord_view_h, coord_view);

  for (int k = 0; k < order + 1; ++k) {
    for (int j = 0; j < order + 1; ++j) {
      for (int i = 0; i < order + 1; ++i) {
        ASSERT_DOUBLE_EQ(
          stk::math::abs(stk::simd::get_data(coord_view_h(0, k, j, i, 0), 0)),
          1);
        ASSERT_DOUBLE_EQ(
          stk::math::abs(stk::simd::get_data(coord_view_h(0, k, j, i, 1), 0)),
          1);
        ASSERT_DOUBLE_EQ(
          stk::math::abs(stk::simd::get_data(coord_view_h(0, k, j, i, 2), 0)),
          1);
      }
    }
  }
}

TEST_F(SimdGatherFixture, gathered_q_field_is_consistent)
{
  const auto map = stk_connectivity_map<order>(mesh, meta.universal_part());
  scalar_view<order> q_view{"q_view", 1};
  field_gather<order>(map, q_field_ngp, q_view);
  vector_view<order> coord_view{"coord_view", 1};
  field_gather<order>(map, coord_field_ngp, coord_view);

  auto coord_view_h = Kokkos::create_mirror_view(coord_view);
  Kokkos::deep_copy(coord_view_h, coord_view);

  auto q_view_h = Kokkos::create_mirror_view(q_view);
  Kokkos::deep_copy(q_view_h, q_view);

  for (int k = 0; k < order + 1; ++k) {
    for (int j = 0; j < order + 1; ++j) {
      for (int i = 0; i < order + 1; ++i) {
        Kokkos::Array<ftype, 3> coord = {
          {coord_view_h(0, k, j, i, 0), coord_view_h(0, k, j, i, 1),
           coord_view_h(0, k, j, i, 2)}};
        ASSERT_DOUBLE_EQ(
          stk::simd::get_data(q_view_h(0, k, j, i), 0),
          stk::simd::get_data(qfunc(coord.data()), 0));
      }
    }
  }
}

TEST_F(SimdGatherFixture, gathered_nodal_q_field_has_values)
{
  const auto map = simd_node_map(mesh, meta.universal_part());
  node_scalar_view q_view{"qn_view", map.extent(0)};

  field_gather(map, q_field_ngp, q_view);

  auto q_view_h = Kokkos::create_mirror_view(q_view);
  Kokkos::deep_copy(q_view_h, q_view);

  double qmax = -1;
  for (int k = 0; k < map.extent_int(0); ++k) {
    for (int n = 0; n < simd_len; ++n) {
      qmax = std::max(
        qmax,
        static_cast<double>(std::abs(stk::simd::get_data(q_view_h(k), n))));
    }
  }
  ASSERT_DOUBLE_EQ(qmax, max_value);
}

TEST_F(SimdGatherFixture, gathered_face_q_is_consistent)
{
  const auto map = face_node_map<order>(mesh, meta.universal_part());
  face_scalar_view<order> q_view("qf_view", map.extent(0));
  field_gather<order>(map, q_field_ngp, q_view);

  face_vector_view<order> coord_view{"coordf_view", map.extent(0)};
  field_gather<order>(map, coord_field_ngp, coord_view);

  auto q_view_h = Kokkos::create_mirror_view(q_view);
  Kokkos::deep_copy(q_view_h, q_view);

  auto coord_view_h = Kokkos::create_mirror_view(coord_view);
  Kokkos::deep_copy(coord_view_h, coord_view);

  for (int j = 0; j < order + 1; ++j) {
    for (int i = 0; i < order + 1; ++i) {
      Kokkos::Array<ftype, 3> coord = {
        {coord_view_h(0, j, i, 0), coord_view_h(0, j, i, 1),
         coord_view_h(0, j, i, 2)}};
      ASSERT_DOUBLE_EQ(
        stk::simd::get_data(q_view_h(0, j, i), 0),
        stk::simd::get_data(qfunc(coord.data()), 0));
    }
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
