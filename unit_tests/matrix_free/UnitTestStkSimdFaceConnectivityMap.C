// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/ValidSimdLength.h"

#include "Kokkos_Core.hpp"

#include "stk_io/IossBridge.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MeshBuilder.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/FEMHelpers.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/SkinBoundary.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_topology/topology.hpp"
#include "stk_util/util/ReportHandler.hpp"

#include <numeric>
#include <string>
#include <vector>

namespace stk {
namespace mesh {
class Part;
}
} // namespace stk
namespace stk {
namespace mesh {
struct Cartesian3d;
}
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

class SimdFaceConnectivityFixture : public ::testing::Test
{
protected:
  SimdFaceConnectivityFixture()
    : bulkPtr(stk::mesh::MeshBuilder(MPI_COMM_WORLD)
                .set_spatial_dimension(3u)
                .create()),
      bulk(*bulkPtr),
      meta(bulk.mesh_meta_data())
  {
    meta.use_simple_fields();
    stk::topology topo(stk::topology::HEX_8);

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
    stk::mesh::create_all_sides(bulk, block_1, allSurfaces, false);

    bulk.modification_end();

    auto surfaceSelector = stk::mesh::selectUnion(allSurfaces);
    stk::mesh::EntityVector all_faces;
    stk::mesh::get_selected_entities(
      surfaceSelector, bulk.get_buckets(meta.side_rank(), surfaceSelector),
      all_faces);
    ThrowRequire(all_faces.size() == topo.num_sides());

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
    mesh = stk::mesh::NgpMesh(bulk);
  }
  std::shared_ptr<stk::mesh::BulkData> bulkPtr;
  stk::mesh::BulkData& bulk;
  stk::mesh::MetaData& meta;
  stk::mesh::NgpMesh mesh;
};

TEST_F(SimdFaceConnectivityFixture, map_has_correct_outermost_index)
{
  const auto map = face_node_map<1>(mesh, meta.universal_part());
  auto map_h = Kokkos::create_mirror_view(map);
  Kokkos::deep_copy(map_h, map);

  constexpr int ans = 6 % simd_len == 0 ? 6 / simd_len : 6 / simd_len + 1;
  ASSERT_EQ(map_h.extent_int(0), ans);
}

TEST_F(SimdFaceConnectivityFixture, map_has_at_correct_number_of_valid_entries)
{
  constexpr int order = 1;
  const auto map = face_node_map<order>(mesh, meta.universal_part());
  auto map_h = Kokkos::create_mirror_view(map);
  Kokkos::deep_copy(map_h, map);

  int valid_count = 0;
  for (int index = 0; index < map.extent_int(0); ++index) {
    for (int j = 0; j < order + 1; ++j) {
      for (int i = 0; i < order + 1; ++i) {
        for (int n = 0; n < simd_len; ++n) {
          if (map_h(index, j, i, n).bucket_id != stk::mesh::InvalidOrdinal) {
            ++valid_count;
          }
        }
      }
    }
  }
  ASSERT_EQ(valid_count, 24);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
