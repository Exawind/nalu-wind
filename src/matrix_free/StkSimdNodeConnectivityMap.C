#include "matrix_free/StkSimdNodeConnectivityMap.h"

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/NodeOrderMap.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdMeshTraverser.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/KokkosFramework.h"

#include <Kokkos_Macros.hpp>
#include <vector>

#include "Kokkos_Core.hpp"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_ngp/NgpMesh.hpp"
#include "stk_topology/topology.hpp"
#include "stk_util/util/StkNgpVector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

node_mesh_index_view
simd_node_map(const stk::mesh::NgpMesh& mesh, const stk::mesh::Selector& active)
{
  node_mesh_index_view node_indices(
    "node_mesh_index_map",
    num_simd_elements(mesh, stk::topology::NODE_RANK, active));

  auto fill_valid_nodes =
    KOKKOS_LAMBDA(int simd_node_index, int simd_index, stk::mesh::Entity ent)
  {
    node_indices(simd_node_index, simd_index) = mesh.fast_mesh_index(ent);
  };

  auto fill_remainder =
    KOKKOS_LAMBDA(int simd_node_index, int simd_index, stk::mesh::Entity)
  {
    node_indices(simd_node_index, simd_index) = stk::mesh::FastMeshIndex{
      stk::mesh::InvalidOrdinal, stk::mesh::InvalidOrdinal};
  };

  simd_traverse(
    mesh, stk::topology::NODE_RANK, active,
    KOKKOS_LAMBDA(int simd_node_index, int simd_index, stk::mesh::Entity ent) {
      node_indices(simd_node_index, simd_index) = mesh.fast_mesh_index(ent);
    },
    KOKKOS_LAMBDA(int simd_node_index, int simd_index, stk::mesh::Entity) {
      node_indices(simd_node_index, simd_index) = stk::mesh::FastMeshIndex{
        stk::mesh::InvalidOrdinal, stk::mesh::InvalidOrdinal};
    });
  return node_indices;
}

node_offset_view
simd_node_offsets(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active,
  ra_entity_row_view_type elid)
{
  node_offset_view node_offsets(
    "node_row_map", num_simd_elements(mesh, stk::topology::NODE_RANK, active));

  auto fill_valid_elids =
    KOKKOS_LAMBDA(int simd_node_index, int ne, stk::mesh::Entity ent)
  {
    node_offsets(simd_node_index, ne) = elid(ent.local_offset());
  };

  auto fill_remainder =
    KOKKOS_LAMBDA(int simd_node_index, int ne, stk::mesh::Entity)
  {
    node_offsets(simd_node_index, ne) = invalid_offset;
  };
  simd_traverse(
    mesh, stk::topology::NODE_RANK, active, fill_valid_elids, fill_remainder);
  return node_offsets;
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
