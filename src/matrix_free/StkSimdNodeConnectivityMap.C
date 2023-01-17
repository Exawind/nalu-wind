// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/StkSimdMeshTraverser.h"
#include "matrix_free/ValidSimdLength.h"

#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_topology/topology.hpp"

#include <KokkosInterface.h>

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
