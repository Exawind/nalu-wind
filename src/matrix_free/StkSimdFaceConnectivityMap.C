// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/NodeOrderMap.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkSimdMeshTraverser.h"
#include "matrix_free/ValidSimdLength.h"

#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_topology/topology.hpp"

#include <KokkosInterface.h>
#include "Kokkos_Macros.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

template <int p>
face_mesh_index_view<p>
face_node_map_t<p>::invoke(
  const stk::mesh::NgpMesh& mesh, const stk::mesh::Selector& active)
{
  constexpr auto map = StkFaceNodeMapping<p>::map;
  face_mesh_index_view<p> face_indices(
    "face_mesh_index_map",
    num_simd_elements(mesh, stk::topology::FACE_RANK, active));

  auto fill_face_connectivity =
    KOKKOS_LAMBDA(int simd_elem_index, int simd_index, stk::mesh::Entity ent)
  {
    const auto nodes =
      mesh.get_nodes(stk::topology::FACE_RANK, mesh.fast_mesh_index(ent));
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        face_indices(simd_elem_index, j, i, simd_index) =
          mesh.fast_mesh_index(nodes[map(j, i)]);
      }
    }
  };

  auto fill_invalid =
    KOKKOS_LAMBDA(int simd_elem_index, int simd_index, stk::mesh::Entity)
  {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        face_indices(simd_elem_index, j, i, simd_index) = invalid_mesh_index;
      }
    }
  };

  simd_traverse(
    mesh, stk::topology::FACE_RANK, active, fill_face_connectivity,
    fill_invalid);
  return face_indices;
}
INSTANTIATE_POLYSTRUCT(face_node_map_t);

template <int p>
face_offset_view<p>
face_offsets_t<p>::invoke(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active,
  ra_entity_row_view_type elid)
{
  constexpr auto map = StkFaceNodeMapping<p>::map;
  face_offset_view<p> face_offsets(
    "face_row_map", num_simd_elements(mesh, stk::topology::FACE_RANK, active));
  simd_traverse(
    mesh, stk::topology::FACE_RANK, active,
    KOKKOS_LAMBDA(int simd_elem_index, int simd_index, stk::mesh::Entity ent) {
      const auto nodes =
        mesh.get_nodes(stk::topology::FACE_RANK, mesh.fast_mesh_index(ent));
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          face_offsets(simd_elem_index, j, i, simd_index) =
            elid(nodes[map(j, i)].local_offset());
        }
      }
    },
    KOKKOS_LAMBDA(int simd_elem_index, int simd_index, stk::mesh::Entity) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          face_offsets(simd_elem_index, j, i, simd_index) = -1;
        }
      }
    });
  return face_offsets;
}
INSTANTIATE_POLYSTRUCT(face_offsets_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
