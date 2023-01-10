// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/NodeOrderMap.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkSimdMeshTraverser.h"
#include "matrix_free/ValidSimdLength.h"

#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_topology/topology.hpp"

#include <KokkosInterface.h>
#include "Kokkos_Macros.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

template <int p>
elem_mesh_index_view<p>
stk_connectivity_map_t<p>::invoke(
  const stk::mesh::NgpMesh& mesh, stk::mesh::Selector active)
{
  constexpr auto map = StkNodeOrderMapping<p>::map;
  elem_mesh_index_view<p> entity_elem(
    "elem_ent_row_map",
    num_simd_elements(mesh, stk::topology::ELEM_RANK, active));

  auto fill_connectivity =
    KOKKOS_LAMBDA(int simd_elem_index, int simd_index, stk::mesh::Entity ent)
  {
    const auto nodes =
      mesh.get_nodes(stk::topology::ELEM_RANK, mesh.fast_mesh_index(ent));
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          entity_elem(simd_elem_index, k, j, i, simd_index) =
            mesh.fast_mesh_index(nodes[map(k, j, i)]);
        }
      }
    }
  };

  auto fill_invalid =
    KOKKOS_LAMBDA(int simd_elem_index, int simd_index, stk::mesh::Entity)
  {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          entity_elem(simd_elem_index, k, j, i, simd_index) =
            invalid_mesh_index;
        }
      }
    }
  };

  simd_traverse(
    mesh, stk::topology::ELEM_RANK, active, fill_connectivity, fill_invalid);
  return entity_elem;
}
INSTANTIATE_POLYSTRUCT(stk_connectivity_map_t);

template <int p>
elem_offset_view<p>
create_offset_map_t<p>::invoke(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active,
  ra_entity_row_view_type elid)
{
  elem_offset_view<p> elem_offset(
    "elem_offset_row_map",
    num_simd_elements(mesh, stk::topology::ELEM_RANK, active));

  constexpr auto map = StkNodeOrderMapping<p>::map;

  auto fill_entity_lids =
    KOKKOS_LAMBDA(int simd_elem_index, int simd_index, stk::mesh::Entity ent)
  {
    const auto nodes =
      mesh.get_nodes(stk::topology::ELEM_RANK, mesh.fast_mesh_index(ent));
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          elem_offset(simd_elem_index, k, j, i, simd_index) =
            elid(nodes[map(k, j, i)].local_offset());
        }
      }
    }
  };

  auto fill_invalid =
    KOKKOS_LAMBDA(int simd_elem_index, int simd_index, stk::mesh::Entity)
  {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          elem_offset(simd_elem_index, k, j, i, simd_index) = invalid_offset;
        }
      }
    }
  };

  simd_traverse(
    mesh, stk::topology::ELEM_RANK, active, fill_entity_lids, fill_invalid);
  return elem_offset;
}

INSTANTIATE_POLYSTRUCT(create_offset_map_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
