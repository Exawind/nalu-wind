// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STK_SIMD_FACE_CONNECTIVITY_MAP_H
#define STK_SIMD_FACE_CONNECTIVITY_MAP_H

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/PolynomialOrders.h"

#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Ngp.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {
template <int p>
struct face_node_map_t
{
  static face_mesh_index_view<p>
  invoke(const stk::mesh::NgpMesh&, const stk::mesh::Selector&);
};
} // namespace impl
P_INVOKEABLE(face_node_map)
namespace impl {
template <int p>
struct face_offsets_t
{
  static face_offset_view<p> invoke(
    const stk::mesh::NgpMesh&,
    const stk::mesh::Selector&,
    ra_entity_row_view_type);
};
} // namespace impl
P_INVOKEABLE(face_offsets)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
