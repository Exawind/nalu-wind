// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STK_SIMD_NODE_CONNECTIVITY_MAP_H
#define STK_SIMD_NODE_CONNECTIVITY_MAP_H

#include "matrix_free/KokkosViewTypes.h"
#include "stk_mesh/base/Ngp.hpp"

#include "stk_mesh/base/Selector.hpp"

namespace stk {
namespace mesh {
struct Entity;
class BulkData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

node_mesh_index_view
simd_node_map(const stk::mesh::NgpMesh&, const stk::mesh::Selector&);
node_offset_view simd_node_offsets(
  const stk::mesh::NgpMesh&,
  const stk::mesh::Selector&,
  ra_entity_row_view_type);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
