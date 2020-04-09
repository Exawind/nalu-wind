#ifndef STK_SIMD_FACE_CONNECTIVITY_MAP_H
#define STK_SIMD_FACE_CONNECTIVITY_MAP_H

#include "matrix_free/KokkosFramework.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

#include "Kokkos_Core.hpp"

#include <memory>

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
