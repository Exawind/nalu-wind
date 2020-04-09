#ifndef STK_SIMD_CONNECTIVITY_MAP_H
#define STK_SIMD_CONNECTIVITY_MAP_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/ValidSimdLength.h"

#include <Kokkos_View.hpp>
#include <memory>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Selector.hpp>

#include "Kokkos_Core.hpp"

#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/Ngp.hpp"

namespace stk {
namespace mesh {
struct Entity;
class BulkData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {
template <int p>
struct stk_connectivity_map_t
{
  static elem_mesh_index_view<p>
  invoke(const stk::mesh::NgpMesh&, stk::mesh::Selector);
};
} // namespace impl
P_INVOKEABLE(stk_connectivity_map)

namespace impl {
template <int p>
struct create_offset_map_t
{
  static elem_offset_view<p> invoke(
    const stk::mesh::NgpMesh&,
    const stk::mesh::Selector&,
    ra_entity_row_view_type);
};
} // namespace impl
P_INVOKEABLE(create_offset_map)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
