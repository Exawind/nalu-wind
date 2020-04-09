#ifndef STK_SIMD_GATHERED_ELEMENT_DATA_H
#define STK_SIMD_GATHERED_ELEMENT_DATA_H

#include <memory>
#include <stk_mesh/base/Field.hpp>

#include "Kokkos_Core.hpp"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/KokkosFramework.h"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_ngp/NgpFieldManager.hpp"

namespace stk {
namespace mesh {
struct Cartesian3d;
class BulkData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {
template <int p>
struct stk_simd_scalar_field_gather_t
{
  static void invoke(
    const_elem_mesh_index_view<p> connectivity,
    const stk::mesh::NgpConstField<double>& field,
    scalar_view<p> simd_element_field);
};
} // namespace impl
P_INVOKEABLE(stk_simd_scalar_field_gather)

namespace impl {
template <int p>
struct stk_simd_vector_field_gather_t
{
  static void invoke(
    const_elem_mesh_index_view<p> connectivity,
    const stk::mesh::NgpConstField<double>& field,
    vector_view<p> simd_element_field);
};
} // namespace impl
P_INVOKEABLE(stk_simd_vector_field_gather)

namespace impl {
template <int p>
struct stk_simd_face_scalar_field_gather_t
{
  static void invoke(
    const_face_mesh_index_view<p> connectivity,
    const stk::mesh::NgpConstField<double>& field,
    face_scalar_view<p> simd_element_field);
};
} // namespace impl
P_INVOKEABLE(stk_simd_face_scalar_field_gather)

namespace impl {
template <int p>
struct stk_simd_face_vector_field_gather_t
{
  static void invoke(
    const_face_mesh_index_view<p> connectivity,
    const stk::mesh::NgpConstField<double>& field,
    face_vector_view<p> simd_element_field);
};
} // namespace impl
P_INVOKEABLE(stk_simd_face_vector_field_gather)

void stk_simd_scalar_node_gather(
  const_node_mesh_index_view,
  const stk::mesh::NgpConstField<double>&,
  node_scalar_view);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
