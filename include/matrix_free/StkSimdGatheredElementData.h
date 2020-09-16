// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STK_SIMD_GATHERED_ELEMENT_DATA_H
#define STK_SIMD_GATHERED_ELEMENT_DATA_H

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/PolynomialOrders.h"

#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"

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
    const stk::mesh::NgpField<double>& field,
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
    const stk::mesh::NgpField<double>& field,
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
    const stk::mesh::NgpField<double>& field,
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
    const stk::mesh::NgpField<double>& field,
    face_vector_view<p> simd_element_field);
};
} // namespace impl
P_INVOKEABLE(stk_simd_face_vector_field_gather)

void stk_simd_scalar_node_gather(
  const_node_mesh_index_view,
  const stk::mesh::NgpField<double>&,
  node_scalar_view);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
