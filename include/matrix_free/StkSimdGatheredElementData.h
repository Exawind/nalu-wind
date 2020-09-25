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
#include "matrix_free/ValidSimdLength.h"

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

template <int p, int simd_len>
struct MeshIndexGetter
{
  KOKKOS_FORCEINLINE_FUNCTION static stk::mesh::FastMeshIndex get(
    const const_elem_mesh_index_view<p>& conn,
    int index,
    int k,
    int j,
    int i,
    int n)
  {
    return valid_mesh_index(conn(index, k, j, i, n)) ? conn(index, k, j, i, n)
                                                     : conn(index, k, j, i, 0);
  }
};

template <int p>
struct MeshIndexGetter<p, 1>
{
  KOKKOS_FORCEINLINE_FUNCTION static stk::mesh::FastMeshIndex get(
    const const_elem_mesh_index_view<p>& conn,
    int index,
    int k,
    int j,
    int i,
    int)
  {
    return conn(index, k, j, i, 0);
  }
};

namespace impl {
template <int p>
struct field_gather_t
{
  static void invoke(
    const_elem_mesh_index_view<p> connectivity,
    const stk::mesh::NgpField<double>& field,
    scalar_view<p> simd_element_field);

  static void invoke(
    const_elem_mesh_index_view<p> connectivity,
    const stk::mesh::NgpField<double>& field,
    vector_view<p> simd_element_field);

  static void invoke(
    const_face_mesh_index_view<p> connectivity,
    const stk::mesh::NgpField<double>& field,
    face_scalar_view<p> simd_element_field);

  static void invoke(
    const_face_mesh_index_view<p> connectivity,
    const stk::mesh::NgpField<double>& field,
    face_vector_view<p> simd_element_field);
};
} // namespace impl
P_INVOKEABLE(field_gather)

void field_gather(
  const_node_mesh_index_view,
  const stk::mesh::NgpField<double>&,
  node_scalar_view);

void field_gather(
  const_node_mesh_index_view,
  const stk::mesh::NgpField<double>&,
  node_vector_view);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
