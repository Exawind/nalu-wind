// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkSimdGatheredElementData.h"

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/ValidSimdLength.h"

#include <KokkosInterface.h>

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_simd/Simd.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

template <int p>
void
field_gather_t<p>::invoke(
  const_elem_mesh_index_view<p> conn,
  const stk::mesh::NgpField<double>& field,
  scalar_view<p> simd_element_field)
{
#if defined(KOKKOS_ENABLE_HIP)
  using policy_type = Kokkos::MDRangePolicy<
    exec_space, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
    Kokkos::Rank<4>, int>;
#else
  using policy_type = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>, int>;
#endif
  const auto range =
    policy_type({0, 0, 0, 0}, {conn.extent_int(0), p + 1, p + 1, p + 1});
  Kokkos::parallel_for(range, KOKKOS_LAMBDA(int index, int k, int j, int i) {
    for (int n = 0; n < simd_len; ++n) {
      const auto mesh_index =
        MeshIndexGetter<p, simd_len>::get(conn, index, k, j, i, n);
      stk::simd::set_data(
        simd_element_field(index, k, j, i), n, field.get(mesh_index, 0));
    }
  });
}

template <int p>
void
field_gather_t<p>::invoke(
  const_elem_mesh_index_view<p> conn,
  const stk::mesh::NgpField<double>& field,
  vector_view<p> simd_element_field)
{
#if defined(KOKKOS_ENABLE_HIP)
  using policy_type = Kokkos::MDRangePolicy<
    exec_space, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
    Kokkos::Rank<4>, int>;
#else
  using policy_type = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<4>, int>;
#endif
  const auto range =
    policy_type({0, 0, 0, 0}, {conn.extent_int(0), p + 1, p + 1, p + 1});
  Kokkos::parallel_for(range, KOKKOS_LAMBDA(int index, int k, int j, int i) {
    for (int n = 0; n < simd_len; ++n) {
      const auto mesh_index =
        MeshIndexGetter<p, simd_len>::get(conn, index, k, j, i, n);

      for (int d = 0; d < 3; ++d) {
        stk::simd::set_data(
          simd_element_field(index, k, j, i, d), n, field.get(mesh_index, d));
      }
    }
  });
}

template <int p>
void
field_gather_t<p>::invoke(
  const_face_mesh_index_view<p> conn,
  const stk::mesh::NgpField<double>& field,
  face_scalar_view<p> simd_element_field)
{
#if defined(KOKKOS_ENABLE_HIP)
  using policy_type = Kokkos::MDRangePolicy<
    exec_space, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
    Kokkos::Rank<3>, int>;
#else
  using policy_type = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>, int>;
#endif
  const auto range = policy_type({0, 0, 0}, {conn.extent_int(0), p + 1, p + 1});
  Kokkos::parallel_for(range, KOKKOS_LAMBDA(int index, int j, int i) {
    for (int n = 0; n < simd_len; ++n) {
      const auto mesh_index = valid_mesh_index(conn(index, 0, 0, n))
                                ? conn(index, j, i, n)
                                : conn(index, j, i, 0);
      stk::simd::set_data(
        simd_element_field(index, j, i), n, field.get(mesh_index, 0));
    }
  });
}

template <int p>
void
field_gather_t<p>::invoke(
  const_face_mesh_index_view<p> conn,
  const stk::mesh::NgpField<double>& field,
  face_vector_view<p> simd_element_field)
{
#if defined(KOKKOS_ENABLE_HIP)
  using policy_type = Kokkos::MDRangePolicy<
    exec_space, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
    Kokkos::Rank<3>, int>;
#else
  using policy_type = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>, int>;
#endif
  const auto range = policy_type({0, 0, 0}, {conn.extent_int(0), p + 1, p + 1});
  Kokkos::parallel_for(range, KOKKOS_LAMBDA(int index, int j, int i) {
    for (int n = 0; n < simd_len; ++n) {
      const auto mesh_index = valid_mesh_index(conn(index, 0, 0, n))
                                ? conn(index, j, i, n)
                                : conn(index, j, i, 0);

      for (int d = 0; d < 3; ++d) {
        stk::simd::set_data(
          simd_element_field(index, j, i, d), n, field.get(mesh_index, d));
      }
    }
  });
}
INSTANTIATE_POLYSTRUCT(field_gather_t);
} // namespace impl

void
field_gather(
  const_node_mesh_index_view conn,
  const stk::mesh::NgpField<double>& field,
  node_scalar_view simd_node_field)
{
  Kokkos::parallel_for(
    DeviceRangePolicy(0, conn.extent_int(0)), KOKKOS_LAMBDA(int index) {
      for (int n = 0; n < simd_len; ++n) {
        const auto simd_mesh_index = conn(index, n);
        const auto mesh_index =
          valid_mesh_index(simd_mesh_index) ? simd_mesh_index : conn(index, 0);
        stk::simd::set_data(
          simd_node_field(index), n, field.get(mesh_index, 0));
      }
    });
}

void
field_gather(
  const_node_mesh_index_view conn,
  const stk::mesh::NgpField<double>& field,
  node_vector_view simd_node_field)
{
  Kokkos::parallel_for(
    DeviceRangePolicy(0, conn.extent_int(0)), KOKKOS_LAMBDA(int index) {
      for (int n = 0; n < simd_len; ++n) {
        const auto simd_mesh_index = conn(index, n);
        const auto mesh_index =
          valid_mesh_index(simd_mesh_index) ? simd_mesh_index : conn(index, 0);
        for (int d = 0; d < 3; ++d) {
          stk::simd::set_data(
            simd_node_field(index, d), n, field.get(mesh_index, d));
        }
      }
    });
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
