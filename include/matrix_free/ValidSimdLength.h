// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STK_SIMD_VALID_LENGTH_H
#define STK_SIMD_VALID_LENGTH_H

#include "matrix_free/KokkosViewTypes.h"

#include "Kokkos_Core.hpp"
#include "stk_simd/Simd.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

inline constexpr int invalid_offset = -1;

constexpr stk::mesh::FastMeshIndex invalid_mesh_index =
  stk::mesh::FastMeshIndex{
    stk::mesh::InvalidOrdinal, stk::mesh::InvalidOrdinal};
KOKKOS_FORCEINLINE_FUNCTION bool
valid_mesh_index(stk::mesh::FastMeshIndex index)
{
  return !(
    index.bucket_id == invalid_mesh_index.bucket_id ||
    index.bucket_ord == invalid_mesh_index.bucket_ord);
}

namespace impl {
template <int p, int len>
struct valid_offset_t
{
  static KOKKOS_FORCEINLINE_FUNCTION int
  valid_offset(int index, const const_elem_offset_view<p>& offsets)
  {
    int n = simd_len - 1;
    while (offsets(index, 0, 0, 0, n) < 0 && n != 0) {
      --n;
    }
    return n + 1;
  }

  static KOKKOS_FORCEINLINE_FUNCTION int
  valid_offset(int index, const const_face_offset_view<p>& offsets)
  {
    int n = simd_len - 1;
    while (offsets(index, 0, 0, n) < 0 && n != 0) {
      --n;
    }
    return n + 1;
  }

  static KOKKOS_FORCEINLINE_FUNCTION int
  valid_offset(int index, const const_node_offset_view& offsets)
  {
    int n = simd_len - 1;
    while (offsets(index, n) < 0 && n != 0) {
      --n;
    }
    return n + 1;
  }

  static KOKKOS_FORCEINLINE_FUNCTION int
  valid_offset(int index, const const_elem_mesh_index_view<p>& offsets)
  {
    int n = simd_len - 1;
    while (!valid_mesh_index(offsets(index, 0, 0, 0, n)) && n != 0) {
      --n;
    }
    return n + 1;
  }
};

template <int p>
struct valid_offset_t<p, 1>
{
  template <typename UnusedType>
  static KOKKOS_FORCEINLINE_FUNCTION int valid_offset(int, const UnusedType&)
  {
    return 1;
  }
};
} // namespace impl
template <int p>
KOKKOS_FORCEINLINE_FUNCTION int
valid_offset(int index, const const_elem_offset_view<p>& offsets)
{
  return impl::valid_offset_t<p, simd_len>::valid_offset(index, offsets);
}

template <int p>
KOKKOS_FORCEINLINE_FUNCTION int
valid_offset(int index, const const_face_offset_view<p>& offsets)
{
  return impl::valid_offset_t<p, simd_len>::valid_offset(index, offsets);
}

KOKKOS_FORCEINLINE_FUNCTION int
valid_offset(int index, const const_node_offset_view& offsets)
{
  return impl::valid_offset_t<0, simd_len>::valid_offset(index, offsets);
}

template <int p>
KOKKOS_FORCEINLINE_FUNCTION int
valid_offset(int index, const const_elem_mesh_index_view<p>& offsets)
{
  return impl::valid_offset_t<p, simd_len>::valid_offset(index, offsets);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
