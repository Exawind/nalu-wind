// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SHUFFLED_ACCESS_H
#define SHUFFLED_ACCESS_H

#include "matrix_free/KokkosFramework.h"
#include "ArrayND.h"

#include "Kokkos_Macros.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {
template <typename ViewType, int d>
struct shuffled_accessor
{
};

template <typename ViewType>
struct shuffled_accessor<ViewType, 0>
{
  KOKKOS_FORCEINLINE_FUNCTION static constexpr typename ViewType::value_type
  access(const ViewType& v, int k, int j, int i) noexcept
  {
    return v(k, j, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION static typename ViewType::value_type&
  access(ViewType& v, int k, int j, int i) noexcept
  {
    return v(k, j, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr typename ViewType::value_type
  access(const ViewType& v, int k, int j, int i, int d) noexcept
  {
    return v(k, j, i, d);
  }

  KOKKOS_FORCEINLINE_FUNCTION static typename ViewType::value_type&
  access(ViewType& v, int k, int j, int i, int d) noexcept
  {
    return v(k, j, i, d);
  }
};

template <typename ViewType>
struct shuffled_accessor<ViewType, 1>
{
  KOKKOS_FORCEINLINE_FUNCTION static constexpr typename ViewType::value_type
  access(const ViewType& v, int k, int j, int i) noexcept
  {
    return v(k, i, j);
  }

  KOKKOS_FORCEINLINE_FUNCTION static typename ViewType::value_type&
  access(ViewType& v, int k, int j, int i) noexcept
  {
    return v(k, i, j);
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr typename ViewType::value_type
  access(const ViewType& v, int k, int j, int i, int d) noexcept
  {
    return v(k, i, j, d);
  }

  KOKKOS_FORCEINLINE_FUNCTION static typename ViewType::value_type&
  access(ViewType& v, int k, int j, int i, int d) noexcept
  {
    return v(k, i, j, d);
  }
};

template <typename ViewType>
struct shuffled_accessor<ViewType, 2>
{
  KOKKOS_FORCEINLINE_FUNCTION static constexpr typename ViewType::value_type
  access(const ViewType& v, int k, int j, int i) noexcept
  {
    return v(i, k, j);
  }

  KOKKOS_FORCEINLINE_FUNCTION static typename ViewType::value_type&
  access(ViewType& v, int k, int j, int i) noexcept
  {
    return v(i, k, j);
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr typename ViewType::value_type
  access(const ViewType& v, int k, int j, int i, int d) noexcept
  {
    return v(i, k, j, d);
  }

  KOKKOS_FORCEINLINE_FUNCTION static typename ViewType::value_type&
  access(ViewType& v, int k, int j, int i, int d) noexcept
  {
    return v(i, k, j, d);
  }
};

template <int d>
struct active_index
{
};

template <>
struct active_index<0>
{
  KOKKOS_FORCEINLINE_FUNCTION static int index_0(int, int, int i) { return i; }
  KOKKOS_FORCEINLINE_FUNCTION static int index_1(int, int j, int) { return j; }
  KOKKOS_FORCEINLINE_FUNCTION static int index_2(int k, int, int) { return k; }
};

template <>
struct active_index<1>
{
  KOKKOS_FORCEINLINE_FUNCTION static int index_0(int, int j, int) { return j; }
  KOKKOS_FORCEINLINE_FUNCTION static int index_1(int, int, int i) { return i; }
  KOKKOS_FORCEINLINE_FUNCTION static int index_2(int k, int, int) { return k; }
};

template <>
struct active_index<2>
{
  KOKKOS_FORCEINLINE_FUNCTION static int index_0(int k, int, int) { return k; }
  KOKKOS_FORCEINLINE_FUNCTION static int index_1(int, int, int i) { return i; }
  KOKKOS_FORCEINLINE_FUNCTION static int index_2(int, int j, int) { return j; }
};

} // namespace impl

template <int d, typename ViewType>
KOKKOS_FORCEINLINE_FUNCTION ftype
shuffled_access(const ViewType& v, int k, int j, int i)
{
  return impl::shuffled_accessor<ViewType, d>::access(v, k, j, i);
}

template <int d, typename ViewType>
KOKKOS_FORCEINLINE_FUNCTION ftype&
shuffled_access(ViewType& v, int k, int j, int i)
{
  return impl::shuffled_accessor<ViewType, d>::access(v, k, j, i);
}

template <int dir, typename ViewType>
KOKKOS_FORCEINLINE_FUNCTION ftype
shuffled_access(const ViewType& v, int k, int j, int i, int d)
{
  return impl::shuffled_accessor<ViewType, dir>::access(v, k, j, i, d);
}

template <int dir, typename ViewType>
KOKKOS_FORCEINLINE_FUNCTION ftype&
shuffled_access(ViewType& v, int k, int j, int i, int d)
{
  return impl::shuffled_accessor<ViewType, dir>::access(v, k, j, i, d);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
