#ifndef SHUFFLED_ACCESS_H
#define SHUFFLED_ACCESS_H

#include <Kokkos_Macros.hpp>
#include <cmath>

#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

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

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
