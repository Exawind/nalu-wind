#ifndef LOCALARRAY_H
#define LOCALARRAY_H

#include "Kokkos_Macros.hpp"
#include "matrix_free/KokkosFramework.h"
#include "stk_simd/Simd.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <typename ArrayType, typename = void>
struct LocalArray
{
};

template <typename ArrayType>
struct alignas(alignment) LocalArray<
  ArrayType,
  typename std::enable_if<std::rank<ArrayType>::value == 1>::type>
{
  static constexpr int Rank = 1;
  using value_type = typename std::remove_all_extents<ArrayType>::type;
  static constexpr int extent_0 = std::extent<ArrayType>::value;
  value_type internal_data_[extent_0];

  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type operator()(int i) const
    noexcept
  {
    return internal_data_[i];
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type operator[](int i) const
    noexcept
  {
    return internal_data_[i];
  }
  KOKKOS_FORCEINLINE_FUNCTION value_type& operator()(int i) noexcept
  {
    return internal_data_[i];
  }
};

template <typename ArrayType>
struct alignas(alignment) LocalArray<
  ArrayType,
  typename std::enable_if<std::rank<ArrayType>::value == 2>::type>
{
  static constexpr int Rank = 2;
  static constexpr int extent_0 = std::extent<ArrayType, 0>::value;
  static constexpr int extent_1 = std::extent<ArrayType, 1>::value;
  using value_type = typename std::remove_all_extents<ArrayType>::type;
  value_type internal_data_[extent_0][extent_1];

  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type
  operator()(int j, int i) const noexcept
  {
    return internal_data_[j][i];
  }
  KOKKOS_FORCEINLINE_FUNCTION value_type& operator()(int j, int i) noexcept
  {
    return internal_data_[j][i];
  }
};

template <typename ArrayType>
struct alignas(alignment) LocalArray<
  ArrayType,
  typename std::enable_if<std::rank<ArrayType>::value == 3>::type>
{
  static constexpr int Rank = 3;
  static constexpr int extent_0 = std::extent<ArrayType, 0>::value;
  static constexpr int extent_1 = std::extent<ArrayType, 1>::value;
  static constexpr int extent_2 = std::extent<ArrayType, 2>::value;
  using value_type = typename std::remove_all_extents<ArrayType>::type;
  value_type internal_data_[extent_0][extent_1][extent_2];

  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type
  operator()(int k, int j, int i) const noexcept
  {
    return internal_data_[k][j][i];
  }
  KOKKOS_FORCEINLINE_FUNCTION value_type&
  operator()(int k, int j, int i) noexcept
  {
    return internal_data_[k][j][i];
  }
};

template <typename ArrayType>
struct alignas(alignment) LocalArray<
  ArrayType,
  typename std::enable_if<std::rank<ArrayType>::value == 4>::type>
{
  static constexpr int Rank = 4;
  static constexpr int extent_0 = std::extent<ArrayType, 0>::value;
  static constexpr int extent_1 = std::extent<ArrayType, 1>::value;
  static constexpr int extent_2 = std::extent<ArrayType, 2>::value;
  static constexpr int extent_3 = std::extent<ArrayType, 3>::value;
  using value_type = typename std::remove_all_extents<ArrayType>::type;
  value_type internal_data_[extent_0][extent_1][extent_2][extent_3];

  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type
  operator()(int l, int k, int j, int i) const noexcept
  {
    return internal_data_[l][k][j][i];
  }
  KOKKOS_FORCEINLINE_FUNCTION value_type&
  operator()(int l, int k, int j, int i) noexcept
  {
    return internal_data_[l][k][j][i];
  }
};

template <typename ArrayType>
struct alignas(alignment) LocalArray<
  ArrayType,
  typename std::enable_if<std::rank<ArrayType>::value == 5>::type>
{
  static constexpr int Rank = 5;
  static constexpr int extent_0 = std::extent<ArrayType, 0>::value;
  static constexpr int extent_1 = std::extent<ArrayType, 1>::value;
  static constexpr int extent_2 = std::extent<ArrayType, 2>::value;
  static constexpr int extent_3 = std::extent<ArrayType, 3>::value;
  static constexpr int extent_4 = std::extent<ArrayType, 4>::value;
  using value_type = typename std::remove_all_extents<ArrayType>::type;
  value_type internal_data_[extent_0][extent_1][extent_2][extent_3][extent_4];

  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type
  operator()(int m, int l, int k, int j, int i) const noexcept
  {
    return internal_data_[m][l][k][j][i];
  }
  KOKKOS_FORCEINLINE_FUNCTION value_type&
  operator()(int m, int l, int k, int j, int i) noexcept
  {
    return internal_data_[m][l][k][j][i];
  }
};
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
