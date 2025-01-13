#ifndef ARRAY_ND_H
#define ARRAY_ND_H

#include "Kokkos_Macros.hpp"
#include <type_traits>

namespace sierra::nalu {

// stack array set to interoperate with Kokkos views
template <typename ArrayType, typename = void>
struct ArrayND
{
};

template <typename ArrayType, decltype(std::rank_v<ArrayType>) r>
using enable_if_rank = std::enable_if_t<std::rank_v<ArrayType> == r>;

template <typename ArrayType>
struct ArrayND<ArrayType, enable_if_rank<ArrayType, 1>>
{
  static constexpr int rank = 1;
  using value_type = std::remove_all_extents_t<ArrayType>;
  value_type internal_data_[std::extent_v<ArrayType, 0>];

  [[nodiscard]] static constexpr int extent_int(int /*unused*/)
  {
    return int(std::extent_v<ArrayType, 0>);
  }

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr value_type
  operator()(int i) const noexcept
  {
    return internal_data_[i];
  }

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr value_type
  operator[](int i) const noexcept
  {
    return internal_data_[i];
  }

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION value_type&
  operator()(int i) noexcept
  {
    return internal_data_[i];
  }
};

template <typename ArrayType>
struct ArrayND<ArrayType, enable_if_rank<ArrayType, 2>>
{
  static constexpr int rank = 2;

  using value_type = std::remove_all_extents_t<ArrayType>;
  value_type internal_data_[std::extent_v<ArrayType, 0>]
                           [std::extent_v<ArrayType, 1>];

  [[nodiscard]] static constexpr int extent_int(int n)
  {
    return (n == 0) ? int(std::extent_v<ArrayType, 0>)
                    : int(std::extent_v<ArrayType, 1>);
  }

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr value_type
  operator()(int j, int i) const noexcept
  {
    return internal_data_[j][i];
  }

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION value_type&
  operator()(int j, int i) noexcept
  {
    return internal_data_[j][i];
  }
};

template <typename ArrayType>
struct ArrayND<ArrayType, enable_if_rank<ArrayType, 3>>
{
  static constexpr int rank = 3;
  [[nodiscard]] static constexpr int extent_int(int n)
  {
    return (n == 0)   ? int(std::extent_v<ArrayType, 0>)
           : (n == 1) ? int(std::extent_v<ArrayType, 1>)
                      : int(std::extent_v<ArrayType, 2>);
  }

  using value_type = std::remove_all_extents_t<ArrayType>;
  value_type internal_data_[std::extent_v<ArrayType, 0>]
                           [std::extent_v<ArrayType, 1>]
                           [std::extent_v<ArrayType, 2>];

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr auto
  operator()(int k, int j, int i) const noexcept
  {
    return internal_data_[k][j][i];
  }

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto&
  operator()(int k, int j, int i) noexcept
  {
    return internal_data_[k][j][i];
  }
};

template <typename ArrayType>
struct ArrayND<ArrayType, enable_if_rank<ArrayType, 4>>
{
  static constexpr int rank = 4;
  [[nodiscard]] static constexpr int extent_int(int n)
  {
    return (n == 0)   ? int(std::extent_v<ArrayType, 0>)
           : (n == 1) ? int(std::extent_v<ArrayType, 1>)
           : (n == 2) ? int(std::extent_v<ArrayType, 2>)
                      : int(std::extent_v<ArrayType, 3>);
  }

  using value_type = std::remove_all_extents_t<ArrayType>;
  value_type
    internal_data_[std::extent_v<ArrayType, 0>][std::extent_v<ArrayType, 1>]
                  [std::extent_v<ArrayType, 2>][std::extent_v<ArrayType, 3>];

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr value_type
  operator()(int l, int k, int j, int i) const noexcept
  {
    return internal_data_[l][k][j][i];
  }

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION value_type&
  operator()(int l, int k, int j, int i) noexcept
  {
    return internal_data_[l][k][j][i];
  }
};

template <typename ArrayType>
struct ArrayND<ArrayType, enable_if_rank<ArrayType, 5>>
{
  static constexpr int rank = 5;
  [[nodiscard]] static constexpr int extent_int(int n)
  {
    return (n == 0)   ? int(std::extent_v<ArrayType, 0>)
           : (n == 1) ? int(std::extent_v<ArrayType, 1>)
           : (n == 2) ? int(std::extent_v<ArrayType, 2>)
           : (n == 3) ? int(std::extent_v<ArrayType, 3>)
                      : int(std::extent_v<ArrayType, 4>);
  }

  using value_type = std::remove_all_extents_t<ArrayType>;
  value_type
    internal_data_[std::extent_v<ArrayType, 0>][std::extent_v<ArrayType, 1>]
                  [std::extent_v<ArrayType, 2>][std::extent_v<ArrayType, 3>]
                  [std::extent_v<ArrayType, 4>];

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr value_type
  operator()(int m, int l, int k, int j, int i) const noexcept
  {
    return internal_data_[m][l][k][j][i];
  }

  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION value_type&
  operator()(int m, int l, int k, int j, int i) noexcept
  {
    return internal_data_[m][l][k][j][i];
  }
};

} // namespace sierra::nalu

#endif