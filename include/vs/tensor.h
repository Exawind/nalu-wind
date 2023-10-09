// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
// Original implementation of this code by Shreyas Ananthan for AMR-Wind
// - (https://github.com/Exawind/amr-wind)
//
// Adapted to use Kokkos

#ifndef VS_TENSOR_H
#define VS_TENSOR_H

#include "Kokkos_Core.hpp"
#include "vs/vstraits.h"
#include "vs/vector.h"

namespace vs {

/** Tensor in 3D vector space
 */
template <typename T>
struct TensorT
{
  T vv[9]{Traits::zero(), Traits::zero(), Traits::zero(),
          Traits::zero(), Traits::zero(), Traits::zero(),
          Traits::zero(), Traits::zero(), Traits::zero()};

  static constexpr int ncomp = 9;
  using size_type = int;
  using value_type = T;
  using reference = T&;
  using iterator = T*;
  using const_iterator = const T*;
  using Traits = DTraits<T>;

  constexpr TensorT() = default;

  KOKKOS_FORCEINLINE_FUNCTION constexpr TensorT(
    const T& xx,
    const T& xy,
    const T& xz,
    const T& yx,
    const T& yy,
    const T& yz,
    const T& zx,
    const T& zy,
    const T& zz)
    : vv{xx, xy, xz, yx, yy, yz, zx, zy, zz}
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION TensorT(
    const VectorT<T>& x,
    const VectorT<T>& y,
    const VectorT<T>& z,
    const bool transpose = false);

  ~TensorT() = default;
  TensorT(const TensorT&) = default;
  TensorT(TensorT&&) = default;
  TensorT& operator=(const TensorT&) & = default;
  TensorT& operator=(const TensorT&) && = delete;
  TensorT& operator=(TensorT&&) & = default;
  TensorT& operator=(TensorT&&) && = delete;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr TensorT<T> zero() noexcept
  {
    return TensorT<T>{Traits::zero(), Traits::zero(), Traits::zero(),
                      Traits::zero(), Traits::zero(), Traits::zero(),
                      Traits::zero(), Traits::zero(), Traits::zero()};
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr TensorT<T> I() noexcept
  {
    return TensorT{Traits::one(),  Traits::zero(), Traits::zero(),
                   Traits::zero(), Traits::one(),  Traits::zero(),
                   Traits::zero(), Traits::zero(), Traits::one()};
  }

  KOKKOS_FORCEINLINE_FUNCTION void
  rows(const VectorT<T>& x, const VectorT<T>& y, const VectorT<T>& z) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION void
  cols(const VectorT<T>& x, const VectorT<T>& y, const VectorT<T>& z) noexcept;

  KOKKOS_FORCEINLINE_FUNCTION VectorT<T> x() const noexcept;
  KOKKOS_FORCEINLINE_FUNCTION VectorT<T> y() const noexcept;
  KOKKOS_FORCEINLINE_FUNCTION VectorT<T> z() const noexcept;

  KOKKOS_FORCEINLINE_FUNCTION VectorT<T> cx() const noexcept;
  KOKKOS_FORCEINLINE_FUNCTION VectorT<T> cy() const noexcept;
  KOKKOS_FORCEINLINE_FUNCTION VectorT<T> cz() const noexcept;

  KOKKOS_FORCEINLINE_FUNCTION T& xx() & noexcept { return vv[0]; }
  KOKKOS_FORCEINLINE_FUNCTION T& xy() & noexcept { return vv[1]; }
  KOKKOS_FORCEINLINE_FUNCTION T& xz() & noexcept { return vv[2]; }

  KOKKOS_FORCEINLINE_FUNCTION T& yx() & noexcept { return vv[3]; }
  KOKKOS_FORCEINLINE_FUNCTION T& yy() & noexcept { return vv[4]; }
  KOKKOS_FORCEINLINE_FUNCTION T& yz() & noexcept { return vv[5]; }

  KOKKOS_FORCEINLINE_FUNCTION T& zx() & noexcept { return vv[6]; }
  KOKKOS_FORCEINLINE_FUNCTION T& zy() & noexcept { return vv[7]; }
  KOKKOS_FORCEINLINE_FUNCTION T& zz() & noexcept { return vv[8]; }

  KOKKOS_FORCEINLINE_FUNCTION const T& xx() const& noexcept { return vv[0]; }
  KOKKOS_FORCEINLINE_FUNCTION const T& xy() const& noexcept { return vv[1]; }
  KOKKOS_FORCEINLINE_FUNCTION const T& xz() const& noexcept { return vv[2]; }

  KOKKOS_FORCEINLINE_FUNCTION const T& yx() const& noexcept { return vv[3]; }
  KOKKOS_FORCEINLINE_FUNCTION const T& yy() const& noexcept { return vv[4]; }
  KOKKOS_FORCEINLINE_FUNCTION const T& yz() const& noexcept { return vv[5]; }

  KOKKOS_FORCEINLINE_FUNCTION const T& zx() const& noexcept { return vv[6]; }
  KOKKOS_FORCEINLINE_FUNCTION const T& zy() const& noexcept { return vv[7]; }
  KOKKOS_FORCEINLINE_FUNCTION const T& zz() const& noexcept { return vv[8]; }

  KOKKOS_FORCEINLINE_FUNCTION T& operator[](size_type pos) & { return vv[pos]; }
  KOKKOS_FORCEINLINE_FUNCTION const T& operator[](size_type pos) const&
  {
    return vv[pos];
  }

  KOKKOS_FORCEINLINE_FUNCTION T* data() noexcept { return vv; }
  KOKKOS_FORCEINLINE_FUNCTION const T* data() const noexcept { return vv; }

  iterator begin() noexcept { return vv; }
  iterator end() noexcept { return vv + ncomp; }
  const_iterator cbegin() const noexcept { return vv; }
  const_iterator cend() const noexcept { return vv + ncomp; }
  size_type size() const noexcept { return ncomp; }
};

using Tensor = TensorT<double>;

} // namespace vs

#include "vs/tensorI.h"

#endif /* VS_TENSOR_H */
