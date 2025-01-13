// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ELEMENT_GRADIENT_H
#define ELEMENT_GRADIENT_H

#include <cmath>

#include "matrix_free/ElementSCSInterpolate.h"
#include "matrix_free/GeometricFunctions.h"
#include "matrix_free/KokkosFramework.h"
#include "ArrayND.h"
#include "matrix_free/ShuffledAccess.h"
#include "matrix_free/TensorOperations.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p, typename BoxArray, typename InpArray>
KOKKOS_FUNCTION
  typename std::enable_if<InpArray::rank == 4, ArrayND<ftype[3][3]>>::type
  gradient_nodal(const BoxArray& box, const InpArray& u, int k, int j, int i)
{
  ArrayND<ftype[3][3]> gu_hat;
  for (int dj = 0; dj < 3; ++dj) {
    gu_hat(dj, 0) = 0;
    gu_hat(dj, 1) = 0;
    gu_hat(dj, 2) = 0;
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto D = Coeffs<p>::D;
      gu_hat(dj, 0) += D(i, q) * u(k, j, q, dj);
      gu_hat(dj, 1) += D(j, q) * u(k, q, i, dj);
      gu_hat(dj, 2) += D(k, q) * u(q, j, i, dj);
    }
  }

  ArrayND<ftype[3][3]> gu;
  inv_transform_t(geom::linear_hex_jacobian<p>(box, k, j, i), gu_hat, gu);
  return gu;
}

template <int p, typename BoxArray, typename InpArray>
KOKKOS_FUNCTION
  typename std::enable_if<InpArray::rank == 3, ArrayND<ftype[3]>>::type
  gradient_nodal(const BoxArray& box, const InpArray& u, int k, int j, int i)
{
  ArrayND<ftype[3]> gu_hat;
  gu_hat(0) = 0;
  gu_hat(1) = 0;
  gu_hat(2) = 0;
  for (int q = 0; q < p + 1; ++q) {
    static constexpr auto D = Coeffs<p>::D;
    gu_hat(0) += D(i, q) * u(k, j, q);
    gu_hat(1) += D(j, q) * u(k, q, i);
    gu_hat(2) += D(k, q) * u(q, j, i);
  }
  ArrayND<ftype[3]> gu;
  inv_transform_t(geom::linear_hex_jacobian<p>(box, k, j, i), gu_hat, gu);
  return gu;
}

template <
  int p,
  int dir,
  typename BoxArrayType,
  typename UArrayType,
  typename UHatArrayType>
KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<UArrayType::rank == 4, ArrayND<ftype[3][3]>>::type
  gradient_scs(
    const BoxArrayType& box,
    const UArrayType& u,
    const UHatArrayType& uhat,
    int l,
    int s,
    int r)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  constexpr int dir_0 = dir;
  constexpr int dir_1 = (dir == XH) ? YH : XH;
  constexpr int dir_2 = (dir == ZH) ? YH : ZH;

  ArrayND<ftype[3][3]> gu_hat;
  for (int d = 0; d < 3; ++d) {
    ftype acc = 0;
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto deriv = Coeffs<p>::Dt;
      acc += deriv(l, q) * shuffled_access<dir>(u, s, r, q, d);
    }
    gu_hat(d, dir_0) = acc;

    acc = 0;
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto deriv = Coeffs<p>::D;
      acc += deriv(r, q) * uhat(s, q, d);
    }
    gu_hat(d, dir_1) = acc;

    acc = 0;
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto deriv = Coeffs<p>::D;
      acc += deriv(s, q) * uhat(q, r, d);
    }
    gu_hat(d, dir_2) = acc;
  }
  ArrayND<ftype[3][3]> gu;
  inv_transform_t(
    geom::linear_hex_jacobian_scs<p, dir>(box, l, s, r), gu_hat, gu);
  return gu;
}

template <
  int p,
  int dir,
  typename BoxArrayType,
  typename UArrayType,
  typename UHatArrayType>
KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<UArrayType::rank == 3, ArrayND<ftype[3]>>::type
  gradient_scs(
    const BoxArrayType& box,
    const UArrayType& u,
    const UHatArrayType& uhat,
    int l,
    int s,
    int r)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  constexpr int dir_0 = dir;
  constexpr int dir_1 = (dir == XH) ? YH : XH;
  constexpr int dir_2 = (dir == ZH) ? YH : ZH;

  ArrayND<ftype[3]> gu_hat;
  for (int d = 0; d < 3; ++d) {
    ftype acc = 0;
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto deriv = Coeffs<p>::Dt;
      acc += deriv(l, q) * shuffled_access<dir>(u, s, r, q);
    }
    gu_hat(dir_0) = acc;

    acc = 0;
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto deriv = Coeffs<p>::D;
      acc += deriv(r, q) * uhat(s, q);
    }
    gu_hat(dir_1) = acc;

    acc = 0;
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto deriv = Coeffs<p>::D;
      acc += deriv(s, q) * uhat(q, r);
    }
    gu_hat(dir_2) = acc;
  }
  ArrayND<ftype[3]> gu;
  inv_transform_t(
    geom::linear_hex_jacobian_scs<p, dir>(box, l, s, r), gu_hat, gu);
  return gu;
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
