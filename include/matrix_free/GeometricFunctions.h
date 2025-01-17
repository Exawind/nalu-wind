// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef GEOMETRIC_FUNCTIONS_H
#define GEOMETRIC_FUNCTIONS_H

#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/TensorOperations.h"

#include "Kokkos_Macros.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {

template <int p, int dk, int dj, int di, typename BoxArray>
KOKKOS_FUNCTION typename BoxArray::value_type
hex_jacobian_component_scs(const BoxArray& box, int l, int s, int r)
{
  enum { LN = 0, RN = 1 };
  enum { XH = 0, YH = 1, ZH = 2 };

  static constexpr auto nlin = Coeffs<p>::Nlin;
  static constexpr auto ntlin = Coeffs<p>::Ntlin;
  typename BoxArray::value_type jac(0);
  switch (dj) {
  case XH: {
    const double lj = (dk == YH)   ? ntlin(LN, l)
                      : (dk == XH) ? nlin(LN, r)
                                   : nlin(LN, s);
    const double rj = (dk == YH)   ? ntlin(RN, l)
                      : (dk == XH) ? nlin(RN, r)
                                   : nlin(RN, s);

    const double lk = (dk == ZH) ? ntlin(LN, l) : nlin(LN, s);
    const double rk = (dk == ZH) ? ntlin(RN, l) : nlin(RN, s);

    jac = -lj * lk * box(di, 0) + lj * lk * box(di, 1) + rj * lk * box(di, 2) -
          rj * lk * box(di, 3) - lj * rk * box(di, 4) + lj * rk * box(di, 5) +
          rj * rk * box(di, 6) - rj * rk * box(di, 7);
    break;
  }
  case YH: {
    const double li = (dk == XH) ? ntlin(LN, l) : nlin(LN, r);
    const double ri = (dk == XH) ? ntlin(RN, l) : nlin(RN, r);

    const double lk = (dk == ZH) ? ntlin(LN, l) : nlin(LN, s);
    const double rk = (dk == ZH) ? ntlin(RN, l) : nlin(RN, s);

    jac = -li * lk * box(di, 0) - ri * lk * box(di, 1) + ri * lk * box(di, 2) +
          li * lk * box(di, 3) - li * rk * box(di, 4) - ri * rk * box(di, 5) +
          ri * rk * box(di, 6) + li * rk * box(di, 7);
    break;
  }
  default: {
    const double li = (dk == XH) ? ntlin(LN, l) : nlin(LN, r);
    const double ri = (dk == XH) ? ntlin(RN, l) : nlin(RN, r);

    const double lj = (dk == YH)   ? ntlin(LN, l)
                      : (dk == XH) ? nlin(LN, r)
                                   : nlin(LN, s);
    const double rj = (dk == YH)   ? ntlin(RN, l)
                      : (dk == XH) ? nlin(RN, r)
                                   : nlin(RN, s);

    jac = -li * lj * box(di, 0) - ri * lj * box(di, 1) - ri * rj * box(di, 2) -
          li * rj * box(di, 3) + li * lj * box(di, 4) + ri * lj * box(di, 5) +
          ri * rj * box(di, 6) + li * rj * box(di, 7);
    break;
  }
  }
  constexpr double isoParametricFactor = 0.5;
  return jac * isoParametricFactor;
}

template <int p, int dk, typename BoxArray>
KOKKOS_FUNCTION ArrayND<typename BoxArray::value_type[3][3]>
linear_hex_jacobian_scs(const BoxArray& box, int k, int j, int i)
{
  enum { XH = 0, YH = 1, ZH = 2 };

  return {
    {{hex_jacobian_component_scs<p, dk, XH, XH>(box, k, j, i),
      hex_jacobian_component_scs<p, dk, XH, YH>(box, k, j, i),
      hex_jacobian_component_scs<p, dk, XH, ZH>(box, k, j, i)},
     {
       hex_jacobian_component_scs<p, dk, YH, XH>(box, k, j, i),
       hex_jacobian_component_scs<p, dk, YH, YH>(box, k, j, i),
       hex_jacobian_component_scs<p, dk, YH, ZH>(box, k, j, i),
     },
     {
       hex_jacobian_component_scs<p, dk, ZH, XH>(box, k, j, i),
       hex_jacobian_component_scs<p, dk, ZH, YH>(box, k, j, i),
       hex_jacobian_component_scs<p, dk, ZH, ZH>(box, k, j, i),
     }}};
}

template <int p, int dk, typename BoxArray>
KOKKOS_FUNCTION ArrayND<typename BoxArray::value_type[3][3]>
linear_hex_invjact_scs(const BoxArray& box, int k, int j, int i)
{
  return invert_transpose_matrix(linear_hex_jacobian_scs<p, dk>(box, k, j, i));
}

template <int p, int dk, typename BoxArrayType>
KOKKOS_FUNCTION ArrayND<ftype[3]>
linear_area(const BoxArrayType& box, int k, int j, int i)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  static constexpr int ds1 = (dk == XH) ? ZH : (dk == YH) ? XH : YH;
  static constexpr int ds2 = (dk == XH) ? YH : (dk == YH) ? ZH : XH;
  const auto dx_ds1 = hex_jacobian_component_scs<p, dk, ds1, XH>(box, k, j, i);
  const auto dx_ds2 = hex_jacobian_component_scs<p, dk, ds2, XH>(box, k, j, i);
  const auto dy_ds1 = hex_jacobian_component_scs<p, dk, ds1, YH>(box, k, j, i);
  const auto dy_ds2 = hex_jacobian_component_scs<p, dk, ds2, YH>(box, k, j, i);
  const auto dz_ds1 = hex_jacobian_component_scs<p, dk, ds1, ZH>(box, k, j, i);
  const auto dz_ds2 = hex_jacobian_component_scs<p, dk, ds2, ZH>(box, k, j, i);
  return ArrayND<ftype[3]>{
    {dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2, dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2,
     dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2}};
}

template <int p, int dk, typename BoxArray>
KOKKOS_FUNCTION ArrayND<ftype[3]>
laplacian_metric(const BoxArray& box, int k, int j, int i)
{
  enum { XH = 0, YH = 1, ZH = 2 };

  const auto jac = linear_hex_jacobian_scs<p, dk>(box, k, j, i);
  const auto adj_jac = adjugate_matrix(jac);
  const auto inv_detj =
    1.0 / (jac(XH, XH) * adj_jac(XH, XH) + jac(YH, XH) * adj_jac(YH, XH) +
           jac(ZH, XH) * adj_jac(ZH, XH));

  switch (dk) {
  case XH: {
    return ArrayND<ftype[3]>{
      {-inv_detj * (adj_jac(XH, XH) * adj_jac(XH, XH) +
                    adj_jac(XH, YH) * adj_jac(XH, YH) +
                    adj_jac(XH, ZH) * adj_jac(XH, ZH)),
       -inv_detj * (adj_jac(XH, XH) * adj_jac(YH, XH) +
                    adj_jac(XH, YH) * adj_jac(YH, YH) +
                    adj_jac(XH, ZH) * adj_jac(YH, ZH)),
       -inv_detj * (adj_jac(XH, XH) * adj_jac(ZH, XH) +
                    adj_jac(XH, YH) * adj_jac(ZH, YH) +
                    adj_jac(XH, ZH) * adj_jac(ZH, ZH))}};
  }
  case YH: {
    return ArrayND<ftype[3]>{
      {-inv_detj * (adj_jac(YH, XH) * adj_jac(YH, XH) +
                    adj_jac(YH, YH) * adj_jac(YH, YH) +
                    adj_jac(YH, ZH) * adj_jac(YH, ZH)),
       -inv_detj * (adj_jac(YH, XH) * adj_jac(XH, XH) +
                    adj_jac(YH, YH) * adj_jac(XH, YH) +
                    adj_jac(YH, ZH) * adj_jac(XH, ZH)),
       -inv_detj * (adj_jac(YH, XH) * adj_jac(ZH, XH) +
                    adj_jac(YH, YH) * adj_jac(ZH, YH) +
                    adj_jac(YH, ZH) * adj_jac(ZH, ZH))}};
  }
  default:
    return ArrayND<ftype[3]>{
      {-inv_detj * (adj_jac(ZH, XH) * adj_jac(ZH, XH) +
                    adj_jac(ZH, YH) * adj_jac(ZH, YH) +
                    adj_jac(ZH, ZH) * adj_jac(ZH, ZH)),
       -inv_detj * (adj_jac(ZH, XH) * adj_jac(XH, XH) +
                    adj_jac(ZH, YH) * adj_jac(XH, YH) +
                    adj_jac(ZH, ZH) * adj_jac(XH, ZH)),
       -inv_detj * (adj_jac(ZH, XH) * adj_jac(YH, XH) +
                    adj_jac(ZH, YH) * adj_jac(YH, YH) +
                    adj_jac(ZH, ZH) * adj_jac(YH, ZH))}};
  }
}

template <int p, int dj, int di, typename CoeffArray, typename BoxArray>
KOKKOS_FUNCTION typename BoxArray::value_type
hex_jacobian_component(
  const CoeffArray& Nlin, const BoxArray& box, int k, int j, int i)
{
  enum { LN = 0, RN = 1 };
  enum { XH = 0, YH = 1, ZH = 2 };
  if (dj == XH) {
    return (-Nlin(LN, j) * Nlin(LN, k) * box(di, 0) +
            Nlin(LN, j) * Nlin(LN, k) * box(di, 1) +
            Nlin(RN, j) * Nlin(LN, k) * box(di, 2) -
            Nlin(RN, j) * Nlin(LN, k) * box(di, 3) -
            Nlin(LN, j) * Nlin(RN, k) * box(di, 4) +
            Nlin(LN, j) * Nlin(RN, k) * box(di, 5) +
            Nlin(RN, j) * Nlin(RN, k) * box(di, 6) -
            Nlin(RN, j) * Nlin(RN, k) * box(di, 7)) *
           0.5;
  } else if (dj == YH) {
    return (-Nlin(LN, i) * Nlin(LN, k) * box(di, 0) -
            Nlin(RN, i) * Nlin(LN, k) * box(di, 1) +
            Nlin(RN, i) * Nlin(LN, k) * box(di, 2) +
            Nlin(LN, i) * Nlin(LN, k) * box(di, 3) -
            Nlin(LN, i) * Nlin(RN, k) * box(di, 4) -
            Nlin(RN, i) * Nlin(RN, k) * box(di, 5) +
            Nlin(RN, i) * Nlin(RN, k) * box(di, 6) +
            Nlin(LN, i) * Nlin(RN, k) * box(di, 7)) *
           0.5;
  } else {
    return (-Nlin(LN, i) * Nlin(LN, j) * box(di, 0) -
            Nlin(RN, i) * Nlin(LN, j) * box(di, 1) -
            Nlin(RN, i) * Nlin(RN, j) * box(di, 2) -
            Nlin(LN, i) * Nlin(RN, j) * box(di, 3) +
            Nlin(LN, i) * Nlin(LN, j) * box(di, 4) +
            Nlin(RN, i) * Nlin(LN, j) * box(di, 5) +
            Nlin(RN, i) * Nlin(RN, j) * box(di, 6) +
            Nlin(LN, i) * Nlin(RN, j) * box(di, 7)) *
           0.5;
  }
}

template <int p, typename BoxArray>
KOKKOS_FUNCTION ArrayND<typename BoxArray::value_type[3][3]>
linear_hex_jacobian(const BoxArray& box, int k, int j, int i)
{
  static constexpr auto coeff = Coeffs<p>::Nlin;
  enum { XH = 0, YH = 1, ZH = 2 };
  ArrayND<typename BoxArray::value_type[3][3]> jac;
  jac(0, 0) = hex_jacobian_component<p, XH, XH>(coeff, box, k, j, i);
  jac(0, 1) = hex_jacobian_component<p, XH, YH>(coeff, box, k, j, i);
  jac(0, 2) = hex_jacobian_component<p, XH, ZH>(coeff, box, k, j, i);
  jac(1, 0) = hex_jacobian_component<p, YH, XH>(coeff, box, k, j, i);
  jac(1, 1) = hex_jacobian_component<p, YH, YH>(coeff, box, k, j, i);
  jac(1, 2) = hex_jacobian_component<p, YH, ZH>(coeff, box, k, j, i);
  jac(2, 0) = hex_jacobian_component<p, ZH, XH>(coeff, box, k, j, i);
  jac(2, 1) = hex_jacobian_component<p, ZH, YH>(coeff, box, k, j, i);
  jac(2, 2) = hex_jacobian_component<p, ZH, ZH>(coeff, box, k, j, i);
  return jac;
}

} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
