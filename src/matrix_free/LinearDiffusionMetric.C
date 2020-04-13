// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearDiffusionMetric.h"

#include <Kokkos_DualView.hpp>
#include <Kokkos_Macros.hpp>

#include "matrix_free/Coefficients.h"
#include "matrix_free/ElementSCSInterpolate.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/TensorOperations.h"
#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {
namespace impl {

namespace {
#define LN 0
#define RN 1
#define XH 0
#define YH 1
#define ZH 2

template <
  int p,
  int dk,
  int dj,
  int di,
  typename BoxArray,
  typename SCSCoeff,
  typename NodalCoeff>
KOKKOS_FUNCTION typename BoxArray::value_type
hex_jacobian_component_scs(
  const SCSCoeff& ntlin,
  const NodalCoeff& nlin,
  const BoxArray& box,
  int k,
  int j,
  int i)
{
  typename BoxArray::value_type jac(0);
  switch (dj) {
  case XH: {
    const double lj = (dk == YH) ? ntlin(LN, j) : nlin(LN, j);
    const double rj = (dk == YH) ? ntlin(RN, j) : nlin(RN, j);

    const double lk = (dk == ZH) ? ntlin(LN, k) : nlin(LN, k);
    const double rk = (dk == ZH) ? ntlin(RN, k) : nlin(RN, k);
    jac = -lj * lk * box(di, 0) + lj * lk * box(di, 1) + rj * lk * box(di, 2) -
          rj * lk * box(di, 3) - lj * rk * box(di, 4) + lj * rk * box(di, 5) +
          rj * rk * box(di, 6) - rj * rk * box(di, 7);
    break;
  }
  case YH: {
    const double li = (dk == XH) ? ntlin(LN, i) : nlin(LN, i);
    const double ri = (dk == XH) ? ntlin(RN, i) : nlin(RN, i);

    const double lk = (dk == ZH) ? ntlin(LN, k) : nlin(LN, k);
    const double rk = (dk == ZH) ? ntlin(RN, k) : nlin(RN, k);

    jac = -li * lk * box(di, 0) - ri * lk * box(di, 1) + ri * lk * box(di, 2) +
          li * lk * box(di, 3) - li * rk * box(di, 4) - ri * rk * box(di, 5) +
          ri * rk * box(di, 6) + li * rk * box(di, 7);
    break;
  }
  case ZH: {
    const double li = (dk == XH) ? ntlin(LN, i) : nlin(LN, i);
    const double ri = (dk == XH) ? ntlin(RN, i) : nlin(RN, i);

    const double lj = (dk == YH) ? ntlin(LN, j) : nlin(LN, j);
    const double rj = (dk == YH) ? ntlin(RN, j) : nlin(RN, j);
    jac = -li * lj * box(di, 0) - ri * lj * box(di, 1) - ri * rj * box(di, 2) -
          li * rj * box(di, 3) + li * lj * box(di, 4) + ri * lj * box(di, 5) +
          ri * rj * box(di, 6) + li * rj * box(di, 7);
    break;
  }
  default:
    break;
  }
  constexpr double isoParametricFactor = 0.5;
  return jac * isoParametricFactor;
}

#undef LN
#undef RN

template <
  int p,
  int dk,
  typename BoxArray,
  typename SCSCoeff,
  typename NodalCoeff>
KOKKOS_FUNCTION LocalArray<typename BoxArray::value_type[3][3]>
linear_hex_jacobian_scs(
  const SCSCoeff& ntlin,
  const NodalCoeff& nlin,
  const BoxArray& box,
  int k,
  int j,
  int i)
{
  return {
    {{hex_jacobian_component_scs<p, dk, XH, XH>(ntlin, nlin, box, k, j, i),
      hex_jacobian_component_scs<p, dk, XH, YH>(ntlin, nlin, box, k, j, i),
      hex_jacobian_component_scs<p, dk, XH, ZH>(ntlin, nlin, box, k, j, i)},
     {
       hex_jacobian_component_scs<p, dk, YH, XH>(ntlin, nlin, box, k, j, i),
       hex_jacobian_component_scs<p, dk, YH, YH>(ntlin, nlin, box, k, j, i),
       hex_jacobian_component_scs<p, dk, YH, ZH>(ntlin, nlin, box, k, j, i),
     },
     {
       hex_jacobian_component_scs<p, dk, ZH, XH>(ntlin, nlin, box, k, j, i),
       hex_jacobian_component_scs<p, dk, ZH, YH>(ntlin, nlin, box, k, j, i),
       hex_jacobian_component_scs<p, dk, ZH, ZH>(ntlin, nlin, box, k, j, i),
     }}};
}

} // namespace

template <int p>
scs_vector_view<p>
diffusion_metric_t<p>::invoke(
  const const_scalar_view<p> alpha, const const_vector_view<p> coordinates)
{
  scs_vector_view<p> metric("diffusion", coordinates.extent_int(0));
  Kokkos::parallel_for(
    "diffusion_metric", coordinates.extent_int(0), KOKKOS_LAMBDA(int index) {
      static constexpr auto ntilde = Coeffs<p>::Nt;
      static constexpr auto nlin = Coeffs<p>::Nlin;
      static constexpr auto ntlin = Coeffs<p>::Ntlin;

      const auto box = hex_vertex_coordinates<p>(index, coordinates);

      LocalArray<ftype[3][p + 1][p + 1][p + 1]> interp;
      {
        const auto alpha_elem = Kokkos::subview(
          alpha, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        interp_scs<p>(alpha_elem, ntilde, interp);
      }

      for (int l = 0; l < p; ++l) {
        for (int s = 0; s < p + 1; ++s) {
          for (int r = 0; r < p + 1; ++r) {
            const auto jac =
              linear_hex_jacobian_scs<p, XH>(ntlin, nlin, box, s, r, l);
            const auto adj_jac = adjugate_matrix(jac);
            const auto inv_detj =
              interp(XH, l, s, r) /
              (jac(0, 0) * adj_jac(0, 0) + jac(1, 0) * adj_jac(1, 0) +
               jac(2, 0) * adj_jac(2, 0));

            metric(index, XH, l, s, r, 0) =
              -inv_detj * (adj_jac(XH, XH) * adj_jac(XH, XH) +
                           adj_jac(XH, YH) * adj_jac(XH, YH) +
                           adj_jac(XH, ZH) * adj_jac(XH, ZH));
            metric(index, XH, l, s, r, 1) =
              -inv_detj * (adj_jac(XH, XH) * adj_jac(YH, XH) +
                           adj_jac(XH, YH) * adj_jac(YH, YH) +
                           adj_jac(XH, ZH) * adj_jac(YH, ZH));
            metric(index, XH, l, s, r, 2) =
              -inv_detj * (adj_jac(XH, XH) * adj_jac(ZH, XH) +
                           adj_jac(XH, YH) * adj_jac(ZH, YH) +
                           adj_jac(XH, ZH) * adj_jac(ZH, ZH));
          }
        }
      }

      for (int l = 0; l < p; ++l) {
        for (int s = 0; s < p + 1; ++s) {
          for (int r = 0; r < p + 1; ++r) {
            const auto jac =
              linear_hex_jacobian_scs<p, YH>(ntlin, nlin, box, s, l, r);
            const auto adj_jac = adjugate_matrix(jac);

            const auto inv_detj =
              interp(YH, l, s, r) /
              (jac(0, 0) * adj_jac(0, 0) + jac(1, 0) * adj_jac(1, 0) +
               jac(2, 0) * adj_jac(2, 0));

            metric(index, YH, l, s, r, 0) =
              -inv_detj * (adj_jac(YH, XH) * adj_jac(YH, XH) +
                           adj_jac(YH, YH) * adj_jac(YH, YH) +
                           adj_jac(YH, ZH) * adj_jac(YH, ZH));
            metric(index, YH, l, s, r, 1) =
              -inv_detj * (adj_jac(YH, XH) * adj_jac(XH, XH) +
                           adj_jac(YH, YH) * adj_jac(XH, YH) +
                           adj_jac(YH, ZH) * adj_jac(XH, ZH));
            metric(index, YH, l, s, r, 2) =
              -inv_detj * (adj_jac(YH, XH) * adj_jac(ZH, XH) +
                           adj_jac(YH, YH) * adj_jac(ZH, YH) +
                           adj_jac(YH, ZH) * adj_jac(ZH, ZH));
          }
        }
      }

      for (int l = 0; l < p; ++l) {
        for (int s = 0; s < p + 1; ++s) {
          for (int r = 0; r < p + 1; ++r) {
            const auto jac =
              linear_hex_jacobian_scs<p, ZH>(ntlin, nlin, box, l, s, r);
            const auto adj_jac = adjugate_matrix(jac);

            const auto inv_detj =
              interp(ZH, l, s, r) /
              (jac(0, 0) * adj_jac(0, 0) + jac(1, 0) * adj_jac(1, 0) +
               jac(2, 0) * adj_jac(2, 0));

            metric(index, ZH, l, s, r, 0) =
              -inv_detj * (adj_jac(ZH, XH) * adj_jac(ZH, XH) +
                           adj_jac(ZH, YH) * adj_jac(ZH, YH) +
                           adj_jac(ZH, ZH) * adj_jac(ZH, ZH));
            metric(index, ZH, l, s, r, 1) =
              -inv_detj * (adj_jac(ZH, XH) * adj_jac(XH, XH) +
                           adj_jac(ZH, YH) * adj_jac(XH, YH) +
                           adj_jac(ZH, ZH) * adj_jac(XH, ZH));
            metric(index, ZH, l, s, r, 2) =
              -inv_detj * (adj_jac(ZH, XH) * adj_jac(YH, XH) +
                           adj_jac(ZH, YH) * adj_jac(YH, YH) +
                           adj_jac(ZH, ZH) * adj_jac(YH, ZH));
          }
        }
      }
    });
  return metric;
}

#undef XH
#undef YH
#undef ZH

INSTANTIATE_POLYSTRUCT(diffusion_metric_t);

} // namespace impl
} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
