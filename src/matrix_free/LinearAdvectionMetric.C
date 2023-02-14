// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearAdvectionMetric.h"

#include "matrix_free/Coefficients.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LocalArray.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ShuffledAccess.h"

#include <KokkosInterface.h>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {
namespace impl {

namespace {

template <int p, int dir, typename PressureArray, typename MomArray>
KOKKOS_FUNCTION void
corrected_momentum_flux_coefficient(
  double scaling,
  int index,
  const_scs_vector_view<p> areav,
  const_scs_vector_view<p> laplacian_metric,
  const PressureArray& pressure,
  const MomArray& rhou_corr,
  scs_scalar_view<p> mdot)
{
  enum { XH = 0, YH = 1, ZH = 2 };

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc_x(0);
        ftype acc_y(0);
        ftype acc_z(0);
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto interp = Coeffs<p>::Nt;
          const auto interp_val = interp(l, q);
          acc_x += interp_val * shuffled_access<dir>(rhou_corr, s, r, q, XH);
          acc_y += interp_val * shuffled_access<dir>(rhou_corr, s, r, q, YH);
          acc_z += interp_val * shuffled_access<dir>(rhou_corr, s, r, q, ZH);
        }
        mdot(index, dir, l, s, r) = acc_x * areav(index, dir, l, s, r, XH) +
                                    acc_y * areav(index, dir, l, s, r, YH) +
                                    acc_z * areav(index, dir, l, s, r, ZH);
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    LocalArray<ftype[p + 1][p + 1]> scratch;
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto flux_point_interpolant = Coeffs<p>::Nt;
          acc += flux_point_interpolant(l, q) *
                 shuffled_access<dir>(pressure, s, r, q);
        }
        scratch(s, r) = acc;
      }
    }

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc_x = 0;
        ftype acc_y = 0;
        ftype acc_z = 0;

        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto flux_point_derivative = Coeffs<p>::Dt;
          static constexpr auto nodal_derivative = Coeffs<p>::D;
          acc_x += flux_point_derivative(l, q) *
                   shuffled_access<dir>(pressure, s, r, q);
          acc_y += nodal_derivative(r, q) * scratch(s, q);
          acc_z += nodal_derivative(s, q) * scratch(q, r);
        }
        mdot(index, dir, l, s, r) -=
          scaling * (acc_x * laplacian_metric(index, dir, l, s, r, 0) +
                     acc_y * laplacian_metric(index, dir, l, s, r, 1) +
                     acc_z * laplacian_metric(index, dir, l, s, r, 2));
      }
    }
  }
}

} // namespace

template <int p>
void
linear_advection_metric_t<p>::invoke(
  double scaling,
  const_scs_vector_view<p> areas,
  const_scs_vector_view<p> laplacian_metric,
  scalar_view<p> density,
  vector_view<p> velocity,
  vector_view<p> proj_pressure_gradient,
  scalar_view<p> pressure,
  scs_scalar_view<p>& mdot)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  Kokkos::parallel_for(
    DeviceRangePolicy(0, areas.extent_int(0)), KOKKOS_LAMBDA(int index) {
      LocalArray<ftype[p + 1][p + 1][p + 1][3]> rhou_corr;
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            const auto rho = density(index, k, j, i);
            for (int d = 0; d < 3; ++d) {
              rhou_corr(k, j, i, d) =
                rho * velocity(index, k, j, i, d) +
                scaling * proj_pressure_gradient(index, k, j, i, d);
            }
          }
        }
      }

      auto pvec = Kokkos::subview(
        pressure, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      corrected_momentum_flux_coefficient<p, 0>(
        scaling, index, areas, laplacian_metric, pvec, rhou_corr, mdot);
      corrected_momentum_flux_coefficient<p, 1>(
        scaling, index, areas, laplacian_metric, pvec, rhou_corr, mdot);
      corrected_momentum_flux_coefficient<p, 2>(
        scaling, index, areas, laplacian_metric, pvec, rhou_corr, mdot);
    });
}
INSTANTIATE_POLYSTRUCT(linear_advection_metric_t);

} // namespace impl
} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
