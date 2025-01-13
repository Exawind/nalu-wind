// Copy1 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain 1s in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/MaxCourantReynolds.h"

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ShuffledAccess.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/ElementVolumeIntegral.h"
#include "matrix_free/ElementSCSInterpolate.h"
#include "ArrayND.h"
#include <KokkosInterface.h>
#include <Kokkos_NumericTraits.hpp>
#include <stk_simd/Simd.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

namespace {

template <int p, int dir, typename CoordArray, typename VelArray>
KOKKOS_FUNCTION ftype
compute_max_courant(double dt, const CoordArray& xc, const VelArray& vel)
{
  ftype max_courant = -1;
  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype udotx = 0;
        ftype dxsq = 0;
        for (int d = 0; d < 3; ++d) {
          const auto uip = interp_scs<p, dir>(vel, l, s, r, d);
          auto dx = shuffled_access<dir>(xc, s, r, l + 1, d) -
                    shuffled_access<dir>(xc, s, r, l + 0, d);
          udotx += uip * dx;
          dxsq += dx * dx;
        }
        const auto courant = dt * stk::math::abs(udotx) / dxsq;
        max_courant = stk::math::max(courant, max_courant);
      }
    }
  }
  return max_courant;
}

template <int p, typename CoordArray, typename VelArray>
KOKKOS_FUNCTION ftype
compute_max_courant(double dt, const CoordArray& xc, const VelArray& vel)
{
  return stk::math::max(
    stk::math::max(
      compute_max_courant<p, 0>(dt, xc, vel),
      compute_max_courant<p, 1>(dt, xc, vel)),
    compute_max_courant<p, 2>(dt, xc, vel));
}

template <
  int p,
  int dir,
  typename CoordArray,
  typename RhoArray,
  typename ViscArray,
  typename VelArray>
KOKKOS_FUNCTION ftype
compute_max_reynolds(
  const CoordArray& xc,
  const RhoArray& rho,
  const ViscArray& visc,
  const VelArray& vel)
{
  ftype max_reynolds = -1;
  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype udotx = 0;
        ftype dxsq = 0;
        for (int d = 0; d < 3; ++d) {
          const auto uip = interp_scs<p, dir>(vel, l, s, r, d);
          auto dx = shuffled_access<dir>(xc, s, r, l + 1, d) -
                    shuffled_access<dir>(xc, s, r, l + 0, d);
          udotx += uip * dx;
          dxsq += dx * dx;
        }

        ftype mu_div_rho = 0;
        {
          const auto ntlin = Coeffs<p>::Ntlin;
          mu_div_rho = ntlin(0, l) * (shuffled_access<dir>(visc, s, r, 0) /
                                      shuffled_access<dir>(rho, s, r, 0)) +
                       ntlin(1, l) * (shuffled_access<dir>(visc, s, r, p) /
                                      shuffled_access<dir>(rho, s, r, p));
        }
        constexpr double small = 1.e-16;
        const auto reynolds = stk::math::abs(udotx) / (mu_div_rho + small);
        max_reynolds = stk::math::max(reynolds, max_reynolds);
      }
    }
  }
  return max_reynolds;
}

template <
  int p,
  typename CoordArray,
  typename RhoArray,
  typename ViscArray,
  typename VelArray>
KOKKOS_FUNCTION ftype
compute_max_reynolds(
  const CoordArray& xc,
  const RhoArray& rho,
  const ViscArray& visc,
  const VelArray& vel)
{
  return stk::math::max(
    stk::math::max(
      compute_max_reynolds<p, 0>(xc, rho, visc, vel),
      compute_max_reynolds<p, 1>(xc, rho, visc, vel)),
    compute_max_reynolds<p, 2>(xc, rho, visc, vel));
}

} // namespace

template <int p>
Kokkos::Array<double, 2>
max_local_courant_reynolds_t<p>::invoke(
  double dt,
  const_vector_view<p> xc,
  const_scalar_view<p> rho,
  const_scalar_view<p> visc,
  const_vector_view<p> vel)
{
  Kokkos::pair<double, double> max_cflre;
  PairReduce<Kokkos::Max<double>> reducer(max_cflre);
  Kokkos::parallel_reduce(
    DeviceRangePolicy(0, xc.extent_int(0)),
    KOKKOS_LAMBDA(int index, Kokkos::pair<double, double>& val) {
      const auto elem_xc = Kokkos::subview(
        xc, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      const auto elem_vel = Kokkos::subview(
        vel, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      const auto elem_visc = Kokkos::subview(
        visc, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      const auto elem_rho = Kokkos::subview(
        rho, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      const auto max_elem_reynolds =
        compute_max_reynolds<p>(elem_xc, elem_rho, elem_visc, elem_vel);

      const auto max_elem_courant =
        compute_max_courant<p>(dt, elem_xc, elem_vel);

      for (int n = 0; n < simd_len; ++n) {
        val.first =
          stk::math::max(stk::simd::get_data(max_elem_courant, n), val.first);
        val.second =
          stk::math::max(stk::simd::get_data(max_elem_reynolds, n), val.second);
      }
    },
    reducer);
  return {{max_cflre.first, max_cflre.second}};
}
INSTANTIATE_POLYSTRUCT(max_local_courant_reynolds_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
