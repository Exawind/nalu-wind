// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/MomentumInterior.h"

#include "matrix_free/ElementGradient.h"
#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/Coefficients.h"
#include "matrix_free/ElementFluxIntegral.h"
#include "matrix_free/ElementVolumeIntegral.h"
#include "matrix_free/GeometricFunctions.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ShuffledAccess.h"
#include "matrix_free/TensorOperations.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LocalArray.h"
#include "matrix_free/ElementSCSInterpolate.h"
#include <KokkosInterface.h>

#include "Kokkos_ScatterView.hpp"
#include "stk_mesh/base/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

namespace {

template <
  int p,
  int dir,
  typename BoxArrayType,
  typename AdvArrayType,
  typename ViscosityArrayType,
  typename UArrayType>
KOKKOS_FORCEINLINE_FUNCTION LocalArray<ftype[3]>
momentum_flux(
  const BoxArrayType& box,
  const AdvArrayType& adv,
  const ViscosityArrayType& visc,
  const UArrayType& u,
  const LocalArray<ftype[p + 1][p + 1][3]>& uhat,
  int l,
  int s,
  int r)
{
  enum { XH = 0, YH = 1, ZH = 2 };

  const auto gu = gradient_scs<p, dir>(box, u, uhat, l, s, r);
  const auto mdot_h = adv(dir, l, s, r);
  const auto visc_ip = stk::math::max(0, interp_scs<p, dir>(visc, l, s, r));
  const auto areav = geom::linear_area<p, dir>(box, l, s, r);

  const auto one_third_divu =
    (1. / 3.) * (gu(XH, XH) + gu(YH, YH) + gu(ZH, ZH));

  LocalArray<ftype[3]> flux;
  flux(0) = 2 * visc_ip *
              ((gu(XH, XH) - one_third_divu) * areav(XH) +
               0.5 * (gu(XH, YH) + gu(YH, XH)) * areav(YH) +
               0.5 * (gu(XH, ZH) + gu(ZH, XH)) * areav(ZH)) -
            mdot_h * uhat(s, r, 0);

  flux(1) = 2 * visc_ip *
              (0.5 * (gu(YH, XH) + gu(XH, YH)) * areav(XH) +
               (gu(YH, YH) - one_third_divu) * areav(YH) +
               0.5 * (gu(YH, ZH) + gu(ZH, YH)) * areav(ZH)) -
            mdot_h * uhat(s, r, 1);

  flux(2) = 2 * visc_ip *
              (0.5 * (gu(ZH, XH) + gu(XH, ZH)) * areav(XH) +
               0.5 * (gu(ZH, YH) + gu(YH, ZH)) * areav(YH) +
               (gu(ZH, ZH) - one_third_divu) * areav(ZH)) -
            mdot_h * uhat(s, r, 2);

  return flux;
}

template <
  int p,
  int dir,
  typename BoxArrayType,
  typename AdvArrayType,
  typename ViscosityArrayType,
  typename VelocityArrayType,
  typename RHSArrayType>
KOKKOS_FUNCTION void
momentum_flux_difference(
  const BoxArrayType& box,
  const AdvArrayType& adv,
  const ViscosityArrayType& visc,
  const VelocityArrayType& vel,
  RHSArrayType& rhs)
{
  for (int l = 0; l < p; ++l) {
    LocalArray<ftype[p + 1][p + 1][3]> flux;
    {
      LocalArray<ftype[p + 1][p + 1][3]> uhat;
      for (int s = 0; s < p + 1; ++s) {
        for (int r = 0; r < p + 1; ++r) {
          for (int d = 0; d < 3; ++d) {
            uhat(s, r, d) = interp_scs<p, dir>(vel, l, s, r, d);
          }
        }
      }

      for (int s = 0; s < p + 1; ++s) {
        for (int r = 0; r < p + 1; ++r) {
          const auto area_weighted_flux =
            momentum_flux<p, dir>(box, adv, visc, vel, uhat, l, s, r);

          flux(s, r, 0) = area_weighted_flux(0);
          flux(s, r, 1) = area_weighted_flux(1);
          flux(s, r, 2) = area_weighted_flux(2);
        }
      }
    }
    for (int d = 0; d < 3; ++d) {
      LocalArray<ftype[p + 1][p + 1]> fbar;
      for (int s = 0; s < p + 1; ++s) {
        for (int r = 0; r < p + 1; ++r) {
          ftype acc = 0;
          for (int q = 0; q < p + 1; ++q) {
            static constexpr auto vandermonde = Coeffs<p>::W;
            acc += vandermonde(r, q) * flux(s, q, d);
          }
          fbar(s, r) = acc;
        }
      }
      for (int s = 0; s < p + 1; ++s) {
        for (int r = 0; r < p + 1; ++r) {
          ftype acc = 0;
          for (int q = 0; q < p + 1; ++q) {
            static constexpr auto vandermonde = Coeffs<p>::W;
            acc += vandermonde(s, q) * fbar(q, r);
          }
          shuffled_access<dir>(rhs, s, r, l + 0, d) -= acc;
          shuffled_access<dir>(rhs, s, r, l + 1, d) += acc;
        }
      }
    }
  }
}

template <int p, typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
momentum_mass(
  int index,
  const Kokkos::Array<double, 3> gammas,
  const const_scalar_view<p>& rho,
  const const_scalar_view<p>& vm1,
  const const_scalar_view<p>& vp0,
  const const_scalar_view<p>& vp1,
  const const_vector_view<p>& um1,
  const const_vector_view<p>& up0,
  const const_vector_view<p>& up1,
  const const_vector_view<p>& gp,
  const const_vector_view<p>& force,
  OutArray& out)
{
  static constexpr auto vandermonde = Coeffs<p>::W;
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      LocalArray<ftype[p + 1][3]> scratch;
      for (int i = 0; i < p + 1; ++i) {
        const auto vm1_val = vm1(index, k, j, i);
        const auto vp0_val = vp0(index, k, j, i);
        const auto vp1_val = vp1(index, k, j, i);
        const auto vol = vp1_val / rho(index, k, j, i);
        for (int d = 0; d < 3; ++d) {
          scratch(i, d) =
            -(gammas[0] * vp1_val * up1(index, k, j, i, d) +
              gammas[1] * vp0_val * up0(index, k, j, i, d) +
              gammas[2] * vm1_val * um1(index, k, j, i, d) +
              vol * (gp(index, k, j, i, d) - force(index, k, j, i, d)));
        }
      }

      for (int i = 0; i < p + 1; ++i) {
        for (int d = 0; d < 3; ++d) {
          ftype acc(0);
          for (int q = 0; q < p + 1; ++q) {
            acc += vandermonde(i, q) * scratch(q, d);
          }
          out(k, j, i, d) = acc;
        }
      }
    }
  }

  for (int d = 0; d < 3; ++d) {
    for (int i = 0; i < p + 1; ++i) {
      LocalArray<ftype[p + 1][p + 1]> scratch;
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          ftype acc(0);
          for (int q = 0; q < p + 1; ++q) {
            acc += vandermonde(j, q) * out(k, q, i, d);
          }
          scratch(k, j) = acc;
        }
      }

      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          ftype acc(0);
          for (int q = 0; q < p + 1; ++q) {
            acc += vandermonde(k, q) * scratch(q, j);
          }
          out(k, j, i, d) = acc;
        }
      }
    }
  }
}
} // namespace

template <int p>
void
momentum_residual_t<p>::invoke(
  Kokkos::Array<double, 3> gammas,
  const_elem_offset_view<p> offsets,
  const_vector_view<p> xc,
  const_scalar_view<p> rho,
  const_scalar_view<p> visc,
  const_scalar_view<p> vm1,
  const_scalar_view<p> vp0,
  const_scalar_view<p> vp1,
  const_vector_view<p> um1,
  const_vector_view<p> up0,
  const_vector_view<p> up1,
  const_vector_view<p> gp,
  const_vector_view<p> force,
  const_scs_scalar_view<p> mdot,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("momentum_residual");
  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    DeviceRangePolicy(0, offsets.extent_int(0)), KOKKOS_LAMBDA(int index) {
      LocalArray<ftype[p + 1][p + 1][p + 1][3]> elem_rhs;
      if (p == 1) {
        static constexpr auto lumped = Coeffs<p>::Wl;
        for (int k = 0; k < p + 1; ++k) {
          const auto Wk = lumped[k];
          for (int j = 0; j < p + 1; ++j) {
            const auto WkWj = Wk * lumped[j];
            for (int i = 0; i < p + 1; ++i) {
              const auto fac = -WkWj * lumped[i];
              const auto scaled_vp1 = fac * vp1(index, k, j, i);
              const auto scaled_vp0 = fac * vp0(index, k, j, i);
              const auto scaled_vm1 = fac * vm1(index, k, j, i);
              const auto vol = scaled_vp1 / rho(index, k, j, i);
              for (int d = 0; d < 3; ++d) {
                elem_rhs(k, j, i, d) =
                  gammas[0] * scaled_vp1 * up1(index, k, j, i, d) +
                  gammas[1] * scaled_vp0 * up0(index, k, j, i, d) +
                  gammas[2] * scaled_vm1 * um1(index, k, j, i, d) +
                  vol * (gp(index, k, j, i, d) - force(index, k, j, i, d));
              }
            }
          }
        }
      } else {
        momentum_mass<p>(
          index, gammas, rho, vm1, vp0, vp1, um1, up0, up1, gp, force,
          elem_rhs);
      }

      auto uvec = Kokkos::subview(
        up1, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      auto muvec = Kokkos::subview(
        visc, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      auto mdotvec = Kokkos::subview(
        mdot, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
        Kokkos::ALL());

      const auto box = hex_vertex_coordinates<p>(index, xc);
      momentum_flux_difference<p, 0>(box, mdotvec, muvec, uvec, elem_rhs);
      momentum_flux_difference<p, 1>(box, mdotvec, muvec, uvec, elem_rhs);
      momentum_flux_difference<p, 2>(box, mdotvec, muvec, uvec, elem_rhs);
      const auto valid_length = valid_offset<p>(index, offsets);

      auto accessor = yout_scatter.access();
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < valid_length; ++n) {
              auto id = offsets(index, k, j, i, n);
              for (int d = 0; d < 3; ++d) {
                accessor(id, d) += stk::simd::get_data(elem_rhs(k, j, i, d), n);
              }
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(momentum_residual_t);

template <int p>
void
momentum_linearized_residual_t<p>::invoke(
  double gamma_0,
  const_elem_offset_view<p> offsets,
  const_scalar_view<p> vp1,
  const_scs_scalar_view<p> mdot,
  const_scs_vector_view<p> diff,
  ra_tpetra_view_type xin,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("momentum_linearized_residual");

  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);

#if defined(KOKKOS_ENABLE_HIP)
  using policy_type = Kokkos::MDRangePolicy<
    exec_space, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
    Kokkos::Rank<2>, int>;
#else
  using policy_type = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>, int>;
#endif
  const auto range = policy_type({0, 0}, {offsets.extent_int(0), 3});
  Kokkos::parallel_for(range, KOKKOS_LAMBDA(int index, int d) {
    const auto length = valid_offset<p>(index, offsets);
    LocalArray<int[p + 1][p + 1][p + 1][simd_len]> idx;
    narray delta;
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          for (int n = 0; n < length; ++n) {
            idx(k, j, i, n) = offsets(index, k, j, i, n);
            stk::simd::set_data(delta(k, j, i), n, xin(idx(k, j, i, n), d));
          }
        }
      }
    }

    narray elem_rhs;
    if (p == 1) {
      lumped_mass_term<p>(index, gamma_0, vp1, delta, elem_rhs);
    } else {
      apply_mass<p>(index, gamma_0, vp1, delta, elem_rhs);
    }
    advdiff_flux<p, 0>(index, mdot, diff, delta, elem_rhs);
    advdiff_flux<p, 1>(index, mdot, diff, delta, elem_rhs);
    advdiff_flux<p, 2>(index, mdot, diff, delta, elem_rhs);

    auto accessor = yout_scatter.access();
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          for (int n = 0; n < length; ++n) {
            accessor(idx(k, j, i, n), d) -=
              stk::simd::get_data(elem_rhs(k, j, i), n);
          }
        }
      }
    }
  });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(momentum_linearized_residual_t);

} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
