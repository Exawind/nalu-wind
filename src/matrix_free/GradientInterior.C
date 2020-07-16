// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/GreenGaussGradientInterior.h"
#include "matrix_free/Coefficients.h"
#include "matrix_free/ElementFluxIntegral.h"
#include "matrix_free/ElementVolumeIntegral.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

#include "Kokkos_ScatterView.hpp"
#include "stk_mesh/base/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

constexpr bool lumped_rhs = false;
constexpr bool lumped_lhs = false;

template <int p>
void
gradient_residual_t<p>::invoke(
  const const_elem_offset_view<p> offsets,
  const const_scs_vector_view<p> areas,
  const const_scalar_view<p> vols,
  const const_scalar_view<p> q,
  const const_vector_view<p> dqdx_predicted,
  typename Tpetra::MultiVector<>::dual_view_type::t_dev yout)
{
  stk::mesh::ProfilingBlock pf("gradient_residual");
  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    offsets.extent_int(0), KOKKOS_LAMBDA(int index) {
      for (int d = 0; d < 3; ++d) {
        LocalArray<ftype[p + 1][p + 1][p + 1]> rhs;
        if (lumped_rhs) {
          for (int k = 0; k < p + 1; ++k) {
            for (int j = 0; j < p + 1; ++j) {
              for (int i = 0; i < p + 1; ++i) {
                constexpr auto Wl = Coeffs<p>::Wl;
                rhs(k, j, i) = -Wl(k) * Wl(j) * Wl(i) * vols(index, k, j, i) *
                               dqdx_predicted(index, k, j, i, d);
              }
            }
          }
        } else {
          LocalArray<ftype[p + 1][p + 1][p + 1]> scratch;
          for (int k = 0; k < p + 1; ++k) {
            for (int j = 0; j < p + 1; ++j) {
              for (int i = 0; i < p + 1; ++i) {
                scratch(k, j, i) =
                  -vols(index, k, j, i) * dqdx_predicted(index, k, j, i, d);
              }
            }
          }
          edge_integral<p, 0>(scratch, rhs);
          edge_integral<p, 1>(rhs, scratch);
          edge_integral<p, 2>(scratch, rhs);
        }

        {
          const auto qv =
            Kokkos::subview(q, index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
          scalar_flux_vector<p, 0>(index, d, areas, qv, rhs);
          scalar_flux_vector<p, 1>(index, d, areas, qv, rhs);
          scalar_flux_vector<p, 2>(index, d, areas, qv, rhs);
        }

        const auto valid_length = valid_offset<p>(index, offsets);
        auto accessor = yout_scatter.access();
        for (int k = 0; k < p + 1; ++k) {
          for (int j = 0; j < p + 1; ++j) {
            for (int i = 0; i < p + 1; ++i) {
              for (int n = 0; n < valid_length; ++n) {
                accessor(offsets(index, k, j, i, n), d) +=
                  stk::simd::get_data(rhs(k, j, i), n);
              }
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(gradient_residual_t);

template <int p>
void
filter_linearized_residual_t<p>::invoke(
  const_elem_offset_view<p> offsets,
  const_scalar_view<p> vols,
  ra_tpetra_view_type xin,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("gradient_linearized_residual");

  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    offsets.extent_int(0), KOKKOS_LAMBDA(int index) {
      for (int d = 0; d < 3; ++d) {
        LocalArray<ftype[p + 1][p + 1][p + 1]> delta;
        LocalArray<int[p + 1][p + 1][p + 1][simd_len]> idx;
        const auto length = valid_offset<p>(index, offsets);

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

        LocalArray<ftype[p + 1][p + 1][p + 1]> rhs;

        if (lumped_lhs) {
          for (int k = 0; k < p + 1; ++k) {
            for (int j = 0; j < p + 1; ++j) {
              for (int i = 0; i < p + 1; ++i) {
                constexpr auto Wl = Coeffs<p>::Wl;
                rhs(k, j, i) =
                  Wl(k) * Wl(j) * Wl(i) * vols(index, k, j, i) * delta(k, j, i);
              }
            }
          }
        } else {
          {
            LocalArray<ftype[p + 1][p + 1][p + 1]> scratch;
            for (int k = 0; k < p + 1; ++k) {
              for (int j = 0; j < p + 1; ++j) {
                for (int i = 0; i < p + 1; ++i) {
                  scratch(k, j, i) = -vols(index, k, j, i) * delta(k, j, i);
                }
              }
            }
            edge_integral<p, 0>(scratch, rhs);
            edge_integral<p, 1>(rhs, scratch);
            edge_integral<p, 2>(scratch, rhs);
          }
        }

        auto accessor = yout_scatter.access();
        for (int k = 0; k < p + 1; ++k) {
          for (int j = 0; j < p + 1; ++j) {
            for (int i = 0; i < p + 1; ++i) {
              for (int n = 0; n < length; ++n) {
                const auto id = idx(k, j, i, n);
                accessor(id, d) -= stk::simd::get_data(rhs(k, j, i), n);
              }
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(filter_linearized_residual_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra