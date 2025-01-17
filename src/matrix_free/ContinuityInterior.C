// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ContinuityInterior.h"

#include "matrix_free/Coefficients.h"
#include "matrix_free/ElementFluxIntegral.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"

#include <KokkosInterface.h>
#include "Kokkos_ScatterView.hpp"

#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "stk_simd/Simd.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

template <int p>
void
continuity_residual_t<p>::invoke(
  double scaling,
  const_elem_offset_view<p> offsets,
  const_scs_scalar_view<p> mdot,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("continuity_residual");

  const auto inv_scaling = 1.0 / scaling;
  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    DeviceRangePolicy(0, offsets.extent_int(0)), KOKKOS_LAMBDA(int index) {
      ArrayND<ftype[p + 1][p + 1][p + 1]> elem_rhs;
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            elem_rhs(k, j, i) = 0;
          }
        }
      }
      scalar_flux_divergence<p, 0>(index, mdot, elem_rhs);
      scalar_flux_divergence<p, 1>(index, mdot, elem_rhs);
      scalar_flux_divergence<p, 2>(index, mdot, elem_rhs);

      const auto valid_length = valid_offset<p>(index, offsets);
      auto accessor = yout_scatter.access();
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            const auto val = -inv_scaling * elem_rhs(k, j, i);
            for (int n = 0; n < valid_length; ++n) {
              const int idx = offsets(index, k, j, i, n);
              accessor(idx, 0) += stk::simd::get_data(val, n);
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(continuity_residual_t);

template <int p>
void
continuity_linearized_residual_t<p>::invoke(
  const_elem_offset_view<p> offsets,
  const_scs_vector_view<p> metric,
  ra_tpetra_view_type xin,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("continuity_linearized_residual");

  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    DeviceRangePolicy(0, offsets.extent_int(0)), KOKKOS_LAMBDA(int index) {
      narray delta;
      ArrayND<int[p + 1][p + 1][p + 1][simd_len]> idx;
      const auto valid_length = valid_offset<p>(index, offsets);
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < valid_length; ++n) {
              idx(k, j, i, n) = offsets(index, k, j, i, n);
              stk::simd::set_data(delta(k, j, i), n, xin(idx(k, j, i, n), 0));
            }
          }
        }
      }

      narray elem_rhs;
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            elem_rhs(k, j, i) = 0;
          }
        }
      }
      diffusive_flux<p, 0>(index, metric, delta, elem_rhs);
      diffusive_flux<p, 1>(index, metric, delta, elem_rhs);
      diffusive_flux<p, 2>(index, metric, delta, elem_rhs);

      auto accessor = yout_scatter.access();
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            const auto val = elem_rhs(k, j, i);
            for (int n = 0; n < valid_length; ++n) {
              accessor(idx(k, j, i, n), 0) -= stk::simd::get_data(val, n);
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(continuity_linearized_residual_t);

} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
