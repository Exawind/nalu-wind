// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionInterior.h"
#include "matrix_free/Coefficients.h"
#include "matrix_free/ElementFluxIntegral.h"
#include "matrix_free/ElementVolumeIntegral.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"

#include <KokkosInterface.h>
#include <Kokkos_ScatterView.hpp>
#include "stk_mesh/base/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

template <int p>
void
conduction_residual_t<p>::invoke(
  Kokkos::Array<double, 3> gammas,
  const_elem_offset_view<p> offsets,
  const_scalar_view<p> qm1,
  const_scalar_view<p> qp0,
  const_scalar_view<p> qp1,
  const_scalar_view<p> volume_metric,
  const_scs_vector_view<p> diffusion_metric,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("conduction_residual");

  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    "conduction_residual",
    Kokkos::RangePolicy<exec_space, int>(0, offsets.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      narray element_rhs;
      if (p > 1) {
        consistent_mass_time_derivative<p>(
          index, gammas, volume_metric, qm1, qp0, qp1, element_rhs);
      } else {
        lumped_time_derivative<p>(
          index, gammas, volume_metric, qm1, qp0, qp1, element_rhs);
      }

      {
        auto qvec = Kokkos::subview(
          qp1, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        diffusive_flux<p, 0>(index, diffusion_metric, qvec, element_rhs);
        diffusive_flux<p, 1>(index, diffusion_metric, qvec, element_rhs);
        diffusive_flux<p, 2>(index, diffusion_metric, qvec, element_rhs);
      }

      const auto valid_length = valid_offset<p>(index, offsets);
      auto accessor = yout_scatter.access();
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < valid_length; ++n) {
              accessor(offsets(index, k, j, i, n), 0) +=
                stk::simd::get_data(element_rhs(k, j, i), n);
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(conduction_residual_t);

template <int p>
void
conduction_linearized_residual_t<p>::invoke(
  double gamma,
  const_elem_offset_view<p> offsets,
  const_scalar_view<p> volume_metric,
  const_scs_vector_view<p> diffusion_metric,
  ra_tpetra_view_type xin,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("conduction_linearized_residual");

  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    "conduction_linop", offsets.extent_int(0), KOKKOS_LAMBDA(int index) {
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

      narray element_rhs;
      if (p > 1) {
        narray scratch;
        mass_term<p>(index, gamma, volume_metric, delta, scratch, element_rhs);
      } else {
        lumped_mass_term<p>(index, gamma, volume_metric, delta, element_rhs);
      }

      diffusive_flux<p, 0>(index, diffusion_metric, delta, element_rhs);
      diffusive_flux<p, 1>(index, diffusion_metric, delta, element_rhs);
      diffusive_flux<p, 2>(index, diffusion_metric, delta, element_rhs);

      auto accessor = yout_scatter.access();
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < valid_length; ++n) {
              accessor(idx(k, j, i, n), 0) -=
                stk::simd::get_data(element_rhs(k, j, i), n);
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(conduction_linearized_residual_t);

} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
