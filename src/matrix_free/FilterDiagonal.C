// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/FilterDiagonal.h"

#include "matrix_free/Coefficients.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LocalArray.h"
#include "matrix_free/LinSysInfo.h"
#include <KokkosInterface.h>

#include "Kokkos_ScatterView.hpp"

#include "stk_mesh/base/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {

template <int p>
void
filter_diagonal_t<p>::invoke(
  const_elem_offset_view<p> offsets,
  const_scalar_view<p> vols,
  tpetra_view_type yout,
  bool lumped)
{
  stk::mesh::ProfilingBlock pf("filter_diagonal");

  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    DeviceRangePolicy(0, offsets.extent_int(0)), KOKKOS_LAMBDA(int index) {
      LocalArray<ftype[p + 1][p + 1][p + 1]> lhs;
      if (lumped) {
        for (int k = 0; k < p + 1; ++k) {
          static constexpr auto Wl = Coeffs<p>::Wl;
          const auto Wk = Wl(k);
          for (int j = 0; j < p + 1; ++j) {
            const auto WjWk = Wl(j) * Wk;
            for (int i = 0; i < p + 1; ++i) {
              lhs(k, j, i) = Wl(i) * WjWk * vols(index, k, j, i);
            }
          }
        }
      } else {
        for (int k = 0; k < p + 1; ++k) {
          static constexpr auto W = Coeffs<p>::W;
          const auto Wk = W(k, k);
          for (int j = 0; j < p + 1; ++j) {
            const auto WjWk = W(j, j) * Wk;
            for (int i = 0; i < p + 1; ++i) {
              lhs(k, j, i) = W(i, i) * WjWk * vols(index, k, j, i);
            }
          }
        }
      }

      auto accessor = yout_scatter.access();
      const int valid_simd_len = valid_offset<p>(index, offsets);
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < valid_simd_len; ++n) {
              accessor(offsets(index, k, j, i, n), 0) +=
                stk::simd::get_data(lhs(k, j, i), n);
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(filter_diagonal_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
