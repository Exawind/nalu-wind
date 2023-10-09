// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/MaxCourantReynolds.h"

#include "matrix_free/LobattoQuadratureRule.h"

#include "gtest/gtest.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
Kokkos::Array<double, 2>
max_cflre()
{
  const int num_elems = 1;

  scalar_view<p> rho{"rho", num_elems};
  scalar_view<p> mu{"mu", num_elems};
  vector_view<p> velocity{"volume_metric", num_elems};
  Kokkos::deep_copy(velocity, 1);
  Kokkos::deep_copy(rho, 1);
  Kokkos::deep_copy(mu, 1. / 8);

  vector_view<p> xc{"coords", num_elems};
  constexpr auto nodes = GLL<p>::nodes;
  Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(int index) {
    for (int k = 0; k < p + 1; ++k) {
      const auto cz = nodes[k];
      for (int j = 0; j < p + 1; ++j) {
        const auto cy = nodes[j];
        for (int i = 0; i < p + 1; ++i) {
          const auto cx = nodes[i];
          xc(index, k, j, i, 0) = 0.5 * cx;
          xc(index, k, j, i, 1) = 0.5 * cy;
          xc(index, k, j, i, 2) = 0.5 * cz;
        }
      }
    }
  });

  double dt = 1;

  return max_local_courant_reynolds<p>(dt, xc, rho, mu, velocity);
}

TEST(MatrixFreeMaxCourantReynolds, gives_correct_values_on_one_element)
{
  auto res = max_cflre<1>();
  ASSERT_NEAR(res[0], 1, 1.e-8);
  ASSERT_NEAR(res[1], 8, 1.e-8);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra