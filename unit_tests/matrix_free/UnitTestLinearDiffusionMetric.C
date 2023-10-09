// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearDiffusionMetric.h"
#include "matrix_free/LobattoQuadratureRule.h"
#include "StkSimdComparisons.h"

#include <Kokkos_Core.hpp>
#include <stk_simd/Simd.hpp>

#include "matrix_free/KokkosViewTypes.h"
#include "gtest/gtest.h"
#include "mpi.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace {

template <int p>
void
diffusion_single_cube_hex_p()
{
  const int num_elems_1D = 32 / p;
  const int num_elems_3D = num_elems_1D * num_elems_1D * num_elems_1D;

  vector_view<p> coords("coordinates", num_elems_3D);
  scalar_view<p> alpha("alpha", num_elems_3D);
  Kokkos::parallel_for(num_elems_3D, KOKKOS_LAMBDA(int index) {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          constexpr auto nodes = GLL<p>::nodes;
          alpha(index, k, j, i) = 1.0;
          coords(index, k, j, i, 0) = nodes[i];
          coords(index, k, j, i, 1) = nodes[j];
          coords(index, k, j, i, 2) = nodes[k];
        }
      }
    }
  });

  const auto diffusion_d = geom::diffusion_metric<p>(alpha, coords);
  auto diffusion_h = Kokkos::create_mirror_view(diffusion_d);
  Kokkos::deep_copy(diffusion_h, diffusion_d);

  for (int index = 0; index < num_elems_3D; ++index) {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p; ++i) {
          ASSERT_DOUBLETYPE_NEAR(diffusion_h(index, 0, i, k, j, 0), -1.0, 1e-10)
            << "index: " << index;
          ASSERT_DOUBLETYPE_NEAR(diffusion_h(index, 0, i, k, j, 1), +0.0, 1e-10)
            << "index: " << index;
          ASSERT_DOUBLETYPE_NEAR(diffusion_h(index, 0, i, k, k, 2), +0.0, 1e-10)
            << "index: " << index;
        }
      }
    }

    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          ASSERT_DOUBLETYPE_NEAR(diffusion_h(index, 1, j, k, i, 0), -1.0, 1e-10)
            << "index: " << index;
          ASSERT_DOUBLETYPE_NEAR(diffusion_h(index, 1, j, k, i, 1), +0.0, 1e-10)
            << "index: " << index;
          ASSERT_DOUBLETYPE_NEAR(diffusion_h(index, 1, j, k, i, 2), +0.0, 1e-10)
            << "index: " << index;
        }
      }
    }

    for (int k = 0; k < p; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          ASSERT_DOUBLETYPE_NEAR(diffusion_h(index, 2, k, j, i, 0), -1.0, 1e-10)
            << "index: " << index;
          ASSERT_DOUBLETYPE_NEAR(diffusion_h(index, 2, k, j, i, 1), +0.0, 1e-10)
            << "index: " << index;
          ASSERT_DOUBLETYPE_NEAR(diffusion_h(index, 2, k, j, i, 2), +0.0, 1e-10)
            << "index: " << index;
        }
      }
    }
  }
}
} // namespace

TEST(linear_diffusion_metric, single_cube_hex8)
{
  diffusion_single_cube_hex_p<1>();
}
TEST(linear_diffusion_metric, single_cube_hex27)
{
  diffusion_single_cube_hex_p<2>();
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
