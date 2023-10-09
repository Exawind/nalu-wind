// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearExposedAreas.h"
#include "matrix_free/LobattoQuadratureRule.h"
#include "StkSimdComparisons.h"

#include <Kokkos_Core.hpp>
#include <stk_simd/Simd.hpp>

#include "gtest/gtest.h"
#include "mpi.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
void
single_cube_hex_p()
{
  const int num_elems_1D = 32 / p;
  const int num_elems_2D = num_elems_1D * num_elems_1D;

  face_vector_view<p> coords("coordinates", num_elems_2D);
  Kokkos::parallel_for(
    num_elems_2D, KOKKOS_LAMBDA(int index) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          constexpr auto nodes = GLL<p>::nodes;
          coords(index, j, i, 0) = nodes[i];
          coords(index, j, i, 1) = nodes[j];
          coords(index, j, i, 2) = 0;
        }
      }
    });

  const auto areas_d = geom::exposed_areas<p>(coords);
  auto areas_h = Kokkos::create_mirror_view(areas_d);
  Kokkos::deep_copy(areas_h, areas_d);

  for (int index = 0; index < num_elems_2D; ++index) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ASSERT_DOUBLETYPE_NEAR(areas_h(index, j, i, 0), +0.0, 1e-10)
          << "index: " << index;
        ASSERT_DOUBLETYPE_NEAR(areas_h(index, j, i, 1), +0.0, 1e-10)
          << "index: " << index;
        ASSERT_DOUBLETYPE_NEAR(areas_h(index, j, i, 2), +1.0, 1e-10)
          << "index: " << index;
      }
    }
  }
}

TEST(linear_exposed_areas, single_cube_hex8) { single_cube_hex_p<1>(); }
TEST(linear_exposed_areas, single_cube_hex27) { single_cube_hex_p<2>(); }
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
