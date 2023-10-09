// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LobattoQuadratureRule.h"
#include "matrix_free/LocalArray.h"
#include "matrix_free/TensorOperations.h"

#include "StkSimdComparisons.h"
#include "gtest/gtest.h"

#include <Kokkos_Core.hpp>
#include <stk_simd/Simd.hpp>

#include <math.h>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace {

template <int p>
void
area_single_cube_hex_p()
{
  constexpr int poly = p;

  LocalArray<double[3][3]> jac = {
    {{+1.1, -2.6, 0}, {-1.2, .7, -0.2}, {10, -std::sqrt(3.), 12}}};

  constexpr auto nodes = GLL<poly>::nodes;
  const int num_elems_1D = 32 / poly;
  const int num_elems_3D = num_elems_1D * num_elems_1D * num_elems_1D;
  vector_view<poly> coords("coordinates", num_elems_3D);

  Kokkos::parallel_for(num_elems_3D, KOKKOS_LAMBDA(int index) {
    for (int k = 0; k < poly + 1; ++k) {
      for (int j = 0; j < poly + 1; ++j) {
        for (int i = 0; i < poly + 1; ++i) {
          const auto old_coords =
            Kokkos::Array<ftype, 3>{{nodes[i], nodes[j], nodes[k]}};
          Kokkos::Array<ftype, 3> new_coords;
          transform(jac, old_coords, new_coords);
          for (int d = 0; d < 3; ++d) {
            coords(index, k, j, i, d) = new_coords[d] + 2;
          }
        }
      }
    }
  });

  auto areas = geom::linear_areas<p>(coords);
  auto areas_h = Kokkos::create_mirror_view(areas);
  Kokkos::deep_copy(areas_h, areas);

  constexpr double tol = 1.e-10;
  const auto adjjac = adjugate_matrix(jac);

  for (int index = 0; index < num_elems_3D; ++index) {
    for (int dj = 0; dj < 3; ++dj) {
      for (int l = 0; l < p; ++l) {
        for (int s = 0; s < p + 1; ++s) {
          for (int r = 0; r < p + 1; ++r) {
            for (int di = 0; di < 3; ++di) {
              ASSERT_DOUBLETYPE_NEAR(
                areas_h(index, dj, l, s, r, di), -adjjac(di, dj), tol)
                << "index: " << index << " dj " << dj << " di " << di;
            }
          }
        }
      }
    }
  }
}

TEST(linear_areas, single_cube_hex8) { area_single_cube_hex_p<1>(); }
TEST(linear_areas, single_cube_hex27) { area_single_cube_hex_p<2>(); }

} // namespace
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
