// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinearAdvectionMetric.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LinearDiffusionMetric.h"
#include "matrix_free/LobattoQuadratureRule.h"
#include "matrix_free/LocalArray.h"
#include "matrix_free/StkSimdComparisons.h"
#include "matrix_free/TensorOperations.h"

#include "Kokkos_Core.hpp"
#include "stk_simd/Simd.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace {

template <int p>
void
advection_single_cube_hex_p()
{
  constexpr int poly = p;
  LocalArray<double[3][3]> jac = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

  constexpr auto nodes = GLL<poly>::nodes;
  const int num_elems_1D = 32 / poly;
  const int num_elems_3D = num_elems_1D * num_elems_1D * num_elems_1D;
  vector_view<poly> coords("coordinates", num_elems_3D);

  scalar_view<p> density("density", num_elems_3D);
  scalar_view<p> pressure("pressure", num_elems_3D);

  vector_view<p> velocity("velocity", num_elems_3D);
  vector_view<p> gp("gp", num_elems_3D);

  Kokkos::parallel_for(
    num_elems_3D, KOKKOS_LAMBDA(int index) {
      for (int k = 0; k < poly + 1; ++k) {
        for (int j = 0; j < poly + 1; ++j) {
          for (int i = 0; i < poly + 1; ++i) {
            const auto old_coords =
              Kokkos::Array<ftype, 3>{{nodes[i], nodes[j], nodes[k]}};
            Kokkos::Array<ftype, 3> new_coords;
            transform(jac, old_coords, new_coords);
            for (int d = 0; d < 3; ++d) {
              coords(index, k, j, i, d) = new_coords[d] + 2;
              velocity(index, k, j, i, d) = (d == 0) ? 1 : 0;
              gp(index, k, j, i, d) = 0;
            }
            pressure(index, k, j, i) = 0;
            density(index, k, j, i) = 1;
          }
        }
      }
    });

  auto areas = geom::linear_areas<p>(coords);
  auto metric = geom::diffusion_metric<p>(coords);

  scs_scalar_view<p> mdot("mdot", num_elems_3D);
  geom::linear_advection_metric<p>(
    1., areas, metric, density, velocity, gp, pressure, mdot);
  auto mdot_h = Kokkos::create_mirror_view(mdot);
  Kokkos::deep_copy(mdot_h, mdot);

  constexpr double tol = 1.e-10;

  for (int index = 0; index < num_elems_3D; ++index) {
    for (int dj = 0; dj < 3; ++dj) {
      for (int l = 0; l < p; ++l) {
        for (int s = 0; s < p + 1; ++s) {
          for (int r = 0; r < p + 1; ++r) {
            ASSERT_DOUBLETYPE_NEAR(
              mdot_h(index, dj, l, s, r), (dj == 0) ? -1 : 0, tol);
          }
        }
      }
    }
  }
}

TEST(linear_advection_metric, single_cube_hex8)
{
  advection_single_cube_hex_p<1>();
}

TEST(linear_advection_metric, single_cube_hex64)
{
  advection_single_cube_hex_p<3>();
}

} // namespace
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
