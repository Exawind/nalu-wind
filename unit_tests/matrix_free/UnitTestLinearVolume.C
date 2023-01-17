// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearVolume.h"
#include "gtest/gtest.h"

#include "matrix_free/LobattoQuadratureRule.h"
#include "matrix_free/TensorOperations.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LocalArray.h"

#include <Kokkos_Core.hpp>
#include <stk_simd/Simd.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace {

template <int poly>
void
single_affine_hex_p(bool cube)
{
  LocalArray<double[3][3]> jac = {{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}};
  if (!cube) {
    jac = {{{2, 1, 1.3333}, {0, 2, -1}, {1, 0, 2}}};
  }
  const auto det = determinant<double>(jac);
  ASSERT_GT(det, 0);

  const int num_elems_1D = 32 / poly;
  const int num_elems_3D = num_elems_1D * num_elems_1D * num_elems_1D;
  vector_view<poly> coords("coordinates", num_elems_3D);
  scalar_view<poly> alpha("alpha", num_elems_3D);

  Kokkos::parallel_for(
    num_elems_3D, KOKKOS_LAMBDA(int index) {
      constexpr auto nodes = GLL<poly>::nodes;
      for (int k = 0; k < poly + 1; ++k) {
        for (int j = 0; j < poly + 1; ++j) {
          for (int i = 0; i < poly + 1; ++i) {
            alpha(index, k, j, i) = 1.0;
            const Kokkos::Array<ftype, 3> old_coords = {
              {nodes[i], nodes[j], nodes[k]}};
            Kokkos::Array<ftype, 3> new_coords;
            transform(jac, old_coords, new_coords);
            for (int d = 0; d < 3; ++d) {
              coords(index, k, j, i, d) = new_coords[d] + 2;
            }
          }
        }
      }
    });
  exec_space().fence();

  const auto volumes_d = geom::volume_metric<poly>(alpha, coords);
  exec_space().fence();

  auto volumes_h = Kokkos::create_mirror_view(volumes_d);
  Kokkos::deep_copy(volumes_h, volumes_d);

  for (int index = 0; index < num_elems_3D; ++index) {
    for (int k = 0; k < poly + 1; ++k) {
      for (int j = 0; j < poly + 1; ++j) {
        for (int i = 0; i < poly + 1; ++i) {
          ASSERT_DOUBLE_EQ(
            stk::simd::get_data(volumes_h(index, k, j, i), 0), det)
            << "index: " << 0;
        }
      }
    }
  }
}
} // namespace
TEST(linear_volume, single_cube_hex8) { single_affine_hex_p<1>(true); }
TEST(linear_volume, single_cube_hex27) { single_affine_hex_p<2>(false); }

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
