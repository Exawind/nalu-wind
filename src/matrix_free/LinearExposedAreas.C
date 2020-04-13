// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearExposedAreas.h"

#include <Kokkos_Macros.hpp>

#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/Coefficients.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/TensorOperations.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {
namespace impl {

namespace {

#define XH 0
#define YH 1
#define ZH 2

template <int di, int dj, typename BoxArray, typename CoeffArray>
KOKKOS_FUNCTION ftype
jacobian_component(
  const BoxArray& base_box, const CoeffArray& nlin, int j, int i)
{
  return (dj == XH)
           ? (-nlin(0, j) * base_box(di, 0) + nlin(0, j) * base_box(di, 1) +
              nlin(1, j) * base_box(di, 2) - nlin(1, j) * base_box(di, 3)) *
               0.5
           : (-nlin(0, i) * base_box(di, 0) - nlin(1, i) * base_box(di, 1) +
              nlin(1, i) * base_box(di, 2) + nlin(0, i) * base_box(di, 3)) *
               0.5;
}

template <typename CoeffArray>
KOKKOS_FUNCTION LocalArray<ftype[3]>
face_area(
  const LocalArray<ftype[3][4]>& base_box, const CoeffArray& nlin, int j, int i)
{
  static constexpr int ds1 = XH;
  static constexpr int ds2 = YH;
  const auto dx_ds1 = jacobian_component<XH, ds1>(base_box, nlin, j, i);
  const auto dx_ds2 = jacobian_component<XH, ds2>(base_box, nlin, j, i);
  const auto dy_ds1 = jacobian_component<YH, ds1>(base_box, nlin, j, i);
  const auto dy_ds2 = jacobian_component<YH, ds2>(base_box, nlin, j, i);
  const auto dz_ds1 = jacobian_component<ZH, ds1>(base_box, nlin, j, i);
  const auto dz_ds2 = jacobian_component<ZH, ds2>(base_box, nlin, j, i);
  return LocalArray<ftype[3]>{{dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2,
                               dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2,
                               dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2}};
}

#undef XH
#undef YH
#undef ZH
} // namespace

template <int p>
face_vector_view<p>
exposed_areas_t<p>::invoke(const const_face_vector_view<p> coordinates)
{
  constexpr auto nlin = Coeffs<p>::Nlin;
  face_vector_view<p> areas("exposed_area_vectors", coordinates.extent_int(0));
  Kokkos::parallel_for(
    "volume", coordinates.extent_int(0), KOKKOS_LAMBDA(int index) {
      const auto base_box = face_vertex_coordinates<p>(index, coordinates);
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          const auto areav = face_area(base_box, nlin, j, i);
          areas(index, j, i, 0) = areav(0);
          areas(index, j, i, 1) = areav(1);
          areas(index, j, i, 2) = areav(2);
        }
      }
    });
  return areas;
}
INSTANTIATE_POLYSTRUCT(exposed_areas_t);

} // namespace impl
} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
