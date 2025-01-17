// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearExposedAreas.h"

#include <KokkosInterface.h>
#include <Kokkos_Macros.hpp>

#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/Coefficients.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {
namespace impl {

namespace {

template <int di, int dj, typename BoxArray, typename CoeffArray>
KOKKOS_FUNCTION ftype
jacobian_component(
  const BoxArray& base_box, const CoeffArray& nlin, int j, int i)
{
  enum { LN = 0, RN = 1 };
  enum { DS1 = 0 };
  return ((dj == DS1)
            ? (-nlin(LN, j) * base_box(di, 0) + nlin(LN, j) * base_box(di, 1) +
               nlin(RN, j) * base_box(di, 2) - nlin(RN, j) * base_box(di, 3))
            : (-nlin(LN, i) * base_box(di, 0) - nlin(RN, i) * base_box(di, 1) +
               nlin(RN, i) * base_box(di, 2) + nlin(LN, i) * base_box(di, 3))) *
         0.5;
}

template <typename CoeffArray>
KOKKOS_FUNCTION ArrayND<ftype[3]>
face_area(
  const ArrayND<ftype[3][4]>& base_box, const CoeffArray& nlin, int j, int i)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  enum { DS1 = 0, DS2 = 1 };

  const auto dx_ds1 = jacobian_component<XH, DS1>(base_box, nlin, j, i);
  const auto dx_ds2 = jacobian_component<XH, DS2>(base_box, nlin, j, i);
  const auto dy_ds1 = jacobian_component<YH, DS1>(base_box, nlin, j, i);
  const auto dy_ds2 = jacobian_component<YH, DS2>(base_box, nlin, j, i);
  const auto dz_ds1 = jacobian_component<ZH, DS1>(base_box, nlin, j, i);
  const auto dz_ds2 = jacobian_component<ZH, DS2>(base_box, nlin, j, i);
  return ArrayND<ftype[3]>{
    {dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2, dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2,
     dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2}};
}
} // namespace

template <int p>
face_vector_view<p>
exposed_areas_t<p>::invoke(const const_face_vector_view<p> coordinates)
{
  constexpr auto nlin = Coeffs<p>::Nlin;
  face_vector_view<p> areas("exposed_area_vectors", coordinates.extent_int(0));
  Kokkos::parallel_for(
    "volume", DeviceRangePolicy(0, coordinates.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
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
