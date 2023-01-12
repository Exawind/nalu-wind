// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearAreas.h"

#include "matrix_free/GeometricFunctions.h"
#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/PolynomialOrders.h"

#include <KokkosInterface.h>
#include "Kokkos_Macros.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {
namespace impl {

template <int p>
scs_vector_view<p>
linear_areas_t<p>::invoke(const_vector_view<p> coordinates)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  scs_vector_view<p> area("area", coordinates.extent_int(0));
  Kokkos::parallel_for(
    DeviceRangePolicy(0, coordinates.extent_int(0)), KOKKOS_LAMBDA(int index) {
      const auto box = hex_vertex_coordinates<p>(index, coordinates);
      for (int l = 0; l < p; ++l) {
        for (int s = 0; s < p + 1; ++s) {
          for (int r = 0; r < p + 1; ++r) {
            auto av = linear_area<p, XH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              area(index, XH, l, s, r, d) = av(d);
            }

            av = linear_area<p, YH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              area(index, YH, l, s, r, d) = av(d);
            }

            av = linear_area<p, ZH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              area(index, ZH, l, s, r, d) = av(d);
            }
          }
        }
      }
    });
  return area;
}

INSTANTIATE_POLYSTRUCT(linear_areas_t);

} // namespace impl
} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
