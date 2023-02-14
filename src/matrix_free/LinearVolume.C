// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearVolume.h"
#include "matrix_free/Coefficients.h"
#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/GeometricFunctions.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LocalArray.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/TensorOperations.h"

#include <KokkosInterface.h>
#include "Kokkos_Macros.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {
namespace impl {

template <int p>
scalar_view<p>
volume_metric_t<p>::invoke(
  const_scalar_view<p> alpha, const_vector_view<p> coordinates)
{
  scalar_view<p> volume("volumes", coordinates.extent_int(0));
  Kokkos::parallel_for(
    "volume", DeviceRangePolicy(0, coordinates.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      const auto box = hex_vertex_coordinates<p>(index, coordinates);
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            volume(index, k, j, i) =
              alpha(index, k, j, i) *
              determinant<ftype>(linear_hex_jacobian<p>(box, k, j, i));
          }
        }
      }
    });
  return volume;
}

template <int p>
scalar_view<p>
volume_metric_t<p>::invoke(const_vector_view<p> coordinates)
{
  scalar_view<p> volume("volumes", coordinates.extent_int(0));
  Kokkos::parallel_for(
    "volume", DeviceRangePolicy(0, coordinates.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      const auto box = hex_vertex_coordinates<p>(index, coordinates);
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            volume(index, k, j, i) =
              determinant<ftype>(linear_hex_jacobian<p>(box, k, j, i));
          }
        }
      }
    });
  return volume;
}
INSTANTIATE_POLYSTRUCT(volume_metric_t);

} // namespace impl
} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
