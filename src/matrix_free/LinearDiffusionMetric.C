// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LinearDiffusionMetric.h"

#include "matrix_free/Coefficients.h"
#include "matrix_free/ElementSCSInterpolate.h"
#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/GeometricFunctions.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LocalArray.h"
#include "matrix_free/PolynomialOrders.h"

#include <KokkosInterface.h>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {
namespace impl {

template <int p>
scs_vector_view<p>
diffusion_metric_t<p>::invoke(
  const_scalar_view<p> alpha, const_vector_view<p> coordinates)
{
  enum { XH = 0, YH = 1, ZH = 2 };

  scs_vector_view<p> metric("diffusion", coordinates.extent_int(0));
  Kokkos::parallel_for(
    "diffusion_metric", DeviceRangePolicy(0, coordinates.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      static constexpr auto ntilde = Coeffs<p>::Nt;

      LocalArray<ftype[3][p + 1][p + 1][p + 1]> interp;
      {
        const auto alpha_elem = Kokkos::subview(
          alpha, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        interp_scs<p>(alpha_elem, ntilde, interp);
      }
      const auto box = hex_vertex_coordinates<p>(index, coordinates);
      for (int l = 0; l < p; ++l) {
        for (int s = 0; s < p + 1; ++s) {
          for (int r = 0; r < p + 1; ++r) {

            auto lm = laplacian_metric<p, XH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              metric(index, XH, l, s, r, d) = interp(XH, l, s, r) * lm(d);
            }

            lm = laplacian_metric<p, YH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              metric(index, YH, l, s, r, d) = interp(YH, l, s, r) * lm(d);
            }

            lm = laplacian_metric<p, ZH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              metric(index, ZH, l, s, r, d) = interp(ZH, l, s, r) * lm(d);
            }
          }
        }
      }
    });
  return metric;
}

template <int p>
scs_vector_view<p>
diffusion_metric_t<p>::invoke(const_vector_view<p> coordinates)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  scs_vector_view<p> metric("diffusion", coordinates.extent_int(0));
  Kokkos::parallel_for(
    "diffusion_metric", DeviceRangePolicy(0, coordinates.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      const auto box = hex_vertex_coordinates<p>(index, coordinates);
      for (int l = 0; l < p; ++l) {
        for (int s = 0; s < p + 1; ++s) {
          for (int r = 0; r < p + 1; ++r) {
            auto lm = laplacian_metric<p, XH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              metric(index, XH, l, s, r, d) = lm(d);
            }

            lm = laplacian_metric<p, YH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              metric(index, YH, l, s, r, d) = lm(d);
            }

            lm = laplacian_metric<p, ZH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              metric(index, ZH, l, s, r, d) = lm(d);
            }
          }
        }
      }
    });
  return metric;
}

INSTANTIATE_POLYSTRUCT(diffusion_metric_t);

} // namespace impl
} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
