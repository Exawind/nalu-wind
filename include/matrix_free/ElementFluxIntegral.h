#ifndef ELEMENT_FLUX_INTEGRAL_H
#define ELEMENT_FLUX_INTEGRAL_H

#include <cmath>

#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"
#include "matrix_free/ShuffledAccess.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <
  int p,
  int dir,
  typename GeometricFactorArray,
  typename DeltaArray,
  typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
diffusive_flux(
  int index,
  const GeometricFactorArray& diffusion_metric,
  const DeltaArray& delta,
  OutArray& out)
{
  enum { LEVEL_0 = 0, LEVEL_1 = 1 };
  for (int l = 0; l < p; ++l) {
    LocalArray<ftype[2][p + 1][p + 1]> scratch;

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto flux_point_interpolant = Coeffs<p>::Nt;
          acc +=
            flux_point_interpolant(l, q) * shuffled_access<dir>(delta, s, r, q);
        }
        scratch(LEVEL_0, s, r) = acc;
      }
    }

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto flux_point_derivative = Coeffs<p>::Dt;
          acc +=
            flux_point_derivative(l, q) * shuffled_access<dir>(delta, s, r, q);
        }
        scratch(LEVEL_1, s, r) = acc * diffusion_metric(index, dir, l, s, r, 0);
      }
    }

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto nodal_derivative = Coeffs<p>::D;
          acc += nodal_derivative(r, q) * scratch(LEVEL_0, s, q);
        }
        scratch(LEVEL_1, s, r) +=
          acc * diffusion_metric(index, dir, l, s, r, 1);
      }
    }

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto nodal_derivative = Coeffs<p>::D;
          acc += nodal_derivative(s, q) * scratch(LEVEL_0, q, r);
        }
        scratch(LEVEL_1, s, r) +=
          acc * diffusion_metric(index, dir, l, s, r, 2);
      }
    }

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto vandermonde = Coeffs<p>::W;
          acc += vandermonde(r, q) * scratch(LEVEL_1, s, q);
        }
        scratch(LEVEL_0, s, r) = acc;
      }
    }

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto vandermonde = Coeffs<p>::W;
          acc += vandermonde(s, q) * scratch(LEVEL_0, q, r);
        }
        shuffled_access<dir>(out, s, r, l + 0) -= acc;
        shuffled_access<dir>(out, s, r, l + 1) += acc;
      }
    }
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
