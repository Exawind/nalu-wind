// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ELEMENT_FLUX_INTEGRAL_H
#define ELEMENT_FLUX_INTEGRAL_H

#include <cmath>

#include "matrix_free/KokkosFramework.h"
#include "ArrayND.h"
#include "matrix_free/ShuffledAccess.h"
#include "matrix_free/ElementSCSInterpolate.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <
  int p,
  int dir,
  typename GeometricFactorArray,
  typename DeltaArray,
  typename ScratchArray>
KOKKOS_FORCEINLINE_FUNCTION void
area_weighted_diffusion_velocity(
  int index,
  int l,
  const GeometricFactorArray& diffusion_metric,
  const DeltaArray& delta,
  ScratchArray& scratch)
{
  enum { LEVEL_0 = 0, LEVEL_1 = 1 };
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
      scratch(LEVEL_1, s, r) += acc * diffusion_metric(index, dir, l, s, r, 1);
    }
  }

  for (int s = 0; s < p + 1; ++s) {
    for (int r = 0; r < p + 1; ++r) {
      ftype acc = 0;
      for (int q = 0; q < p + 1; ++q) {
        static constexpr auto nodal_derivative = Coeffs<p>::D;
        acc += nodal_derivative(s, q) * scratch(LEVEL_0, q, r);
      }
      scratch(LEVEL_1, s, r) += acc * diffusion_metric(index, dir, l, s, r, 2);
    }
  }
}

template <int p, int dir, typename InArray, typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
scalar_flux_divergence(int index, const InArray& in, OutArray& out)
{
  for (int l = 0; l < p; ++l) {
    ArrayND<ftype[p + 1][p + 1]> scratch;
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto vandermonde = Coeffs<p>::W;
          acc += vandermonde(r, q) * in(index, dir, l, s, q);
        }
        scratch(s, r) = acc;
      }
    }

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto vandermonde = Coeffs<p>::W;
          acc += vandermonde(s, q) * scratch(q, r);
        }
        shuffled_access<dir>(out, s, r, l + 0) -= acc;
        shuffled_access<dir>(out, s, r, l + 1) += acc;
      }
    }
  }
}

template <
  int p,
  int dir,
  typename AdvectionMetricArray,
  typename DiffusionMetricArray,
  typename DeltaArray,
  typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
advdiff_flux(
  int index,
  const AdvectionMetricArray& advection_metric,
  const DiffusionMetricArray& diffusion_metric,
  const DeltaArray& delta,
  OutArray& out)
{
  for (int l = 0; l < p; ++l) {
    enum { LEVEL_0 = 0, LEVEL_1 = 1 };
    ArrayND<ftype[2][p + 1][p + 1]> scratch;
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto flux_point_interpolant = Coeffs<p>::Nt;
          acc +=
            flux_point_interpolant(l, q) * shuffled_access<dir>(delta, s, r, q);
        }
        scratch(LEVEL_0, s, r) = acc;
        scratch(LEVEL_1, s, r) = -acc * advection_metric(index, dir, l, s, r);
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
        scratch(LEVEL_1, s, r) +=
          acc * diffusion_metric(index, dir, l, s, r, 0);
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
    ArrayND<ftype[2][p + 1][p + 1]> scratch;

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

template <
  int p,
  int dir,
  typename GeometricFactorArray,
  typename InArray,
  typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
scalar_flux_vector(
  int index,
  int d,
  const GeometricFactorArray& area,
  const InArray& in,
  OutArray& out)
{
  enum { LEVEL_0 = 0, LEVEL_1 = 1 };
  for (int l = 0; l < p; ++l) {
    ArrayND<ftype[2][p + 1][p + 1]> scratch;
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype in_scs(0);
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto interp = Coeffs<p>::Nt;
          in_scs += interp(l, q) * shuffled_access<dir>(in, s, r, q);
        }
        scratch(LEVEL_0, s, r) = in_scs * area(index, dir, l, s, r, d);
      }
    }

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto vandermonde = Coeffs<p>::W;
          acc += vandermonde(r, q) * scratch(LEVEL_0, s, q);
        }
        scratch(LEVEL_1, s, r) = acc;
      }
    }

    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto vandermonde = Coeffs<p>::W;
          acc += vandermonde(s, q) * scratch(LEVEL_1, q, r);
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
