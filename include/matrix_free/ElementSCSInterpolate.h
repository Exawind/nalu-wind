// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ELEMENT_SCS_INTERPOLATE_H
#define ELEMENT_SCS_INTERPOLATE_H

#include <cmath>

#include "matrix_free/Coefficients.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"
#include "matrix_free/ShuffledAccess.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p, int dir, typename ScalarView>
KOKKOS_FORCEINLINE_FUNCTION ftype
interp_scs(const ScalarView& phi, int l, int s, int r)
{
  ftype val(0);
  for (int q = 0; q < p + 1; ++q) {
    static constexpr auto interp = Coeffs<p>::Nt;
    val += interp(l, q) * shuffled_access<dir>(phi, s, r, q);
  }
  return val;
}

template <int p, int dir, typename ScalarView>
KOKKOS_FORCEINLINE_FUNCTION ftype
interp_scs(const ScalarView& phi, int l, int s, int r, int d)
{
  ftype val(0);
  for (int q = 0; q < p + 1; ++q) {
    static constexpr auto interp = Coeffs<p>::Nt;
    val += interp(l, q) * shuffled_access<dir>(phi, s, r, q, d);
  }
  return val;
}

template <int p, int dir, int dk, typename ScalarView>
KOKKOS_FORCEINLINE_FUNCTION ftype
grad_scs(const ScalarView& phi, int l, int s, int r)
{
  constexpr int dir_0 = dir;
  constexpr int dir_1 = (dir == 0) ? 1 : 0;

  switch (dk) {
  case dir_0: {
    ftype val(0);
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto grad = Coeffs<p>::Dt;
      val += grad(l, q) * shuffled_access<dir>(phi, s, r, q);
    }
    return val;
  }
  case dir_1: {
    ArrayND<ftype[p + 1]> val_array;
    for (int q = 0; q < p + 1; ++q) {
      val_array(q) = interp_scs<p, dir>(phi, l, s, q);
    }

    ftype val(0);
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto grad = Coeffs<p>::D;
      val += grad(r, q) * val_array(q);
    }
    return val;
  }
  default: {
    ArrayND<ftype[p + 1]> val_array;
    for (int q = 0; q < p + 1; ++q) {
      val_array(q) = interp_scs<p, dir>(phi, l, q, r);
    }

    ftype val(0);
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto grad = Coeffs<p>::D;
      val += grad(s, q) * val_array(q);
    }
    return val;
  }
  }
}

template <int p, int dir, int dk, typename ScalarView>
KOKKOS_FORCEINLINE_FUNCTION ftype
grad_scs(const ScalarView& phi, int l, int s, int r, int d)
{
  constexpr int dir_0 = dir;
  constexpr int dir_1 = (dir == 0) ? 1 : 0;
  constexpr int dir_2 = (dir == 2) ? 1 : 2;

  if (dk == dir_0) {
    ftype val(0);
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto grad = Coeffs<p>::Dt;
      val += grad(l, q) * shuffled_access<dir>(phi, s, r, q, d);
    }
    return val;
  }

  if (dk == dir_1) {
    ArrayND<ftype[p + 1]> val_array;
    for (int q = 0; q < p + 1; ++q) {
      val_array(q) = interp_scs<p, dir>(phi, l, s, q, d);
    }

    ftype val(0);
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto grad = Coeffs<p>::D;
      val += grad(r, q) * val_array(q);
    }
    return val;
  }

  if (dk == dir_2) {
    ArrayND<ftype[p + 1]> val_array;
    for (int q = 0; q < p + 1; ++q) {
      val_array(q) = interp_scs<p, dir>(phi, l, q, r, d);
    }

    ftype val(0);
    for (int q = 0; q < p + 1; ++q) {
      static constexpr auto grad = Coeffs<p>::D;
      val += grad(s, q) * val_array(q);
    }
    return val;
  }
}

template <int p, typename CoeffArray, typename InArray, typename OutArray>
KOKKOS_FUNCTION void
interp_scs(const InArray& in, const CoeffArray& coeff, OutArray& interp)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  for (int i = 0; i < p; ++i) {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += coeff(i, q) * in(k, j, q);
        }
        interp(XH, i, k, j) = acc;
      }
    }
  }

  for (int j = 0; j < p; ++j) {
    for (int k = 0; k < p + 1; ++k) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += coeff(j, q) * in(k, q, i);
        }
        interp(YH, j, k, i) = acc;
      }
    }
  }

  for (int k = 0; k < p; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += coeff(k, q) * in(q, j, i);
        }
        interp(ZH, k, j, i) = acc;
      }
    }
  }
}

template <int p, typename InArray, typename OutArray>
KOKKOS_FUNCTION void
interp_scs_vector(const InArray& in, OutArray& interp)
{
  enum { XH = 0, YH = 1, ZH = 2 };

  static constexpr auto coeff = Coeffs<p>::Nt;

  for (int d = 0; d < 3; ++d) {
    for (int i = 0; i < p; ++i) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          ftype acc(0);
          for (int q = 0; q < p + 1; ++q) {
            acc += coeff(i, q) * in(k, j, q, d);
          }
          interp(XH, i, k, j, d) = acc;
        }
      }
    }

    for (int j = 0; j < p; ++j) {
      for (int k = 0; k < p + 1; ++k) {
        for (int i = 0; i < p + 1; ++i) {
          ftype acc(0);
          for (int q = 0; q < p + 1; ++q) {
            acc += coeff(j, q) * in(k, q, i, d);
          }
          interp(YH, j, k, i, d) = acc;
        }
      }
    }

    for (int k = 0; k < p; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          ftype acc(0);
          for (int q = 0; q < p + 1; ++q) {
            acc += coeff(k, q) * in(q, j, i, d);
          }
          interp(ZH, k, j, i, d) = acc;
        }
      }
    }
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
