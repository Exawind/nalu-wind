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

#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

#define XH 0
#define YH 1
#define ZH 2

template <int p, typename CoeffArray, typename InArray, typename OutArray>
KOKKOS_FUNCTION void
interp_scs(const InArray& in, const CoeffArray& coeff, OutArray& interp)
{
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

#undef XH
#undef YH
#undef ZH

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
