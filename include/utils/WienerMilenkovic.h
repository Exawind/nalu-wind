// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef WIENERMILENKOVIC_H_
#define WIENERMILENKOVIC_H_

#include <vs/vector_space.H>

using Vector = vs::Vector;

namespace WienerMilenkovic {

namespace {

KOKKOS_FORCEINLINE_FUNCTION
double
compute_coeff0(Vector v)
{
  return 2.0 - 0.125 * (v & v);
}

} // namespace

KOKKOS_FORCEINLINE_FUNCTION
Vector
apply_rotation(Vector wMilenkovic, Vector vector, double transpose = 1.0)
{
  const double wm0 = compute_coeff0(wMilenkovic);
  const double nu = 2.0 / (4.0 - wm0);
  const double cosPhiO2 = 0.5 * wm0 * nu;
  const Vector wmCrossVec = wMilenkovic ^ vector;

  return vector + transpose * nu * cosPhiO2 * wmCrossVec +
           0.5 * nu * nu * wMilenkovic ^
         wmCrossVec;
}

KOKKOS_FORCEINLINE_FUNCTION
Vector
compose(Vector p, Vector q, double transposeP = 1.0, double transposeQ = 1.0)
{
  const double p0 = compute_coeff0(p);
  const double q0 = compute_coeff0(q);

  const double delta1 = (4.0 - p0) * (4.0 - q0);
  const double delta2 = p0 * q0 - transposeP * (p & q);

  double premultFac;

  if (delta2 < 0.0) {
    premultFac = -4.0 / (delta1 - delta2);
  } else {
    premultFac = 4.0 / (delta1 + delta2);
  }

  return premultFac * (transposeQ * p0 * q + transposeP * q0 * p +
                       transposeP * transposeQ * (p ^ q));
}
} // namespace WienerMilenkovic

#endif
