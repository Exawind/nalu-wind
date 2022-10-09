// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef WIENER_MILENKOVIC_H_
#define WIENER_MILENKOVIC_H_

#include <stk_math/StkMath.hpp>
#include <vs/vector_space.h>

// Wiener-Milenkovic Parameters (WMP)
namespace wmp {

namespace {
KOKKOS_FORCEINLINE_FUNCTION
double
compute_coeff_zero(vs::Vector vec)
{
  return 2.0 - 0.125 * (vec & vec);
}

//! Convert a boolean into a 1.0 for false or a -1.0 for true
KOKKOS_FORCEINLINE_FUNCTION
double
bool_sign(const bool condition)
{
  return 1.0 - 2.0 * static_cast<double>(condition);
}

} // namespace

KOKKOS_FORCEINLINE_FUNCTION
double
generator(const double phi)
{
  return 4.0 * stk::math::tan(phi * 0.25);
}

KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
create_wm_param(vs::Vector axis, const double angle)
{
  axis.normalize();
  return generator(angle) * axis;
}

//! Compose Wiener-Milenkovic parameters 'wmP' and 'wmQ'
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
compose(
  const vs::Vector wmP,
  const vs::Vector wmQ,
  bool transposeP = false,
  bool transposeQ = false)
{
  const double tP = bool_sign(transposeP);
  const double tQ = bool_sign(transposeQ);

  const double p0 = compute_coeff_zero(wmP);
  const double q0 = compute_coeff_zero(wmQ);

  const double delta1 = (4.0 - p0) * (4.0 - q0);
  const double delta2 = p0 * q0 - tP * (wmP & wmQ);

  const double sign = bool_sign(delta2 < 0.0);
  const double preFac = sign * 4.0 / (delta1 + sign * delta2);

  return preFac * (tQ * p0 * wmQ + tP * q0 * wmP + tP * tQ * (wmP ^ wmQ));
}

//! Apply a Wiener-Milenkovic rotation 'wmP' to a vector 'vec'
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
rotate(const vs::Vector wmP, const vs::Vector vec, const bool transpose = false)
{
  const double trans = bool_sign(transpose);
  const double wm0 = compute_coeff_zero(wmP);
  const double nu = 2.0 / (4.0 - wm0);
  const double cosPhiO2 = 0.5 * wm0 * nu;
  const vs::Vector crossWmVec = wmP ^ vec;

  return vec + trans * nu * cosPhiO2 * crossWmVec +
         0.5 * nu * nu * (wmP ^ crossWmVec);
}

} // namespace wmp

#endif
