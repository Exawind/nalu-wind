// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef DEFLECTION_RAMPING_H
#define DEFLECTION_RAMPING_H

#include <aero/aero_utils/displacements.h>

namespace fsi {

double KOKKOS_FORCEINLINE_FUNCTION
linear_ramp_span(
  const double spanLocation,
  const double zeroRampDistance)
{
  return stk::math::max((zeroRampDistance-spanLocation)/zeroRampDistance, 0.0);
}

} // namespace fsi

#endif
