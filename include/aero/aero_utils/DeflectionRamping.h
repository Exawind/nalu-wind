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
linear_ramp_span(const double spanLocation, const double zeroRampDistance)
{
  return 1.0 - stk::math::max(
                 (zeroRampDistance - spanLocation) / zeroRampDistance, 0.0);
}

//! ramp from 0 to 1 to allow turbines to only experience rigid body blade
//! motion until time == startRamp, then linearly ramp to full bladeDeflections
//! over the time window startRamp to endRamp
double KOKKOS_FORCEINLINE_FUNCTION
temporal_ramp(const double time, const double startRamp, const double endRamp)
{
  const double denom = stk::math::max(endRamp - startRamp, 1e-12);
  return stk::math::max(0.0, stk::math::min(1.0, (time - startRamp) / denom));
}

} // namespace fsi

#endif
