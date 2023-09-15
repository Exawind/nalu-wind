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
  const double zeroRampDistance,
  const bool useRamp = true)
{
  return 1.0 - stk::math::max(
                 (zeroRampDistance - spanLocation) / zeroRampDistance, 0.0) *
                 static_cast<double>(useRamp);
}

double KOKKOS_FORCEINLINE_FUNCTION
linear_ramp_theta(
  const aero::SixDOF& hub,
  const vs::Vector& root,
  const vs::Vector& position,
  const double rampSpan,
  const double zeroRampLoc,
  const bool useRamp = true)
{
  auto v1 = (root - hub.position_).normalize();
  auto v2 = (position - hub.position_).normalize();

  // make sure vectors are in the plane of rotation to compute the angle between
  // them
  const vs::Vector rotationAxis =
    wmp::rotate(hub.orientation_, vs::Vector::ihat(), true).normalize();
  v1 = v1 - vs::project(v1, rotationAxis);
  v2 = v2 - vs::project(v2, rotationAxis);

  const auto angle = vs::angle(v1, v2);

  return stk::math::min(
    1.0, stk::math::max(
           static_cast<double>(!useRamp), (zeroRampLoc - angle) / rampSpan));
}

//! ramp from 0 to 1 to allow turbines to only experience rigid body blade
//! motion until time == startRamp, then linearly ramp to full bladeDeflections
//! over the time window startRamp to endRamp
double KOKKOS_FORCEINLINE_FUNCTION
temporal_ramp(
  const double time,
  const double startRamp,
  const double endRamp,
  const bool useRamp = true)
{
  const double denom = stk::math::max(endRamp - startRamp, 1e-12);
  return stk::math::max(
    static_cast<double>(!useRamp),
    stk::math::min(1.0, (time - startRamp) / denom));
}

} // namespace fsi

#endif
