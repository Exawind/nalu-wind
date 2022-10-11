// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef DISPLACEMENTS_H_
#define DISPLACEMENTS_H_

#include <aero/aero_utils/WienerMilenkovic.h>

namespace aero {

//! Implementation of a pitch deformation strategy that ramps to the true
//! applied pitch with a hyperbolic tangent
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
pitch_displacement_contribution(
  const vs::Vector distance,
  const vs::Vector root,
  const double pitch,
  const double rLocation,
  const double rampFactor = 2.0)
{
  const auto globZ = wmp::rotate(root, vs::Vector::khat());
  auto pitchRotWM = wmp::create_wm_param(globZ, pitch);

  const auto pitchRot = wmp::rotate(pitchRotWM, distance);

  const double rampPitch = pitch *
                           (1.0 - stk::math::exp(-rampFactor * rLocation)) /
                           (1.0 + stk::math::exp(-rampFactor * rLocation));

  pitchRotWM = pitch_wm(rampPitch, globZ);
  const auto rampPitchRot = wmp::rotate(pitchRotWM, distance);
  return rampPitchRot - pitchRot;
}

//! Convert one array of 6 deflections (transX, transY, transZ, wmX, wmY,
//! wmZ) into one vector of translational displacement at a given node on the
//! turbine surface CFD mesh.
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
compute_translational_displacements(
  const vs::Vector cfdPos,
  const vs::Vector totalPosOffset,
  const vs::Vector totDispNode)
{

  const vs::Vector distance = cfdPos - totalPosOffset;
  const vs::Vector pointLocal = wmp::rotate(totalPosOffset, distance);
  const vs::Vector rotation = wmp::rotate(totDispNode, pointLocal, true);
  return totDispNode + rotation - distance;
}

//! Accounting for pitch, convert one array of 6 deflections (transX, transY,
//! transZ, wmX, wmY, wmZ) into one vector of translational displacement at a
//! given node on the turbine surface CFD mesh.
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
compute_translational_displacements(
  const vs::Vector cfdPos,
  const vs::Vector totalPosOffset,
  const vs::Vector totDispNode,
  const vs::Vector root,
  const double pitch,
  const double rLoc)
{
  auto disp =
    compute_translational_displacements(cfdPos, totalPosOffset, totDispNode);
  return disp + pitch_displacement_contribution(
                  cfdPos - totalPosOffset, root, pitch, rLoc);
}

} // namespace aero

#endif
