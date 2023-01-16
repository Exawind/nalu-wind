// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FSIUTILS_H
#define FSIUTILS_H
#include "aero/aero_utils/displacements.h"
#include "vs/vector.h"
#include <Kokkos_Macros.hpp>

namespace fsi {

//! compute displacements from the net motions of the hub (including rotations)
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
translational_displacements_from_hub_motion(
  const aero::SixDOF& hubRef,
  const aero::SixDOF& hubDisp,
  const aero::SixDOF& bladeRef)
{
  const auto refRelToHub =
    aero::local_aero_coordinates(bladeRef.position_, hubRef);
  // apply opposite rotation since it is the displacement i.e. what needs to be
  // added to get to the current position
  const auto positionDueToRotation =
    wmp::rotate(hubDisp.orientation_, refRelToHub, true);
  const auto referencePosition = bladeRef.position_ - hubRef.position_;
  return positionDueToRotation - referencePosition + hubDisp.position_;
}

// TODO(psakiev) write a unit test for this.
//! assemble the orientation just from changes at the root (no pitch, includes
//! hub)
KOKKOS_FORCEINLINE_FUNCTION vs::Vector
orientation_displacments_from_hub_motion(
  const aero::SixDOF& rootRef,
  const aero::SixDOF& rootDisp,
  const aero::SixDOF& bladeRef)
{
  const auto rootRelativeRefOrientation = wmp::rotate(
    rootRef.orientation_,
    wmp::pop(rootRef.orientation_, bladeRef.orientation_));
  const auto rootRelativeTwist =
    wmp::rotate(rootDisp.orientation_, rootRelativeRefOrientation, true);
  // apply twist first and then rotate out of root reference frame
  return wmp::push(rootDisp.orientation_, rootRelativeTwist);
}

KOKKOS_FORCEINLINE_FUNCTION
aero::SixDOF
displacements_from_hub_motion(
  const aero::SixDOF& hubRef,
  const aero::SixDOF& hubDisp,
  const aero::SixDOF& rootRef,
  const aero::SixDOF& rootDisp,
  const aero::SixDOF& bladeRef)
{
  return aero::SixDOF(
    translational_displacements_from_hub_motion(hubRef, hubDisp, bladeRef),
    orientation_displacments_from_hub_motion(rootRef, rootDisp, bladeRef));
}

} // namespace fsi

#endif
