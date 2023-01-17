// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include "gtest/gtest.h"
#include "aero/fsi/FSIUtils.h"

namespace aero {
namespace {

TEST(FSIUtils, translational_component_from_blade_rotation_displacement)
{
  // hub at (1,1,1), orientation is such that upstream is -ihat
  // (upstream reference hub for this test is considered local ihat)
  const SixDOF hubReference(
    vs::Vector::one(), wmp::create_wm_param(vs::Vector::khat(), M_PI));

  // blade point of reference is (1,1,2), and twisted 45 degrees relative to
  // hub
  const SixDOF bladeReference(
    vs::Vector(1.0, 1.0, 2.0),
    wmp::create_wm_param(vs::Vector::khat(), M_PI_4));

  // hub has rotated 90 degrees about the upstream (-ihat) direction
  const SixDOF hubDisp(
    vs::Vector::zero(), wmp::create_wm_param(vs::Vector::ihat(-1.0), M_PI_2));

  const vs::Vector newPosition(1.0, 0.0, 1.0);

  const auto goldRotationDisp = newPosition - bladeReference.position_;

  const auto bldCurRelativeToHub =
    fsi::translation_displacements_from_hub_motion(
      hubReference, hubDisp, bladeReference);

  EXPECT_DOUBLE_EQ(goldRotationDisp.x(), bldCurRelativeToHub.x());
  EXPECT_DOUBLE_EQ(goldRotationDisp.y(), bldCurRelativeToHub.y());
  EXPECT_DOUBLE_EQ(goldRotationDisp.z(), bldCurRelativeToHub.z());
}

} // namespace
} // namespace aero
