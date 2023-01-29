// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <gtest/gtest.h>
#include <aero/aero_utils/DeflectionRamping.h>

namespace {

TEST(DeflectionRamping, linearRampingAlongSpanInsideRamp)
{

  const double spanLocation = 0.3;
  const double zeroRampLocation = 0.4;
  const double expectedRampFactor = 0.25;
  const double calcRampFactor =
    fsi::linear_ramp_span(spanLocation, zeroRampLocation);
  EXPECT_DOUBLE_EQ(expectedRampFactor, calcRampFactor);
}

TEST(DeflectionRamping, linearRampingAlongSpanOutsideRamp)
{

  const double spanLocation = 0.5;
  const double zeroRampLocation = 0.4;
  const double expectedRampFactor = 0.00;
  const double calcRampFactor =
    fsi::linear_ramp_span(spanLocation, zeroRampLocation);
  EXPECT_DOUBLE_EQ(expectedRampFactor, calcRampFactor);
}
} // namespace
