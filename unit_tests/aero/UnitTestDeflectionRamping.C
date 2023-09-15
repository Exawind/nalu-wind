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
  const double expectedRampFactor = 0.75;
  const double calcRampFactor =
    fsi::linear_ramp_span(spanLocation, zeroRampLocation);
  EXPECT_DOUBLE_EQ(expectedRampFactor, calcRampFactor);
}

TEST(DeflectionRamping, linearRampingAlongSpanOutsideRamp)
{

  const double spanLocation = 0.5;
  const double zeroRampLocation = 0.4;
  const double expectedRampFactor = 1.00;
  const double calcRampFactor =
    fsi::linear_ramp_span(spanLocation, zeroRampLocation);
  EXPECT_DOUBLE_EQ(expectedRampFactor, calcRampFactor);
}

TEST(DeflectionRamping, linearRampingAlongThetaInsideRamp)
{
  const double theta = utils::radians(110.0);
  const double rampSpan = utils::radians(20.0);
  const double thetaZero = utils::radians(120.0);
  const double goldRampFactor = 0.5;

  const aero::SixDOF hub(vs::Vector::zero(), vs::Vector::ihat());
  const vs::Vector root(vs::Vector::one());
  const auto rotation = wmp::create_wm_param(vs::Vector::ihat(), theta);

  const auto pClockWise = wmp::rotate(rotation, root);
  const auto rampClockWise =
    fsi::linear_ramp_theta(hub, root, pClockWise, rampSpan, thetaZero);
  const auto pCClockWise = wmp::rotate(rotation, root, true);
  const auto rampCClockWise =
    fsi::linear_ramp_theta(hub, root, pCClockWise, rampSpan, thetaZero);
  EXPECT_NEAR(goldRampFactor, rampClockWise, 1e-10);
  EXPECT_NEAR(rampClockWise, rampCClockWise, 1e-10);
}

TEST(DeflectionRamping, linearRampingAlongThetaPreRamp)
{
  const double theta = utils::radians(90.0);
  const double rampSpan = utils::radians(20.0);
  const double thetaZero = utils::radians(120.0);
  const double goldRampFactor = 1.0;

  const aero::SixDOF hub(vs::Vector::zero(), vs::Vector::ihat());
  const vs::Vector root(vs::Vector::one());
  const auto rotation = wmp::create_wm_param(vs::Vector::ihat(), theta);

  const auto pClockWise = wmp::rotate(rotation, root);
  const auto rampClockWise =
    fsi::linear_ramp_theta(hub, root, pClockWise, rampSpan, thetaZero);
  const auto pCClockWise = wmp::rotate(rotation, root, true);
  const auto rampCClockWise =
    fsi::linear_ramp_theta(hub, root, pCClockWise, rampSpan, thetaZero);
  EXPECT_DOUBLE_EQ(goldRampFactor, rampClockWise);
  EXPECT_DOUBLE_EQ(rampClockWise, rampCClockWise);
}

TEST(DeflectionRamping, linearRampingAlongThetaPostRamp)
{
  const double theta = utils::radians(121.0);
  const double rampSpan = utils::radians(20.0);
  const double thetaZero = utils::radians(120.0);
  const double goldRampFactor = 0.0;

  const aero::SixDOF hub(vs::Vector::zero(), vs::Vector::ihat());
  const vs::Vector root(vs::Vector::one());
  const auto rotation = wmp::create_wm_param(vs::Vector::ihat(), theta);

  const auto pClockWise = wmp::rotate(rotation, root);
  const auto rampClockWise =
    fsi::linear_ramp_theta(hub, root, pClockWise, rampSpan, thetaZero);
  const auto pCClockWise = wmp::rotate(rotation, root, true);
  const auto rampCClockWise =
    fsi::linear_ramp_theta(hub, root, pCClockWise, rampSpan, thetaZero);
  EXPECT_DOUBLE_EQ(goldRampFactor, rampClockWise);
  EXPECT_DOUBLE_EQ(rampClockWise, rampCClockWise);
}

TEST(DeflectionRamping, temporalRampingPhases)
{
  const double startRamp = 1.0;
  const double endRamp = 2.0;
  EXPECT_DOUBLE_EQ(0.0, fsi::temporal_ramp(0.999, startRamp, endRamp));
  EXPECT_DOUBLE_EQ(0.0, fsi::temporal_ramp(startRamp, startRamp, endRamp));
  EXPECT_DOUBLE_EQ(0.5, fsi::temporal_ramp(1.5, startRamp, endRamp));
  EXPECT_DOUBLE_EQ(1.0, fsi::temporal_ramp(endRamp, startRamp, endRamp));
  EXPECT_DOUBLE_EQ(1.0, fsi::temporal_ramp(5.0, startRamp, endRamp));
}

TEST(DeflectionRamping, booleanDisablesTemporal)
{
  const double startRamp = 1.0;
  const double endRamp = 2.0;
  const double time = 0.0;
  EXPECT_DOUBLE_EQ(1.0, fsi::temporal_ramp(time, startRamp, endRamp, false));
  EXPECT_DOUBLE_EQ(0.0, fsi::temporal_ramp(time, startRamp, endRamp, true));
}

TEST(DeflectionRamping, booleanDisablesTheta)
{
  const double theta = utils::radians(90.0);
  const double rampSpan = utils::radians(20.0);
  const double thetaZero = utils::radians(60.0);

  const aero::SixDOF hub(vs::Vector::zero(), vs::Vector::ihat());
  const vs::Vector root(vs::Vector::one());
  const auto rotation = wmp::create_wm_param(vs::Vector::ihat(), theta);

  const auto pClockWise = wmp::rotate(rotation, root);
  EXPECT_DOUBLE_EQ(0.0, fsi::linear_ramp_theta(hub, root, pClockWise, rampSpan, thetaZero, true));
  EXPECT_DOUBLE_EQ(1.0, fsi::linear_ramp_theta(hub, root, pClockWise, rampSpan, thetaZero, false));
}

TEST(DeflectionRamping, booleanDisablesSpan)
{
  const double spanLocation = 0.3;
  const double zeroRampLocation = 0.4;
  EXPECT_DOUBLE_EQ(0.75, fsi::linear_ramp_span(spanLocation, zeroRampLocation, true));
  EXPECT_DOUBLE_EQ(1.0, fsi::linear_ramp_span(spanLocation, zeroRampLocation, false));
}
} // namespace
