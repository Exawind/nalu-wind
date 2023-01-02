// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "aero/aero_utils/WienerMilenkovic.h"
#include <gtest/gtest.h>
#include <aero/aero_utils/displacements.h>

namespace test_displacements {
TEST(AeroDisplacements, creation_from_pointer)
{
  std::vector<double> openfastSurrogate(6, 1.0);
  aero::Displacement disp(openfastSurrogate.data());
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(openfastSurrogate[i], disp.translation_[i]);
    EXPECT_DOUBLE_EQ(openfastSurrogate[i + 3], disp.rotation_[i]);
  }
}

TEST(AeroDisplacements, creation_from_vs_vector)
{
  aero::Displacement disp(vs::Vector::one(), 2.0 * vs::Vector::one());
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(1.0, disp.translation_[i]);
    EXPECT_DOUBLE_EQ(2.0, disp.rotation_[i]);
  }
}

//! Test that two WM Params give the same end location for a point
void
test_wiener_milenkovic(
  vs::Vector goldWmp, vs::Vector testWmp, vs::Vector testPoint, double eps)
{
  auto goldPnt = wmp::rotate(goldWmp, testPoint);
  auto testPnt = wmp::rotate(testWmp, testPoint);
  EXPECT_NEAR(goldPnt.x(), testPnt.x(), eps)
    << "Gold WMP: " << goldWmp << " testWmp: " << testWmp;
  EXPECT_NEAR(goldPnt.y(), testPnt.y(), eps)
    << "Gold WMP: " << goldWmp << " testWmp: " << testWmp;
  EXPECT_NEAR(goldPnt.z(), testPnt.z(), eps)
    << "Gold WMP: " << goldWmp << " testWmp: " << testWmp;
}

// Test displacements interpolation along the (1,0,0) axis, and for rotations
// that start at 0 and end at 1.0 degree along the segment
// We use a small angle and a small offset from the axis of rotation since the
// WMP paramters reduce in accuracy as the angle increases
TEST(AeroDisplacements, linear_interp_total_displacements)
{
  const double angle = 1.0 / 90.0 * M_PI_4;
  const double interpFactor = 0.5;
  const auto axis = vs::Vector::ihat();
  const aero::Displacement start(
    vs::Vector::zero(), wmp::create_wm_param(axis, 0.0));
  const aero::Displacement end(
    vs::Vector::one(), wmp::create_wm_param(axis, angle));

  auto interpDisp =
    aero::linear_interp_total_displacement(start, end, interpFactor);

  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(interpFactor, interpDisp.translation_[i])
      << "Failed i: " << i;
  }

  const double goldAngle = angle * interpFactor;
  // put the point on the axis + a small offset
  const auto axisOffset = (vs::Vector::one() - axis) * 1e-3;
  const vs::Vector testPoint = axis * interpFactor + axisOffset;
  auto wmpGold = wmp::create_wm_param(axis, goldAngle);
  test_wiener_milenkovic(wmpGold, interpDisp.rotation_, testPoint, 1e-10);
}

TEST(
  AeroDisplacements,
  compute_translational_displacments_translation_only_deflections)
{
  const double eps = 1e-14;
  const double delta = 0.1;
  const double angleRot = 5.0 / 180.0 * M_PI;

  // cfd pos will be based on simple solid body rotation around the x axis
  const vs::Vector rotX = wmp::create_wm_param(vs::Vector::ihat(), angleRot);
  // benign reference position on the z axis
  // looks like openfast updates this so assuming solid body rotation is stored
  // in the referencePos
  const aero::Displacement referencePos(vs::Vector::khat(), rotX);

  const vs::Vector cfdPos = wmp::rotate(rotX, referencePos.translation_);

  const aero::Displacement deflections(
    vs::Vector::one() * delta, vs::Vector::zero());

  const auto displacements = aero::compute_translational_displacements(
    deflections, referencePos, cfdPos);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(delta, displacements[i], eps);
  }
}

TEST(
  AeroDisplacements,
  compute_translational_displacments_rotation_only_deflections)
{
  const double eps = 1e-14;
  const double angleRot = 5.0 / 180.0 * M_PI;
  const double angleDef = 10.0 / 180.0 * M_PI;

  // cfd pos will be based on simple solid body rotation around the x axis
  const vs::Vector rotX = wmp::create_wm_param(vs::Vector::ihat(), angleRot);

  // benign reference position on the z axis
  const aero::Displacement referencePos(vs::Vector::khat(), rotX);

  const vs::Vector cfdPos = wmp::rotate(rotX, referencePos.translation_);

  const aero::Displacement deflections(
    vs::Vector::zero(), wmp::create_wm_param(vs::Vector::jhat(), angleDef));

  const auto displacements = aero::compute_translational_displacements(
    deflections, referencePos, cfdPos);

  // deflections should be based on the delta from y axis rotation
  const vs::Vector goldDef = {
    stk::math::sin(angleDef), 0.0, stk::math::cos(angleDef)};

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(goldDef[i], displacements[i], eps);
  }
}
} // namespace test_displacements
