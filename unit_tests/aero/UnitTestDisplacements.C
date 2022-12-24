// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

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

// Test displacements interpolation along the (1,1,1) axis, and for rotations
// that start at 0 and end at 90 degrees along the segment
TEST(AeroDisplacements, linear_interp_total_displacements)
{
  const double angle = M_PI_4;
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
  const vs::Vector testPoint = {3.0, 3.0, 3.0};
  auto wmpGold = wmp::create_wm_param(axis, goldAngle);
  // TODO(psakiev) figure out why this isn't passing with Ganesh
  // The current diff is O(1e-2).
  test_wiener_milenkovic(wmpGold, interpDisp.rotation_, testPoint, 1e-12);
}
} // namespace test_displacements
