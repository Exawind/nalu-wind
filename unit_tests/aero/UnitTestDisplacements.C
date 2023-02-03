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

testing::Message&
operator<<(testing::Message& out, const vs::Vector& vec)
{
  out << "(" << vec.x() << " " << vec.y() << " " << vec.z() << ")";
  return out;
}

namespace test_displacements {
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

TEST(AeroDisplacements, creation_from_pointer)
{
  std::vector<double> openfastSurrogate(6, 1.0);
  aero::SixDOF disp(openfastSurrogate.data());
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(openfastSurrogate[i], disp.position_[i]);
    EXPECT_DOUBLE_EQ(openfastSurrogate[i + 3], disp.orientation_[i]);
  }
}

TEST(AeroDisplacements, creation_from_vs_vector)
{
  aero::SixDOF disp(vs::Vector::one(), 2.0 * vs::Vector::one());
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(1.0, disp.position_[i]);
    EXPECT_DOUBLE_EQ(2.0, disp.orientation_[i]);
  }
}

TEST(AeroDisplacements, add_six_dof_together)
{
  const auto dispA = vs::Vector::ihat();
  const auto dispB = vs::Vector::khat();

  const auto orientA =
    wmp::create_wm_param(vs::Vector::jhat(), utils::radians(15.0));
  const auto orientB =
    wmp::create_wm_param(vs::Vector::ihat(), utils::radians(15.0));

  const aero::SixDOF a(dispA, orientA);
  const aero::SixDOF b(dispB, orientB);

  const auto c = a + b;

  const auto goldTrans = dispA + dispB;
  // adding b to a, so pushing b wmp onto a stack
  const auto goldRot = wmp::push(orientB, orientA);
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(goldTrans[i], c.position_[i]) << i;
  }
  test_wiener_milenkovic(goldRot, c.orientation_, vs::Vector::one(), 1e-12);
}

TEST(AeroDisplacements, subtract_six_dof)
{
  const auto dispA = vs::Vector::ihat();
  const auto dispB = vs::Vector::khat();

  const auto orientA =
    wmp::create_wm_param(vs::Vector::jhat(), utils::radians(15.0));
  const auto orientB =
    wmp::create_wm_param(vs::Vector::ihat(), utils::radians(15.0));

  const aero::SixDOF a(dispA, orientA);
  const aero::SixDOF b(dispB, orientB);

  const auto c = a - b;
  const auto goldTrans = dispA - dispB;
  const auto goldRot = wmp::pop(orientB, orientA);
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(goldTrans[i], c.position_[i]) << i;
  }
  test_wiener_milenkovic(goldRot, c.orientation_, vs::Vector::one(), 1e-12);
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
  const aero::SixDOF start(vs::Vector::zero(), wmp::create_wm_param(axis, 0.0));
  const aero::SixDOF end(vs::Vector::one(), wmp::create_wm_param(axis, angle));

  auto interpDisp =
    aero::linear_interp_total_displacement(start, end, interpFactor);

  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(interpFactor, interpDisp.position_[i])
      << "Failed i: " << i;
  }

  const double goldAngle = angle * interpFactor;
  // put the point on the axis + a small offset
  const auto axisOffset = (vs::Vector::one() - axis) * 1e-3;
  const vs::Vector testPoint = axis * interpFactor + axisOffset;
  auto wmpGold = wmp::create_wm_param(axis, goldAngle);
  test_wiener_milenkovic(wmpGold, interpDisp.orientation_, testPoint, 1e-10);
}

TEST(
  AeroDisplacements,
  compute_translational_displacments_translation_only_deflections)
{
  const double eps = 1e-14;
  const double delta = 0.1;
  const double angleRot = 5.0 / 180.0 * M_PI;

  // cfd pos will be based on simple solid body rotation around the x axis
  const vs::Vector rotX = wmp::create_wm_param(vs::Vector::ihat(), -angleRot);
  // benign reference position on the z axis
  // looks like openfast updates this so assuming solid body rotation is stored
  // in the referencePos
  const aero::SixDOF referencePos(vs::Vector::khat(), rotX);

  // CFD Pos is using "coordinates" field so it is likely fixed in time. Need to
  // confirm with Ganesh for now we will treat this as an offset from the
  // referencePos in openfast
  const vs::Vector cfdPos = referencePos.position_ + vs::Vector::ihat();

  const aero::SixDOF deflections(vs::Vector::one() * delta, vs::Vector::zero());

  const auto displacements = aero::compute_translational_displacements(
    deflections, referencePos, cfdPos);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(delta, displacements[i], eps) << "i: " << i;
  }
}

TEST(AeroDisplacements, convert_to_local_coordiantes)
{
  const double tol = 1e-16;
  const double eps = 0.1;
  // refenence rotations to be assembled into one WienerMilenkovic param
  const auto cone =
    wmp::create_wm_param(vs::Vector::jhat(), utils::radians(5.0));
  const auto rotation =
    wmp::create_wm_param(vs::Vector::ihat(), utils::radians(25.0));
  const auto yaw =
    wmp::create_wm_param(vs::Vector::khat(), utils::radians(2.0));

  const auto totalRotations = wmp::push(yaw, wmp::push(rotation, cone));

  const vs::Vector initialPosition = {0.0, 0.0, 1.0};

  const aero::SixDOF refPos(initialPosition, totalRotations);

  // this would be like the position on the surface of an airfoil relative to
  // it's aerodynamic coodinate system
  const vs::Vector localPosGold = {eps, eps, eps};
  // take the initialPosition and add the offset in the inertialFram i.e. undo
  // rotations for converting from inertial to local
  const auto inertialPos =
    initialPosition + wmp::rotate(totalRotations, localPosGold, true);

  const auto localPos = aero::local_aero_coordinates(inertialPos, refPos);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(localPosGold[i], localPos[i], tol) << "i: " << i;
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
  const vs::Vector rotX = wmp::create_wm_param(vs::Vector::ihat(), -angleRot);

  // benign reference position on the z axis
  const aero::SixDOF referencePos(vs::Vector::khat(), rotX);

  // CFD Pos is using "coordinates" field so it is likely fixed in time. Need to
  // confirm with Ganesh for now we will treat this as an offset from the
  // referencePos in openfast
  const vs::Vector cfdPos = referencePos.position_ + vs::Vector::ihat();

  const aero::SixDOF deflections(
    vs::Vector::zero(), wmp::create_wm_param(vs::Vector::jhat(), angleDef));

  const auto displacements = aero::compute_translational_displacements(
    deflections, referencePos, cfdPos);

  // deflections will be based on the rotation of the difference between the cfd
  // and the referencePos and are applied in the opposite direction of the angle
  // supplied
  //
  // subtration
  const vs::Vector goldDef =
    vs::Vector({stk::math::cos(angleDef), 0.0, stk::math::sin(angleDef)}) -
    vs::Vector::ihat();

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(goldDef[i], displacements[i], eps) << "i: " << i;
  }
}
} // namespace test_displacements
