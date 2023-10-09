// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <gtest/gtest.h>
#include <aero/actuator/UtilitiesActuator.h>
#include <algorithm>
#include <functional>
#include <random>
#include <sstream>

namespace sierra {
namespace nalu {

namespace {

template <class T>
double
F_distance(T& p1, T& p2)
{
  double d = 0;
  for (int i = 0; i < 3; i++) {
    d += std::pow(p1[i] - p2[i], 2.0);
  }
  return std::sqrt(d);
}

Point
F_center(Point& A, Point& B, Point& C)
{
  Point center = {
    (A[0] + B[0] + C[0]) / 3.0, (A[1] + B[1] + C[1]) / 3.0,
    (A[2] + B[2] + C[2]) / 3.0};
  return center;
}

std::vector<double>
F_RotateAboutAxis(
  int axis,
  double angle,
  const std::vector<double>& p,
  const std::vector<double>& h)
{
  std::vector<double> pPrime(3);
  const double dCos{std::cos(angle)}, dSin{std::sin(angle)};
  int j = (axis + 1) % 3;
  int k = (j + 1) % 3;
  pPrime[axis] = p[axis];
  pPrime[j] = dCos * (p[j] - h[j]) - dSin * (p[k] - h[k]) + h[j];
  pPrime[k] = dCos * (p[k] - h[k]) + dSin * (p[j] - h[j]) + h[k];
  return pPrime;
}

} // namespace

TEST(ActuatorSweptPointLocator, NGP_PointsOnACircle)
{
  const double PI = std::acos(-1.0);
  const std::vector<double> origin(3, 0.0);
  std::vector<double> hub(3, 0.0);
  std::mt19937::result_type seed = std::time(0);
  auto fn_np_real_rand = std::bind(
    std::uniform_real_distribution<double>(-1.0, 1.0), std::mt19937(seed));
  auto fn_int_rand =
    std::bind(std::uniform_int_distribution<int>(0, 2), std::mt19937(seed));

  // check rotation function
  {
    std::vector<double> a(3), b(3);
    a = {1, 0, 0};
    b = {4, 1, 0};
    auto c = F_RotateAboutAxis(2, 0.5 * PI, a, origin);
    ASSERT_NEAR(0.0, c[0], 1e-12);
    ASSERT_NEAR(1.0, c[1], 1e-12);
    ASSERT_NEAR(0.0, c[2], 1e-12);

    a[1] = 1.0;
    c = F_RotateAboutAxis(2, 0.5 * PI, b, a);
    ASSERT_NEAR(1.0, c[0], 1e-12);
    ASSERT_NEAR(4.0, c[1], 1e-12);
    ASSERT_NEAR(0.0, c[2], 1e-12);
  }

  int i = 0;
  while (i < 20) {
    std::ostringstream message;
    for (int ih = 0; ih < 3; ih++) {
      hub[ih] = fn_np_real_rand();
    }

    const int index = fn_int_rand();

    // three random points with unique directions
    // with respect to the hub
    std::vector<double> p1(3), p2(3), p3(3);
    p1 = {fn_np_real_rand(), fn_np_real_rand(), fn_np_real_rand()};
    p2 = F_RotateAboutAxis(index, 2.0 * PI / 3.0, p1, hub);
    p3 = F_RotateAboutAxis(index, 4.0 * PI / 3.0, p1, hub);

    // All the points are equidistance from the hub
    Point A{p1[0], p1[1], p1[2]}, B{p2[0], p2[1], p2[2]},
      C{p3[0], p3[1], p3[2]};

    actuator_utils::SweptPointLocator locator;
    locator.update_point_location(0, A);
    locator.update_point_location(1, B);
    locator.update_point_location(2, C);

    Point center = F_center(A, B, C);
    const double radius = F_distance<Point>(A, center);

    auto contPnts = locator.get_control_points();

    // just in case it fails...
    message << "Failure for points: " << std::endl
            << "A: " << A[0] << ", " << A[1] << ", " << A[2] << std::endl
            << "B: " << B[0] << ", " << B[1] << ", " << B[2] << std::endl
            << "C: " << C[0] << ", " << C[1] << ", " << C[2] << std::endl
            << "Hub: " << hub[0] << ", " << hub[1] << ", " << hub[2]
            << std::endl
            << "Center: " << center[0] << ", " << center[1] << ", " << center[2]
            << std::endl
            << "Control Point 0: " << contPnts[0][0] << ", " << contPnts[0][1]
            << ", " << contPnts[0][2] << std::endl
            << "Control Point 1: " << contPnts[1][0] << ", " << contPnts[1][1]
            << ", " << contPnts[1][2] << std::endl
            << "Control Point 2: " << contPnts[2][0] << ", " << contPnts[2][1]
            << ", " << contPnts[2][2] << std::endl
            << "Rotation Axis: " << index << std::endl;

    int match1{(index + 1) % 3}, match2{(index + 2) % 3};
    ASSERT_NEAR(hub[match1], center[match1], 1e-12) << message.str();
    ASSERT_NEAR(hub[match2], center[match2], 1e-12) << message.str();

    double t = fn_np_real_rand() * PI;
    Point Ap = locator(t);

    // is point on circle that intersected all three points
    EXPECT_NEAR(0.0, std::fabs(F_distance<Point>(Ap, center) - radius), 1e-12)
      << message.str() << "Failure at t=: " << t << " was calculated as "
      << Ap[0] << ", " << Ap[1] << ", " << Ap[2] << std::endl;
    i++;
  }
}

} // namespace nalu
} // namespace sierra
