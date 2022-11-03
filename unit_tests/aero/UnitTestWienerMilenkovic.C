#include "vs/vector.h"
#include "vs/vstraits.h"
#include <gtest/gtest.h>
#include <aero/aero_utils/WienerMilenkovic.h>
#include <vs/trig_ops.h>
#include "KokkosInterface.h"

namespace {
using KVector =
  Kokkos::View<vs::Vector*, Kokkos::LayoutRight, sierra::nalu::MemSpace>;
using KDouble =
  Kokkos::View<double*, Kokkos::LayoutRight, sierra::nalu::MemSpace>;
const double eps_test = vs::DTraits<double>::eps() * 1.e2;

//! compare WM rotation to standard tensor rotation
void
impl_test_WM_rotation(
  vs::Vector axis, vs::Vector point, double angle, bool transpose = false)
{

  // device memory
  KVector dEnd("end", 1);
  KVector dEndGold("end_gold", 1);
  KVector dAxis("axis", 1);
  KVector dPoint("point", 1);
  KDouble dAngle("angle_in_degrees", 1);
  // host mirrors
  auto hEnd = Kokkos::create_mirror_view(dEnd);
  auto hEndGold = Kokkos::create_mirror_view(dEndGold);
  auto hAxis = Kokkos::create_mirror_view(dAxis);
  auto hPoint = Kokkos::create_mirror_view(dPoint);
  auto hAngle = Kokkos::create_mirror_view(dAngle);

  hAxis(0) = axis;
  hPoint(0) = point;
  hAngle(0) = angle;

  Kokkos::deep_copy(dAxis, hAxis);
  Kokkos::deep_copy(dPoint, hPoint);
  Kokkos::deep_copy(dAngle, hAngle);

  bool tp = transpose;

  Kokkos::parallel_for(
    1, KOKKOS_LAMBDA(int) {
      if (tp)
        dEndGold(0) = dPoint(0) & vs::quaternion(dAxis(0), -dAngle(0));
      else
        dEndGold(0) = dPoint(0) & vs::quaternion(dAxis(0), dAngle(0));

      // WM setup
      const auto wmAxis =
        wmp::create_wm_param(dAxis(0), utils::radians(dAngle(0)));
      dEnd(0) = wmp::rotate(wmAxis, dPoint(0), transpose);
    });

  Kokkos::deep_copy(hEnd, dEnd);
  Kokkos::deep_copy(hEndGold, dEndGold);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(hEndGold(0)[i], hEnd(0)[i], eps_test);
  }
}

void
impl_test_WM_compose_two_rot(
  vs::Vector point, vs::Vector v1, vs::Vector v2, double angle1, double angle2)
{

  // device memory
  KVector dEnd("end", 1);
  KVector dEndGold("end_gold", 1);
  KVector dAxis("axis", 2);
  KVector dPoint("point", 1);
  KDouble dAngle("angle_in_degrees", 2);
  // host mirrors
  auto hEnd = Kokkos::create_mirror_view(dEnd);
  auto hEndGold = Kokkos::create_mirror_view(dEndGold);
  auto hAxis = Kokkos::create_mirror_view(dAxis);
  auto hPoint = Kokkos::create_mirror_view(dPoint);
  auto hAngle = Kokkos::create_mirror_view(dAngle);

  hAxis(0) = v1;
  hAxis(1) = v2;
  hPoint(0) = point;
  hAngle(0) = angle1;
  hAngle(1) = angle2;

  Kokkos::deep_copy(dAxis, hAxis);
  Kokkos::deep_copy(dPoint, hPoint);
  Kokkos::deep_copy(dAngle, hAngle);

  Kokkos::parallel_for(
    1, KOKKOS_LAMBDA(int) {
      dEndGold(0) = (dPoint(0) & vs::quaternion(dAxis(0), dAngle(0))) &
                    vs::quaternion(dAxis(1), dAngle(1));

      // WM setup
      const auto wmAxis1 =
        wmp::create_wm_param(dAxis(0), utils::radians(dAngle(0)));
      const auto wmAxis2 =
        wmp::create_wm_param(dAxis(1), utils::radians(dAngle(1)));
      const auto wmCompose = wmp::push(wmAxis2, wmAxis1);
      dEnd(0) = wmp::rotate(wmCompose, dPoint(0));
    });

  Kokkos::deep_copy(hEnd, dEnd);
  Kokkos::deep_copy(hEndGold, dEndGold);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(hEndGold(0)[i], hEnd(0)[i], eps_test);
  }
}

void
impl_test_WM_compose_add_sub(vs::Vector v1, vs::Vector v2, vs::Vector point)
{
  KVector dEnd("end", 1);
  KVector dAxis("axis", 2);
  KVector dPoint("point", 1);
  // host mirrors
  auto hEnd = Kokkos::create_mirror_view(dEnd);
  auto hAxis = Kokkos::create_mirror_view(dAxis);
  auto hPoint = Kokkos::create_mirror_view(dPoint);

  hAxis(0) = v1;
  hAxis(1) = v2;
  hPoint(0) = point;

  const auto gold = wmp::rotate(v1, point);

  Kokkos::deep_copy(dAxis, hAxis);
  Kokkos::deep_copy(dPoint, hPoint);

  Kokkos::parallel_for(
    1, KOKKOS_LAMBDA(int) {
      // add v1 and v2 togther
      const auto v3 = wmp::push(dAxis(1), dAxis(0));
      // subtract v2 from v3
      const auto v4 = wmp::pop(dAxis(1), v3);

      dEnd(0) = wmp::rotate(v4, point);
    });

  Kokkos::deep_copy(hEnd, dEnd);

  for (int i = 0; i < 3; ++i) {
    ASSERT_NEAR(gold[i], hEnd(0)[i], eps_test) << i << " index failed.";
  }
}
} // namespace

namespace test_wmp {
TEST(WienerMilenkovic, NGP_rotation_major_axis)
{
  // Test major axis rotations
  impl_test_WM_rotation(vs::Vector::khat(), vs::Vector::ihat(), 90.0);
  impl_test_WM_rotation(vs::Vector::ihat(), vs::Vector::khat(), 90.0);
  impl_test_WM_rotation(vs::Vector::jhat(), vs::Vector::khat(), 90.0);
  impl_test_WM_rotation(
    vs::Vector::jhat() * 10.0, vs::Vector::ihat() * 3.0, 90.0);
}

TEST(WienerMilenkovic, NGP_rotation_arbitrary_axis)
{
  // test arbitrary axis rotation
  impl_test_WM_rotation(vs::Vector::one(), vs::Vector::khat(), 90.0);
}

TEST(WienerMilenkovic, NGP_negative_rotation)
{
  impl_test_WM_rotation(vs::Vector::khat(), vs::Vector::ihat(), 90.0, true);
  // negative angle is the same as transpose.  not sure we need both...
  impl_test_WM_rotation(vs::Vector::khat(), vs::Vector::ihat(), -90.0);
}

TEST(WienerMilenkovic, NGP_compose_two_rotations_same_as_two_quaternions)
{
  impl_test_WM_compose_two_rot(
    vs::Vector::ihat(), vs::Vector::khat(), vs::Vector::jhat(), 90.0, 45.0);
}

TEST(WienerMilenkovic, NGP_compose_push_then_pop_param)
{
  const auto wmp1 = wmp::create_wm_param(vs::Vector::khat(), 30.0);
  const auto wmp2 = wmp::create_wm_param({1.0, 1.0, 1.0}, 25.0);
  const vs::Vector point = {1.0, 1.0, 1.0};
  impl_test_WM_compose_add_sub(wmp1, wmp2, point);
}

} // namespace test_wmp
