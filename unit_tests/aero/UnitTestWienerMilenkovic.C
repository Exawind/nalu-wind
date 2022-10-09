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

//! compare WM rotation to standard tensor rotation
void
impl_test_WM_rotation(vs::Vector axis, vs::Vector point, double angle)
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

  Kokkos::parallel_for(
    1, KOKKOS_LAMBDA(int) {
      dEndGold(0) = dPoint(0) & vs::quaternion(dAxis(0), dAngle(0));

      // WM setup
      const auto wmAxis =
        wmp::create_wm_param(dAxis(0), utils::radians(dAngle(0)));
      dEnd(0) = wmp::rotate(wmAxis, dPoint(0));
    });

  Kokkos::deep_copy(hEnd, dEnd);
  Kokkos::deep_copy(hEndGold, dEndGold);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(hEndGold(0)[i], hEnd(0)[i], vs::DTraits<double>::eps());
  }
}

void
impl_test_WM_compose(vs::Vector v1, vs::Vector v2)
{
  // add v1 and v2 togther
  const auto v3 = wmp::compose(v2, v1);
  // subtract v2 from v3
  const auto v4 = wmp::compose(v2, v3, true);

  // v1 and v4 should be equal
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(v1[i], v4[i], vs::DTraits<double>::eps());
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

TEST(WienerMilenkovic, NGP_composee_add_then_subtract)
{
  impl_test_WM_compose(
    wmp::create_wm_param(vs::Vector::khat(), 30.0),
    wmp::create_wm_param({1.0, 1.0, 1.0}, 25.0));
}

} // namespace test_wmp
