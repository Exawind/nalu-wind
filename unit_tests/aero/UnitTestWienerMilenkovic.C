#include "vs/vstraits.h"
#include <gtest/gtest.h>
#include <aero/aero_utils/WienerMilenkovic.h>
#include <vs/trig_ops.h>
#include "KokkosInterface.h"

namespace {
using KVector =
  Kokkos::View<vs::Vector*, Kokkos::LayoutRight, sierra::nalu::MemSpace>;

//! compare WM rotation to standard tensor rotation
void
impl_test_WM_rotation()
{

  // device memory
  KVector end("end", 1);
  KVector end_gold("end_gold", 1);
  // host mirrors
  KVector::HostMirror h_end = Kokkos::create_mirror_view(end);
  KVector::HostMirror h_end_gold = Kokkos::create_mirror_view(end_gold);

  Kokkos::parallel_for(
    1, KOKKOS_LAMBDA(int) {
      const double angleDegrees = 90.0;
      const vs::Vector start = {1.0, 0.0, 0.0};
      end_gold(0) = start & vs::zrot(angleDegrees);

      // WM setup
      const auto axis = vs::Vector::khat();
      const auto wmAxis = wmp::generator(utils::radians(angleDegrees)) * axis;
      end(0) = wmp::apply(wmAxis, start);
    });

  // copy data back to host
  Kokkos::deep_copy(h_end, end);
  Kokkos::deep_copy(h_end_gold, end_gold);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(h_end_gold(0)[i], h_end(0)[i], vs::DTraits<double>::eps());
  }
}

TEST(WienerMilenkovic, NGP_rotation) { impl_test_WM_rotation(); }
} // namespace
