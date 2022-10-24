#include <gtest/gtest.h>

#include "wind_energy/LidarPatterns.h"
#include "NaluParsing.h"
#include "UnitTestUtils.h"
#include "master_element/TensorOps.h"

#include <yaml-cpp/yaml.h>

#include <ostream>
#include <memory>
#include <array>

namespace sierra {
namespace nalu {

std::array<vs::Vector, 8> box{{
  {-1, -1, -1},
  {+1, -1, -1},
  {+1, +1, -1},
  {-1, +1, -1},
  {-1, -1, +1},
  {+1, -1, +1},
  {+1, +1, +1},
  {-1, +1, +1},
}};

TEST(box_intersect, line_goes_straight_through)
{
  vs::Vector origin = {-10, 0, 0};
  vs::Vector line = {1, 0, 0};

  auto [found, seg] = details::line_intersection_with_box(box, origin, line);
  ASSERT_TRUE(found);

  ASSERT_DOUBLE_EQ(seg.tail_[0], -1);
  ASSERT_DOUBLE_EQ(seg.tail_[1], 0);
  ASSERT_DOUBLE_EQ(seg.tail_[2], 0);

  ASSERT_DOUBLE_EQ(seg.tip_[0], +1);
  ASSERT_DOUBLE_EQ(seg.tip_[1], 0);
  ASSERT_DOUBLE_EQ(seg.tip_[2], 0);
}

TEST(box_intersect, line_goes_straight_through_abl_box)
{
  vs::Vector origin = {-1000, 2500, 100};

  vs::Vector line = {0.996195, -0.0871557, 0};
  std::array<vs::Vector, 8> abl_box;
  for (int n = 0; n < 8; ++n) {
    const auto new_x = 2500 * (1 + box[n][0]);
    const auto new_y = 2500 * (1 + box[n][1]);
    const auto new_z = 500 * (1 + box[n][2]);
    abl_box[n] = vs::Vector(new_x, new_y, new_z);
  }

  auto [found, seg] =
    details::line_intersection_with_box(abl_box, origin, line);
  ASSERT_TRUE(found);
  ASSERT_NEAR(seg.tail_[0], 0, 1e-10);
  ASSERT_NEAR(seg.tail_[2], 100, 1e-10);
  ASSERT_NEAR(seg.tip_[0], 5000, 1e-10);
  ASSERT_NEAR(seg.tip_[2], 100, 1e-10);
}

TEST(box_intersect, line_goes_through_corners)
{
  vs::Vector origin = {-2, -2, -2};
  vs::Vector line = {1, 1, 1};

  auto [found, seg] = details::line_intersection_with_box(box, origin, line);
  ASSERT_TRUE(found);

  ASSERT_DOUBLE_EQ(seg.tail_[0], -1);
  ASSERT_DOUBLE_EQ(seg.tail_[1], -1);
  ASSERT_DOUBLE_EQ(seg.tail_[2], -1);

  ASSERT_DOUBLE_EQ(seg.tip_[0], +1);
  ASSERT_DOUBLE_EQ(seg.tip_[1], +1);
  ASSERT_DOUBLE_EQ(seg.tip_[2], +1);
}

TEST(box_intersect, line_tangent_to_box)
{
  vs::Vector corner = box[0];
  const double theta = -M_PI_4;
  vs::Vector origin = {
    box[0][0] + std::cos(theta), box[0][1] + std::sin(theta), box[0][2]};
  vs::Vector line = {-std::cos(theta), -std::sin(theta), 0};

  auto [found, seg] = details::line_intersection_with_box(box, origin, line);
  ASSERT_TRUE(found);

  ASSERT_DOUBLE_EQ(seg.tail_[0], -1);
  ASSERT_DOUBLE_EQ(seg.tail_[1], -1);
  ASSERT_DOUBLE_EQ(seg.tail_[2], -1);

  ASSERT_DOUBLE_EQ(seg.tip_[0], -1);
  ASSERT_DOUBLE_EQ(seg.tip_[1], -1);
  ASSERT_DOUBLE_EQ(seg.tip_[2], -1);
}

TEST(box_intersect, line_goes_along_face)
{
  // tolerance-y

  vs::Vector origin = {-2, 0, -1};
  vs::Vector line = {1, 0, 0};

  auto [found, seg] = details::line_intersection_with_box(box, origin, line);
  ASSERT_TRUE(found);

  ASSERT_DOUBLE_EQ(seg.tail_[0], -1);
  ASSERT_DOUBLE_EQ(seg.tail_[1], 0);
  ASSERT_DOUBLE_EQ(seg.tail_[2], -1);

  ASSERT_DOUBLE_EQ(seg.tip_[0], +1);
  ASSERT_DOUBLE_EQ(seg.tip_[1], 0);
  ASSERT_DOUBLE_EQ(seg.tip_[2], -1);
}

TEST(box_intersect, line_goes_along_edge)
{
  // tolerance-y

  vs::Vector edge = box[0] - box[1];
  vs::Vector origin = vs::Vector(-2, 0, 0) + (box[1] - box[0]);
  vs::Vector line = {1, 0, 0};

  auto [found, seg] = details::line_intersection_with_box(box, origin, line);
  ASSERT_TRUE(found);

  ASSERT_DOUBLE_EQ(seg.tail_[0], 1);
  ASSERT_DOUBLE_EQ(seg.tail_[1], 0);
  ASSERT_DOUBLE_EQ(seg.tail_[2], 0);

  ASSERT_DOUBLE_EQ(seg.tip_[0], 1);
  ASSERT_DOUBLE_EQ(seg.tip_[1], 0);
  ASSERT_DOUBLE_EQ(seg.tip_[2], 0);
}

TEST(box_intersect, line_does_not_intersect)
{
  vs::Vector edge = box[0] - box[1];
  vs::Vector origin = vs::Vector(-100, 0, 0);
  vs::Vector line = {0, 1, 0};

  auto [found, seg] = details::line_intersection_with_box(box, origin, line);
  ASSERT_FALSE(found);
}

class RadarParseFixture : public ::testing::Test
{
public:
  RadarParseFixture() {}
  RadarSegmentGenerator slgen;

  std::string radar_nobox =
    "  radar_specifications:                                \n"
    "    angular_speed: 1 # deg/s                           \n"
    "    sweep_angle: 20 # degrees                          \n"
    "    center: [-10000,0,0]                               \n"
    "    beam_length: 1e5                                   \n"
    "    axis: [1,0,0]                                      \n";
};

TEST_F(RadarParseFixture, load_one_corner_throws)
{
  const std::string spec_str =
    radar_nobox + "    box_0: [-2500,-2500,0]                             \n";

  auto spec = YAML::Load(spec_str)["radar_specifications"];
  ASSERT_THROW(slgen.load(spec), std::runtime_error);
}

TEST_F(RadarParseFixture, load_all_corners_degenerate)
{
  const std::string spec_str =
    radar_nobox + "    box_1: [-2500,-2500,0]                             \n"
                  "    box_2: [-2500,-2500,0]                             \n"
                  "    box_3: [-2500,-2500,0]                             \n"
                  "    box_4: [-2500,-2500,0]                             \n"
                  "    box_5: [-2500,-2500,0]                             \n"
                  "    box_6: [-2500,-2500,0]                             \n"
                  "    box_7: [-2500,-2500,0]                             \n"
                  "    box_8: [-2500,-2500,0]                             \n";

  auto spec = YAML::Load(spec_str)["radar_specifications"];
  ASSERT_THROW(slgen.load(spec), std::runtime_error);
}

TEST_F(RadarParseFixture, zero_indexed_throws)
{
  const std::string spec_str =
    radar_nobox + "    box_0: [-2500,-2500,0]                             \n"
                  "    box_1: [ 2500,-2500,0]                             \n"
                  "    box_2: [ 2500, 2500,0]                             \n"
                  "    box_3: [-2500, 2500,0]                             \n"
                  "    box_4: [-2500,-2500,3000]                          \n"
                  "    box_5: [2500,-2500,3000]                           \n"
                  "    box_6: [2500,2500,3000]                            \n"
                  "    box_7: [-2500,2500,3000]                           \n";

  auto spec = YAML::Load(spec_str)["radar_specifications"];
  ASSERT_THROW(slgen.load(spec), std::runtime_error);
}

TEST_F(RadarParseFixture, load_all_corners_valid)
{
  const std::string spec_str =
    radar_nobox + "    box_1: [-2500,-2500,0]                             \n"
                  "    box_2: [ 2500,-2500,0]                             \n"
                  "    box_3: [ 2500, 2500,0]                             \n"
                  "    box_4: [-2500, 2500,0]                             \n"
                  "    box_5: [-2500,-2500,3000]                          \n"
                  "    box_6: [2500,-2500,3000]                           \n"
                  "    box_7: [2500,2500,3000]                            \n"
                  "    box_8: [-2500,2500,3000]                           \n";

  auto spec = YAML::Load(spec_str)["radar_specifications"];
  ASSERT_NO_THROW(slgen.load(spec));
}

TEST_F(RadarParseFixture, bbox_load_valid)
{
  const std::string spec_str =
    radar_nobox + "    bbox: [-2500,-2500,0,2500,2500,100]                \n";

  auto spec = YAML::Load(spec_str)["radar_specifications"];
  ASSERT_NO_THROW(slgen.load(spec));
}

TEST_F(RadarParseFixture, bbox_load_invalid)
{
  const std::string spec_str =
    radar_nobox + "    bbox: [-2500,-2500,0,2500,2500,0]                \n";

  auto spec = YAML::Load(spec_str)["radar_specifications"];
  ASSERT_THROW(slgen.load(spec), std::runtime_error);
}

class RadarScanFixture : public ::testing::Test
{
public:
  RadarScanFixture()
  {
    slgen.load(YAML::Load(radar_spec_str)["radar_specifications"]);
  }
  RadarSegmentGenerator slgen;
  std::string radar_spec_str =
    "  radar_specifications:                                \n"
    "    angular_speed: 10 # deg/s                          \n"
    "    sweep_angle: 20 # degrees                          \n"
    "    center: [-10000,0,90]                              \n"
    "    beam_length: 100000                                \n"
    "    reset_time_delta: 1.0                              \n"
    "    axis: [1,0,0]                                      \n"
    "    box_1: [-2500,-2500,0]                             \n"
    "    box_2: [ 2500,-2500,0]                             \n"
    "    box_3: [ 2500, 2500,0]                             \n"
    "    box_4: [-2500, 2500,0]                             \n"
    "    box_5: [-2500,-2500,3000]                          \n"
    "    box_6: [2500,-2500,3000]                           \n"
    "    box_7: [2500,2500,3000]                            \n"
    "    box_8: [-2500,2500,3000]                           \n";

  double tol_ = 1e-10;
  double reset_time_ = 1;
  double sweep_time_ = 2; //(20/10)
  double sweep_angle_ = convert::degrees_to_radians(20);

  const double x_orig_ = -10000;
  const double x_near_ = -2500;
  const double x_far_ = 2500;
  const double z_near_ = 90;
  const double z_far_ = 90;
};

TEST_F(RadarScanFixture, correct_at_time_zero)
{
  const double time = 0;
  auto seg = slgen.generate(time);

  const double angle = -sweep_angle_ / 2;

  const auto y_near = (x_near_ - x_orig_) * std::tan(angle);
  const auto y_far = (x_far_ - x_orig_) * std::tan(angle);

  ASSERT_NEAR(seg.tail_[0], x_near_, tol_);
  ASSERT_NEAR(seg.tail_[1], y_near, tol_);
  ASSERT_NEAR(seg.tail_[2], z_near_, tol_);

  ASSERT_NEAR(seg.tip_[0], x_far_, tol_);
  ASSERT_NEAR(seg.tip_[1], y_far, tol_);
  ASSERT_NEAR(seg.tip_[2], z_far_, tol_);
}

TEST_F(RadarScanFixture, correct_at_end_of_sweep)
{
  const double time = sweep_time_;
  auto seg = slgen.generate(time);

  const double angle = sweep_angle_ / 2;

  const auto y_near = (x_near_ - x_orig_) * std::tan(angle);
  const auto y_far = (x_far_ - x_orig_) * std::tan(angle);

  ASSERT_NEAR(seg.tail_[0], x_near_, tol_);
  ASSERT_NEAR(seg.tail_[1], y_near, tol_);
  ASSERT_NEAR(seg.tail_[2], z_near_, tol_);

  ASSERT_NEAR(seg.tip_[0], x_far_, tol_);
  ASSERT_NEAR(seg.tip_[1], y_far, tol_);
  ASSERT_NEAR(seg.tip_[2], z_far_, tol_);
}

TEST_F(RadarScanFixture, pauses_at_end_of_forward_sweep)
{
  const double time = sweep_time_ + reset_time_;
  auto seg = slgen.generate(time);

  const double angle = sweep_angle_ / 2;

  const auto y_near = (x_near_ - x_orig_) * std::tan(angle);
  const auto y_far = (x_far_ - x_orig_) * std::tan(angle);

  ASSERT_NEAR(seg.tail_[0], x_near_, tol_);
  ASSERT_NEAR(seg.tail_[1], y_near, tol_);
  ASSERT_NEAR(seg.tail_[2], z_near_, tol_);

  ASSERT_NEAR(seg.tip_[0], x_far_, tol_);
  ASSERT_NEAR(seg.tip_[1], y_far, tol_);
  ASSERT_NEAR(seg.tip_[2], z_far_, tol_);
}

TEST_F(RadarScanFixture, mid_sweep_is_zero_angle)
{
  const double time = 3 * sweep_time_ / 2 + reset_time_;
  auto seg = slgen.generate(time);

  const double angle = 0;
  const auto y_near = (x_near_ - x_orig_) * std::tan(angle);
  const auto y_far = (x_far_ - x_orig_) * std::tan(angle);

  ASSERT_NEAR(seg.tail_[0], x_near_, tol_);
  ASSERT_NEAR(seg.tail_[1], y_near, tol_);
  ASSERT_NEAR(seg.tail_[2], z_near_, tol_);

  ASSERT_NEAR(seg.tip_[0], x_far_, tol_);
  ASSERT_NEAR(seg.tip_[1], y_far, tol_);
  ASSERT_NEAR(seg.tip_[2], z_far_, tol_);
}

TEST_F(RadarScanFixture, returns_to_start)
{
  const double time = 2 * sweep_time_ + reset_time_;
  auto seg = slgen.generate(time);
  auto seg_start = slgen.generate(0);

  ASSERT_NEAR(seg.tail_[0], seg_start.tail_[0], tol_);
  ASSERT_NEAR(seg.tail_[1], seg_start.tail_[1], tol_);
  ASSERT_NEAR(seg.tail_[2], seg_start.tail_[2], tol_);
  ASSERT_NEAR(seg.tip_[0], seg_start.tip_[0], tol_);
  ASSERT_NEAR(seg.tip_[1], seg_start.tip_[1], tol_);
  ASSERT_NEAR(seg.tip_[2], seg_start.tip_[2], tol_);
}

TEST_F(RadarScanFixture, pauses_at_end)
{
  const double time = 2 * sweep_time_ + 2 * reset_time_;
  auto seg = slgen.generate(time);

  const double angle = -sweep_angle_ / 2;
  const auto y_near = (x_near_ - x_orig_) * std::tan(angle);
  const auto y_far = (x_far_ - x_orig_) * std::tan(angle);

  ASSERT_NEAR(seg.tail_[0], x_near_, tol_);
  ASSERT_NEAR(seg.tail_[1], y_near, tol_);
  ASSERT_NEAR(seg.tail_[2], z_near_, tol_);

  ASSERT_NEAR(seg.tip_[0], x_far_, tol_);
  ASSERT_NEAR(seg.tip_[1], y_far, tol_);
  ASSERT_NEAR(seg.tip_[2], z_far_, tol_);
}

TEST_F(RadarScanFixture, cycles)
{
  const double time = 3 * sweep_time_ / 2;
  auto seg = slgen.generate(time);
  auto seg_periodic = slgen.generate(time + 2 * (sweep_time_ + reset_time_));

  ASSERT_NEAR(seg.tail_[0], seg_periodic.tail_[0], tol_);
  ASSERT_NEAR(seg.tail_[1], seg_periodic.tail_[1], tol_);
  ASSERT_NEAR(seg.tail_[2], seg_periodic.tail_[2], tol_);
  ASSERT_NEAR(seg.tip_[0], seg_periodic.tip_[0], tol_);
  ASSERT_NEAR(seg.tip_[1], seg_periodic.tip_[1], tol_);
  ASSERT_NEAR(seg.tip_[2], seg_periodic.tip_[2], tol_);
}

} // namespace nalu
} // namespace sierra
