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

class ScanningLidarFixture : public ::testing::Test
{
public:
  ScanningLidarFixture()
  {
    lidar_spec = YAML::Load(lidar_spec_str)["lidar_specifications"];
    scan_spec = lidar_spec["scanning_lidar_specifications"];
    freq = lidar_spec["frequency"].as<double>();
    slgen.load(scan_spec);
  }

  double vector_angle(
    std::array<double, 3> u, std::array<double, 3> v, std::array<double, 3> n)
  {
    std::array<double, 3> cross;
    cross3(u.data(), v.data(), cross.data());
    return 180 / M_PI *
           std::atan2(
             ddot(cross.data(), n.data(), 3), ddot(v.data(), u.data(), 3));
  }

  std::array<double, 3> rotate_euler_vec(
    const std::array<double, 3>& axis, double angle, std::array<double, 3> vec)
  {
    enum { XH = 0, YH = 1, ZH = 2 };
    normalize_vec3(vec.data());
    std::array<double, 9> nX = {
      {0, -axis[ZH], +axis[YH], +axis[ZH], 0, -axis[XH], -axis[YH], +axis[XH],
       0}};
    const double cosTheta = std::cos(angle);
    std::array<double, 9> rot = {
      {cosTheta, 0, 0, 0, cosTheta, 0, 0, 0, cosTheta}};

    const double sinTheta = std::sin(angle);
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        rot[j * 3 + i] +=
          (1 - cosTheta) * axis[i] * axis[j] + sinTheta * nX[j * 3 + i];
      }
    }
    std::array<double, 3> vecprime;
    matvec33(rot.data(), vec.data(), vecprime.data());
    return vecprime;
  }

  double angle(double time)
  {
    const auto axis_coord = scan_spec["axis"].as<Coordinates>();
    const auto center_coord = scan_spec["center"].as<Coordinates>();

    std::array<double, 3> axis{axis_coord.x_, axis_coord.y_, axis_coord.z_};
    normalize_vec3(axis.data());

    std::array<double, 3> center{
      center_coord.x_, center_coord.y_, center_coord.z_};
    const auto length = scan_spec["beam_length"].as<double>();
    std::array<double, 3> normalized_tip_loc;

    auto ele = scan_spec["elevation_angles"].as<std::vector<double>>();
    ThrowRequire(!ele.empty());
    const double pitch = -ele.front() * M_PI / 180;
    const std::array<double, 3> normal = {0, 0, 1};
    for (int d = 0; d < 3; ++d) {
      normalized_tip_loc[d] =
        (slgen.generate(time).tip_[d] - center[d]) / length;
    }

    std::array<double, 3> yaxis;
    cross3(axis.data(), normal.data(), yaxis.data());
    return vector_angle(
      axis, normalized_tip_loc, rotate_euler_vec(yaxis, pitch, normal));
  }

  double elevation_angle(double time)
  {
    const auto axis_coord = scan_spec["axis"].as<Coordinates>();
    const auto center_coord = scan_spec["center"].as<Coordinates>();

    std::array<double, 3> axis{axis_coord.x_, axis_coord.y_, axis_coord.z_};
    normalize_vec3(axis.data());

    std::array<double, 3> center{
      center_coord.x_, center_coord.y_, center_coord.z_};
    const auto length = scan_spec["beam_length"].as<double>();
    std::array<double, 3> normalized_tip_loc;

    const std::array<double, 3> normal{0, 0, 1};
    for (int d = 0; d < 3; ++d) {
      normalized_tip_loc[d] =
        (slgen.generate(time).tip_[d] - center[d]) / length;
    }

    std::array<double, 3> cross;
    cross3(axis.data(), normal.data(), cross.data());
    return vector_angle(normalized_tip_loc, axis, cross);
  }

  YAML::Node scan_spec;
  YAML::Node lidar_spec;
  ScanningLidarSegmentGenerator slgen;
  double freq{1};

  const std::string lidar_spec_str =
    "lidar_specifications:                                  \n"
    "  from_target_part: [unused]                           \n"
    "  type: scanning                                       \n"
    "  scanning_lidar_specifications:                       \n"
    "    stare_time: 1 #seconds                             \n"
    "    step_delta_angle: 1 #degrees                       \n"
    "    sweep_angle: 20 #degrees                           \n"
    "    reset_time_delta: 1 #second                        \n"
    "    center: [500.,500.,100.]                           \n"
    "    beam_length: 50.                                   \n"
    "    axis: [1,1,0]                                      \n"
    "    elevation_angles: [30]                             \n"
    "  frequency: 2  #Hz                                    \n"
    "  points_along_line: 2                                 \n"
    "  output: text                                         \n"
    "  name: lidar-los                                      \n";
};

TEST_F(ScanningLidarFixture, print_tip_location)
{
  const double dt = 1.0 / freq;
  const double time = 21;
  const int samples = 1 + std::round(time / dt);
  auto center = scan_spec["center"].as<Coordinates>();

  std::ofstream outputFile("ScanningLidar.pattern.txt");
  outputFile << "x,y,z" << std::endl;
  for (int j = 0; j < samples; ++j) {
    const double time = dt * j;
    auto seg = slgen.generate(time);
    ASSERT_DOUBLE_EQ(seg.tail_.at(0), center.x_);
    ASSERT_DOUBLE_EQ(seg.tail_.at(1), center.y_);
    ASSERT_DOUBLE_EQ(seg.tail_.at(2), center.z_);
    outputFile << std::setprecision(15) << seg.tip_.at(0) << ", "
               << seg.tip_.at(1) << ", " << seg.tip_.at(2) << std::endl;
  }
}

TEST_F(ScanningLidarFixture, check_angles)
{
  const auto sweep = scan_spec["sweep_angle"].as<double>();
  const auto stare = scan_spec["stare_time"].as<double>();
  const auto reset = scan_spec["reset_time_delta"].as<double>();

  const double start_time = 0;
  ASSERT_NEAR(angle(start_time), sweep / 2, 1e-12);

  const double forward_phase_end = start_time + sweep / stare;
  ASSERT_NEAR(angle(forward_phase_end), -sweep / 2, 1e-12);

  const double mid_reset_time = forward_phase_end + reset / 2;
  ASSERT_NEAR(angle(mid_reset_time), 0, 1e-12);

  const double end_time = forward_phase_end + reset;
  ASSERT_NEAR(angle(end_time), sweep / 2, 1e-12);
}

TEST_F(ScanningLidarFixture, check_elevation)
{
  const auto sweep = scan_spec["sweep_angle"].as<double>();
  const auto stare = scan_spec["stare_time"].as<double>();
  const auto reset = scan_spec["reset_time_delta"].as<double>();
  const auto ele = scan_spec["elevation_angles"].as<std::vector<double>>();

  const double start_time = 0;
  ASSERT_NEAR(elevation_angle(start_time), ele.front(), 1e-12);

  const double forward_phase_end = start_time + sweep / stare;
  ASSERT_NEAR(elevation_angle(forward_phase_end), ele.front(), 1e-12);
}

TEST_F(ScanningLidarFixture, stares)
{
  const auto stare = scan_spec["stare_time"].as<double>();
  const double some_time = std::floor(11 / stare) * stare;
  const double some_time_frac = 0.1 * stare + some_time;
  for (int d = 0; d < 3; ++d) {
    ASSERT_DOUBLE_EQ(
      slgen.generate(some_time).tip_.at(d),
      slgen.generate(some_time_frac).tip_.at(d));
  }
}

} // namespace nalu
} // namespace sierra
