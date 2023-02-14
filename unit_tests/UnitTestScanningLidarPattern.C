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

  double vector_angle(vs::Vector u, vs::Vector v, vs::Vector n)
  {
    return 180 / M_PI * std::atan2((u ^ v) & n, u & v);
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

  vs::Vector to_vec3(std::array<double, 3> x) { return {x[0], x[1], x[2]}; }

  double angle(double time)
  {
    const auto axis_coord = scan_spec["axis"].as<Coordinates>();
    const auto center_coord = scan_spec["center"].as<Coordinates>();

    vs::Vector axis{axis_coord.x_, axis_coord.y_, axis_coord.z_};
    normalize_vec3(axis.data());

    vs::Vector center{center_coord.x_, center_coord.y_, center_coord.z_};
    const auto length = scan_spec["beam_length"].as<double>();

    const auto sight =
      (to_vec3(slgen.generate(time).tip_) - center).normalize();

    vs::Vector normal = {0, 0, 1};
    const auto proj_sight = (sight - normal * (sight & normal)).normalize();

    std::array<double, 3> yaxis;
    cross3(axis.data(), normal.data(), yaxis.data());
    return vector_angle(axis, proj_sight, {0, 0, 1});
  }

  double elevation_angle(double time)
  {
    const auto center_coord = scan_spec["center"].as<Coordinates>();
    vs::Vector center{center_coord.x_, center_coord.y_, center_coord.z_};
    const vs::Vector normal{0, 0, 1};
    const auto tip = (to_vec3(slgen.generate(time).tip_) - center).normalize();
    return 180 / M_PI *
           std::atan2(tip & normal, vs::mag(tip - normal * (tip & normal)));
  }

  YAML::Node scan_spec;
  YAML::Node lidar_spec;
  ScanningLidarSegmentGenerator slgen;
  double freq{1};
  double total_time{22};

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
    "    axis: [1,0,0]                                      \n"
    "    elevation_angles: [30,60]                          \n"
    "  frequency: 2  #Hz                                    \n"
    "  points_along_line: 2                                 \n"
    "  output: text                                         \n"
    "  name: lidar-los                                      \n";
};

TEST_F(ScanningLidarFixture, print_tip_location)
{
  const double dt = 1.0 / freq;
  const double time = total_time;
  const int samples = std::round(time / dt);
  auto center = scan_spec["center"].as<Coordinates>();

  std::string outputFileName("ScanningLidar.pattern.txt");
  std::ofstream outputFile(outputFileName);
  outputFile << "x,y,z" << std::endl;

  const auto start_height = slgen.generate(0).tip_[2];
  for (int j = 0; j < samples; ++j) {
    const double time = dt * j;
    auto seg = slgen.generate(time);
    ASSERT_DOUBLE_EQ(seg.tail_.at(0), center.x_);
    ASSERT_DOUBLE_EQ(seg.tail_.at(1), center.y_);
    ASSERT_DOUBLE_EQ(seg.tail_.at(2), center.z_);
    ASSERT_DOUBLE_EQ(
      seg.tip_.at(2), start_height); // z-coordinate doesn't change in sweep
    outputFile << std::setprecision(15) << seg.tip_.at(0) << ", "
               << seg.tip_.at(1) << ", " << seg.tip_.at(2) << "\n";
  }
  unlink(outputFileName.c_str());
}

TEST_F(ScanningLidarFixture, check_angles)
{
  const auto sweep = scan_spec["sweep_angle"].as<double>();
  const auto stare = scan_spec["stare_time"].as<double>();
  const auto step = scan_spec["step_delta_angle"].as<double>();
  const auto reset = scan_spec["reset_time_delta"].as<double>();

  const double start_time = 0;
  ASSERT_NEAR(angle(start_time), sweep / 2, 1e-12);

  const double forward_phase_end =
    start_time + std::floor(sweep / step) * stare;
  ASSERT_NEAR(angle(forward_phase_end), -sweep / 2, 1e-12);

  const double mid_reset_time = forward_phase_end + stare + reset / 2;
  ASSERT_NEAR(angle(mid_reset_time), 0, 1e-12);

  const double end_time = forward_phase_end + stare + reset;
  ASSERT_NEAR(angle(end_time), sweep / 2, 1e-12);
}

TEST_F(ScanningLidarFixture, check_elevation)
{
  const auto sweep = scan_spec["sweep_angle"].as<double>();
  const auto stare = scan_spec["stare_time"].as<double>();
  const auto step = scan_spec["step_delta_angle"].as<double>();
  const auto reset = scan_spec["reset_time_delta"].as<double>();
  const auto ele = scan_spec["elevation_angles"].as<std::vector<double>>();
  ASSERT_EQ(ele.size(), 2U);

  const double start_time = 0;
  ASSERT_NEAR(elevation_angle(start_time), ele[0], 1e-12);

  const double forward_phase_end =
    start_time + std::floor(sweep / step) * stare;
  ASSERT_NEAR(elevation_angle(forward_phase_end), ele[0], 1e-12);

  const double mid_reset_time = forward_phase_end + stare + reset / 2;
  ASSERT_NEAR(elevation_angle(mid_reset_time), ele[0], 1e-12);

  const double end_time = forward_phase_end + stare;
  ASSERT_NEAR(elevation_angle(end_time), ele[0], 1e-12);

  const double restart = forward_phase_end + stare;
  ASSERT_NEAR(elevation_angle(end_time), ele[0], 1e-12);

  const double end_time_plus_a_bit =
    forward_phase_end + stare + reset + forward_phase_end / 2;
  ASSERT_NEAR(elevation_angle(end_time_plus_a_bit), ele[1], 1e-12);
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

TEST_F(ScanningLidarFixture, stares_at_end)
{
  const auto stare = scan_spec["stare_time"].as<double>();
  const double final_position_time = 20;
  const double some_time_after = 0.1 * stare + final_position_time;
  for (int d = 0; d < 3; ++d) {
    ASSERT_DOUBLE_EQ(
      slgen.generate(final_position_time).tip_.at(d),
      slgen.generate(some_time_after).tip_.at(d));
  }
}

} // namespace nalu
} // namespace sierra
