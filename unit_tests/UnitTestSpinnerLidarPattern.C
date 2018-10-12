#include <gtest/gtest.h>

#include <wind_energy/SyntheticLidar.h>
#include "UnitTestUtils.h"

#include <yaml-cpp/yaml.h>

#include <ostream>
#include <memory>
#include <array>

namespace sierra {
namespace nalu {

TEST(SpinnerLidar, print_tip_location)
{
  const std::string lidarSpec =
     "lidar_specifications:                                  \n"
     "  from_target_part: [unused]                           \n"
     "  inner_prism_initial_theta: 90                        \n"
     "  inner_prism_rotation_rate: 3.5                       \n"
     "  inner_prism_azimuth: 15.2                            \n"
     "  outer_prism_initial_theta: 90                        \n"
     "  outer_prism_rotation_rate: 6.5                       \n"
     "  outer_prism_azimuth: 15.2                            \n"
     "  scan_time: 2 #seconds                                \n"
     "  number_of_samples: 984                               \n"
     "  points_along_line: 100                               \n"
     "  center: [500,500,100]                                \n"
     "  beam_length: 1.0                                     \n"
     "  axis: [1,1,0]                                        \n"
     "  ground_direction: [0,0,1]                            \n";

  YAML::Node lidarSpecNode = YAML::Load(lidarSpec)["lidar_specifications"];

  const double nsamp = 984;
  const double dt = 2.0 / nsamp;
  SpinnerLidarSegmentGenerator slgen;
  slgen.load(lidarSpecNode);

  std::ofstream outputFile("SpinnerLidar.pattern.txt");
  outputFile << "x,y,z" << std::endl;

  for (int j = 0; j < nsamp; ++j) {
    const double time = dt * j;
    auto seg = slgen.generate_path_segment(time);
    ASSERT_TRUE(seg.tip_.at(0) > seg.tail_.at(0));
    ASSERT_TRUE(seg.tip_.at(1) > seg.tail_.at(1));
    ASSERT_DOUBLE_EQ(seg.tail_.at(0), 500);
    ASSERT_DOUBLE_EQ(seg.tail_.at(1), 500);
    outputFile
     << seg.tip_.at(0)
     << ", "
     << seg.tip_.at(1)
     << ", "
     << seg.tip_.at(2)
     << std::endl;
  }
}

}}
