#include <gtest/gtest.h>

#include <wind_energy/SyntheticLidar.h>
#include "UnitTestUtils.h"

#include <stk_mesh/base/MeshBuilder.hpp>
#include <yaml-cpp/yaml.h>

#include <ostream>
#include <memory>
#include <array>

namespace sierra {
namespace nalu {

const auto lidarSpec =
  R"(lidar_specifications:
    from_target_part: [unused]
    inner_prism_initial_theta: 90
    inner_prism_rotation_rate: 3.5
    inner_prism_azimuth: 15.2
    outer_prism_initial_theta: 90
    outer_prism_rotation_rate: 6.5
    outer_prism_azimuth: 15.2
    scan_time: 2 #seconds
    number_of_samples: 984
    points_along_line: 4
    center: [500,500,100]
    beam_length: 50.
    axis: [1,1,0]
    ground_direction: [0,0,1]
    output: text
    name: lidar-los
)";

TEST(SpinnerLidar, print_tip_location)
{

  YAML::Node lidarSpecNode = YAML::Load(lidarSpec)["lidar_specifications"];

  const double nsamp = 984;
  const double dt = 2.0 / nsamp;
  SpinnerLidarSegmentGenerator slgen;
  slgen.load(lidarSpecNode);

  for (int j = 0; j < nsamp; ++j) {
    const double time = dt * j;
    auto seg = slgen.generate(time);
    ASSERT_TRUE(seg.tip_.at(0) > seg.tail_.at(0));
    ASSERT_TRUE(seg.tip_.at(1) > seg.tail_.at(1));
    ASSERT_DOUBLE_EQ(seg.tail_.at(0), 500);
    ASSERT_DOUBLE_EQ(seg.tail_.at(1), 500);
  }
}

vs::Vector
velocity_func(const double* x, double time)
{
  return {
    time + 1 + 2.1 * x[0] / 500 + 3.2 * x[1] / 500 + 4.3 * x[2] / 100,
    time - 2 + 1.2 * x[0] / 500 + 2.3 * x[1] / 500 + 3.4 * x[2] / 100,
    time + 3 - 2.1 * x[0] / 500 + 3.2 * x[1] / 500 - 4.3 * x[2] / 100};
}

TEST(SpinnerLidar, volume_interp)
{

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA);
  builder.set_spatial_dimension(3U);
  auto bulk = builder.create();
  auto& meta = bulk->mesh_meta_data();
  stk::io::StkMeshIoBroker io(bulk->parallel());
  io.set_bulk_data(*bulk);

  const int n = 16;
  const auto nx = std::to_string(n);
  const auto ny = std::to_string(n);
  const auto nz = std::to_string(n / 2);
  auto mesh_name = "generated:" + nx + "x" + ny + "x" + nz +
                   "|bbox:-1000,-1000,0,1000,1000,1000|sideset:xXyYzZ";
  io.add_mesh_database(mesh_name, stk::io::READ_MESH);
  io.create_input_mesh();

  using vector_field_type = stk::mesh::Field<double, stk::mesh::Cartesian3d>;
  auto node_rank = stk::topology::NODE_RANK;
  auto& vel_field =
    meta.declare_field<vector_field_type>(node_rank, "velocity", 2);
  stk::mesh::put_field_on_entire_mesh(vel_field);
  io.populate_bulk_data();

  auto& vel_field_old = vel_field.field_of_state(stk::mesh::StateN);

  const auto& coord_field =
    *dynamic_cast<const vector_field_type*>(meta.coordinate_field());

  const auto& node_buckets =
    bulk->get_buckets(stk::topology::NODE_RANK, meta.universal_part());
  for (const auto* ib : node_buckets) {
    for (const auto& node : *ib) {
      auto* uptr = stk::mesh::field_data(vel_field, node);
      auto* uptr_old = stk::mesh::field_data(vel_field_old, node);
      const auto* xptr = stk::mesh::field_data(coord_field, node);
      const auto vel_at_x = velocity_func(xptr, 0);
      for (int d = 0; d < 3; ++d) {
        uptr[d] = vel_at_x[d];
        uptr_old[d] = uptr[d];
      }
    }
  }

  YAML::Node lidarSpecNode = YAML::Load(lidarSpec)["lidar_specifications"];
  const auto fileName = lidarSpecNode["name"].as<std::string>() + ".txt";
  unlink(fileName.c_str());

  {
    LidarLineOfSite los;
    los.load(lidarSpecNode);
    los.set_time(0);
    los.output(*bulk, meta.universal_part(), "coordinates", 0);
  }

  if (bulk->parallel_rank() == 0) {
    std::string line;
    std::ifstream myfile("lidar-los.txt");
    if (myfile.is_open()) {
      // skip first two lines
      std::getline(myfile, line); // header
      std::getline(myfile, line); // center point

      while (std::getline(myfile, line)) {
        std::stringstream iline(line);
        std::string word;
        std::vector<double> values;
        while (std::getline(iline, word, ',')) {
          values.push_back(std::stod(word));
        }
        ASSERT_EQ(values.size(), 5u);

        constexpr int coord_start = 1;
        constexpr int vel_start = 4;

        const auto& exact_vel = velocity_func(&values[coord_start], 0);
        const auto& center = lidarSpecNode["center"].as<std::vector<double>>();
        const vs::Vector lineVector(
          values[coord_start + 0] - center[0],
          values[coord_start + 1] - center[1],
          values[coord_start + 2] - center[2]);

        const auto exact_los_velocity =
          lineVector & exact_vel / vs::mag(lineVector);

        EXPECT_NEAR(exact_los_velocity, values.at(vel_start), 1e-10);
      }
      myfile.close();
    }
  }
  unlink(fileName.c_str());
}

TEST(Spinner, invalid_predictor_throws)
{
  const std::string lidar_nc =
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
    "  points_along_line: 4                                 \n"
    "  center: [500,500,100]                                \n"
    "  beam_length: 500.                                    \n"
    "  axis: [1,1,0]                                        \n"
    "  ground_direction: [0,0,1]                            \n"
    "  output: netcdf                                       \n"
    "  time_step: 0.01                                      \n"
    "  name: lidar/lidar-1                                  \n";

  const std::string bad_predictor = "  predictor: foo";
  const std::string good_predictor = "  predictor: nearest";

  const auto bad_spec =
    YAML::Load(lidar_nc + bad_predictor)["lidar_specifications"];
  const auto good_spec =
    YAML::Load(lidar_nc + good_predictor)["lidar_specifications"];

  {
    LidarLineOfSite los;
    EXPECT_NO_THROW(los.load(good_spec));
  }
  {
    LidarLineOfSite los;
    EXPECT_THROW(los.load(bad_spec), std::runtime_error);
  }
}

} // namespace nalu
} // namespace sierra
