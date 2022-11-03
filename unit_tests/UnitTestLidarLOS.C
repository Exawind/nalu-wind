#include <gtest/gtest.h>

#include "wind_energy/SyntheticLidar.h"
#include "NaluParsing.h"
#include "UnitTestUtils.h"
#include "stk_mesh/base/MeshBuilder.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"

#include <yaml-cpp/yaml.h>

#include <ostream>
#include <memory>
#include <array>
#if !defined(KOKKOS_ENABLE_HIP)
#include <filesystem>
#endif
namespace sierra {
namespace nalu {

const std::string db_spec =
  "data_probes:                                             \n"
  "  output_frequency: 10                                   \n"
  "  search_method: stk_kdtree                              \n"
  "  search_tolerance: 1.0e-3                               \n"
  "  search_expansion_factor: 2.0                           \n"
  "  lidar_specifications:                                  \n";

const std::string scan_spec =
  "      - name: lidar/scan                                  \n"
  "        type: scanning                                    \n"
  "        frequency: 5                                      \n"
  "        points_along_line: 10                             \n"
  "        output: text                                      \n"
  "        scanning_lidar_specifications:                    \n"
  "          center: [500,500,100]                           \n"
  "          beam_length: 20                                 \n"
  "          axis: [-1,0,0]                                  \n"
  "          stare_time: 1                                   \n"
  "          sweep_angle: 30                                 \n"
  "          step_delta_angle: 2                             \n"
  "          reset_time_delta: 0                             \n"
  "          elevation_angles: [-5,0,5]                      \n";

const std::string radar_spec =
  "      - name: lidar/radar                                 \n"
  "        type: radar                                       \n"
  "        frequency: 1                                      \n"
  "        points_along_line: 2                              \n"
  "        reuse_search_data: no                             \n"
  "        output: text                                      \n"
  "        radar_cone_grid:                                  \n"
  "          cone_angle: 0.5                                 \n"
  "          num_circles: 2                                  \n"
  "          lines_per_cone_circle: 6                        \n"
  "        radar_specifications:                             \n"
  "          bbox: [-1000,-1000,0,1000,1000,1000]            \n"
  "          center: [-5000,0,100]                           \n"
  "          beam_length: 10000                              \n"
  "          angular_speed: 10                               \n"
  "          axis: [1,0,0]                                   \n"
  "          sweep_angle: 10                                 \n"
  "          reset_time:  0                                  \n"
  "          elevation_angles: [0,1,2,3]                     \n";

const std::string spec_str = db_spec + scan_spec + radar_spec;

class LidarLOSFixture : public ::testing::Test
{
public:
  LidarLOSFixture()
    : bulkptr(stk::mesh::MeshBuilder(MPI_COMM_WORLD)
                .set_aura_option((stk::mesh::BulkData::NO_AUTO_AURA))
                .set_spatial_dimension(3U)
                .create()),
      bulk(*bulkptr),
      meta(bulk.mesh_meta_data())
  {
    stk::io::StkMeshIoBroker io(bulk.parallel());
    io.set_bulk_data(bulk);

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
    stk::mesh::field_fill(1., vel_field.field_of_state(stk::mesh::StateNP1));
    stk::mesh::field_fill(1., vel_field.field_of_state(stk::mesh::StateN));

    // lidar will write new files if they exist. Delete them here
    // to adding new files ad infinitum`
#if !defined(KOKKOS_ENABLE_HIP)
    std::filesystem::remove("lidar/scan.txt");
    for (int j = 0; j < 13; ++j) {
      std::filesystem::remove("lidar/radar-grid-" + std::to_string(j) + ".txt");
    }
#else
    throw std::runtime_error(
      "LidarLOSFixture() filesystem not supported on HIP");
#endif
  }

private:
  std::shared_ptr<stk::mesh::BulkData> bulkptr;

public:
  stk::mesh::BulkData& bulk;
  stk::mesh::MetaData& meta;
  YAML::Node spec = YAML::Load(spec_str)["data_probes"];
  LidarLOS los;
};

TEST_F(LidarLOSFixture, load) { los.load(spec, nullptr); }

TEST_F(LidarLOSFixture, write)
{
  EXPECT_NO_THROW(los.load(spec, nullptr));
  los.set_time_for_all(0);
  for (int num_steps = 0; num_steps < 10; ++num_steps) {
    los.output(
      bulk, !stk::mesh::Selector{}, "coordinates", 0.5, num_steps * 0.5);
  }
}

TEST(make_radar_grid, first_is_axis)
{
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  vs::Vector axis(coeff(rng), coeff(rng), coeff(rng));
  axis.normalize();
  auto rays =
    details::make_radar_grid(convert::degrees_to_radians(45), 4, 10, axis);
  ASSERT_DOUBLE_EQ(rays[0][0], axis[0]);
  ASSERT_DOUBLE_EQ(rays[0][1], axis[1]);
  ASSERT_DOUBLE_EQ(rays[0][2], axis[2]);
}

TEST(make_radar_grid, check_opposite)
{
  vs::Vector axis(0, 0, -1);
  axis.normalize();
  auto rays =
    details::make_radar_grid(convert::degrees_to_radians(45), 4, 10, axis);
  ASSERT_DOUBLE_EQ(rays[0][0], axis[0]);
  ASSERT_DOUBLE_EQ(rays[0][1], axis[1]);
  ASSERT_DOUBLE_EQ(rays[0][2], axis[2]);
}

} // namespace nalu
} // namespace sierra
