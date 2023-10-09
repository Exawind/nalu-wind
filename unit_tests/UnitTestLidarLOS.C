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
#include "Ioss_FileInfo.h"
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
  "      - name: lidar_scan                                  \n"
  "        type: scanning                                    \n"
  "        frequency: 5                                      \n"
  "        points_along_line: 10                             \n"
  "        output: text                                      \n"
  "        scanning_lidar_specifications:                    \n"
  "          center: [500,500,100]                           \n"
  "          beam_length: 20                                 \n"
  "          axis: [0,1,0]                                   \n"
  "          stare_time: 1                                   \n"
  "          sweep_angle: 30                                 \n"
  "          step_delta_angle: 2                             \n"
  "          reset_time_delta: 0                             \n"
  "          elevation_angles: [-5,0,5]                      \n";

const std::string radar_spec =
  "      - name: lidar_radar                                 \n"
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

const std::string radar_cone_spec =
  "      - name: lidar_radar-filtered                        \n"
  "        type: radar                                       \n"
  "        frequency: 4                                      \n"
  "        points_along_line: 10                              \n"
  "        reuse_search_data: no                             \n"
  "        output: text                                      \n"
  "        radar_cone_filter:                                \n"
  "          cone_angle: 0.25                                \n"
  "          quadrature_type: radau                          \n"
  "          lines_per_cone_circle: 20                       \n"
  "        radar_specifications:                             \n"
  "          bbox: [-1000,-1000,0,1000,1000,1000]            \n"
  "          center: [-10000,0,100]                          \n"
  "          beam_length: 10000                              \n"
  "          angular_speed: 2                                \n"
  "          axis: [1,0,0]                                   \n"
  "          sweep_angle: 10                                 \n"
  "          reset_time: 0                                   \n"
  "          elevation_angles: [0]                           \n";

const std::string spec_str = db_spec + scan_spec + radar_spec + radar_cone_spec;

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
    remove_files();
  }

  ~LidarLOSFixture() { remove_files(); }

private:
  std::shared_ptr<stk::mesh::BulkData> bulkptr;

  void remove_files()
  {
    Ioss::FileInfo("lidar_scan.txt").remove_file();
    Ioss::FileInfo("lidar_radar-filtered.txt").remove_file();
    for (int j = 0; j < 13; ++j) {
      Ioss::FileInfo("lidar_radar-grid-" + std::to_string(j) + ".txt")
        .remove_file();
    }
  }

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
  for (int num_steps = 0; num_steps < 20; ++num_steps) {
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

TEST(spherical_cap_quadrature, integrate_constant)
{
  const auto ang = M_PI / 3;
  auto[rays, weights] = details::spherical_cap_radau(ang, 6, 4);
  auto cart_const = [](vs::Vector x) { return 1; };
  double integral = 0;
  for (int j = 0; j < int(weights.size()); ++j) {
    integral += cart_const(rays[j]) * weights[j];
  }
  ASSERT_NEAR(integral, 4 * M_PI * std::pow(std::sin(ang / 2), 2), 1e-12);
}

TEST(spherical_cap_quadrature, integrate_constant_halfpower)
{
  const auto ang = M_PI / 3;
  auto[rays, weights] = details::spherical_cap_radau(ang, 6, 9, [](double x) {
    return 1.234529105942581469654 * std::pow(2, -x * x);
  });
  auto cart_const = [](vs::Vector x) { return 1; };
  double integral = 0;
  for (int j = 0; j < int(weights.size()); ++j) {
    integral += cart_const(rays[j]) * weights[j];
  }
  ASSERT_NEAR(integral, 4 * M_PI * std::pow(std::sin(ang / 2), 2), 1e-12);
}

TEST(spherical_cap_quadrature, integrate_constant_trunc_normal_1)
{
  const auto ang = M_PI / 3;
  auto[rays, weights] = details::spherical_cap_truncated_normal(
    ang, 6, details::NormalRule::SIGMA1);
  auto cart_const = [](vs::Vector x) { return 1; };
  double integral = 0;
  for (int j = 0; j < int(weights.size()); ++j) {
    integral += cart_const(rays[j]) * weights[j];
  }
  ASSERT_NEAR(integral, 4 * M_PI * std::pow(std::sin(ang / 2), 2), 1e-12);
}

TEST(spherical_cap_quadrature, integrate_constant_trunc_normal_2)
{
  const auto ang = M_PI / 4;
  auto[rays, weights] = details::spherical_cap_truncated_normal(
    ang, 6, details::NormalRule::SIGMA2);
  auto cart_const = [](vs::Vector x) { return 1; };
  double integral = 0;
  for (int j = 0; j < int(weights.size()); ++j) {
    integral += cart_const(rays[j]) * weights[j];
  }
  ASSERT_NEAR(integral, 4 * M_PI * std::pow(std::sin(ang / 2), 2), 1e-12);
}

TEST(spherical_cap_quadrature, integrate_constant_trunc_normal_3)
{
  const auto ang = M_PI / 5;
  auto[rays, weights] = details::spherical_cap_truncated_normal(
    ang, 6, details::NormalRule::SIGMA3);
  auto cart_const = [](vs::Vector x) { return 1; };
  double integral = 0;
  for (int j = 0; j < int(weights.size()); ++j) {
    integral += cart_const(rays[j]) * weights[j];
  }
  ASSERT_NEAR(integral, 4 * M_PI * std::pow(std::sin(ang / 2), 2), 1e-12);
}

TEST(spherical_cap_quadrature, integrate_constant_trunc_normal_halfpower)
{
  const auto ang = M_PI / 5;
  auto[rays, weights] = details::spherical_cap_truncated_normal(
    ang, 6, details::NormalRule::HALFPOWER);
  auto cart_const = [](vs::Vector x) { return 1; };
  double integral = 0;
  for (int j = 0; j < int(weights.size()); ++j) {
    integral += cart_const(rays[j]) * weights[j];
  }
  ASSERT_NEAR(integral, 4 * M_PI * std::pow(std::sin(ang / 2), 2), 1e-12);
}

TEST(spherical_cap_quadrature, integrate_linear)
{
  const auto ang = M_PI / 4;
  auto[rays, weights] = details::spherical_cap_radau(ang, 6, 4);
  auto cart_lin = [](vs::Vector x) { return x[0] + x[1] + x[2]; };
  double integral = 0;
  for (int j = 0; j < int(weights.size()); ++j) {
    integral += cart_lin(rays[j]) * weights[j];
  }
  ASSERT_NEAR(integral, M_PI * std::sin(ang) * std::sin(ang), 1e-12);
}

TEST(spherical_cap_quadrature, integrate_quadratic)
{
  const auto ang = 0.2423891; // some number < pi/2
  auto[rays, weights] = details::spherical_cap_radau(ang, 6, 4);
  auto cart_quad = [](vs::Vector x) {
    return (x[0] - 1) * (x[0] - 1) + 2 * x[1] * x[1] + x[2];
  };
  double integral = 0;
  for (int j = 0; j < int(weights.size()); ++j) {
    integral += cart_quad(rays[j]) * weights[j];
  }
  const double ans =
    -M_PI * (-9 + std::cos(2 * ang)) * std::pow(std::sin(ang / 2), 2);
  ASSERT_NEAR(integral, ans, 1e-12);
}

TEST(spherical_cap_quadrature, error_improves_with_points_for_nonpoly)
{
  const auto ang = M_PI / 3; // some number < pi/2

  auto ramp = [](double x) { return x > 0 ? x : 0; };
  auto cart_ramp = [ramp](vs::Vector x) {
    return ramp(x[0] * x[1]) * ramp(x[0] * x[1]) + x[2];
  };

  const double actual_integral = 2.3995550177556133;
  auto error = [&](int ntheta, int nphi) {
    auto[rays, weights] = details::spherical_cap_radau(ang, ntheta, nphi);
    double integral = 0;
    for (int j = 0; j < int(weights.size()); ++j) {
      integral += cart_ramp(rays[j]) * weights[j];
    }
    return std::abs(actual_integral - integral);
  };
  ASSERT_LT(
    error(5, 6), error(3, 3)); // not necessarily the case for all functions
  ASSERT_NEAR(error(121, 9), 0, 1e-8); // should be small with this many points
}

} // namespace nalu
} // namespace sierra
