// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gcl/UnitTestGCL.h"

namespace {

namespace hex8_golds_x_rot {
namespace mesh_velocity {
static constexpr double swept_vol[12] = {0.0,    -0.0625, 0.0,     -0.0625,
                                         0.0,    0.0625,  0.0,     0.0625,
                                         0.0625, 0.0625,  -0.0625, -0.0625};

static constexpr double face_vel_mag[12] = {0.0,   -0.125, 0.0,    -0.125,
                                            0.0,   0.125,  0.0,    0.125,
                                            0.125, 0.125,  -0.125, -0.125};
} // namespace mesh_velocity
} // namespace hex8_golds_x_rot

namespace hex8_golds_y_rot {
namespace mesh_velocity {
static constexpr double swept_vol[12] = {0.0625,  0.0,    -0.0625, 0.0,
                                         -0.0625, 0.0,    0.0625,  0.0,
                                         -0.0625, 0.0625, 0.0625,  -0.0625};

static constexpr double face_vel_mag[12] = {0.125,  0.0,   -0.125, 0.0,
                                            -0.125, 0.0,   0.125,  0.0,
                                            -0.125, 0.125, 0.125,  -0.125};
} // namespace mesh_velocity
} // namespace hex8_golds_y_rot
} // namespace

TEST_F(GCLTest, rigid_rotation_elem)
{
  if (bulk_.parallel_size() > 1)
    return;

  realm_.realmUsesEdges_ = false;
  const std::string meshDims = "3x3x3|offset:0,65,0";
  const bool secondOrder = false;
  const double deltaT = 0.003; // approx 0.25 deg motion for given omega
  const std::string mesh_motion =
    "mesh_motion:                                                          \n"
    "  - name: interior                                                    \n"
    "    mesh_parts: [ block_1 ]                                           \n"
    "    motion:                                                           \n"
    "      - type: rotation                                                \n"
    "        omega: 1.5707963267948966                                     \n"
    "        axis: [1.0, 0.0, 0.0]                                         \n"
    "        centroid: [0.0, 0.0, 0.0]                                     \n";

  fill_mesh_and_init_fields(meshDims);
  init_time_integrator(secondOrder, deltaT);
  register_algorithms(mesh_motion);
  init_states();
  compute_div_mesh_vel();
  compute_dvoldt();
  compute_absolute_error();
}

TEST_F(GCLTest, rigid_rotation_edge)
{
  if (bulk_.parallel_size() > 1)
    return;

  realm_.realmUsesEdges_ = true;
  const std::string meshDims = "3x3x3|offset:0,65,0";
  const bool secondOrder = false;
  const double deltaT = 0.003; // approx 0.25 deg motion for given omega
  const std::string mesh_motion =
    "mesh_motion:                                                          \n"
    "  - name: interior                                                    \n"
    "    mesh_parts: [ block_1 ]                                           \n"
    "    motion:                                                           \n"
    "      - type: rotation                                                \n"
    "        omega: 1.5707963267948966                                     \n"
    "        axis: [1.0, 0.0, 0.0]                                         \n"
    "        centroid: [0.0, 0.0, 0.0]                                     \n";

  fill_mesh_and_init_fields(meshDims);
  init_time_integrator(secondOrder, deltaT);
  register_algorithms(mesh_motion);
  init_states();
  compute_div_mesh_vel();
  compute_dvoldt();
  compute_absolute_error();
}

TEST_F(GCLTest, rigid_scaling_elem)
{
  if (bulk_.parallel_size() > 1)
    return;

  realm_.realmUsesEdges_ = false;
  const std::string meshDims = "3x3x3|offset:0,65,0";
  const bool secondOrder = false;
  const double deltaT = 0.003; // approx 0.25 deg motion for given omega
  const std::string mesh_motion =
    "mesh_motion:                                                          \n"
    "  - name: interior                                                    \n"
    "    mesh_parts: [ block_1 ]                                           \n"
    "    motion:                                                           \n"
    "      - type: scaling                                                 \n"
    "        rate: [1.0, 1.0, 1.0]                                         \n"
    "        centroid: [0.0, 0.0, 0.0]                                     \n";

  fill_mesh_and_init_fields(meshDims);
  init_time_integrator(secondOrder, deltaT);
  register_algorithms(mesh_motion);
  init_states();
  compute_div_mesh_vel();
  compute_dvoldt();
  compute_relative_error();
}

TEST_F(GCLTest, rigid_scaling_edge)
{
  if (bulk_.parallel_size() > 1)
    return;

  realm_.realmUsesEdges_ = true;
  const std::string meshDims = "3x3x3|offset:0,65,0";
  const bool secondOrder = false;
  const double deltaT = 0.003; // approx 0.25 deg motion for given omega
  const std::string mesh_motion =
    "mesh_motion:                                                          \n"
    "  - name: interior                                                    \n"
    "    mesh_parts: [ block_1 ]                                           \n"
    "    motion:                                                           \n"
    "      - type: scaling                                                 \n"
    "        rate: [1.0, 1.0, 1.0]                                         \n"
    "        centroid: [0.0, 0.0, 0.0]                                     \n";

  fill_mesh_and_init_fields(meshDims);
  init_time_integrator(secondOrder, deltaT);
  register_algorithms(mesh_motion);
  init_states();
  compute_div_mesh_vel();
  compute_dvoldt();
  compute_relative_error();
}

TEST_F(GCLTest, rigid_translation)
{
  if (bulk_.parallel_size() > 1)
    return;

  const std::string meshDims = "3x3x3|offset:0,65,0";
  const bool secondOrder = false;
  const double deltaT = 0.003; // approx 0.25 deg motion for given omega
  const std::string mesh_motion =
    "mesh_motion:                                                          \n"
    "  - name: interior                                                    \n"
    "    mesh_parts: [ block_1 ]                                           \n"
    "    motion:                                                           \n"
    "      - type: translation                                             \n"
    "        velocity: [1.0, 0.0, 0.0]                                     \n";

  fill_mesh_and_init_fields(meshDims);
  init_time_integrator(secondOrder, deltaT);
  register_algorithms(mesh_motion);
  init_states();
  compute_div_mesh_vel();
  compute_dvoldt();
  compute_absolute_error();
}

TEST_F(GCLTest, mesh_velocity_x_rot)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1)
    return;

  const std::string meshDims = "1x1x1";
  realm_.realmUsesEdges_ = false;
  const bool secondOrder = false;
  const double deltaT = 0.5;
  const std::string mesh_motion =
    "mesh_motion:                                                          \n"
    "  - name: interior                                                    \n"
    "    mesh_parts: [ block_1 ]                                           \n"
    "    motion:                                                           \n"
    "      - type: rotation                                                \n"
    "        omega: -3.141592653589793                                     \n"
    "        axis: [1.0, 0.0, 0.0]                                         \n"
    "        centroid: [0.5, 0.5, 0.5]                                     \n";

  fill_mesh_and_init_fields(meshDims, false);
  init_time_integrator(secondOrder, deltaT, 1);
  register_algorithms(mesh_motion);
  init_states();

  const double tol = 1.0e-15;
  namespace gold_values = ::hex8_golds_x_rot::mesh_velocity;
  {
    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::ELEM_RANK, sel);
    int counter = 0;
    for (const auto* b : bkts) {
      const double* sv = stk::mesh::field_data(*sweptVol_, *b, 0);
      const double* fvm = stk::mesh::field_data(*faceVelMag_, *b, 0);
      for (int i = 0; i < 12; i++) {
        EXPECT_NEAR(gold_values::swept_vol[i], sv[i], tol);
        counter++;
        EXPECT_NEAR(gold_values::face_vel_mag[i], fvm[i], tol);
        counter++;
      }
    }
    EXPECT_EQ(counter, 24);
  } // namespace =::hex8_golds_x_rot::mesh_velocity;
}

TEST_F(GCLTest, mesh_velocity_y_rot)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1)
    return;
  realm_.realmUsesEdges_ = false;

  const std::string meshDims = "1x1x1";
  const bool secondOrder = false;
  const double deltaT = 0.5;
  const std::string mesh_motion =
    "mesh_motion:                                                          \n"
    "  - name: interior                                                    \n"
    "    mesh_parts: [ block_1 ]                                           \n"
    "    motion:                                                           \n"
    "      - type: rotation                                                \n"
    "        omega: -3.141592653589793                                     \n"
    "        axis: [0.0,1.0,0.0]                                           \n"
    "        centroid: [0.5, 0.5, 0.5]                                     \n";

  fill_mesh_and_init_fields(meshDims, false);
  init_time_integrator(secondOrder, deltaT, 1);
  register_algorithms(mesh_motion);
  init_states();

  const double tol = 1.0e-15;
  namespace gold_values = ::hex8_golds_y_rot::mesh_velocity;
  {
    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::ELEM_RANK, sel);
    int counter = 0;
    for (const auto* b : bkts) {
      const double* sv = stk::mesh::field_data(*sweptVol_, *b, 0);
      const double* fvm = stk::mesh::field_data(*faceVelMag_, *b, 0);
      for (int i = 0; i < 12; i++) {
        EXPECT_NEAR(gold_values::swept_vol[i], sv[i], tol);
        counter++;
        EXPECT_NEAR(gold_values::face_vel_mag[i], fvm[i], tol);
        counter++;
      }
    }
    EXPECT_EQ(counter, 24);
  } // namespace =::hex8_golds_y_rot::mesh_velocity;
}

TEST_F(GCLTest, mesh_velocity_y_rot_scs_center)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1)
    return;

  const std::string meshDims = "1x1x1";
  const bool secondOrder = false;
  const double deltaT = 0.25;
  const std::string mesh_motion =
    "mesh_motion:                                                          \n"
    "  - name: interior                                                    \n"
    "    mesh_parts: [ block_1 ]                                           \n"
    "    motion:                                                           \n"
    "      - type: rotation                                                \n"
    "        omega: -3.141592653589793                                     \n"
    "        axis: [0.0,1.0,0.0]                                           \n"
    "        centroid: [0.5, 0.5, 0.75]                                    \n";

  fill_mesh_and_init_fields(meshDims, false);
  init_time_integrator(secondOrder, deltaT);
  register_algorithms(mesh_motion);
  init_states();

  const double tol = 1.0e-15;
  namespace gold_values = ::hex8_golds_y_rot::mesh_velocity;
  {
    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::ELEM_RANK, sel);
    int counter = 0;
    for (const auto* b : bkts) {
      const double* sv = stk::mesh::field_data(*sweptVol_, *b, 0);
      const double* fvm = stk::mesh::field_data(*faceVelMag_, *b, 0);
      for (int i = 0; i < 12; i++) {
        // check only for scs through the center of which rotation axis passes
        // in addition to all scs perpendicular to rotation axis - total 6 scs
        if (
          (i == 1) || (i == 3) || (i == 4) || (i == 5) || (i == 6) ||
          (i == 7)) {
          EXPECT_NEAR(0.0, sv[i], tol);
          counter++;
          EXPECT_NEAR(0.0, fvm[i], tol);
          counter++;
        }
      }
    }
    EXPECT_EQ(counter, 12);
  } // namespace =::hex8_golds_y_rot::mesh_velocity;
}

TEST_F(GCLTest, mesh_airy_waves)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1)
    return;

  const std::string meshDims = "3x3x3";
  const bool secondOrder = true;
  const double deltaT = 0.25;
  const std::string mesh_motion =
    "mesh_motion:                                                          \n"
    "  - name: WaveTest                                                    \n"
    "    mesh_parts: [ block_1 ]                                           \n"
    "    motion:                                                           \n"
    "      - type: waving_boundary                                         \n"
    "        wave_model:  Idealized                                        \n"
    "        wave_height: 0.1                                              \n"
    "        wave_length: 1.                                               \n"
    "        phase_velocity: 25.                                           \n"
    "        mesh_damping_length: 1.                                       \n"
    "        mesh_damping_coeff : 3                                        \n";

  fill_mesh_and_init_fields(meshDims);
  init_time_integrator(secondOrder, deltaT);
  register_algorithms(mesh_motion);
  init_states();
  compute_div_mesh_vel();
  compute_dvoldt();
  compute_absolute_error();
}
