// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details

#include <gtest/gtest.h>
#include <actuator/ActuatorTypes.h>
#include <actuator/ActuatorBulkSimple.h>
#include <actuator/ActuatorParsingSimple.h>
#include <actuator/ActuatorFunctorsSimple.h>
#include <actuator/ActuatorFLLC.h>
#include <yaml-cpp/yaml.h>

namespace sierra {
namespace nalu {

namespace {
const char* actuatorPars = R"act(actuator:
  type: ActLineSimple
  n_simpleblades: 1
  Blade0:
    num_force_pts_blade: 1
    epsilon: [3.0, 3.0, 3.0]
    p1: [0, -4, 0] 
    p2: [0,  4, 0]
    p1_zero_alpha_dir: [1, 0, 0]
    chord_table: [1.0]
    twist_table: [0.0]
    aoa_table: [-180, 0, 180]
    cl_table:  [2, 2, 2]
    cd_table:  [1.2])act";

TEST(ActuatorFLLC, NGP_ComputeLiftForceDistribution)
{
  const YAML::Node y_node = YAML::Load(actuatorPars);
  auto actMeta = ActuatorMeta(1, ActuatorType::ActLineSimpleNGP);
  auto actMetaSim = actuator_Simple_parse(y_node, actMeta);
  actMetaSim.useFLLC_ = true;
  ActuatorBulkSimple actBulk(actMetaSim);

  auto vel = actBulk.velocity_.view_host();
  auto relVel = actBulk.relativeVelocity_.view_host();
  auto density = actBulk.density_.view_host();
  auto spanDir = actMetaSim.spanDir_.view_host();

  auto range_policy=actBulk.local_range_policy();
  Kokkos::parallel_for(
    "init velocities", range_policy, KOKKOS_LAMBDA(int i) {
      double windSpeedSpan = 0.0;
      for (int j = 0; j < 3; ++j) {
        vel(i, j) = 1.0;
        windSpeedSpan += vel(i, j) * spanDir(0, j);
      }
      // compute 2d velocity for computing lift/drag to compare
      for (int j = 0; j < 3; ++j) {
        relVel(i, j) = vel(i, j) - windSpeedSpan * spanDir(0, j);
      }
      density(i) = 1.0;
    });

  Kokkos::parallel_for("compute forces", range_policy, ActSimpleComputeForce(actBulk, actMetaSim));

  // given a CL, CD, chord and U vector we can compute a lift and total force
  // then ensure our computation gives the expected lift force distribution from
  // the paper

  ActFixScalarDbl G("G-paper", vel.extent_int(0));
  auto area = actMetaSim.elemAreaDv_.view_host();

  const double chord = 1.0;
  const double Cl = 2.0;

  Kokkos::parallel_for(
    "compute G like paper", range_policy, KOKKOS_LAMBDA(int i) {
      const double umag2 = relVel(i, 0) * relVel(i, 0) +
                           relVel(i, 1) * relVel(i, 1) +
                           relVel(i, 2) * relVel(i, 2);
      G(i) = 0.5 * chord * Cl * umag2; // chord is 1.0 and Cl is 2.0 everywhere
    });

  actuator_utils::reduce_view_on_host(G);

  FLLC::compute_lift_force_distribution(actBulk);

  auto fllc_lift_force = actBulk.liftForceDistribution_.view_host();
    // assert that the two lift forces are equal
  Kokkos::parallel_for(
    "check values", range_policy, KOKKOS_LAMBDA(int i) {
      double gmag = 0.0;
      for (int j = 0; j < 3; ++j)
        gmag += fllc_lift_force(i, j) * fllc_lift_force(i, j);
      gmag = std::sqrt(gmag) / area(0, i) / density(i);
      EXPECT_DOUBLE_EQ(G(i), gmag);
    });
}

} // namespace
} // namespace nalu
} // namespace sierra