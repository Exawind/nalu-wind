// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorParsingFAST.h>
#include <aero/actuator/ActuatorFunctorsFAST.h>
#include <aero/actuator/ActuatorBulkFAST.h>
#include <aero/actuator/UtilitiesActuator.h>
#include "UnitTestActuatorUtil.h"
#include <gtest/gtest.h>

namespace sierra {
namespace nalu {

namespace {

//-----------------------------------------------------------------
class ActuatorFunctorFastTests : public ::testing::Test
{
protected:
  std::string inputFileSurrogate_;
  const double tol_;
  std::vector<std::string> fastParseParams_{actuator_unit::nrel5MWinputs};
  const ActuatorMeta actMeta_;

  ActuatorFunctorFastTests()
    : tol_(1e-8), actMeta_(1, ActuatorType::ActLineFASTNGP)
  {
  }
};

TEST_F(ActuatorFunctorFastTests, NGP_runUpdatePoints)
{
  const YAML::Node y_node = actuator_unit::create_yaml_node(fastParseParams_);

  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_);
  ActuatorBulkFAST actBulk(actMetaFast, 0.0625);

  fast::OpenFAST& fast = actBulk.openFast_;
  const int turbineID = actBulk.localTurbineId_;

  ASSERT_EQ(actMetaFast.numPointsTotal_, 41);

  auto points = actBulk.pointCentroid_.view_host();

  RunActFastUpdatePoints(actBulk);

  // test for empty points
  for (int i = 0; i < actMetaFast.numPointsTotal_; ++i) {
    if (i != actMetaFast.get_fast_index(fast::TOWER, 0, 0)) {
      EXPECT_TRUE(points(i, 0) != 0 || points(i, 1) != 0 || points(i, 2) != 0)
        << "Index failure: " << i
        << " on rank: " << NaluEnv::self().parallel_rank();
    }
  }

  if (turbineID == fast.get_procNo(actBulk.localTurbineId_)) {
    EXPECT_EQ(fast::HUB, fast.getForceNodeType(turbineID, 0));
    std::vector<double> hubPos(3);
    fast.getHubPos(hubPos, turbineID);
    EXPECT_DOUBLE_EQ(hubPos[0], points(0, 0));
    EXPECT_DOUBLE_EQ(hubPos[1], points(0, 1));
    EXPECT_DOUBLE_EQ(hubPos[2], points(0, 2));

    int i = 1;
    for (; i <= fast.get_numForcePtsBlade(turbineID) * 3; ++i) {
      EXPECT_EQ(fast::BLADE, fast.getForceNodeType(turbineID, i))
        << "Index is: " << i;
    }
    for (; i < fast.get_numForcePts(turbineID); ++i) {
      EXPECT_EQ(fast::TOWER, fast.getForceNodeType(turbineID, i))
        << "Index is: " << i;
    }
  }
}

TEST_F(ActuatorFunctorFastTests, NGP_runAssignVelAndComputeForces)
{
  const YAML::Node y_node = actuator_unit::create_yaml_node(fastParseParams_);

  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_);
  ActuatorBulkFAST actBulk(actMetaFast, 0.0625);

  fast::OpenFAST& fast = actBulk.openFast_;
  const int turbineID = actBulk.localTurbineId_;

  ASSERT_EQ(actMetaFast.numPointsTotal_, 41);

  auto vel = actBulk.velocity_.view_host();
  auto force = actBulk.actuatorForce_.view_host();

  auto localRangePolicy = actBulk.local_range_policy();

  for (int i = 0; i < vel.extent_int(0); ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(0.0, vel(i, j));
    }
  }

  actBulk.velocity_.modify_host();
  actBulk.actuatorForce_.modify_host();

  Kokkos::deep_copy(actBulk.velocity_.view_host(), 1.0);

  Kokkos::parallel_for(
    "testAssignVel", localRangePolicy, ActFastAssignVel(actBulk));

  actBulk.interpolate_velocities_to_fast();
  actBulk.step_fast();

  RunActFastComputeForce(actBulk);

  ActFixVectorDbl fastForces(
    "forcesComputedFromFAST", actMetaFast.numPointsTotal_);

  if (fast.get_procNo(turbineID) == turbineID) {
    std::vector<double> tempForce(3);
    const int offset = actBulk.turbIdOffset_.h_view(turbineID);

    for (int i = offset; i < fast.get_numForcePts(turbineID) + offset; ++i) {
      fast.getForce(tempForce, i - offset, turbineID);

      for (int j = 0; j < 3; ++j) {
        fastForces(i, j) = tempForce[j];
      }
    }
  }

  actuator_utils::reduce_view_on_host(fastForces);

  Kokkos::parallel_for(
    "checkAnswers",
    Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(
      0, actMetaFast.numPointsTotal_),
    [&](int i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_DOUBLE_EQ(fastForces(i, j), force(i, j));
      }
    });
}

TEST_F(ActuatorFunctorFastTests, NGP_spreadForceWhProjIdentity)
{
  // skipping for now.  There is some issue with the openfast files getting
  // created when using it in the unit tests.  This test passes in isolation but
  // fails when all the tests run.
  GTEST_SKIP();
  auto epsLoc = std::find_if(
    fastParseParams_.begin(), fastParseParams_.end(),
    [](std::string val) { return val.find("epsilon:") != std::string::npos; });

  *epsLoc = "    epsilon: [1.0, 0.5, 2.0]\n";
  fastParseParams_.push_back("    epsilon_tower: 5.0\n");
  const YAML::Node y_node = actuator_unit::create_yaml_node(fastParseParams_);

  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_);
  ActuatorBulkFAST actBulk(actMetaFast, 0.0625);

  auto fastRangePolicy = actBulk.local_range_policy();

  actBulk.velocity_.modify_host();
  actBulk.actuatorForce_.modify_host();

  auto vel = actBulk.velocity_.view_host();
  auto force = actBulk.actuatorForce_.view_host();
  auto points = actBulk.pointCentroid_.view_host();

  Kokkos::deep_copy(vel, 8.0);

  RunActFastUpdatePoints(actBulk);

  Kokkos::parallel_for(
    "testAssignVel", fastRangePolicy, ActFastAssignVel(actBulk));

  actBulk.interpolate_velocities_to_fast();
  actBulk.step_fast();

  RunActFastComputeForce(actBulk);
  RunActFastStashOrientVecs(actBulk);

  // compute source term contributions at the hub location
  // from the tower actuator points
  auto hubLocation =
    Kokkos::subview(actBulk.pointCentroid_.view_host(), 0, Kokkos::ALL);
  const int nPntsTower =
    actMetaFast.fastInputs_.globTurbineData[0].numForcePtsTwr;

  ActFastSpreadForceWhProjInnerLoop projInner(actBulk);
  SpreadForceInnerLoop isoInner(actBulk);

  for (int i = 0; i < nPntsTower; i++) {
    double sourceTermNoProj[3] = {0, 0, 0};
    double sourceTermWhProj[3] = {0, 0, 0};
    const uint64_t id =
      static_cast<uint64_t>(actMetaFast.get_fast_index(fast::TOWER, 0, i));

    auto epsilon =
      Kokkos::subview(actBulk.epsilon_.view_host(), id, Kokkos::ALL);

    for (int j = 0; j < 3; j++) {
      ASSERT_DOUBLE_EQ(5.0, epsilon(j)) << "point index: " << id << " j: " << j;
    }

    // note offset is zero so can just use id
    projInner(id, hubLocation.data(), &sourceTermWhProj[0], 1.0, 1.0);
    isoInner(id, hubLocation.data(), &sourceTermNoProj[0], 1.0, 1.0);

    for (int j = 0; j < 3; j++) {
      EXPECT_NEAR(sourceTermNoProj[j], sourceTermWhProj[j], 1e-8);
    }
  }
}

} // namespace

} /* namespace nalu */
} /* namespace sierra */
