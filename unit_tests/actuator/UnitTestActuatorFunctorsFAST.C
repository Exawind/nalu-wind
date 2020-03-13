// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorParsingFAST.h>
#include <actuator/ActuatorFunctorsFAST.h>
#include <actuator/ActuatorBulkFAST.h>
#include <actuator/UtilitiesActuator.h>
#include <yaml-cpp/yaml.h>
#include <gtest/gtest.h>

namespace sierra {
namespace nalu {

namespace{

YAML::Node
create_yaml_node(const std::vector<std::string>& testFile)
{
  std::string temp;
  for (auto&& line : testFile) {
    temp += line;
  }
  return YAML::Load(temp);
}

//-----------------------------------------------------------------
class ActuatorFunctorFASTTests : public ::testing::Test
{
protected:
  std::string inputFileSurrogate_;
  const double tol_;
  std::vector<std::string> fastParseParams_;
  ActuatorMeta actMeta_;

  ActuatorFunctorFASTTests()
    : tol_(1e-8),
      actMeta_(1)
  {}

  void SetUp()
  {
    fastParseParams_.push_back("actuator:\n");
    fastParseParams_.push_back("  t_start: 0\n");
    fastParseParams_.push_back("  simStart: init\n");
    fastParseParams_.push_back("  n_every_checkpoint: 1\n");
    fastParseParams_.push_back("  dt_fast: 0.00625\n");
    fastParseParams_.push_back("  t_max: 0.0625\n");
    fastParseParams_.push_back("  dry_run: no\n");
    fastParseParams_.push_back("  debug: yes\n");
    fastParseParams_.push_back("  Turbine0:\n");
    fastParseParams_.push_back("    turbine_name: turbinator\n");
    fastParseParams_.push_back("    epsilon: [5.0, 5.0, 5.00]\n");
    fastParseParams_.push_back("    turb_id: 0\n");
    fastParseParams_.push_back("    fast_input_filename: reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst\n");
    fastParseParams_.push_back("    restart_filename: blah\n");
    fastParseParams_.push_back("    num_force_pts_blade: 10\n");
    fastParseParams_.push_back("    num_force_pts_tower: 10\n");
    fastParseParams_.push_back("    turbine_base_pos: [0,0,0]\n");
    fastParseParams_.push_back("    air_density:  1.0\n");
    fastParseParams_.push_back("    nacelle_area:  1.0\n");
    fastParseParams_.push_back("    nacelle_cd:  1.0\n");
  }

};


TEST_F(ActuatorFunctorFASTTests, initializeActuatorBulk){
  const YAML::Node y_node = create_yaml_node(fastParseParams_);
  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_, 1.0);

  const fast::fastInputs& fi = actMetaFast.fastInputs_;
  ASSERT_EQ(fi.comm , NaluEnv::self().parallel_comm());
  ASSERT_EQ(fi.globTurbineData.size(),1);
  ASSERT_EQ(fi.debug , true);
  ASSERT_EQ(fi.dryRun , false);
  ASSERT_EQ(fi.nTurbinesGlob , 1);
  ASSERT_EQ(fi.tStart , 0.0);
  ASSERT_EQ(fi.simStart , fast::init);
  ASSERT_EQ(fi.nEveryCheckPoint , 1);
  ASSERT_EQ(fi.dtFAST , 0.00625);
  ASSERT_EQ(fi.tMax , 0.0625);

  ASSERT_EQ(fi.globTurbineData[0].FASTInputFileName , "reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst");
  ASSERT_EQ(fi.globTurbineData[0].FASTRestartFileName ,"blah");
  ASSERT_EQ(fi.globTurbineData[0].TurbID,0);
  ASSERT_EQ(fi.globTurbineData[0].numForcePtsBlade , 10);
  ASSERT_EQ(fi.globTurbineData[0].numForcePtsTwr,10);
  ASSERT_EQ(fi.globTurbineData[0].air_density,1.0);
  ASSERT_EQ(fi.globTurbineData[0].nacelle_area,1.0);
  ASSERT_EQ(fi.globTurbineData[0].nacelle_cd,1.0);

  try{
    ActuatorBulkFAST actBulk(actMetaFast);
    EXPECT_TRUE(actBulk.openFast_.isDebug());
  } catch ( std::exception const& err){
    FAIL()<<err.what();
  }
}

TEST_F(ActuatorFunctorFASTTests, runActFastZero){
  const YAML::Node y_node = create_yaml_node(fastParseParams_);

  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_, 1.0);
  ActuatorBulkFAST actBulk(actMetaFast);

  ASSERT_EQ(actBulk.totalNumPoints_, 41);

  auto velHost = actBulk.velocity_.view_host();
  auto frcHost = actBulk.actuatorForce_.view_host();

  actBulk.actuatorForce_.modify_device();
  actBulk.velocity_.modify_device();

  for(int i=0; i<actBulk.totalNumPoints_; ++i){
    for(int j=0; j<3; ++j){
      actBulk.actuatorForce_.h_view(i,j) = 1.0;
      actBulk.velocity_.h_view(i,j) = 1.0;
    }
  }

  actBulk.actuatorForce_.sync_device();
  actBulk.velocity_.sync_device();

  for(int i = 0; i<velHost.extent_int(0); ++i){
    for(int j=0; j<3; ++j){
      EXPECT_DOUBLE_EQ(1.0, velHost(i,j));
      EXPECT_DOUBLE_EQ(1.0, frcHost(i,j));
    }
  }

  Kokkos::parallel_for("testActFastZero", actBulk.totalNumPoints_,ActFastZero(actBulk));

  for(int i = 0; i<velHost.extent_int(0); ++i){
    for(int j=0; j<3; ++j){
      EXPECT_DOUBLE_EQ(0.0, velHost(i,j));
      EXPECT_DOUBLE_EQ(0.0, frcHost(i,j));
    }
  }
}

TEST_F(ActuatorFunctorFASTTests, runUpdatePoints){
  const YAML::Node y_node = create_yaml_node(fastParseParams_);

  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_, 1.0);
  ActuatorBulkFAST actBulk(actMetaFast);

  fast::OpenFAST& fast = actBulk.openFast_;
  const int turbineID = actBulk.localTurbineId_;

  ASSERT_EQ(actBulk.totalNumPoints_, 41);

  auto points = actBulk.pointCentroid_.view_host();
  auto localRangePolicy = actBulk.local_range_policy(actMeta_);
  Kokkos::parallel_for("testActFastZero", actBulk.totalNumPoints_,ActFastZero(actBulk));

  for(int i = 0; i<points.extent_int(0); ++i){
    for(int j=0; j<3; ++j){
      EXPECT_DOUBLE_EQ(0.0, points(i,j));
    }
  }

  Kokkos::parallel_for("testUpdatePoints", localRangePolicy, ActFastUpdatePoints(actBulk));
  actuator_utils::reduce_view_on_host(points);

  if(turbineID == fast.get_procNo(actBulk.localTurbineId_)){
    EXPECT_EQ(fast::HUB, fast.getForceNodeType(turbineID,0));
    std::vector<double> hubPos(3);
    fast.getHubPos(hubPos, turbineID);
    EXPECT_DOUBLE_EQ(hubPos[0], points(0,0));
    EXPECT_DOUBLE_EQ(hubPos[1], points(0,1));
    EXPECT_DOUBLE_EQ(hubPos[2], points(0,2));

    int i =1;
    for(; i<=fast.get_numForcePtsBlade(turbineID)*3; ++i){
      EXPECT_EQ(fast::BLADE, fast.getForceNodeType(turbineID, i)) << "Index is: "<<i;
    }
    for(; i<fast.get_numForcePts(turbineID); ++i){
      EXPECT_EQ(fast::TOWER, fast.getForceNodeType(turbineID, i)) << "Index is: "<<i;
    }
  }
}


TEST_F(ActuatorFunctorFASTTests, runAssignVelAndComputeForces){
  const YAML::Node y_node = create_yaml_node(fastParseParams_);

  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_, 1.0);
  ActuatorBulkFAST actBulk(actMetaFast);

  fast::OpenFAST& fast = actBulk.openFast_;
  const int turbineID = actBulk.localTurbineId_;

  ASSERT_EQ(actBulk.totalNumPoints_, 41);

  auto vel = actBulk.velocity_.view_host();
  auto force = actBulk.actuatorForce_.view_host();

  auto localRangePolicy = actBulk.local_range_policy(actMeta_);
  Kokkos::parallel_for("testActFastZero", actBulk.totalNumPoints_,ActFastZero(actBulk));

  for(int i = 0; i<vel.extent_int(0); ++i){
    for(int j=0; j<3; ++j){
      EXPECT_DOUBLE_EQ(0.0, vel(i,j));
    }
  }

  actBulk.velocity_.modify_host();
  actBulk.actuatorForce_.modify_host();

  Kokkos::parallel_for("initUniformVelocity",localRangePolicy,[&](int i){
   actBulk.velocity_.h_view(i,0) = 1.0;
  });
  actuator_utils::reduce_view_on_host(vel);

  Kokkos::parallel_for("testAssignVel", localRangePolicy, ActFastAssignVel(actBulk));
  Kokkos::parallel_for("testComputeForces", localRangePolicy, ActFastComputeForce(actBulk));

  actuator_utils::reduce_view_on_host(force);

  ActFixVectorDbl fastForces("forcesComputedFromFAST", actBulk.totalNumPoints_);

  if(fast.get_procNo(turbineID)==turbineID){
    std::vector<double> tempForce(3);
    const int offset = actBulk.turbIdOffset_.h_view(turbineID);

    for(int i=offset; i<fast.get_numForcePts(turbineID)+offset; ++i){
      fast.getForce(tempForce, i-offset, turbineID);

      for(int j=0; j<3; ++j){
        fastForces(i,j)=tempForce[j];
      }
    }
  }

  actuator_utils::reduce_view_on_host(fastForces);

  Kokkos::parallel_for("checkAnswers",
    Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0,actBulk.totalNumPoints_),
    KOKKOS_LAMBDA(int i){
    for(int j=0; j<3; ++j){
      EXPECT_DOUBLE_EQ(fastForces(i,j), force(i,j));
    }
  });
}

}

} /* namespace nalu */
} /* namespace sierra */
