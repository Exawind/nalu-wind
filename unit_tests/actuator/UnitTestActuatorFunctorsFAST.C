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
#include <UnitTestUtils.h>
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
  stk::mesh::MetaData stkMeta_;
  stk::mesh::BulkData stkBulk_;
  const double tol_;
  const VectorFieldType* coordinates_{nullptr};
  VectorFieldType* velocity_{nullptr};
  VectorFieldType* actuatorForce_{nullptr};
  std::vector<std::string> fastParseParams_;
  ActuatorMeta actMeta_;

  ActuatorFunctorFASTTests()
    : stkMeta_(3),
      stkBulk_(stkMeta_, MPI_COMM_WORLD),
      tol_(1e-8),
      coordinates_(nullptr),
      velocity_(&stkMeta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "velocity")),
      actuatorForce_(&stkMeta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "actuator_source")),
      actMeta_(1)
  {
    stk::mesh::put_field_on_mesh(
      *velocity_, stkMeta_.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(
      *actuatorForce_, stkMeta_.universal_part(), 3, nullptr);
  }

  void SetUp()
  {
    const std::string meshSpec = "generated:5x5x5";
    unit_test_utils::fill_hex8_mesh(meshSpec, stkBulk_);
    coordinates_ =
      static_cast<const VectorFieldType*>(stkMeta_.coordinate_field());
    const stk::mesh::Selector selector =
      stkMeta_.locally_owned_part() | stkMeta_.globally_shared_part();
    const auto& buckets =
      stkBulk_.get_buckets(stk::topology::NODE_RANK, selector);
    for (const stk::mesh::Bucket* bptr : buckets) {
      for (stk::mesh::Entity node : *bptr) {
        const double* coords = stk::mesh::field_data(*coordinates_, node);
        double* vel = stk::mesh::field_data(*velocity_, node);
        double* aF = stk::mesh::field_data(*actuatorForce_, node);
        for (int i = 0; i < 3; i++) {
          vel[i] = coords[i];
          aF[i] = 0.0;
        }
      }
    }
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
    fastParseParams_.push_back("    epsilon: [1.0, 0, 0]\n");
    fastParseParams_.push_back("    turb_id: 0\n");
    fastParseParams_.push_back("    fast_input_filename: reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst\n");
    fastParseParams_.push_back("    restart_filename: blah\n");
    fastParseParams_.push_back("    num_force_pts_blade: 10\n");
    fastParseParams_.push_back("    num_force_pts_tower: 10\n");
    fastParseParams_.push_back("    turbine_base_pos: [0,0,0]\n");
    fastParseParams_.push_back("    turbine_hub_pos:  [0,0,60.0]\n");
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
    ActuatorBulkFAST actBulk(actMetaFast, stkBulk_);
    EXPECT_TRUE(actBulk.openFast_.isDebug());
  } catch ( std::exception const& err){
    FAIL()<<err.what();
  }
}

TEST_F(ActuatorFunctorFASTTests, runActFastZero){
  const YAML::Node y_node = create_yaml_node(fastParseParams_);

  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_, 1.0);
  ActuatorBulkFAST actBulk(actMetaFast, stkBulk_);

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

//TODO(psakiev) run updatePoints
TEST_F(ActuatorFunctorFASTTests, runUpdatePoints){
  const YAML::Node y_node = create_yaml_node(fastParseParams_);

  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_, 1.0);
  ActuatorBulkFAST actBulk(actMetaFast, stkBulk_);

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
  actBulk.reduce_view_on_host(points, NaluEnv::self().parallel_comm());

  // hub
  EXPECT_DOUBLE_EQ(0.0, points(0,0));
  EXPECT_DOUBLE_EQ(0.0, points(0,1));
  EXPECT_DOUBLE_EQ(60.0, points(0,2));
}
//TODO(psakiev) run assign vel
//TODO(psakiev) run compute forces

}

} /* namespace nalu */
} /* namespace sierra */
