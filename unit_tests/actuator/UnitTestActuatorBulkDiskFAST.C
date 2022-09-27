// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorBulkDiskFAST.h>
#include <aero/actuator/ActuatorParsingFAST.h>
#include "UnitTestActuatorUtil.h"
#include <aero/actuator/UtilitiesActuator.h>
#include <memory>
#include <gtest/gtest.h>

namespace sierra {
namespace nalu {

namespace {

class ActuatorBulkDiskFastTest : public ::testing::Test
{
protected:
  void SetUp()
  {
    actMeta_ = std::make_unique<ActuatorMeta>(1, ActuatorType::ActDiskFASTNGP);
  }

  std::vector<std::string> inputs_{actuator_unit::nrel5MWinputs};
  std::unique_ptr<ActuatorMeta> actMeta_;
};

// TODO(psakeiv) move this to a more appropriate location
TEST_F(ActuatorBulkDiskFastTest, NGP_fastPointIndexLocator)
{
  auto y_node = actuator_unit::create_yaml_node(inputs_);
  auto myMeta = actuator_FAST_parse(y_node, *actMeta_);
  ActuatorBulkDiskFAST actBulk(myMeta, 0.0625);

  fast::OpenFAST& fast = actBulk.openFast_;

  if (NaluEnv::self().parallel_rank() == 0) {
    const int nPntsBlade = fast.get_numForcePtsBlade(0);
    const int nPntsTower = fast.get_numForcePtsTwr(0);
    {
      const int index = myMeta.get_fast_index(fast::HUB, 0);
      EXPECT_EQ(fast::HUB, fast.getForceNodeType(0, index));
    }
    for (int i = 0; i < nPntsBlade; i++) {
      for (int j = 0; j < 3; j++) {
        const int index = myMeta.get_fast_index(fast::BLADE, 0, i, j);
        EXPECT_EQ(fast::BLADE, fast.getForceNodeType(0, index));
      }
    }
    for (int j = 0; j < nPntsTower; j++) {
      const int index = myMeta.get_fast_index(fast::TOWER, 0, j);
      EXPECT_EQ(fast::TOWER, fast.getForceNodeType(0, index));
    }
  }
}

TEST_F(ActuatorBulkDiskFastTest, NGP_computeSweptPointCountUniform)
{
  inputs_.push_back("    num_swept_pts: 2\n");
  auto y_node = actuator_unit::create_yaml_node(inputs_);
  auto myMeta = actuator_FAST_parse(y_node, *actMeta_);
  ASSERT_EQ(1, myMeta.nPointsSwept_.extent(0));
  ASSERT_EQ(2, myMeta.nPointsSwept_(0));
  ASSERT_TRUE(myMeta.useUniformAziSampling_(0));
  ActuatorBulkDiskFAST actBulk(myMeta, 0.0625);
  EXPECT_EQ(101, myMeta.numPointsTotal_);
  EXPECT_EQ(101, myMeta.numPointsTurbine_.h_view(0));
}

TEST_F(ActuatorBulkDiskFastTest, NGP_computeSweptPointCountVaried)
{
  auto y_node = actuator_unit::create_yaml_node(inputs_);
  auto myMeta = actuator_FAST_parse(y_node, *actMeta_);
  ASSERT_EQ(1, myMeta.nPointsSwept_.extent(0));
  ASSERT_EQ(0, myMeta.nPointsSwept_(0));
  ASSERT_FALSE(myMeta.useUniformAziSampling_(0));
  ActuatorBulkDiskFAST actBulk(myMeta, 0.0625);
  EXPECT_EQ(296, myMeta.numPointsTotal_);
  EXPECT_EQ(296, myMeta.numPointsTurbine_.h_view(0));
}

TEST_F(ActuatorBulkDiskFastTest, NGP_sweptPointsPopulatedUniform)
{
  inputs_.push_back("    num_swept_pts: 2\n");
  auto y_node = actuator_unit::create_yaml_node(inputs_);
  auto myMeta = actuator_FAST_parse(y_node, *actMeta_);
  ASSERT_TRUE(myMeta.useUniformAziSampling_(0));
  ActuatorBulkDiskFAST actBulk(myMeta, 0.0625);

  if (NaluEnv::self().parallel_rank() == 0) {
    const int nPntsBlade = actBulk.openFast_.get_numForcePtsBlade(0);
    ASSERT_EQ(nPntsBlade, actBulk.numSweptCount_.size());
    ASSERT_EQ(nPntsBlade, actBulk.numSweptOffset_.size());
  }

  const int start = actuator_utils::get_fast_point_index(
    myMeta.fastInputs_, 0, 3, fast::TOWER, 9);
  auto points = actBulk.pointCentroid_.view_host();
  auto epsilon = actBulk.epsilon_.view_host();
  auto searchRad = actBulk.searchRadius_.view_host();

  // show values have been initialized
  for (int i = 0; i < myMeta.numPointsTotal_; ++i) {
    EXPECT_TRUE(
      epsilon(i, 0) != 0.0 || epsilon(i, 1) != 0.0 || epsilon(i, 2) != 0.0)
      << "Index failed at: " << i;
    EXPECT_TRUE(searchRad(i) != 0.0) << "Index failed at: " << i;
  }

  for (int i = start; i < myMeta.numPointsTotal_; ++i) {
    EXPECT_TRUE(
      points(i, 0) != 0.0 || points(i, 1) != 0.0 || points(i, 2) != 0.0)
      << "Index failed at: " << i;
  }
}

TEST_F(ActuatorBulkDiskFastTest, NGP_sweptPointsPopulatedVaried)
{
  auto y_node = actuator_unit::create_yaml_node(inputs_);
  auto myMeta = actuator_FAST_parse(y_node, *actMeta_);
  ASSERT_FALSE(myMeta.useUniformAziSampling_(0));
  ActuatorBulkDiskFAST actBulk(myMeta, 0.0625);

  if (NaluEnv::self().parallel_rank() == 0) {
    const int nPntsBlade = actBulk.openFast_.get_numForcePtsBlade(0);
    ASSERT_EQ(nPntsBlade, actBulk.numSweptCount_.size());
    ASSERT_EQ(nPntsBlade, actBulk.numSweptOffset_.size());
  }

  const int start = actuator_utils::get_fast_point_index(
    myMeta.fastInputs_, 0, 3, fast::TOWER, 9);
  auto points = actBulk.pointCentroid_.view_host();
  auto epsilon = actBulk.epsilon_.view_host();
  auto searchRad = actBulk.searchRadius_.view_host();

  // show values have been initialized
  for (int i = 0; i < myMeta.numPointsTotal_; ++i) {
    EXPECT_TRUE(
      epsilon(i, 0) != 0.0 || epsilon(i, 1) != 0.0 || epsilon(i, 2) != 0.0)
      << "Index failed at: " << i;
    EXPECT_TRUE(searchRad(i) != 0.0) << "Index failed at: " << i;
  }

  for (int i = start; i < myMeta.numPointsTotal_; ++i) {
    EXPECT_TRUE(
      points(i, 0) != 0.0 || points(i, 1) != 0.0 || points(i, 2) != 0.0)
      << "Index failed at: " << i;
  }
}

} // namespace

} /* namespace nalu */
} /* namespace sierra */
