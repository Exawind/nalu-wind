// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorBulkDiskFAST.h>
#include <actuator/ActuatorParsingFAST.h>
#include "UnitTestActuatorUtil.h"
#include <nalu_make_unique.h>
#include <gtest/gtest.h>

namespace sierra
{
namespace nalu
{

namespace{

class ActuatorBulkDiskFastTest : public ::testing::Test{
protected:

  void SetUp(){
    actMeta_ = make_unique<ActuatorMeta>(1, ActuatorType::ActDiskFAST);
  }

  std::vector<std::string> inputs_{actuator_unit::nrel5MWinputs};
  std::unique_ptr<ActuatorMeta> actMeta_;
};

TEST_F(ActuatorBulkDiskFastTest, computeSweptPointCountUniform){
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

TEST_F(ActuatorBulkDiskFastTest, computeSweptPointCountVaried){
  auto y_node = actuator_unit::create_yaml_node(inputs_);
  auto myMeta = actuator_FAST_parse(y_node, *actMeta_);
  ASSERT_EQ(1, myMeta.nPointsSwept_.extent(0));
  ASSERT_EQ(0, myMeta.nPointsSwept_(0));
  ASSERT_FALSE(myMeta.useUniformAziSampling_(0));
  ActuatorBulkDiskFAST actBulk(myMeta, 0.0625);
  EXPECT_EQ(296, myMeta.numPointsTotal_);
  EXPECT_EQ(296, myMeta.numPointsTurbine_.h_view(0));
}

}

} /* namespace nalu */
} /* namespace sierra */
