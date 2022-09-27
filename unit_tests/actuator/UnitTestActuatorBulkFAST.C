// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
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
class ActuatorBulkFastTests : public ::testing::Test
{
protected:
  std::string inputFileSurrogate_;
  const double tol_;
  std::vector<std::string> fastParseParams_{actuator_unit::nrel5MWinputs};
  const ActuatorMeta actMeta_;

  ActuatorBulkFastTests()
    : tol_(1e-8), actMeta_(1, ActuatorType::ActLineFASTNGP)
  {
  }
};

TEST_F(ActuatorBulkFastTests, NGP_initializeActuatorBulk)
{
  std::vector<std::string> modInputs(fastParseParams_);
  modInputs[4] = "  dt_fast: 0.005\n";
  modInputs[5] = "  t_max: 0.29\n";
  const YAML::Node y_node = actuator_unit::create_yaml_node(modInputs);
  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_);

  const fast::fastInputs& fi = actMetaFast.fastInputs_;
  ASSERT_EQ(fi.comm, NaluEnv::self().parallel_comm());
  ASSERT_EQ(fi.globTurbineData.size(), 1);
  ASSERT_EQ(fi.debug, false);
  ASSERT_EQ(fi.dryRun, false);
  ASSERT_EQ(fi.nTurbinesGlob, 1);
  ASSERT_EQ(fi.tStart, 0.0);
  ASSERT_EQ(fi.simStart, fast::init);
  ASSERT_EQ(fi.nEveryCheckPoint, 1);
  ASSERT_EQ(fi.dtFAST, 0.005);
  ASSERT_EQ(fi.tMax, 0.29);

  ASSERT_EQ(fi.globTurbineData[0].FASTInputFileName, "nrel5mw.fst");
  ASSERT_EQ(fi.globTurbineData[0].FASTRestartFileName, "blah");
  ASSERT_EQ(fi.globTurbineData[0].TurbID, 0);
  ASSERT_EQ(fi.globTurbineData[0].numForcePtsBlade, 10);
  ASSERT_EQ(fi.globTurbineData[0].numForcePtsTwr, 10);
  ASSERT_EQ(fi.globTurbineData[0].air_density, 1.0);
  ASSERT_EQ(fi.globTurbineData[0].nacelle_area, 1.0);
  ASSERT_EQ(fi.globTurbineData[0].nacelle_cd, 1.0);

  try {
    ActuatorBulkFAST actBulk(actMetaFast, 0.29);
    EXPECT_EQ(actBulk.tStepRatio_, 58);
    EXPECT_FALSE(actBulk.openFast_.isDebug());
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
}

TEST_F(ActuatorBulkFastTests, NGP_epsilonTowerAndAnisotropicEpsilon)
{

  auto epsLoc = std::find_if(
    fastParseParams_.begin(), fastParseParams_.end(),
    [](std::string val) { return val.find("epsilon:") != std::string::npos; });

  *epsLoc = "    epsilon: [1.0, 0.5, 2.0]\n";
  fastParseParams_.push_back("    epsilon_tower: [5.0, 5.0, 5.0]\n");

  const YAML::Node y_node = actuator_unit::create_yaml_node(fastParseParams_);
  auto actMetaFast = actuator_FAST_parse(y_node, actMeta_);
  try {
    ActuatorBulkFAST actBulk(actMetaFast, 0.0625);
    auto epsilon = actBulk.epsilon_.view_host();
    auto orient = actBulk.orientationTensor_.view_host();

    // check blades
    for (int i = 1; i < 31; i++) {
      EXPECT_DOUBLE_EQ(1.0, epsilon(i, 0));
      EXPECT_DOUBLE_EQ(0.5, epsilon(i, 1));
      EXPECT_DOUBLE_EQ(2.0, epsilon(i, 2));
    }

    // check tower
    for (int i = 31; i < 41; i++) {
      for (int j = 1; j < 3; j++) {
        EXPECT_DOUBLE_EQ(5.0, epsilon(i, j));
      }
    }

    EXPECT_FALSE(actMetaFast.isotropicGaussian_);
    EXPECT_EQ(actMetaFast.numPointsTotal_, orient.extent_int(0));

  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
}

} // namespace

} // namespace nalu
} // namespace sierra
