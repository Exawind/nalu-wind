// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorParsingFAST.h>
#include <aero/actuator/ActuatorBulkFAST.h>
#include <aero/actuator/ActuatorParsing.h>
#include <NaluParsing.h>
#include <gtest/gtest.h>

namespace sierra {
namespace nalu {
namespace {

YAML::Node
create_yaml_node(const std::vector<std::string>& testFile)
{
  std::string temp;
  for (auto&& line : testFile) {
    temp += line;
  }
  return YAML::Load(temp);
}

// Ensure errors are clear nalu-wind errors and not yaml mysteries
void
test_wo_lines(
  const std::vector<std::string>& testFile, const ActuatorMeta& actMeta)
{
  // capture std::cout print statements
  std::stringstream buffer;
  std::streambuf* sbuf = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());

  for (unsigned i = 0; i < testFile.size(); i++) {
    std::vector<std::string> localCopy(testFile);
    localCopy[i] = "";
    try {
      auto y_node = create_yaml_node(localCopy);
      try {
        EXPECT_THROW(actuator_FAST_parse(y_node, actMeta), std::runtime_error)
          << " when missing: " << testFile[i];
      } catch (
        std::exception const& err) { // yaml or some other error sliped through
        EXPECT_TRUE(false) << "Missing line: " << testFile[i]
                           << "Leads to exception: " << err.what();
      }
    } catch (
      std::exception const& err) { // yaml-error throws during node creation
      // Do nothing for now, but should create a wrapper to catch these errors
      // and make them more helpful EXPECT_TRUE(false) << "Missing line: "<<
      // testFile[i]<<"Leads to exception: "<< err.what();
    }
  }
  std::cout.rdbuf(sbuf);
}

class ActuatorParsingFastTests : public ::testing::Test
{
public:
  std::vector<std::string> inputFileLines_;

private:
  void SetUp()
  {
    inputFileLines_.push_back("actuator:\n");
    inputFileLines_.push_back("  t_start: 0\n");
    inputFileLines_.push_back("  simStart: init\n");
    inputFileLines_.push_back("  n_every_checkpoint: 1\n");
    inputFileLines_.push_back("  dt_fast: 0.001\n");
    inputFileLines_.push_back("  t_max: 1000.0\n");
    inputFileLines_.push_back("  Turbine0:\n");
    inputFileLines_.push_back("    turbine_name: turbinator\n");
    inputFileLines_.push_back("    epsilon: [1.0, 0.5, 2.0]\n");
    inputFileLines_.push_back("    turb_id: 0\n");
    inputFileLines_.push_back("    fast_input_filename: UnitFast.inp\n");
    inputFileLines_.push_back("    restart_filename: restart.dat\n");
    inputFileLines_.push_back("    num_force_pts_blade: 10\n");
    inputFileLines_.push_back("    num_force_pts_tower: 10\n");
    inputFileLines_.push_back("    turbine_base_pos: [0,0,0]\n");
  }
};

TEST_F(ActuatorParsingFastTests, NGP_minimumRequired)
{
  ActuatorMeta actMeta(1, ActuatorTypeMap["ActLineFASTNGP"]);
  try {
    auto y_node = create_yaml_node(inputFileLines_);
    auto actMetaFAST = actuator_FAST_parse(y_node, actMeta);
    EXPECT_DOUBLE_EQ(1.0, actMetaFAST.epsilonTower_.h_view(0, 0));
    EXPECT_DOUBLE_EQ(0.5, actMetaFAST.epsilonTower_.h_view(0, 1));
    EXPECT_DOUBLE_EQ(2.0, actMetaFAST.epsilonTower_.h_view(0, 2));
    EXPECT_DOUBLE_EQ(1.0, actMetaFAST.epsilon_.h_view(0, 0));
    EXPECT_DOUBLE_EQ(0.5, actMetaFAST.epsilon_.h_view(0, 1));
    EXPECT_DOUBLE_EQ(2.0, actMetaFAST.epsilon_.h_view(0, 2));
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
  test_wo_lines(inputFileLines_, actMeta);
}

TEST_F(ActuatorParsingFastTests, NGP_minimumRequiredAAL)
{
  ActuatorMeta actMeta(1, ActuatorTypeMap["AdvActLineFASTNGP"]);
  inputFileLines_[8] = "    epsilon_chord: [1.0, 1.0, 1.0]\n";
  inputFileLines_.push_back("    epsilon_min: [0.1, 0.1, 0.1]\n");
  try {
    auto y_node = create_yaml_node(inputFileLines_);
    auto actMetaFAST = actuator_FAST_parse(y_node, actMeta);
    SUCCEED();
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
  test_wo_lines(inputFileLines_, actMeta);
}

TEST_F(ActuatorParsingFastTests, NGP_oneValueEpsilonParses)
{
  ActuatorMeta actMeta(1, ActuatorTypeMap["ActLineFASTNGP"]);
  inputFileLines_[8] = "    epsilon: 1.0\n";
  try {
    auto y_node = create_yaml_node(inputFileLines_);
    auto actMetaFAST = actuator_FAST_parse(y_node, actMeta);
    SUCCEED();
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
  test_wo_lines(inputFileLines_, actMeta);
}

TEST_F(ActuatorParsingFastTests, NGP_epsilonTower)
{
  ActuatorMeta actMeta(1, ActuatorTypeMap["ActLineFASTNGP"]);
  inputFileLines_.push_back("    epsilon_tower: [5.0, 5.0, 5.0]\n");
  try {
    auto y_node = create_yaml_node(inputFileLines_);
    auto actMetaFAST = actuator_FAST_parse(y_node, actMeta);
    for (int i = 0; i < 3; i++)
      EXPECT_DOUBLE_EQ(5.0, actMetaFAST.epsilonTower_.h_view(0, i));
    EXPECT_DOUBLE_EQ(1.0, actMetaFAST.epsilon_.h_view(0, 0));
    EXPECT_DOUBLE_EQ(0.5, actMetaFAST.epsilon_.h_view(0, 1));
    EXPECT_DOUBLE_EQ(2.0, actMetaFAST.epsilon_.h_view(0, 2));
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
}

TEST_F(ActuatorParsingFastTests, useFLLC)
{
  const char* actuatorYaml = R"blk(actuator:
  search_target_part: dummy
  search_method: stk_kdtree
  type: ActLineFASTNGP
  n_turbines_glob: 1
  t_start: 0
  simStart: init
  n_every_checkpoint: 1
  dt_fast: 0.00625
  t_max: 0.0625
  debug: no
  Turbine0:
    fllt_correction: yes
    turbine_name: turbinator
    epsilon_min: [5.0, 5.0, 5.00]
    epsilon_chord: [1.0, 1.0, 1.00]
    turb_id: 0
    fast_input_filename: nrel5mw.fst
    restart_filename: blah
    num_force_pts_blade: 10
    num_force_pts_tower: 10
    turbine_base_pos: [0,0,0]
    air_density:  1.0
    nacelle_area:  1.0
    nacelle_cd:  1.0
 )blk";
  const YAML::Node actuatorNode = YAML::Load(actuatorYaml);
  auto actBase = actuator_parse(actuatorNode);
  ASSERT_TRUE(actBase.useFLLC_);
  auto actMeta = actuator_FAST_parse(actuatorNode, actBase);
  ASSERT_TRUE(actMeta.useFLLC_);
}

} // namespace
} // namespace nalu
} // namespace sierra
