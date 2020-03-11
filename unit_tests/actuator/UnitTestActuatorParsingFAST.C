// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorParsingFAST.h>
#include <actuator/ActuatorBulkFAST.h>
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
        EXPECT_THROW(actuator_FAST_parse(y_node, actMeta, 1.0), std::runtime_error)
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

class ActuatorParsingFASTTest : public ::testing::Test
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
    inputFileLines_.push_back("    epsilon: [1.0, 0, 0]\n");
    inputFileLines_.push_back("    turb_id: 0\n");
    inputFileLines_.push_back("    fast_input_filename: UnitFast.inp\n");
    inputFileLines_.push_back("    restart_filename: restart.dat\n");
    inputFileLines_.push_back("    num_force_pts_blade: 10\n");
    inputFileLines_.push_back("    num_force_pts_tower: 10\n");
    inputFileLines_.push_back("    turbine_base_pos: [0,0,0]\n");
  }
};

TEST_F(ActuatorParsingFASTTest, minimumRequired)
{
  ActuatorMeta actMeta(1, ActuatorTypeMap["ActLineFAST"]);
  try {
    auto y_node = create_yaml_node(inputFileLines_);
    auto actMetaFAST = actuator_FAST_parse(y_node, actMeta, 1.0);
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
  test_wo_lines(inputFileLines_, actMeta);
}

TEST_F(ActuatorParsingFASTTest, minimumRequiredFLLC)
{
  ActuatorMeta actMeta(1, ActuatorTypeMap["ActLineFAST"]);
  inputFileLines_[8] = "    epsilon_chord: [1.0, 1.0, 1.0]\n";
  inputFileLines_.push_back("    epsilon_min: [10.0, 0.0, 0.0]\n");
  try {
    auto y_node = create_yaml_node(inputFileLines_);
    auto actMetaFAST = actuator_FAST_parse(y_node, actMeta, 1.0);
    SUCCEED();
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
  test_wo_lines(inputFileLines_, actMeta);
}

} // namespace
} // namespace nalu
} // namespace sierra
