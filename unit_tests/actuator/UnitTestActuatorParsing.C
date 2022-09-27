// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <aero/actuator/ActuatorParsing.h>

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
test_wo_lines(const std::vector<std::string>& testFile)
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
        EXPECT_THROW(actuator_parse(y_node), std::runtime_error)
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

class ActuatorParsingTest : public ::testing::Test
{
public:
  std::vector<std::string> inputFileLines_;

private:
  void SetUp()
  {
    inputFileLines_.push_back("actuator:\n");
    inputFileLines_.push_back("  type: ActLineFAST\n");
    inputFileLines_.push_back("  n_turbines_glob: 1\n");
    inputFileLines_.push_back("  search_target_part: [part1, part2]\n");
    inputFileLines_.push_back("  Turbine0:\n");
    inputFileLines_.push_back("   num_force_pts_blade: 2\n");
  }
};

/// Check the minimum parse terms are not violated and
/// that defaults haven't changed
TEST_F(ActuatorParsingTest, NGP_bareMinimumParse)
{
  try {
    auto y_actuator = create_yaml_node(inputFileLines_);
    ActuatorMeta actMeta = actuator_parse(y_actuator);
    EXPECT_EQ(stk::search::KDTREE, actMeta.searchMethod_);
    EXPECT_EQ(2u, actMeta.searchTargetNames_.size());
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
  test_wo_lines(inputFileLines_);
}

} // namespace

} // namespace nalu
} // namespace sierra
