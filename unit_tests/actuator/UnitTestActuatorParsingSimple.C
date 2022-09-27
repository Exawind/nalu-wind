// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorParsingSimple.h>
#include <aero/actuator/ActuatorBulkSimple.h>
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
        EXPECT_THROW(actuator_Simple_parse(y_node, actMeta), std::runtime_error)
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

class ActuatorParsingSimpleTests : public ::testing::Test
{
public:
  std::vector<std::string> inputFileLines_;

private:
  void SetUp()
  {
    inputFileLines_.push_back("actuator:\n");
    inputFileLines_.push_back("  n_simpleblades: 1\n");
    inputFileLines_.push_back("  Blade0:\n");
    inputFileLines_.push_back("    num_force_pts_blade: 10\n");
    inputFileLines_.push_back("    epsilon: [3.0, 3.0, 3.0]\n");
    inputFileLines_.push_back("    p1: [-25, -4, 0]\n");
    inputFileLines_.push_back("    p2: [-25,  4, 0]\n");
    inputFileLines_.push_back("    p1_zero_alpha_dir: [1, 0, 0]\n");
    inputFileLines_.push_back("    chord_table: [1.0]\n");
    inputFileLines_.push_back("    twist_table: [0.0]\n");
    inputFileLines_.push_back("    aoa_table: [-180, 0, 180]\n");
    inputFileLines_.push_back("    cl_table:  [-10,  0, 10]\n");
    inputFileLines_.push_back("    cd_table:  [0]\n");
  }
};

TEST_F(ActuatorParsingSimpleTests, NGP_minimumRequired)
{
  ActuatorMeta actMeta(1, ActuatorTypeMap["ActLineSimpleNGP"]);
  try {
    auto y_node = create_yaml_node(inputFileLines_);
    auto actMetaSimple = actuator_Simple_parse(y_node, actMeta);
    EXPECT_EQ(10, actMetaSimple.num_force_pts_blade_.h_view(0));
    // Check epsilon
    EXPECT_DOUBLE_EQ(3.0, actMetaSimple.epsilon_.h_view(0, 0));
    EXPECT_DOUBLE_EQ(3.0, actMetaSimple.epsilon_.h_view(0, 1));
    EXPECT_DOUBLE_EQ(3.0, actMetaSimple.epsilon_.h_view(0, 2));
    // Check p1_ and p2_ values
    EXPECT_DOUBLE_EQ(-25.0, actMetaSimple.p1_.h_view(0, 0));
    EXPECT_DOUBLE_EQ(-4.0, actMetaSimple.p1_.h_view(0, 1));
    EXPECT_DOUBLE_EQ(0.0, actMetaSimple.p1_.h_view(0, 2));
    EXPECT_DOUBLE_EQ(-25.0, actMetaSimple.p2_.h_view(0, 0));
    EXPECT_DOUBLE_EQ(+4.0, actMetaSimple.p2_.h_view(0, 1));
    EXPECT_DOUBLE_EQ(0.0, actMetaSimple.p2_.h_view(0, 2));
    // Check p1ZeroAlphaDir_
    EXPECT_DOUBLE_EQ(1.0, actMetaSimple.p1ZeroAlphaDir_.h_view(0, 0));
    EXPECT_DOUBLE_EQ(0.0, actMetaSimple.p1ZeroAlphaDir_.h_view(0, 1));
    EXPECT_DOUBLE_EQ(0.0, actMetaSimple.p1ZeroAlphaDir_.h_view(0, 2));

    // Check the chord/twist tables (DV)
    for (int i = 0; i < actMetaSimple.num_force_pts_blade_.h_view(0); i++) {
      EXPECT_DOUBLE_EQ(1.0, actMetaSimple.chord_tableDv_.h_view(0, i));
      EXPECT_DOUBLE_EQ(0.0, actMetaSimple.twistTableDv_.h_view(0, i));
    }
    // Check the polar tables
    for (int i = 0; i < 3; i++) {
      EXPECT_DOUBLE_EQ(0.0, actMetaSimple.cdPolarTableDv_.h_view(0, i));
    }
    EXPECT_DOUBLE_EQ(-10.0, actMetaSimple.clPolarTableDv_.h_view(0, 0));
    EXPECT_DOUBLE_EQ(0.0, actMetaSimple.clPolarTableDv_.h_view(0, 1));
    EXPECT_DOUBLE_EQ(10.0, actMetaSimple.clPolarTableDv_.h_view(0, 2));
    EXPECT_DOUBLE_EQ(-180.0, actMetaSimple.aoaPolarTableDv_.h_view(0, 0));
    EXPECT_DOUBLE_EQ(0.0, actMetaSimple.aoaPolarTableDv_.h_view(0, 1));
    EXPECT_DOUBLE_EQ(180.0, actMetaSimple.aoaPolarTableDv_.h_view(0, 2));

  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
  test_wo_lines(inputFileLines_, actMeta);
}

TEST_F(ActuatorParsingSimpleTests, NGP_oneValueEpsilonParses)
{
  ActuatorMeta actMeta(1, ActuatorTypeMap["ActLineSimpleNGP"]);
  inputFileLines_[4] = "    epsilon: 3.0\n";
  try {
    auto y_node = create_yaml_node(inputFileLines_);
    auto actMetaSimple = actuator_Simple_parse(y_node, actMeta);
    SUCCEED();
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
  test_wo_lines(inputFileLines_, actMeta);
}

} // namespace
} // namespace nalu
} // namespace sierra
