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
#include <actuator/ActuatorParsing.h>

namespace sierra {
namespace nalu {

namespace {

/// Check the minimum parse terms are not violated and
/// that defaults haven't changed
//TODO(psakiev) add robust testing like FAST parsing function
TEST(ActuatorParse, bareMinimumParse)
{
  std::string testFile = "actuator:\n"
                         "  type: ActLineFAST\n"
                         "  n_turbines_glob: 1\n"
                         "  search_target_part: [part1, part2]";
  YAML::Node y_actuator = YAML::Load(testFile);
  try {
    ActuatorMeta actMeta = actuator_parse(y_actuator);
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }

  ActuatorMeta actMeta = actuator_parse(y_actuator);
  EXPECT_EQ(stk::search::KDTREE, actMeta.searchMethod_);
  EXPECT_EQ(2, actMeta.searchTargetNames_.size());
}

} // namespace

} // namespace nalu
} // namespace sierra
