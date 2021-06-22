// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "OutputInfo.h"
#include "gtest/gtest.h"

#include "NaluParsing.h"

namespace sierra {
namespace nalu {
namespace {
TEST(OutputManagerTest, can_read_multiple_output_nodes)
{
  const char* parseText =
    R"par(  output:
    output_data_base_name: test1
  output:
    output_data_base_name: test2
  )par";
  const YAML::Node y_node = YAML::Load(parseText);
  OutputInfo oInfo;
  ASSERT_NO_THROW(oInfo.load(y_node));
}
} // namespace

} // namespace nalu
} // namespace sierra
