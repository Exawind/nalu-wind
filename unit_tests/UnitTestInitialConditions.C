// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include <yaml-cpp/yaml.h>
#include "InitialConditions.h"
#include "NaluParsing.h"

namespace sierra {
namespace nalu {

TEST(InitialCondition, createICs)
{
  // constant IC's
  {
    const char* ic_input = R"INPUT(
    initial_conditions:
      - constant: ic_1
        target_name: [fluid_part]
        value:
          pressure: 0.0
          velocity: [7.250462296293199, 3.380946093925596, 0.0])INPUT";

    const YAML::Node node = YAML::Load(ic_input);

    auto vec = InitialConditionCreator(false).create_ic_vector(node);

    ASSERT_EQ(1, vec.size());
    auto* type_ptr = dynamic_cast<ConstantInitialConditionData*>(vec[0].get());
    ASSERT_TRUE(nullptr != type_ptr);
  }
  // user fcn IC's
  {
    const char* ic_input = R"INPUT(
    initial_conditions:

     - user_function: icUser
       target_name: block_1
       user_function_name:
         velocity: TaylorGreen
         pressure: TaylorGreen)INPUT";

    const YAML::Node node = YAML::Load(ic_input);

    auto vec = InitialConditionCreator(false).create_ic_vector(node);

    ASSERT_EQ(1, vec.size());
    auto* type_ptr =
      dynamic_cast<UserFunctionInitialConditionData*>(vec[0].get());
    ASSERT_TRUE(nullptr != type_ptr);
  }
}
} // namespace nalu
} // namespace sierra
