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
#include <gtest/gtest.h>

namespace sierra
{
namespace nalu
{

namespace{

TEST(ActuatorBulkDiskFAST, construction){
  ActuatorMeta actMeta(1);
  auto y_node = actuator_unit::create_yaml_node(actuator_unit::nrel5MWinputs);
  auto actMetaFast = actuator_FAST_parse(y_node, actMeta);
  ActuatorBulkDiskFAST actBulk(actMetaFast, 0.0625);
  EXPECT_EQ(30, actBulk.epsilon_.extent_int(0));
}

}

} /* namespace nalu */
} /* namespace sierra */
