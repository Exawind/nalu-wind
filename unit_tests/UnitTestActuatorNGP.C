// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include<gtest/gtest.h>
#include<UnitTestActuatorNGP.h>

namespace sierra{
namespace nalu{

namespace{
TEST(ActuatorNGP, testExecuteOnHostOnly){
  ActuatorMeta meta(1);
  ActuatorInfoNGP infoTurb0;
  infoTurb0.turbineName_ = "Turbine0";
  infoTurb0.numPoints_ = 20;
  meta.add_turbine(infoTurb0);
  TestActuatorHostOnly actuator(meta);
  ASSERT_NO_THROW(actuator.execute());
  const ActuatorBulk& bulk = actuator.actuator_bulk();
  EXPECT_DOUBLE_EQ(3.0, bulk.epsilon_.h_view(1,0));
  EXPECT_DOUBLE_EQ(6.0, bulk.epsilon_.h_view(1,1));
  EXPECT_DOUBLE_EQ(9.0, bulk.epsilon_.h_view(1,2));

  EXPECT_DOUBLE_EQ(1.0, bulk.pointCentroid_.h_view(1,0));
  EXPECT_DOUBLE_EQ(0.5, bulk.pointCentroid_.h_view(1,1));
  EXPECT_DOUBLE_EQ(0.25, bulk.pointCentroid_.h_view(1,2));

  EXPECT_DOUBLE_EQ(2.5, bulk.velocity_.h_view(1,0));
  EXPECT_DOUBLE_EQ(5.0, bulk.velocity_.h_view(1,1));
  EXPECT_DOUBLE_EQ(7.5, bulk.velocity_.h_view(1,2));

  EXPECT_DOUBLE_EQ(3.1, bulk.actuatorForce_.h_view(1,0));
  EXPECT_DOUBLE_EQ(6.2, bulk.actuatorForce_.h_view(1,1));
  EXPECT_DOUBLE_EQ(9.3, bulk.actuatorForce_.h_view(1,2));
}

TEST(ActuatorNGP, testExecuteOnHostAndDevice){
  ActuatorMeta meta(1);
  ActuatorInfoNGP infoTurb0;
  infoTurb0.turbineName_ = "Turbine0";
  infoTurb0.numPoints_ = 20;
  meta.add_turbine(infoTurb0);
  TestActuatorHostDev actuator(meta);
  ASSERT_NO_THROW(actuator.execute());
  const ActuatorBulkMod& bulk = actuator.actuator_bulk();
  const double expectVal = bulk.velocity_.h_view(1,1)*bulk.pointCentroid_.h_view(1,0);
  EXPECT_DOUBLE_EQ(expectVal, bulk.scalar_.h_view(1));
}

}

} //namespace nalu
} //namespace sierra
