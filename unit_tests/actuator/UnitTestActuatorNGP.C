// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/UnitTestActuatorNGP.h>
#include <NaluEnv.h>
#include <gtest/gtest.h>

namespace sierra {
namespace nalu {

namespace {
TEST(ActuatorNGP, testExecuteOnHostOnly)
{
  ActuatorMeta actMeta(1);
  ActuatorInfoNGP infoTurb0;
  infoTurb0.turbineName_ = "Turbine0";
  infoTurb0.numPoints_ = 20;
  actMeta.add_turbine(infoTurb0);
  TestActuatorHostOnly actuator(actMeta);
  ASSERT_NO_THROW(actuator.execute());
  const ActuatorBulk& actBulk = actuator.actuator_bulk();
  EXPECT_DOUBLE_EQ(3.0, actBulk.epsilon_.h_view(1, 0));
  EXPECT_DOUBLE_EQ(6.0, actBulk.epsilon_.h_view(1, 1));
  EXPECT_DOUBLE_EQ(9.0, actBulk.epsilon_.h_view(1, 2));

  EXPECT_DOUBLE_EQ(1.0, actBulk.pointCentroid_.h_view(1, 0));
  EXPECT_DOUBLE_EQ(0.5, actBulk.pointCentroid_.h_view(1, 1));
  EXPECT_DOUBLE_EQ(0.25, actBulk.pointCentroid_.h_view(1, 2));

  EXPECT_DOUBLE_EQ(2.5, actBulk.velocity_.h_view(1, 0));
  EXPECT_DOUBLE_EQ(5.0, actBulk.velocity_.h_view(1, 1));
  EXPECT_DOUBLE_EQ(7.5, actBulk.velocity_.h_view(1, 2));

  EXPECT_DOUBLE_EQ(3.1, actBulk.actuatorForce_.h_view(1, 0));
  EXPECT_DOUBLE_EQ(6.2, actBulk.actuatorForce_.h_view(1, 1));
  EXPECT_DOUBLE_EQ(9.3, actBulk.actuatorForce_.h_view(1, 2));
}

TEST(ActuatorNGP, testExecuteOnHostAndDevice)
{
  ActuatorMeta actMeta(1);
  ActuatorInfoNGP infoTurb0;
  infoTurb0.turbineName_ = "Turbine0";
  infoTurb0.numPoints_ = 20;
  actMeta.add_turbine(infoTurb0);
  TestActuatorHostDev actuator(actMeta);
  ASSERT_NO_THROW(actuator.execute());
  const ActuatorBulkMod& actBulk = actuator.actuator_bulk();
  const double expectVal =
    actBulk.velocity_.h_view(1, 1) * actBulk.pointCentroid_.h_view(1, 0);
  EXPECT_DOUBLE_EQ(expectVal, actBulk.scalar_.h_view(1));
}

} // namespace

} // namespace nalu
} // namespace sierra
