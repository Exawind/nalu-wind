// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include <gtest/gtest.h>
#include <aero/actuator/ActuatorBulk.h>
#include <aero/actuator/ActuatorInfo.h>

// to allocate need turbine info
// compute offsets need num procs
// add fields/fixed fields
// allocate
namespace sierra {
namespace nalu {

namespace {

TEST(ActuatorMeta, NGP_constructor)
{
  const int numTurbines = 2;
  ActuatorMeta fieldMeta(numTurbines);
  EXPECT_EQ(numTurbines, fieldMeta.numberOfActuators_);
  EXPECT_EQ(0, fieldMeta.numPointsTurbine_.h_view(0));
  EXPECT_EQ(0, fieldMeta.numPointsTurbine_.h_view(1));
}

TEST(ActuatorMeta, addTurbine)
{
  const int numTurbines = 1;
  ActuatorMeta fieldMeta(numTurbines);
  ActuatorInfoNGP dummyInfo;
  dummyInfo.numPoints_ = 1024;
  fieldMeta.add_turbine(dummyInfo);
  EXPECT_EQ(1024, fieldMeta.numPointsTurbine_.h_view(0));
}

TEST(ActuatorMeta, NGP_copyCtor)
{
  const int numTurbines = 2;
  ActuatorMeta fieldMeta(numTurbines);
  ActuatorInfoNGP actInfo1;
  actInfo1.numPoints_ = 30;
  actInfo1.turbineId_ = 0;
  ActuatorInfoNGP actInfo2;
  actInfo2.numPoints_ = 24;
  actInfo2.turbineId_ = 1;
  fieldMeta.add_turbine(actInfo1);
  fieldMeta.add_turbine(actInfo2);
  EXPECT_EQ(30, fieldMeta.numPointsTurbine_.h_view(0));
  EXPECT_EQ(24, fieldMeta.numPointsTurbine_.h_view(1));

  ActuatorMeta fieldMeta2(fieldMeta);
  EXPECT_EQ(30, fieldMeta.numPointsTurbine_.h_view(0));
  EXPECT_EQ(24, fieldMeta.numPointsTurbine_.h_view(1));
}

TEST(ActuatorBulk, NGP_constructor)
{
  const int numTurbines = 2;
  ActuatorMeta fieldMeta(numTurbines);
  ActuatorInfoNGP dummyInfo;
  dummyInfo.numPoints_ = 36;
  dummyInfo.turbineId_ = 0;
  ActuatorInfoNGP dummyInfo2;
  dummyInfo2.numPoints_ = 40;
  dummyInfo2.turbineId_ = 1;
  fieldMeta.add_turbine(dummyInfo);
  fieldMeta.add_turbine(dummyInfo2);
  ActuatorBulk actBulkData(fieldMeta);
  EXPECT_EQ(76, actBulkData.actuatorForce_.extent_int(0));
  EXPECT_EQ(0, actBulkData.turbIdOffset_.h_view(0));
  EXPECT_EQ(36, actBulkData.turbIdOffset_.h_view(1));
}

} // namespace
} // namespace nalu
} // namespace sierra
