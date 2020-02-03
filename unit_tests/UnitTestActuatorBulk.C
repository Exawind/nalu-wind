// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include <gtest/gtest.h>
#include <actuator/ActuatorBulk.h>
#include <actuator/ActuatorInfo.h>

// to allocate need turbine info
// compute offsets need num procs
// add fields/fixed fields
// allocate
namespace sierra{
namespace nalu{

namespace{

TEST(ActuatorMeta, constructor){
 const int numTurbines = 2;
 ActuatorMeta fieldMeta(numTurbines);
 EXPECT_EQ(numTurbines, fieldMeta.num_actuators());
 EXPECT_EQ(0, fieldMeta.num_points_turbine(0));
 EXPECT_EQ(0, fieldMeta.num_points_turbine(1));
}

TEST(ActuatorMeta, addTurbine){
  const int numTurbines = 1;
  ActuatorMeta fieldMeta(numTurbines);
  ActuatorInfoNGP dummyInfo;
  dummyInfo.numPoints_=1024;
  fieldMeta.add_turbine(0, dummyInfo);
  EXPECT_EQ(1024, fieldMeta.num_points_turbine(0));
}

TEST(ActuatorMeta, copyCtor){
  const int numTurbines = 2;
  ActuatorMeta fieldMeta(numTurbines);
  ActuatorInfoNGP actInfo1;
  actInfo1.numPoints_= 30;
  ActuatorInfoNGP actInfo2;
  actInfo2.numPoints_= 24;
  fieldMeta.add_turbine(0, actInfo1);
  fieldMeta.add_turbine(1, actInfo2);
  EXPECT_EQ(30, fieldMeta.num_points_turbine(0));
  EXPECT_EQ(24, fieldMeta.num_points_turbine(1));

  ActuatorMeta fieldMeta2(fieldMeta);
  EXPECT_EQ(30, fieldMeta2.num_points_turbine(0));
  EXPECT_EQ(24, fieldMeta2.num_points_turbine(1));

}

TEST(ActuatorBulk, constructor){
  const int numTurbines = 1;
  ActuatorMeta fieldMeta(numTurbines);
  ActuatorInfoNGP dummyInfo;
  dummyInfo.numPoints_=36;
  fieldMeta.add_turbine(0, dummyInfo);
  ActuatorBulk bulkData(fieldMeta);
  EXPECT_EQ(36, bulkData.totalNumPoints_);
}

}
}
}
