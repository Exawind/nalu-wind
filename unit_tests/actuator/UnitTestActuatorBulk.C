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
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>

// to allocate need turbine info
// compute offsets need num procs
// add fields/fixed fields
// allocate
namespace sierra {
namespace nalu {

namespace {

TEST(ActuatorMeta, constructor)
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

TEST(ActuatorMeta, copyCtor)
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

TEST(ActuatorBulk, constructor)
{
  stk::mesh::MetaData stkMeta(3);
  stk::mesh::BulkData stkBulk(stkMeta, MPI_COMM_WORLD);
  const int numTurbines = 1;
  ActuatorMeta fieldMeta(numTurbines);
  ActuatorInfoNGP dummyInfo;
  dummyInfo.numPoints_ = 36;
  dummyInfo.turbineId_ = 0;
  fieldMeta.add_turbine(dummyInfo);
  ActuatorBulk actBulkData(fieldMeta, stkBulk);
  EXPECT_EQ(36, actBulkData.totalNumPoints_);
}

} // namespace
} // namespace nalu
} // namespace sierra
