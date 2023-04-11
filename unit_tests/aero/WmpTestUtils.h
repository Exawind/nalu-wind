// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <gtest/gtest.h>
#include "vs/vector.h"
#include "vs/vstraits.h"
#include <aero/aero_utils/WienerMilenkovic.h>

inline testing::Message&
operator<<(testing::Message& out, const vs::Vector& vec)
{
  out << "(" << vec.x() << " " << vec.y() << " " << vec.z() << ")";
  return out;
}

//! Test that two WM Params give the same end location for a point
inline void
test_wiener_milenkovic(
  vs::Vector goldWmp, vs::Vector testWmp, vs::Vector testPoint, double eps)
{
  auto goldPnt = wmp::rotate(goldWmp, testPoint);
  auto testPnt = wmp::rotate(testWmp, testPoint);
  EXPECT_NEAR(goldPnt.x(), testPnt.x(), eps)
    << "Gold WMP: " << goldWmp << " testWmp: " << testWmp;
  EXPECT_NEAR(goldPnt.y(), testPnt.y(), eps)
    << "Gold WMP: " << goldWmp << " testWmp: " << testWmp;
  EXPECT_NEAR(goldPnt.z(), testPnt.z(), eps)
    << "Gold WMP: " << goldWmp << " testWmp: " << testWmp;
}
