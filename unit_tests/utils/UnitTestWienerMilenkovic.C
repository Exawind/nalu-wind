// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <gtest/gtest.h>
#include <utils/WienerMilenkovic.h>

TEST(WienerMilenkovic, rotationMatrixIsEquivalent)
{
  vs::Vector v1 = vs::Vector::ihat();
  vs::Vector v2 = vs::Vector::khat();
  auto v3 = v1 ^ v2;
}
