// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <gtest/gtest.h>
#include <aero/aero_utils/displacements.h>

namespace test_displacements {
TEST(AeroDisplacements, NGP_creation_from_std_vector)
{
  std::vector<double> openfastSurrogate(6, 1.0);
  aero::Displacement disp(openfastSurrogate);
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(openfastSurrogate[i], disp.translation_[i]);
    EXPECT_DOUBLE_EQ(openfastSurrogate[i + 3], disp.rotation_[i]);
  }
}

TEST(AeroDisplacements, NGP_creation_from_vs_vector)
{
  aero::Displacement disp(vs::Vector::one(), 2.0 * vs::Vector::one());
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(1.0, disp.translation_[i]);
    EXPECT_DOUBLE_EQ(2.0, disp.rotation_[i]);
  }
}
} // namespace test_displacements
