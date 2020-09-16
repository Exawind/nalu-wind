// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LocalArray.h"
#include "matrix_free/KokkosFramework.h"

#include "StkSimdComparisons.h"
#include <gtest/gtest.h>

namespace sierra {
namespace nalu {
namespace matrix_free {

TEST(local_array, fill_double_1)
{
  LocalArray<double[4]> y = {{1, 1, 1, 1}};
  for (int i = 0; i < 4; ++i) {
    ASSERT_DOUBLE_EQ(y(i), 1.0);
    ASSERT_DOUBLE_EQ(y(i), y[i]);
  }
}

TEST(local_array, fill_ftypedouble_1)
{
  LocalArray<ftype[4]> y = {{1, 1, 1, 1}};
  for (int i = 0; i < 4; ++i) {
    ASSERT_DOUBLETYPE_NEAR(y(i), 1.0, 1.0e-16);
    ASSERT_DOUBLETYPE_NEAR(y(i), y[i], 1.0e-16);
  }
}

TEST(local_array, fill_double_2)
{
  LocalArray<double[3][3]> y = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      ASSERT_DOUBLE_EQ(y(j, i), (i == j) ? 1.0 : 0.0);
    }
  }
}

TEST(local_array, fill_double_3)
{
  LocalArray<double[3][3][3]> y;
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        y(k, j, i) = 2.0;
        ASSERT_DOUBLE_EQ(y(k, j, i), 2.0);
      }
    }
  }
}

TEST(local_array, fill_double_4)
{
  LocalArray<double[3][3][4][3]> y;
  for (int l = 0; l < 3; ++l) {
    for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 3; ++i) {
          y(l, k, j, i) = 2.0;
          ASSERT_DOUBLE_EQ(y(l, k, j, i), 2.0);
        }
      }
    }
  }
}

TEST(local_array, fill_double_5)
{
  LocalArray<double[7][3][3][4][3]> y;
  for (int m = 0; m < 7; ++m) {
    for (int l = 0; l < 3; ++l) {
      for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 3; ++i) {
            y(m, l, k, j, i) = 6.0;
            ASSERT_DOUBLE_EQ(y(m, l, k, j, i), 6.0);
          }
        }
      }
    }
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
