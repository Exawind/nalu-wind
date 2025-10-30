// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ArrayND.h"
#include "master_element/CompileTimeElements.h"
#include "matrix_free/KokkosFramework.h"

#include "StkSimdComparisons.h"
#include <gtest/gtest.h>

namespace sierra::nalu {

TEST(array_nd, fill_double_1)
{
  ArrayND<double[4]> y = {{1, 1, 1, 1}};
  for (int i = 0; i < 4; ++i) {
    ASSERT_DOUBLE_EQ(y(i), 1.0);
    ASSERT_DOUBLE_EQ(y(i), y[i]);
  }
}

TEST(array_nd, fill_ftypedouble_1)
{
  ArrayND<DoubleType[4]> y = {{1, 1, 1, 1}};
  for (int i = 0; i < 4; ++i) {
    ASSERT_DOUBLETYPE_NEAR(y(i), 1.0, 1.0e-16);
    ASSERT_DOUBLETYPE_NEAR(y(i), y[i], 1.0e-16);
  }
}

TEST(array_nd, fill_double_2)
{
  ArrayND<double[3][3]> y = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      ASSERT_DOUBLE_EQ(y(j, i), (i == j) ? 1.0 : 0.0);
    }
  }
}

TEST(array_nd, fill_double_3)
{
  ArrayND<double[3][3][3]> y;
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        y(k, j, i) = 2.0;
        ASSERT_DOUBLE_EQ(y(k, j, i), 2.0);
      }
    }
  }
}

TEST(array_nd, fill_double_4)
{
  ArrayND<double[3][3][4][3]> y;
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

TEST(array_nd, fill_double_5)
{
  ArrayND<double[7][3][3][4][3]> y;
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

TEST(array_nd, static_rank)
{
  {
    ArrayND<double[2]> s;
    static_assert(decltype(s)::rank == 1);
    static_assert(s.extent_int(0) == 2);
  }

  {
    ArrayND<double[2][3]> s;
    static_assert(decltype(s)::rank == 2);
    static_assert(s.extent_int(0) == 2);
    static_assert(s.extent_int(1) == 3);
  }

  {
    ArrayND<double[2][3][4]> s;
    static_assert(decltype(s)::rank == 3);
    static_assert(s.extent_int(0) == 2);
    static_assert(s.extent_int(1) == 3);
    static_assert(s.extent_int(2) == 4);
  }

  {
    ArrayND<double[2][3][4][5]> s;
    static_assert(decltype(s)::rank == 4);
    static_assert(s.extent_int(0) == 2);
    static_assert(s.extent_int(1) == 3);
    static_assert(s.extent_int(2) == 4);
    static_assert(s.extent_int(3) == 5);
  }

  {
    ArrayND<double[2][3][4][5][6]> s;
    static_assert(decltype(s)::rank == 5);
    static_assert(s.extent_int(0) == 2);
    static_assert(s.extent_int(1) == 3);
    static_assert(s.extent_int(2) == 4);
    static_assert(s.extent_int(3) == 5);
    static_assert(s.extent_int(4) == 6);
  }
}

} // namespace sierra::nalu
