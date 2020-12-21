// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorBladeDistributor.h>
#include <gtest/gtest.h>

namespace sierra {
namespace nalu {
namespace {

struct TestInputs
{
  int nTotalBlades_;
  int globBladeNum_;
  int numRanks_;
  int rank_;
  bool answer_;
};

class BladeDistributorTest : public testing::TestWithParam<TestInputs>
{
};

TEST_P(BladeDistributorTest, checkSpecificValues)
{
  const bool result = does_blade_belong_on_this_rank(
    GetParam().nTotalBlades_, GetParam().globBladeNum_, GetParam().numRanks_,
    GetParam().rank_);
  EXPECT_EQ(GetParam().answer_, result);
}

INSTANTIATE_TEST_SUITE_P(
  BladeTestCases,
  BladeDistributorTest,
  testing::Values(
    TestInputs({3, 0, 4, 0, true}), TestInputs({3, 0, 4, 3, false})));

} // namespace
} // namespace nalu
} // namespace sierra