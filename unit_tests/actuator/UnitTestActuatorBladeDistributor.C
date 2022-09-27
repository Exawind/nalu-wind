// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorBladeDistributor.h>
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

std::ostream&
operator<<(std::ostream& os, const TestInputs ti)
{
  return os << "nTotalBlades: " << ti.nTotalBlades_
            << " globBladeNum: " << ti.globBladeNum_
            << " numRanks: " << ti.numRanks_ << " rank: " << ti.rank_
            << " answer: " << ti.answer_;
}

INSTANTIATE_TEST_SUITE_P(
  BladeTestCases,
  BladeDistributorTest,
  testing::Values(
    TestInputs({3, 0, 4, 0, true}),
    TestInputs({3, 1, 4, 0, false}),
    TestInputs({3, 2, 4, 0, false}),
    TestInputs({3, 0, 4, 1, false}),
    TestInputs({3, 1, 4, 1, true}),
    TestInputs({3, 2, 4, 1, false}),
    TestInputs({3, 0, 4, 2, false}),
    TestInputs({3, 1, 4, 2, false}),
    TestInputs({3, 2, 4, 2, true}),
    TestInputs({3, 0, 4, 3, false}),
    TestInputs({3, 1, 4, 3, false}),
    TestInputs({3, 2, 4, 3, false}),
    TestInputs({6, 0, 4, 3, false}),
    TestInputs({9, 1, 4, 0, true}),
    TestInputs({3, 2, 36, 2, true})));

TEST_P(BladeDistributorTest, checkSpecificValues)
{
  const bool result = blade_belongs_on_this_rank(
    GetParam().nTotalBlades_, GetParam().globBladeNum_, GetParam().numRanks_,
    GetParam().rank_);
  EXPECT_EQ(GetParam().answer_, result);
}

TEST_P(BladeDistributorTest, allBladesAreUsedOnlyOnce)
{
  std::vector<int> counter(GetParam().nTotalBlades_);
  for (int r = 0; r < GetParam().numRanks_; r++) {
    for (int b = 0; b < GetParam().nTotalBlades_; b++) {
      if (blade_belongs_on_this_rank(
            GetParam().nTotalBlades_, b, GetParam().numRanks_, r)) {
        counter[b]++;
      }
    }
  }
  for (int i = 0; i < GetParam().nTotalBlades_; i++) {
    EXPECT_EQ(1, counter[i]) << "Failed for index: " << i;
  }
}

} // namespace
} // namespace nalu
} // namespace sierra
