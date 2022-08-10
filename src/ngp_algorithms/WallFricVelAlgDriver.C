// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/WallFricVelAlgDriver.h"
#include "Realm.h"
#include "wind_energy/BdyLayerStatistics.h"

#include "stk_util/parallel/ParallelReduce.hpp"

namespace sierra {
namespace nalu {

WallFricVelAlgDriver::WallFricVelAlgDriver(Realm& realm) : NgpAlgDriver(realm)
{
}

void
WallFricVelAlgDriver::pre_work()
{
  // Reset the accumulator
  utauAreaSum_[0] = 0.0;
  utauAreaSum_[1] = 0.0;
}

void
WallFricVelAlgDriver::post_work()
{
  // Post actions only need to be performed if the ABL statistics is active
  if ((realm_.bdyLayerStats_ == nullptr) || (!realm_.isFinalOuterIter_))
    return;

  double utauSumLocal[2] = {0.0, 0.0};
  double utauSumGlobal[2];

  for (int i = 0; i < simdLen; ++i) {
    utauSumLocal[0] += stk::simd::get_data(utauAreaSum_[0], i);
    utauSumLocal[1] += stk::simd::get_data(utauAreaSum_[1], i);
  }

  stk::all_reduce_sum(
    NaluEnv::self().parallel_comm(), utauSumLocal, utauSumGlobal, 2);

  double utau_average = utauSumGlobal[0] / utauSumGlobal[1];
  realm_.bdyLayerStats_->set_utau_avg(utau_average);
}

} // namespace nalu
} // namespace sierra
