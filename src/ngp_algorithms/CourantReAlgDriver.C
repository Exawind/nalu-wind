// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/CourantReAlgDriver.h"
#include "Realm.h"
#include "SimdInterface.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"


namespace sierra {
namespace nalu {

CourantReAlgDriver::CourantReAlgDriver(
  Realm& realm
) : NgpAlgDriver(realm)
{}

void CourantReAlgDriver::update_max_cfl_rey(const double cfl, const double rey)
{
  maxCFL_ = stk::math::max(maxCFL_, cfl);
  maxRe_ = stk::math::max(maxRe_, rey);
}

void CourantReAlgDriver::pre_work()
{
  maxCFL_ = -1.0e6;
  maxRe_ = -1.0e6;
}

void CourantReAlgDriver::post_work()
{
  double local[2] = {maxCFL_, maxRe_};
  double global[2] = {-1.0e6, -1.0e6};

  stk::all_reduce_max(realm_.bulk_data().parallel(), local, global, 2);

  realm_.maxCourant_ = global[0];
  realm_.maxReynolds_ = global[1];
}

}  // nalu
}  // sierra
