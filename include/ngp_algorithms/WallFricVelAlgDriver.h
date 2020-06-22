// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef WALLFRICVELALGDRIVER_H
#define WALLFRICVELALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class Realm;

/** Wall friction velocity computation driver
 *
 *  Orchestrate the computation of wall friction velocity on a wall boundary
 *  with multiple element topologies. For ABL wall function, this driver
 *  provides an additional functionality to compute an area-weighted average of
 *  the friction velocity (utau) that is used in the BdyLayerStatistics class.
 *
 *  \sa ABLWallFrictionVelAlg, BdyLayerStatistics
 */
class WallFricVelAlgDriver : public NgpAlgDriver
{
public:
  WallFricVelAlgDriver(Realm&);

  virtual ~WallFricVelAlgDriver() = default;

  //! Reset utau accumulators before topology specialized algorithms are executed
  virtual void pre_work() override;

  //! Perform global integration of utau and update ABL statistics instance
  virtual void post_work() override;

  /** Accumulate partial sum from topology-specific element algorithms
   *
   *  @param utau_area_sum Partial sum of (utau * area) over the integration points
   *  @param area_sum Partially integrated area for the integration points
   */
  inline void accumulate_utau_area_sum(
    const DoubleType& utau_area_sum,
    const DoubleType& area_sum)
  {
    utauAreaSum_[0] += utau_area_sum;
    utauAreaSum_[1] += area_sum;
  }

private:
  /** Accumulator for area-weighted friction-velocity average
   *
   *  The first entry stores the (utau * area) sum
   *  The second entry stores wall area sum
   *
   *  utau_average = (utau * area) / total_area;
   */
  DoubleType utauAreaSum_[2];

};

}  // nalu
}  // sierra

#endif /* WALLFRICVELALGDRIVER_H */
