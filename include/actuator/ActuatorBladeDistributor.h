// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORBLADEDISTRIBUTOR_H_
#define ACTUATORBLADEDISTRIBUTOR_H_

#include <vector>
#include <utility>

namespace sierra {
namespace nalu {

struct ActuatorBulk;
struct ActuatorMeta;
/**
 * @brief Data structure for caching blade specific info
 *
 */
struct BladeDistributionInfo
{
  int offset_;
  int nPoints_;
  int nNeighbors_;
};

/**
 * @brief Compute the maximum parallelization of blades to loop over
 *
 * @param actMeta
 * @param actBulk
 * @return std::vector<BladeDistributionInfo> - first item is the offset where
 the blade can be found and the second is the number of points along the blade
 */
std::vector<BladeDistributionInfo>
compute_blade_distributions(const ActuatorMeta& actMeta, ActuatorBulk& actBulk);
/**
 * @brief determine if a blade's lifting line correction should be computed on
 * the reference processor
 *
 * @param totalNumBlades  the total number of actuator blades in the simulation
 * @param globBladeNum the index of the current blade being evaluated
 * @param numRanks the rank being evaluated with the globBladeNum to see if they
 * match
 * @param ranks the total number of ranks in the simulation
 * @return true
 * @return false
 */
bool blade_belongs_on_this_rank(
  int totalNumBlades, int globBladeNum, int numRanks, int ranks);
} // namespace nalu
} // namespace sierra

#endif /* ACTUATORBLADEDISTRIBUTOR_H_ */
