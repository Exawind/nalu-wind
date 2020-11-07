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

namespace sierra{
namespace nalu{

struct ActuatorBulk;
struct ActuatorMeta;

/**
 * @brief Compute the maximum parallelization of blades to loop over
 * 
 * @param actMeta 
 * @param actBulk 
 * @return std::vector<std::pair<int,int>> - first item is the offset where the blade can be found and the second
  is the number of points along the blade
 */
std::vector<std::pair<int,int>> compute_blade_distributions(const ActuatorMeta& actMeta, ActuatorBulk& actBulk);

}
}

#endif /* ACTUATORBLADEDISTRIBUTOR_H_ */
