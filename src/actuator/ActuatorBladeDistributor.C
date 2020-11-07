// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorBladeDistributor.h>
#include <actuator/ActuatorBulkSimple.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

std::vector<std::pair<int, int>>
compute_blade_distributions(const ActuatorMeta& actMeta, ActuatorBulk& actBulk)
{
  std::vector<std::pair<int, int>> results;
  const int rank = NaluEnv::self().parallel_rank();

  switch (actMeta.actuatorType_) {
  case (ActuatorType::ActLineSimpleNGP): {
    auto actMetaSimp = dynamic_cast<const ActuatorMetaSimple&>(actMeta);
    // one blade per processor for this case, but we could change this
    if (rank == actBulk.localTurbineId_) {
      const int iBlade = actBulk.localTurbineId_;
      const int offset = actBulk.turbIdOffset_.h_view(iBlade);
      const int nPoints = actMetaSimp.num_force_pts_blade_.h_view(iBlade);
      results.push_back(std::make_pair(offset, nPoints));
    }
    break;
  }
  default:
    throw std::runtime_error(
      "compute_blade_distribution::invalid actuator type hit");
  }

  return results;
}

} // namespace nalu
} // namespace sierra