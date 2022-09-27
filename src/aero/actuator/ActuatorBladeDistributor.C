// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorBladeDistributor.h>
#include <aero/actuator/ActuatorBulkSimple.h>
#include <NaluEnv.h>
#ifdef NALU_USES_OPENFAST
#include <aero/actuator/ActuatorBulkFAST.h>
#include <aero/actuator/UtilitiesActuator.h>
#endif

namespace sierra {
namespace nalu {

bool
blade_belongs_on_this_rank(
  int numBladesTotal, int globBladeNum, int numRanks, int rank)
{
  const int div = numBladesTotal / numRanks;
  const bool isInDivisionIncrement =
    globBladeNum >= (div * rank) && globBladeNum < div * (rank + 1);
  const bool isInRemainderIncrement = (globBladeNum - div * numRanks) == rank;
  ThrowAssert(!(isInDivisionIncrement && isInRemainderIncrement));
  return isInDivisionIncrement || isInRemainderIncrement;
}

std::vector<BladeDistributionInfo>
compute_blade_distributions(const ActuatorMeta& actMeta, ActuatorBulk& actBulk)
{
  std::vector<BladeDistributionInfo> results;
  const int rank = NaluEnv::self().parallel_rank();

  switch (actMeta.actuatorType_) {
  case (ActuatorType::ActLineSimpleNGP): {
    auto actMetaSimp = dynamic_cast<const ActuatorMetaSimple&>(actMeta);
    // one blade per processor for this case, but we could change this
    if (!actMeta.entityFLLC_(actBulk.localTurbineId_))
      break;
    if (rank == actBulk.localTurbineId_) {
      const int iBlade = actBulk.localTurbineId_;
      const int offset = actBulk.turbIdOffset_.h_view(iBlade);
      const int nPoints = actMetaSimp.num_force_pts_blade_.h_view(iBlade);
      const int nNeighbor = actMetaSimp.numNearestPointsFllcInt_.h_view(iBlade);
      results.push_back({offset, nPoints, nNeighbor});
    }
    break;
  }
  case (ActuatorType::ActDiskFASTNGP):
  case (ActuatorType::ActLineFASTNGP): {
#ifdef NALU_USES_OPENFAST
    const int numRanks = NaluEnv::self().parallel_size();
    auto actMetaFast = dynamic_cast<const ActuatorMetaFAST&>(actMeta);
    int numBladesTotal = 0;
    // compute the total number of blades
    for (int iTurb = 0; iTurb < actMeta.numberOfActuators_; ++iTurb) {
      // skip this entity if fllc isn't active
      if (!actMeta.entityFLLC_(iTurb))
        continue;
      numBladesTotal += actMetaFast.nBlades_(iTurb);
    }

    // loop through and assign them to the processors
    for (int iTurb = 0, globBladeNum = 0; iTurb < actMeta.numberOfActuators_;
         ++iTurb) {

      // skip this turbine if fllc isn't active
      if (!actMeta.entityFLLC_(iTurb))
        continue;

      const int turbOffset = actBulk.turbIdOffset_.h_view(iTurb);
      const int nBlades = actMetaFast.nBlades_(iTurb);
      const int nNeighbors = actMeta.numNearestPointsFllcInt_.h_view(iTurb);

      for (int iBlade = 0; iBlade < nBlades; ++iBlade) {
        const int bladeStart = actuator_utils::get_fast_point_index(
          actMetaFast.fastInputs_, iTurb, nBlades,
          fast::ActuatorNodeType::BLADE, 0, iBlade);

        const int offset = turbOffset + bladeStart;

        const int nPoints =
          actMetaFast.fastInputs_.globTurbineData[iTurb].numForcePtsBlade;

        if (blade_belongs_on_this_rank(
              numBladesTotal, globBladeNum, numRanks, rank)) {
          results.push_back({offset, nPoints, nNeighbors});
        }

        globBladeNum++;
      }
    }
    break;
#else
    throw std::runtime_error(
      "The code was not compiled with OpenFAST support.  Please recompile to "
      "use a FAST variant of the actuator models.");
#ifndef KOKKOS_ENABLE_CUDA
    break;
#endif
#endif
  }
  default:
    // should never hit this execpt through developer mistake
    throw std::runtime_error(
      "compute_blade_distribution::invalid actuator type hit");
  }

  return results;
}

} // namespace nalu
} // namespace sierra
