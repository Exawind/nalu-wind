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
#ifdef NALU_USES_OPENFAST
#include <actuator/ActuatorBulkFAST.h>
#include <actuator/UtilitiesActuator.h>
#endif

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
  case (ActuatorType::ActDiskFASTNGP):
  case (ActuatorType::ActDiskFAST):
  case (ActuatorType::ActLineFAST):
  case (ActuatorType::ActLineFASTNGP): {
#ifdef NALU_USES_OPENFAST
    const int numRanks = NaluEnv::self().parallel_size();
    auto actMetaFast = dynamic_cast<const ActuatorMetaFAST&>(actMeta);
    int numBladesTotal = 0;
    // compute the total number of blades
    for (int iTurb = 0; iTurb < actMeta.numberOfActuators_; ++iTurb) {
      numBladesTotal += actMetaFast.nBlades_(iTurb);
    }

    const int div = numRanks / numBladesTotal;
    const int remainder = numRanks % numBladesTotal;

    // loop through and assign them to the processors

    for (int iTurb = 0, gBlade = 0; iTurb < actMeta.numberOfActuators_;
         ++iTurb) {
      const int turbOffset = actBulk.turbIdOffset_.h_view(iTurb);
      const int nBlades = actMetaFast.nBlades_(iTurb);

      for (int iBlade = 0; iBlade < nBlades; ++iBlade) {
        const int bladeStart = actuator_utils::get_fast_point_index(
          actMetaFast.fastInputs_, iTurb, nBlades,
          fast::ActuatorNodeType::BLADE, 0, iBlade);

        const int offset = turbOffset + bladeStart;
        const int nPoints =
          actMetaFast.fastInputs_.globTurbineData[iTurb].numForcePtsBlade;
        const bool isInDivisionIncrement =
          gBlade >= (div * rank) && gBlade < div * (rank + 1);
        const bool isInRemainderIncrement = rank % numRanks == gBlade;
        if (isInDivisionIncrement || isInRemainderIncrement) {
          results.push_back(std::make_pair(offset, nPoints));
        }
        gBlade++;
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