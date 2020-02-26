// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorFunctorsFAST.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

template <>
ActFastUpdatePoints::ActuatorFunctor(ActuatorBulkFAST& actBulk)
  : actBulk_(actBulk)
{
  TOUCH_DUAL_VIEW(actBulk_.pointCentroid_, memory_space);
}

template <>
void
ActFastUpdatePoints::operator()(const int& index) const
{
  fast::OpenFAST& FAST = actBulk_.openFast_;
  auto points = GET_LOCAL_VIEW(actBulk_.pointCentroid_, memory_space);
  auto offsets = GET_LOCAL_VIEW(actBulk_.turbIdOffset_, memory_space);

  // if local fast owns point
  int turbId = 0;
  const int nTurbs = offsets.extent_int(0);
  for (int i = 0; i < nTurbs; i++) {
    if (offsets(i) > index) {
      turbId = i - 1;
      break;
    }
  }

  int owningRank = FAST.get_procNo(turbId);

  if (owningRank == NaluEnv::self().parallel_rank()) {
    // compute location
    std::vector<double> tempCoords(3, 0.0);
    FAST.getForceNodeCoordinates(tempCoords, owningRank, turbId);
    for (int i = 0; i < 3; i++) {
      points(index, i) = tempCoords[i];
    }
  } else {
    return;
  }
}

} /* namespace nalu */
} /* namespace sierra */
