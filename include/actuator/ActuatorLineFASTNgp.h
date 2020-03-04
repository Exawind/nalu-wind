// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORLINEFASTNGP_H_
#define ACTUATORLINEFASTNGP_H_

#include <actuator/ActuatorNGP.h>
#include <actuator/ActuatorBulkFAST.h>
#include <actuator/ActuatorFunctorsFAST.h>

namespace sierra
{
namespace nalu
{
using ActuatorNgpFAST = ActuatorNGP<ActuatorMetaFAST, ActuatorBulkFAST>;

template <>
void
ActuatorNgpFAST::execute()
{
  auto velReduce   = actBulk_.velocity_.template      view<ActuatorFixedMemSpace>();
  auto forceReduce = actBulk_.actuatorForce_.template view<ActuatorFixedMemSpace>();

  Kokkos::parallel_for("zeroQuantitiesActuatorNgpFAST", numActPoints_, ActFastZero(actBulk_));

  // set range policy to only operating over points owned by local fast turbine
  auto fastRangePolicy = actBulk_.local_range_policy(actMeta_);
  Kokkos::parallel_for("updatePointLocationsActuatorNgpFAST", fastRangePolicy, ActFastUpdatePoints(actBulk_));

  actBulk_.stk_search_act_pnts(actMeta_);
  const int localSizeCoarseSearch = actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

  Kokkos::parallel_for("interpolateVelocitiesActuatorNgpFAST", numActPoints_, InterpolateActVel(actBulk_));

  actBulk_.reduce_view_on_host(velReduce, NaluEnv::self().parallel_comm());

  Kokkos::parallel_for("assignFastVelActuatorNgpFAST", fastRangePolicy, ActFastAssignVel(actBulk_));

  actBulk_.step_fast();

  Kokkos::parallel_for("computeForcesActuatorNgpFAST", fastRangePolicy, ActFastComputeForce(actBulk_));

  actBulk_.reduce_view_on_host(forceReduce, NaluEnv::self().parallel_comm());

  Kokkos::parallel_for("spreadForcesActuatorNgpFAST", localSizeCoarseSearch, SpreadActForce(actBulk_));
  // TODO(psakiev) compute thrust
}

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORLINEFASTNGP_H_ */
