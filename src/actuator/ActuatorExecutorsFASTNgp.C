// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <actuator/ActuatorExecutorsFASTNgp.h>

namespace sierra {
namespace nalu {

ActuatorLineFastNGP::ActuatorLineFastNGP(
  const ActuatorMetaFAST& actMeta,
  ActuatorBulkFAST& actBulk,
  stk::mesh::BulkData& stkBulk)
  : actMeta_(actMeta),
    actBulk_(actBulk),
    stkBulk_(stkBulk),
    numActPoints_(actMeta_.numPointsTotal_)
{
}

void
ActuatorLineFastNGP::operator()()
{
  auto forceReduce = actBulk_.actuatorForce_.view_host();

  actBulk_.zero_source_terms(stkBulk_);

  if (actBulk_.fast_is_time_zero()) {
    update();
  }
  actBulk_.interpolate_velocities_to_fast();

  update();

  actBulk_.step_fast();

  // set range policy to only operating over points owned by local fast turbine
  auto fastRangePolicy = actBulk_.local_range_policy();

  Kokkos::parallel_for(
    "computeForcesActuatorNgpFAST", fastRangePolicy,
    ActFastComputeForce(actBulk_));

  actuator_utils::reduce_view_on_host(forceReduce);

  const int localSizeCoarseSearch =
    actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

  // TODO(psakiev) apply re-orientation and distance projection
  // maybe compute distance at time of coarse search and cache to save a loop
  Kokkos::parallel_for(
    "spreadForcesActuatorNgpFAST", localSizeCoarseSearch,
    SpreadActuatorForce(actBulk_, stkBulk_));

  actBulk_.parallel_sum_source_term(stkBulk_);

  if (actBulk_.openFast_.isDebug()) {
    Kokkos::parallel_for(
      "setUpTorqueCalc", actMeta_.numberOfActuators_,
      ActFastSetUpThrustCalc(actBulk_));

    actuator_utils::reduce_view_on_host(actBulk_.hubLocations_);
    actuator_utils::reduce_view_on_host(actBulk_.hubOrientation_);

    Kokkos::parallel_for(
      "computeTorque", localSizeCoarseSearch,
      ActFastComputeThrust(actBulk_, stkBulk_));
    actuator_utils::reduce_view_on_host(actBulk_.turbineThrust_);
    actuator_utils::reduce_view_on_host(actBulk_.turbineTorque_);
    actBulk_.output_torque_info();
  }
}

void
ActuatorLineFastNGP::update()
{
  auto velReduce = actBulk_.velocity_.view_host();
  auto pointReduce = actBulk_.pointCentroid_.view_host();

  Kokkos::parallel_for(
    "zeroQuantitiesActuatorNgpFAST", numActPoints_, ActFastZero(actBulk_));

  // set range policy to only operating over points owned by local fast turbine
  auto fastRangePolicy = actBulk_.local_range_policy();

  Kokkos::parallel_for(
    "updatePointLocationsActuatorNgpFAST", fastRangePolicy,
    ActFastUpdatePoints(actBulk_));
  actuator_utils::reduce_view_on_host(pointReduce);

  actBulk_.stk_search_act_pnts(actMeta_, stkBulk_);

  Kokkos::parallel_for(
    "interpolateVelocitiesActuatorNgpFAST", numActPoints_,
    InterpActuatorVel(actBulk_, stkBulk_));

  actuator_utils::reduce_view_on_host(velReduce);

  Kokkos::parallel_for(
    "assignFastVelActuatorNgpFAST", fastRangePolicy,
    ActFastAssignVel(actBulk_));
}

ActuatorDiskFastNGP::ActuatorDiskFastNGP(
  const ActuatorMetaFAST& actMeta,
  ActuatorBulkDiskFAST& actBulk,
  stk::mesh::BulkData& stkBulk)
  : actMeta_(actMeta),
    actBulk_(actBulk),
    stkBulk_(stkBulk),
    numActPoints_(actMeta_.numPointsTotal_)
{
}

void
ActuatorDiskFastNGP::operator()()
{
  auto velReduce = actBulk_.velocity_.view_host();
  auto pointReduce = actBulk_.pointCentroid_.view_host();

  Kokkos::parallel_for(
    "zeroFieldsDiskNgp", numActPoints_, KOKKOS_LAMBDA(int index) {
      for (int i = 0; i < 3; ++i) {
        actBulk_.actuatorForce_.d_view(index, i) = 0.0;
        actBulk_.velocity_.d_view(index, i) = 0.0;
      }
    });

  if (!actBulk_.searchExecuted_) {
    actBulk_.stk_search_act_pnts(actMeta_, stkBulk_);
  }
  actBulk_.zero_source_terms(stkBulk_);

  Kokkos::parallel_for(
    "interpolateVelocitiesActuatorNgpFAST", numActPoints_,
    InterpActuatorVel(actBulk_, stkBulk_));

  actuator_utils::reduce_view_on_host(velReduce);

  auto fastRangePolicy = actBulk_.local_range_policy();

  Kokkos::parallel_for(
    "assignFastVelActuatorNgpFAST", fastRangePolicy,
    ActFastAssignVel(actBulk_));

  auto forceReduce = actBulk_.actuatorForce_.view_host();

  actBulk_.interpolate_velocities_to_fast();

  actBulk_.step_fast();

  Kokkos::parallel_for(
    "computeForcesActuatorNgpFAST", fastRangePolicy,
    ActFastComputeForce(actBulk_));

  actuator_utils::reduce_view_on_host(forceReduce);

  actBulk_.spread_forces_over_disk(actMeta_);

  const int localSizeCoarseSearch =
    actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

  Kokkos::parallel_for(
    "spreadForcesActuatorNgpFAST", localSizeCoarseSearch,
    SpreadActuatorForce(actBulk_, stkBulk_));

  actBulk_.parallel_sum_source_term(stkBulk_);

  if (actBulk_.openFast_.isDebug()) {
    Kokkos::parallel_for(
      "setUpTorqueCalc", actMeta_.numberOfActuators_,
      ActFastSetUpThrustCalc(actBulk_));

    actuator_utils::reduce_view_on_host(actBulk_.hubLocations_);
    actuator_utils::reduce_view_on_host(actBulk_.hubOrientation_);

    Kokkos::parallel_for(
      "computeTorque", localSizeCoarseSearch,
      ActFastComputeThrust(actBulk_, stkBulk_));
    actuator_utils::reduce_view_on_host(actBulk_.turbineThrust_);
    actuator_utils::reduce_view_on_host(actBulk_.turbineTorque_);
    actBulk_.output_torque_info();
  }
}

} // namespace nalu
} // namespace sierra
