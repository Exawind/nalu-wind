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
  actBulk_.zero_source_terms(stkBulk_);

  // set range policy to only operating over points owned by local fast turbine
  auto fastRangePolicy = actBulk_.local_range_policy();

  RunInterpActuatorVel(actBulk_, stkBulk_);

  Kokkos::parallel_for(
    "assignFastVelActuatorNgpFAST", fastRangePolicy,
    ActFastAssignVel(actBulk_));

  actBulk_.interpolate_velocities_to_fast();

  RunActFastUpdatePoints(actBulk_);

  actBulk_.stk_search_act_pnts(actMeta_, stkBulk_);

  actBulk_.step_fast();

  RunActFastComputeForce(actBulk_);

  const int localSizeCoarseSearch =
    actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

  if (actMeta_.isotropicGaussian_) {
    Kokkos::parallel_for(
      "spreadForcesActuatorNgpFAST", localSizeCoarseSearch,
      SpreadActuatorForce(actBulk_, stkBulk_));
  }
  else {
    RunActFastStashOrientVecs(actBulk_);

    Kokkos::parallel_for(
      "spreadForceUsingProjDistance", localSizeCoarseSearch,
      ActFastSpreadForceWhProjection(actBulk_, stkBulk_));
  }

  actBulk_.parallel_sum_source_term(stkBulk_);

  if (actBulk_.openFast_.isDebug()) {
;
    actBulk_.output_torque_info(stkBulk_);
  }
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
  actBulk_.zero_source_terms(stkBulk_);

  RunInterpActuatorVel(actBulk_, stkBulk_);

  auto fastRangePolicy = actBulk_.local_range_policy();

  Kokkos::parallel_for(
    "assignFastVelActuatorNgpFAST", fastRangePolicy,
    ActFastAssignVel(actBulk_));

  auto forceReduce = actBulk_.actuatorForce_.view_host();

  actBulk_.interpolate_velocities_to_fast();

  actBulk_.step_fast();

  RunActFastComputeForce(actBulk_);

  actBulk_.spread_forces_over_disk(actMeta_);

  const int localSizeCoarseSearch =
    actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

  Kokkos::parallel_for(
    "spreadForcesActuatorNgpFAST", localSizeCoarseSearch,
    SpreadActuatorForce(actBulk_, stkBulk_));

  actBulk_.parallel_sum_source_term(stkBulk_);

  if (actBulk_.openFast_.isDebug()) {
    actBulk_.output_torque_info(stkBulk_);
  }
}

} // namespace nalu
} // namespace sierra
