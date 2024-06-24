// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <aero/actuator/ActuatorExecutorsFASTNgp.h>

namespace sierra {
namespace nalu {

ActuatorLineFastNGP::ActuatorLineFastNGP(
  const ActuatorMetaFAST& actMeta,
  ActuatorBulkFAST& actBulk,
  stk::mesh::BulkData& stkBulk)
  : ActuatorExecutor(actMeta, actBulk),
    actMeta_(actMeta),
    actBulk_(actBulk),
    stkBulk_(stkBulk)
{
}

void
ActuatorLineFastNGP::operator()()
{
  //Zero the (body-force) actuator source term 
  actBulk_.zero_source_terms(stkBulk_); 

  //set range policy to only operating over points owned by local fast turbine
  auto fastRangePolicy = actBulk_.local_range_policy();

  //Interpolate velocity to actuator points. 
  RunInterpActuatorVel(actBulk_, stkBulk_);

  // Add FLLC correction to velocity field. 
  apply_fllc(actBulk_);

  Kokkos::parallel_for(
    "assignFastVelActuatorNgpFAST", fastRangePolicy,
    ActFastAssignVel(actBulk_));

  // get relative velocity from openFAST
  ActFastCacheRelativeVelocities(actBulk_);

  // Compute filtered lifting line correction 
  compute_fllc();

  // Send interpolated velocities at actuator points to openFAST
  actBulk_.interpolate_velocities_to_fast();

  // Get actuator point centroids 
  RunActFastUpdatePoints(actBulk_);

  // Execute fine and coarse search given point centroids 
  actBulk_.stk_search(actMeta_, stkBulk_); 

  // call openfast and step
  actBulk_.step_fast();

  // compute the force from openfast at actuator points
  RunActFastComputeForce(actBulk_);

  // Loop over all coarse points
  const int localSizeCoarseSearch =
    actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

  if (actMeta_.isotropicGaussian_) {
    Kokkos::parallel_for(
      "spreadForcesActuatorNgpFAST", HostRangePolicy(0, localSizeCoarseSearch),
      SpreadActuatorForce(actBulk_, stkBulk_));
  } else {
    RunActFastStashOrientVecs(actBulk_);

    Kokkos::parallel_for(
      "spreadForceUsingProjDistance", HostRangePolicy(0, localSizeCoarseSearch),
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
  : ActuatorExecutor(actMeta, actBulk),
    actMeta_(actMeta),
    actBulk_(actBulk),
    stkBulk_(stkBulk)
{
}

void
ActuatorDiskFastNGP::operator()()
{
  actBulk_.zero_source_terms(stkBulk_);

  RunInterpActuatorVel(actBulk_, stkBulk_);

  apply_fllc(actBulk_);

  auto fastRangePolicy = actBulk_.local_range_policy();

  Kokkos::parallel_for(
    "assignFastVelActuatorNgpFAST", fastRangePolicy,
    ActFastAssignVel(actBulk_));

  ActFastCacheRelativeVelocities(actBulk_);

  auto forceReduce = actBulk_.actuatorForce_.view_host();

  actBulk_.interpolate_velocities_to_fast();

  if (actBulk_.adm_points_need_updating) {

    RunActFastUpdatePoints(actBulk_);

    actBulk_.update_ADM_points(actMeta_);

    actBulk_.stk_search(actMeta_, stkBulk_,true);
  }

  actBulk_.step_fast();

  RunActFastComputeForce(actBulk_);

  compute_fllc();

  actBulk_.spread_forces_over_disk(actMeta_);

  const int localSizeCoarseSearch =
    actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

  if (actMeta_.turbineLevelSearch_) {
    Kokkos::parallel_for(
      "spreadForcesActuatorNgpFAST", HostRangePolicy(0, localSizeCoarseSearch),
      SpreadActuatorForceTurbineSearch(actBulk_, stkBulk_));
  }
  else {
    Kokkos::parallel_for(
      "spreadForcesActuatorNgpFAST", HostRangePolicy(0, localSizeCoarseSearch),
      SpreadActuatorForce(actBulk_, stkBulk_));
  }

  actBulk_.parallel_sum_source_term(stkBulk_);

  if (actBulk_.openFast_.isDebug()) {
    actBulk_.output_torque_info(stkBulk_);
  }
}

} // namespace nalu
} // namespace sierra
