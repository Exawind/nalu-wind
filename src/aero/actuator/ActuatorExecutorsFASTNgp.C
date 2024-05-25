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

  //Range for Kokkos parallel-for -- set range policy to only operating over points owned by local fast turbine
  auto fastRangePolicy = actBulk_.local_range_policy();

  //Interpolate velocity to actuator points. This happens before fine search? Is this from previous timestep
  RunInterpActuatorVel(actBulk_, stkBulk_);

  // Add FLLC correction to velocity field. Is this term treated explicitly? Also seems like it’s from previous step
  apply_fllc(actBulk_);

  //assign velocity data to a point and turbine ID from openfast?
  Kokkos::parallel_for(
    "assignFastVelActuatorNgpFAST", fastRangePolicy,
    ActFastAssignVel(actBulk_));

  // get relative velocity from openFAST
  ActFastCacheRelativeVelocities(actBulk_);

  // Compute filtered lifting line correction and store in fllc(I,j)
  compute_fllc();

  // Send interpolated velocities at actuator points to openFAST
  actBulk_.interpolate_velocities_to_fast();

  // Get actuator point centroids (from openfast or from basis-functions?). Again, this after the interpolation step?
  RunActFastUpdatePoints(actBulk_);

  // Execute fine and coarse search given point centroids (see next slides)
  // Do not need to update fine and coarse search when performing turbine level search 
  if !(actMeta_->turbineLevelSearch_) { 
    actBulk_.stk_search_act_pnts(actMeta_, stkBulk_); // this is the fine and coarse searching. 
  }

  // call openfast and step
  actBulk_.step_fast();

  // compute the force from openfast at actuator points
  RunActFastComputeForce(actBulk_);

  // Loop over all coarse points, the spread the force to. 
  // Does this perform the loop over coarse search for each point centroid?
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

  // sum up force contributions
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

    actBulk_.stk_search_act_pnts(actMeta_, stkBulk_);
  }

  actBulk_.step_fast();

  RunActFastComputeForce(actBulk_);

  compute_fllc();

  actBulk_.spread_forces_over_disk(actMeta_);

  const int localSizeCoarseSearch =
    actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

  Kokkos::parallel_for(
    "spreadForcesActuatorNgpFAST", HostRangePolicy(0, localSizeCoarseSearch),
    SpreadActuatorForce(actBulk_, stkBulk_));

  actBulk_.parallel_sum_source_term(stkBulk_);

  if (actBulk_.openFast_.isDebug()) {
    actBulk_.output_torque_info(stkBulk_);
  }
}

} // namespace nalu
} // namespace sierra
