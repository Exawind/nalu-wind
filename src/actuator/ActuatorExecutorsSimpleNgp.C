// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <actuator/ActuatorExecutorsSimpleNgp.h>

namespace sierra {
namespace nalu {

ActuatorLineSimpleNGP::ActuatorLineSimpleNGP(
  const ActuatorMetaSimple& actMeta,
  ActuatorBulkSimple& actBulk,
  stk::mesh::BulkData& stkBulk)
  : actMeta_(actMeta),
    actBulk_(actBulk),
    stkBulk_(stkBulk),
    numActPoints_(actMeta_.numPointsTotal_)
{
}

void
ActuatorLineSimpleNGP::operator()()
{
  auto forceReduce = actBulk_.actuatorForce_.view_host();

  actBulk_.zero_source_terms(stkBulk_);

  update();

  // set range policy to only operating over points owned by local fast turbine
  auto fastRangePolicy = actBulk_.local_range_policy();


  Kokkos::parallel_for(
    "computeForcesActuatorNgpSimple", fastRangePolicy,
    ActSimpleComputeForce(actBulk_, actMeta_));

  actuator_utils::reduce_view_on_host(forceReduce);

  const int localSizeCoarseSearch =
    actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

  // === Always use SpreadActuatorForce() ===
  // -- for both isotropic and anisotropic Guassians ---
  Kokkos::parallel_for(
    "spreadForcesActuatorNgpSimple", localSizeCoarseSearch,
    SpreadActuatorForce(actBulk_, stkBulk_));
  // -- Uncomment below to use ActSimpleSpreadForceWhProjection
  // Kokkos::parallel_for(
  //   "spreadForceUsingProjDistance", localSizeCoarseSearch,
  //   ActSimpleSpreadForceWhProjection(actBulk_, stkBulk_));
  // ====

  actBulk_.parallel_sum_source_term(stkBulk_);

}

void
ActuatorLineSimpleNGP::update()
{

  auto velReduce = actBulk_.velocity_.view_host();
  auto pointReduce = actBulk_.pointCentroid_.view_host();
  actBulk_.zero_open_fast_views();

  // set range policy to only operating over points owned by local fast turbine
  auto fastRangePolicy = actBulk_.local_range_policy();

  // Uncomment this section if you want to update the point positions
  // ====
  // -- Get p1 and p2 for blade geometry --
  // double p1[3]; 
  // double p2[3]; 
  // for (int j=0; j<3; j++) { 
  //   p1[j] = actMeta_.p1_.h_view(actBulk_.localTurbineId_,j);
  //   p2[j] = actMeta_.p2_.h_view(actBulk_.localTurbineId_,j);
  // }
  // int Npts=actMeta_.num_force_pts_blade_.h_view(actBulk_.localTurbineId_);
  // -- functor to update points -- 
  // Kokkos::parallel_for(
  //   "updatePointLocationsActuatorNgpSimple", fastRangePolicy,
  //   ActSimpleUpdatePoints(actBulk_, Npts));
  // actuator_utils::reduce_view_on_host(pointReduce);
  // =====
  actBulk_.stk_search_act_pnts(actMeta_, stkBulk_);

  Kokkos::parallel_for(
    "interpolateVelocitiesActuatorNgpSimple", numActPoints_,
    InterpActuatorVel(actBulk_, stkBulk_));
  actuator_utils::reduce_view_on_host(velReduce);

  Kokkos::parallel_for(
    "interpolateDensityActuatorNgpSimple", numActPoints_,
    InterpActuatorDensity(actBulk_, stkBulk_));
  auto rhoReduce = actBulk_.density_.view_host();
  actuator_utils::reduce_view_on_host(rhoReduce);

  // This is for output purposes
  Kokkos::parallel_for(
    "assignSimpleVelActuatorNgpSimple", fastRangePolicy,
    ActSimpleAssignVel(actBulk_));

}

} // namespace nalu
} // namespace sierra
