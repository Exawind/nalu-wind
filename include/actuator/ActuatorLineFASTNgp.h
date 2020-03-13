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
#include <actuator/UtilitiesActuator.h>

namespace sierra
{
namespace nalu
{

struct ActuatorLineFastNGP{

  ActuatorLineFastNGP(const ActuatorMetaFAST& actMeta,
    ActuatorBulkFAST& actBulk,
    stk::mesh::BulkData& stkBulk):
    actMeta_(actMeta),
    actBulk_(actBulk),
    stkBulk_(stkBulk),
    numActPoints_( actBulk_.totalNumPoints_)
  {}

  inline void operator()()
  {
    auto forceReduce = actBulk_.actuatorForce_.view_host();

    actBulk_.zero_source_terms(stkBulk_);

    if(actBulk_.fast_is_time_zero()){
      update();
    }
    actBulk_.interpolate_velocities_to_fast();

    update();

    actBulk_.step_fast();

    // set range policy to only operating over points owned by local fast turbine
    auto fastRangePolicy = actBulk_.local_range_policy(actMeta_);

    Kokkos::parallel_for("computeForcesActuatorNgpFAST", fastRangePolicy, ActFastComputeForce(actBulk_));

    actuator_utils::reduce_view_on_host(forceReduce);

    const int localSizeCoarseSearch = actBulk_.coarseSearchElemIds_.view_host().extent_int(0);

    Kokkos::parallel_for("spreadForcesActuatorNgpFAST", localSizeCoarseSearch, SpreadActuatorForce(actBulk_, stkBulk_));

    // TODO(psakiev) compute thrust
    actBulk_.parallel_sum_source_term(stkBulk_);
  }

  void update(){
    auto velReduce   = actBulk_.velocity_.view_host();
    auto pointReduce = actBulk_.pointCentroid_.view_host();

    Kokkos::parallel_for("zeroQuantitiesActuatorNgpFAST", numActPoints_, ActFastZero(actBulk_));

    Kokkos::parallel_for("zeroQuantitiesActuatorNgpFAST", numActPoints_, ActFastZero(actBulk_));

    // set range policy to only operating over points owned by local fast turbine
    auto fastRangePolicy = actBulk_.local_range_policy(actMeta_);

    Kokkos::parallel_for("updatePointLocationsActuatorNgpFAST", fastRangePolicy, ActFastUpdatePoints(actBulk_));
    actuator_utils::reduce_view_on_host(pointReduce);

    actBulk_.stk_search_act_pnts(actMeta_, stkBulk_);

    Kokkos::parallel_for("interpolateVelocitiesActuatorNgpFAST", numActPoints_, InterpActuatorVel(actBulk_, stkBulk_));

    actuator_utils::reduce_view_on_host(velReduce);

    Kokkos::parallel_for("assignFastVelActuatorNgpFAST", fastRangePolicy, ActFastAssignVel(actBulk_));
  }
  const ActuatorMetaFAST& actMeta_;
  ActuatorBulkFAST& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  const int numActPoints_;
};

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORLINEFASTNGP_H_ */
