// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORFUNCTORSFAST_H_
#define ACTUATORFUNCTORSFAST_H_

#include <actuator/ActuatorNGP.h>
#include <actuator/ActuatorBulkFAST.h>

namespace sierra {
namespace nalu {

namespace actfast {
// tags
struct ComputeLocations
{
};
} // namespace actfast

// typedefs
using ActuatorNgpFAST = Actuator<ActuatorMetaFAST, ActuatorBulkFAST>;

using ActFastUpdatePoints = ActuatorFunctor<
  ActuatorBulkFAST,
  actfast::ComputeLocations,
  Kokkos::DefaultHostExecutionSpace>;

// declarations
template <>
ActFastUpdatePoints::ActuatorFunctor(ActuatorBulkFAST& actBulk);

template <>
void
ActuatorNgpFAST::execute()
{
  // compute point locations
  Kokkos::parallel_for(
    "updatePointLocationsActuatorNgpFAST", numActPoints_,
    ActFastUpdatePoints(actBulk_));
  // find points
  actBulk_.stk_search_act_pnts(actMeta_);
  // interpolate velocities
  // compute forces in FAST
  // spread forces to nodes
  // compute thrust
}

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORFUNCTORSFAST_H_ */
