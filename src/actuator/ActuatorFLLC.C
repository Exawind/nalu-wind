// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorFLLC.h>
#include <actuator/ActuatorBulk.h>

namespace sierra {
namespace nalu {
namespace FLLC {

void
compute_lift_force_distribution(ActuatorBulk& actBulk)
{
  // need force
  // relative velocity
  ActDualViewHelper<ActuatorFixedMemSpace> helper;
  auto vel = helper.get_local_view(actBulk.relativeVelocity_);
  auto force = helper.get_local_view(actBulk.actuatorForce_);
  auto G = helper.get_local_view(actBulk.liftForceDistribution_);

  // auto range_policy = actBulk.local_range_policy(); TODO(psakiev)
  auto range_policy =
    Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0, vel.extent_int(0));
  Kokkos::parallel_for(
    "extract lift", range_policy, KOKKOS_LAMBDA(int i) {
      const double vmag2 =
        vel(i, 0) * vel(i, 0) + vel(i, 1) * vel(i, 1) + vel(i, 2) * vel(i, 2);
      const double fv = vel(i, 0) * force(i, 0) + vel(i, 1) * force(i, 1) +
                        vel(i, 2) * force(i, 2);

      for (int j = 0; j < 3; ++j) {
        G(i, j) = force(i, j) - vel(i, j) * fv / vmag2;
      }
    });
}

} // namespace FLLC
} // namespace nalu
} // namespace sierra