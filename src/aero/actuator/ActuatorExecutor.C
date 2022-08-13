// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorExecutor.h>

namespace sierra {
namespace nalu {

ActuatorExecutor::ActuatorExecutor(
  const ActuatorMeta& actMeta, ActuatorBulk& actBulk)
  : fLiftLineCorr_(actMeta, actBulk)
{
}

void
ActuatorExecutor::compute_fllc()
{
  if (!fLiftLineCorr_.is_active())
    return;
  fLiftLineCorr_.compute_lift_force_distribution();
  fLiftLineCorr_.grad_lift_force_distribution();
  fLiftLineCorr_.compute_induced_velocities();
}

void
ActuatorExecutor::apply_fllc(ActuatorBulk& actBulk)
{
  if (!fLiftLineCorr_.is_active())
    return;
  ActDualViewHelper<ActuatorMemSpace> helper;

  helper.sync(actBulk.fllc_);
  helper.touch_dual_view(actBulk.velocity_);

  auto fllc = helper.get_local_view(actBulk.fllc_);
  auto vel = helper.get_local_view(actBulk.velocity_);

  Kokkos::parallel_for(
    "apply fllc",
    Kokkos::RangePolicy<ActuatorExecutionSpace>(0, vel.extent_int(0)),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < 3; ++j) {
        vel(i, j) += fllc(i, j);
      }
    });
}

} // namespace nalu
} // namespace sierra
