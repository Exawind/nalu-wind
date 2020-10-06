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
#include <actuator/UtilitiesActuator.h>
#include <cmath>

namespace sierra {
namespace nalu {
namespace FLLC {

void
compute_lift_force_distribution(
  ActuatorBulk& actBulk, const ActuatorMeta& actMeta)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper;
  auto vel = helper.get_local_view(actBulk.relativeVelocity_);
  auto force = helper.get_local_view(actBulk.actuatorForce_);
  auto G = helper.get_local_view(actBulk.liftForceDistribution_);

  helper.touch_dual_view(actBulk.deltaLiftForceDistribution_);

  Kokkos::deep_copy(G, 0.0);

  auto range_policy = actBulk.local_range_policy(actMeta);
  Kokkos::parallel_for(
    "extract lift", range_policy, KOKKOS_LAMBDA(int i) {
      const double vmag2 =
        vel(i, 0) * vel(i, 0) + vel(i, 1) * vel(i, 1) + vel(i, 2) * vel(i, 2);
      const double fv = vel(i, 0) * force(i, 0) + vel(i, 1) * force(i, 1) +
                        vel(i, 2) * force(i, 2);

      for (int j = 0; j < 3; ++j) {
        double temp = 0.0;
        temp = force(i, j) - vel(i, j) * fv / vmag2;
        G(i) += temp * temp;
      }
      G(i) = std::sqrt(G(i));
    });

  actuator_utils::reduce_view_on_host(G);
}

void
grad_lift_force_distribution(ActuatorBulk& actBulk, const ActuatorMeta& actMeta)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper;
  auto G = helper.get_local_view(actBulk.liftForceDistribution_);
  auto deltaG = helper.get_local_view(actBulk.deltaLiftForceDistribution_);

  const int offset =
    helper.get_local_view(actBulk.turbIdOffset_)(actBulk.localTurbineId_);

  const int numEntityPoints =
    helper.get_local_view(actMeta.numPointsTurbine_)(actBulk.localTurbineId_);

  helper.touch_dual_view(actBulk.deltaLiftForceDistribution_);

  Kokkos::deep_copy(deltaG, 0.0);

  auto range_policy = actBulk.local_range_policy(actMeta);
  Kokkos::parallel_for(
    "compute dG", range_policy, KOKKOS_LAMBDA(int i) {
      const int index = i - offset;
      if (index == 0 || index == numEntityPoints - 1) {
        deltaG(i) = G(i);
      } else {
        deltaG(i) = 0.5 * (G(i + 1) - G(i - 1));
      }
    });

  actuator_utils::reduce_view_on_host(deltaG);
}

} // namespace FLLC
} // namespace nalu
} // namespace sierra