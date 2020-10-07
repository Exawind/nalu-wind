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

// TODO(psakiev) - do we NEED to do local range policy on any of these?, other
// parallelization options?

void
compute_lift_force_distribution(
  ActuatorBulk& actBulk, const ActuatorMeta& actMeta)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper;
  helper.touch_dual_view(actBulk.deltaLiftForceDistribution_);

  auto vel = helper.get_local_view(actBulk.relativeVelocity_);
  auto force = helper.get_local_view(actBulk.actuatorForce_);
  auto G = helper.get_local_view(actBulk.liftForceDistribution_);

  Kokkos::deep_copy(G, 0.0);

  auto range_policy = actBulk.local_range_policy(actMeta);

  // surrogate for equation 5.3
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
  helper.touch_dual_view(actBulk.deltaLiftForceDistribution_);

  auto G = helper.get_local_view(actBulk.liftForceDistribution_);
  auto deltaG = helper.get_local_view(actBulk.deltaLiftForceDistribution_);

  const int offset =
    helper.get_local_view(actBulk.turbIdOffset_)(actBulk.localTurbineId_);

  const int numEntityPoints =
    helper.get_local_view(actMeta.numPointsTurbine_)(actBulk.localTurbineId_);

  Kokkos::deep_copy(deltaG, 0.0);

  auto range_policy = actBulk.local_range_policy(actMeta);

  // equations 5.4 and 5.5 a/b
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

void
compute_induced_velocities(ActuatorBulk& actBulk, const ActuatorMeta& actMeta)
{
  using mem_space = ActuatorFixedMemSpace;
  using mem_layout = ActuatorFixedMemLayout;

  ActDualViewHelper<mem_space> helper;
  helper.touch_dual_view(actBulk.fllVelocityCorrection_);

  auto deltaG = helper.get_local_view(actBulk.deltaLiftForceDistribution_);
  auto epsilon = helper.get_local_view(actBulk.epsilon_);
  auto epsilonOpt = helper.get_local_view(actBulk.epsilonOpt_);
  auto point = helper.get_local_view(actBulk.pointCentroid_);
  auto relVel = helper.get_local_view(actBulk.relativeVelocity_);
  auto deltaU = helper.get_local_view(actBulk.fllVelocityCorrection_);

  const int nTurb = actBulk.localTurbineId_;
  const int offset = helper.get_local_view(actBulk.turbIdOffset_)(nTurb);
  const int nPoints = helper.get_local_view(actMeta.numPointsTurbine_)(nTurb);

  const double dx[3] = {
    point(offset, 0) - point(offset + 1, 0),
    point(offset, 1) - point(offset + 1, 1),
    point(offset, 2) - point(offset + 1, 2)};

  const double dR = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
  const double normal[3] = {dx[0] / dR, dx[1] / dR, dx[2] / dR};
  const double relaxation_factor = 0.1;

  // copy deltaU so we can zero it for our data reduction strategy
  Kokkos::View<double* [3], mem_layout, mem_space> deltaU_stash(
    "temp copy", deltaU.extent_int(0));
  Kokkos::deep_copy(deltaU_stash, deltaU);
  Kokkos::deep_copy(deltaU, 0.0);

  auto range_policy = actBulk.local_range_policy(actMeta);

  Kokkos::parallel_for(
    "induced velocities", range_policy, KOKKOS_LAMBDA(int index) {
      double optInd = 0;
      double lesInd = 0;

      const int i = index - offset;
      const double Uinf = std::sqrt(
                            relVel(index, 0) * relVel(index, 0) +
                            relVel(index, 1) * relVel(index, 1) +
                            relVel(index, 2) * relVel(index, 2)) +
                          1.e-12;

      const double oneOverUinf = 1.0 / Uinf;

      // Compute equation 5.7 in reference paper
      // could do clock arithmatic to avoid if statement
      // for (int j = (i+1)%nPoints, k=0; k < nPoints-1; ++k, j=(j+1)%nPoints) {
      for (int j = 0; j < nPoints; ++j) {
        if (i == j)
          continue;
        // constant point spacing
        const double dr = dR * (i - j);
        const double coefficient = deltaG(j + offset) / (4.0 * M_PI * dr);
        optInd +=
          coefficient *
          (1.0 -
           std::exp(-dr * dr / (epsilonOpt(index, 0) * epsilonOpt(index, 0))));

        lesInd +=
          coefficient *
          (1.0 - std::exp(-dr * dr / (epsilon(index, 0) * epsilon(index, 0))));
      }
      const double deltaU_N = oneOverUinf * (optInd - lesInd);

      // update the correction term with relaxation
      // equation 5.8
      for (int j = 0; j < 3; ++j) {
        deltaU(index, j) = relaxation_factor * deltaU_N * normal[j] +
                           (1.0 - relaxation_factor) * deltaU_stash(index, j);
      }
    });

  actuator_utils::reduce_view_on_host(deltaU);
};

} // namespace FLLC

void
compute_fllc(ActuatorBulk& actBulk, const ActuatorMeta& actMeta)
{
  if (!actMeta.useFLLC_)
    return;
  FLLC::compute_lift_force_distribution(actBulk, actMeta);
  FLLC::grad_lift_force_distribution(actBulk, actMeta);
  FLLC::compute_induced_velocities(actBulk, actMeta);
}

} // namespace nalu
} // namespace sierra