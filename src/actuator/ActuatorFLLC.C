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
#include <actuator/ActuatorScalingFLLC.h>
#include <cmath>

namespace sierra {
namespace nalu {
namespace FLLC {

// free functions for vector operations
KOKKOS_INLINE_FUNCTION double
dot(double* u, double* v)
{
  double result = 0.0;
  for (int i = 0; i < 3; ++i) {
    result += u[i] * v[i];
  }
  return result;
}

// TODO(psakiev) - do we NEED to do local range policy on any of these?, other
// parallelization options?
// TODO(psakiev) - need to set this up to run per blade and not over entire
// turbine for the openfast case
// TODO(psakiev) - add option to run over portion of the blade

void
compute_lift_force_distribution(
  ActuatorBulk& actBulk, const ActuatorMeta& actMeta)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper;
  helper.touch_dual_view(actBulk.deltaLiftForceDistribution_);
  helper.touch_dual_view(actBulk.relativeVelocityMagnitude_);

  auto vel = helper.get_local_view(actBulk.relativeVelocity_);
  auto force = helper.get_local_view(actBulk.actuatorForce_);
  auto G = helper.get_local_view(actBulk.liftForceDistribution_);
  auto Uinf = helper.get_local_view(actBulk.relativeVelocityMagnitude_);
  auto points = helper.get_local_view(actBulk.pointCentroid_);

  Kokkos::deep_copy(G, 0.0);
  Kokkos::deep_copy(Uinf, 0.0);

  auto range_policy = actBulk.local_range_policy(actMeta);

  // for now let's just worry about constant span direction (no blade
  // deformation)
  ActFixScalarDbl span_dir("span normal", 1);

  // surrogate for equation 5.3
  Kokkos::parallel_for(
    "extract lift", range_policy, KOKKOS_LAMBDA(int i) {
      auto v = Kokkos::subview(vel, i, Kokkos::ALL);
      auto f = Kokkos::subview(force, i, Kokkos::ALL);

      const double fv = dot(f.data(), v.data());
      const double vmag2 = dot(v.data(), v.data());
      Uinf(i) = std::sqrt(vmag2);

      for (int j = 0; j < 3; ++j) {
        G(i, j) = force(i, j) - vel(i, j) * fv / vmag2;
      }
  });

  scale_lift_force(actBulk, actMeta, range_policy, helper);
  
  actuator_utils::reduce_view_on_host(G);
  actuator_utils::reduce_view_on_host(Uinf);
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
      for (int j = 0; j < 3; ++j) {
        if (index == 0) {
          deltaG(i, j) = G(i, j);
        } else if (index == numEntityPoints - 1) {
          deltaG(i, j) = -1.0 * G(i, j);
        } else {
          deltaG(i, j) = 0.5 * (G(i + 1, j) - G(i - 1, j));
        }
      }
    });

  actuator_utils::reduce_view_on_host(deltaG);
}

void
compute_induced_velocities(
  ActuatorBulk& actBulk,
  const ActuatorMeta& actMeta)
{
  using mem_space = ActuatorMemSpace;
  using mem_layout = ActuatorMemLayout;

  ActDualViewHelper<mem_space> helper;
  helper.touch_dual_view(actBulk.fllc_);

  auto deltaG = helper.get_local_view(actBulk.deltaLiftForceDistribution_);
  auto epsilon = helper.get_local_view(actBulk.epsilon_);
  auto epsilonOpt = helper.get_local_view(actBulk.epsilonOpt_);
  auto point = helper.get_local_view(actBulk.pointCentroid_);
  auto relVel = helper.get_local_view(actBulk.relativeVelocity_);
  auto deltaU = helper.get_local_view(actBulk.fllc_);
  auto Uinf = helper.get_local_view(actBulk.relativeVelocityMagnitude_);

  const int nTurb = actBulk.localTurbineId_;
  const int offset = helper.get_local_view(actBulk.turbIdOffset_)(nTurb);
  const int nPoints = helper.get_local_view(actMeta.numPointsTurbine_)(nTurb);

  const double dx[3] = {
    point(offset, 0) - point(offset + 1, 0),
    point(offset, 1) - point(offset + 1, 1),
    point(offset, 2) - point(offset + 1, 2)};

  const double dR = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
  const double relaxation_factor = 0.1;

  // copy deltaU so we can zero it for our data reduction strategy
  Kokkos::View<double* [3], mem_layout, mem_space> deltaU_stash(
    "temp copy", deltaU.extent_int(0));
  Kokkos::deep_copy(deltaU_stash, deltaU);
  Kokkos::deep_copy(deltaU, 0.0);

  auto range_policy = actBulk.local_range_policy(actMeta);

  Kokkos::parallel_for("compute flucs", range_policy, KOKKOS_LAMBDA(int index) {
      double optInd[3] = {0, 0, 0};
      double lesInd[3] = {0, 0, 0};

      const int i = index - offset;

      const double epsLes2 = epsilon(index, 0) * epsilon(index, 0);
      const double epsOpt2 = epsilonOpt(index,0)*epsilonOpt(index,0);
      
      // Compute equation 5.7 in reference paper
      for (int j = 0; j < nPoints; ++j) {
        if (i == j)
          continue;
        // constant point spacing
        const double dr = dR * (i - j);
        const double dr2 = dr * dr;

        const double coefficient = 1.0 / (-4.0 * M_PI * dr * Uinf(j + offset));
        const double coefOpt = 1.0 - std::exp(-dr2 / epsOpt2);
        const double coefLes = 1.0 - std::exp(-dr2 / epsLes2);

        for (int dir = 0; dir < 3; ++dir) {
          optInd[dir] -= deltaG(j + offset, dir) * coefficient * coefOpt;
          lesInd[dir] -= deltaG(j + offset, dir) * coefficient * coefLes;
        }
      }
      // update the correction term with relaxation
      // equation 5.8
      for (int j = 0; j < 3; ++j) {
        deltaU(index, j) = relaxation_factor * (optInd[j] - lesInd[j]) +
                           (1.0 - relaxation_factor) * deltaU_stash(index, j);
      }
    });

  actuator_utils::reduce_view_on_host(deltaU);
};

} // namespace FLLC

void
Compute_FLLC(ActuatorBulk& actBulk, const ActuatorMeta& actMeta)
{
  if (!actMeta.useFLLC_)
    return;
  FLLC::compute_lift_force_distribution(actBulk, actMeta);
  FLLC::grad_lift_force_distribution(actBulk, actMeta);
  FLLC::compute_induced_velocities(actBulk, actMeta);
}

void
Apply_FLLC(ActuatorBulk& actBulk, const ActuatorMeta& actMeta)
{
  if (!actMeta.useFLLC_)
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
