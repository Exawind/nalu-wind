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
#include <actuator/ActuatorBladeDistributor.h>
#include <cmath>

namespace sierra {
namespace nalu {

// free functions for vector operations
inline double
dot(double* u, double* v)
{
  double result = 0.0;
  for (int i = 0; i < 3; ++i) {
    result += u[i] * v[i];
  }
  return result;
}

FilteredLiftingLineCorrection::FilteredLiftingLineCorrection(
  const ActuatorMeta& actMeta, ActuatorBulk& actBulk)
  : actBulk_(actBulk), actMeta_(actMeta)
{
  bladeDistInfo_ = compute_blade_distributions(actMeta, actBulk);
}

void
FilteredLiftingLineCorrection::compute_lift_force_distribution()
{
  ActDualViewHelper<mem_space> helper;
  helper.touch_dual_view(actBulk_.deltaLiftForceDistribution_);
  helper.touch_dual_view(actBulk_.relativeVelocityMagnitude_);

  auto vel = helper.get_local_view(actBulk_.relativeVelocity_);
  auto force = helper.get_local_view(actBulk_.actuatorForce_);
  auto G = helper.get_local_view(actBulk_.liftForceDistribution_);
  auto Uinf = helper.get_local_view(actBulk_.relativeVelocityMagnitude_);
  auto points = helper.get_local_view(actBulk_.pointCentroid_);

  Kokkos::deep_copy(G, 0.0);
  Kokkos::deep_copy(Uinf, 0.0);

  for (auto&& info : bladeDistInfo_) {

    const auto offset = info.offset_;
    const auto nPoints = info.nPoints_;
    // debug size checks
    ThrowAssert(offset + nPoints < static_cast<int>(G.size()));

    auto range_policy =
      Kokkos::RangePolicy<exec_space>(offset, offset + nPoints);

    // surrogate for equation 5.3
    Kokkos::parallel_for(
      "extract lift", range_policy, ACTUATOR_LAMBDA(int i) {
        auto v = Kokkos::subview(vel, i, Kokkos::ALL);
        auto f = Kokkos::subview(force, i, Kokkos::ALL);

        const double fv = dot(f.data(), v.data());
        const double vmag2 = dot(v.data(), v.data());
        Uinf(i) = std::sqrt(vmag2);

        for (int j = 0; j < 3; ++j) {
          G(i, j) = force(i, j) - vel(i, j) * fv / vmag2;
        }
      });
    FLLC::scale_lift_force(
      actBulk_, actMeta_, range_policy, helper, offset, nPoints);
  }

  actuator_utils::reduce_view_on_host(G);
  actuator_utils::reduce_view_on_host(Uinf);
}

void
FilteredLiftingLineCorrection::grad_lift_force_distribution()
{
  ActDualViewHelper<mem_space> helper;
  helper.touch_dual_view(actBulk_.deltaLiftForceDistribution_);

  auto G = helper.get_local_view(actBulk_.liftForceDistribution_);
  auto deltaG = helper.get_local_view(actBulk_.deltaLiftForceDistribution_);

  Kokkos::deep_copy(deltaG, 0.0);
  for (auto&& info : bladeDistInfo_) {

    const auto offset = info.offset_;
    const auto nPoints = info.nPoints_;
    ThrowAssert(offset + nPoints < static_cast<int>(G.size()));
    ThrowAssert(offset + nPoints < static_cast<int>(deltaG.size()));

    auto range_policy =
      Kokkos::RangePolicy<exec_space>(offset, offset + nPoints);

    // equations 5.4 and 5.5 a/b
    Kokkos::parallel_for(
      "compute dG", range_policy, ACTUATOR_LAMBDA(int i) {
        const int index = i - offset;
        for (int j = 0; j < 3; ++j) {
          if (index == 0) {
            deltaG(i, j) = G(i, j);
          } else if (index == nPoints - 1) {
            deltaG(i, j) = -1.0 * G(i, j);
          } else {
            deltaG(i, j) = 0.5 * (G(i + 1, j) - G(i - 1, j));
          }
        }
      });
  }

  actuator_utils::reduce_view_on_host(deltaG);
}

void
FilteredLiftingLineCorrection::compute_induced_velocities()
{
  ActDualViewHelper<mem_space> helper;
  helper.touch_dual_view(actBulk_.fllc_);

  auto deltaG = helper.get_local_view(actBulk_.deltaLiftForceDistribution_);
  auto epsilon = helper.get_local_view(actBulk_.epsilon_);
  auto epsilonOpt = helper.get_local_view(actBulk_.epsilonOpt_);
  auto point = helper.get_local_view(actBulk_.pointCentroid_);
  auto deltaU = helper.get_local_view(actBulk_.fllc_);
  auto Uinf = helper.get_local_view(actBulk_.relativeVelocityMagnitude_);

  const double relaxationFactor = 0.1;

  // copy deltaU so we can zero it for our data reduction strategy
  Kokkos::View<double* [3], mem_layout, mem_space> deltaU_stash(
    "temp copy", deltaU.extent_int(0));
  Kokkos::deep_copy(deltaU_stash, deltaU);
  Kokkos::deep_copy(deltaU, 0.0);

  for (auto&& info : bladeDistInfo_) {

    const auto offset = info.offset_;
    const auto nPoints = info.nPoints_;
    const auto nNeighbors = info.nNeighbors_;

    auto pointHost = actBulk_.pointCentroid_.view_host();
    const double dx[3] = {
      pointHost(offset, 0) - pointHost(offset + 1, 0),
      pointHost(offset, 1) - pointHost(offset + 1, 1),
      pointHost(offset, 2) - pointHost(offset + 1, 2)};

    const double dR = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);

    auto range_policy =
      Kokkos::RangePolicy<exec_space>(offset, offset + nPoints);

    Kokkos::parallel_for(
      "compute flucs", range_policy, ACTUATOR_LAMBDA(int index) {
        double optInd[3] = {0, 0, 0};
        double lesInd[3] = {0, 0, 0};

        const int i = index - offset;

        const double epsLes2 = epsilon(index, 0) * epsilon(index, 0);
        const double epsOpt2 = epsilonOpt(index, 0) * epsilonOpt(index, 0);

        // limits to approximate integral and speed up computation
        const int start = std::max(i - nNeighbors, 0);
        const int end = std::min(i + nNeighbors, nPoints);
        // Compute equation 5.7 in reference paper
        for (int j = start; j < end; ++j)

        {
          if (i == j)
            continue;
          // constant point spacing
          const double dr = dR * (i - j);
          const double dr2 = dr * dr;

          const double coefficient =
            1.0 / (-4.0 * M_PI * dr * Uinf(j + offset));
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
          deltaU(index, j) = relaxationFactor * (optInd[j] - lesInd[j]) +
                             (1.0 - relaxationFactor) * deltaU_stash(index, j);
        }
      });
  }
  actuator_utils::reduce_view_on_host(deltaU);
}

bool
FilteredLiftingLineCorrection::is_active()
{
  return actMeta_.useFLLC_;
}

} // namespace nalu
} // namespace sierra
