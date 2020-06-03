// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef UNITTESTACTUATORNGP_H_
#define UNITTESTACTUATORNGP_H_

#include <actuator/ActuatorNGP.h>
#include <actuator/ActuatorBulk.h>
#include <actuator/ActuatorInfo.h>
#include <actuator/ActuatorSearch.h>
#include <actuator/UtilitiesActuator.h>
#include <actuator/ActuatorFunctors.h>

namespace sierra {
namespace nalu {

struct ComputePointLocation
{
};
struct InterpolateValues
{
};
struct SpreadForces
{
};
struct PostIter
{
};

// host only examples
inline void
ActPreIter(ActuatorBulk& actBulk)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  helper_.touch_dual_view(actBulk.epsilon_);

  auto epsilon = helper_.get_local_view(actBulk.epsilon_);

  Kokkos::parallel_for(
    "ActPreIter",
    Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0, epsilon.extent_int(0)),
    [epsilon](int index) {
      epsilon(index, 0) = index * 3.0;
      epsilon(index, 1) = index * 6.0;
      epsilon(index, 2) = index * 9.0;
    });
}

inline void
ActCompPnt(ActuatorBulk& actBulk)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  helper_.touch_dual_view(actBulk.pointCentroid_);

  auto points = helper_.get_local_view(actBulk.pointCentroid_);

  Kokkos::parallel_for(
    "ActCompPnt",
    Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0, points.extent_int(0)),
    [points](int index) {
      points(index, 0) = index;
      points(index, 1) = index * 0.5;
      points(index, 2) = index * 0.25;
    });
}

inline void
ActInterp(ActuatorBulk& actBulk)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  helper_.touch_dual_view(actBulk.velocity_);

  auto velocity = helper_.get_local_view(actBulk.velocity_);

  Kokkos::parallel_for(
    "ActInterp",
    Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0, velocity.extent_int(0)),
    [velocity](int index) {
      velocity(index, 0) = index * 2.5;
      velocity(index, 1) = index * 5.0;
      velocity(index, 2) = index * 7.5;
    });
}

inline void
ActSpread(ActuatorBulk& actBulk)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  helper_.touch_dual_view(actBulk.actuatorForce_);

  auto force = helper_.get_local_view(actBulk.actuatorForce_);

  Kokkos::parallel_for(
    "ActSpread",
    Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0, force.extent_int(0)),
    [force](int index) {
      force(index, 0) = index * 3.1;
      force(index, 1) = index * 6.2;
      force(index, 2) = index * 9.3;
    });
}

using TestActuatorHostOnly = ActuatorNGP<ActuatorMeta, ActuatorBulk>;
template <>
void
TestActuatorHostOnly::execute()
{
  ActPreIter(actBulk_);
  ActCompPnt(actBulk_);
  ActInterp(actBulk_);
  ActSpread(actBulk_);
}

// Create a different bulk data that will allow execution on device and host
// for functors
struct ActuatorBulkMod : public ActuatorBulk
{
  ActuatorBulkMod(ActuatorMeta meta)
    : ActuatorBulk(meta), scalar_("scalar", meta.numPointsTotal_)
  {
  }
  ActScalarDblDv scalar_;
};
//-----------------------------------------------------------------
// host or device execution example

 void
 ActPostIter(ActuatorBulkMod& actBulk)
{
  ActDualViewHelper<ActuatorMemSpace> helper;
  helper.touch_dual_view(actBulk.scalar_);
  actBulk.velocity_.sync_device();;
  actBulk.pointCentroid_.sync_device();

  auto scalar = actBulk.scalar_.view_device();
  auto vel = actBulk.velocity_.view_device();
  auto point = actBulk.pointCentroid_.view_device();

  Kokkos::parallel_for(
    "ActPostIter",
    Kokkos::RangePolicy<ActuatorExecutionSpace>(0,
    actBulk.scalar_.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      scalar(index) = point(index, 0) * vel(index, 1);
    });
}

using TestActuatorHostDev = ActuatorNGP<ActuatorMeta, ActuatorBulkMod>;
template <>
void
TestActuatorHostDev::execute()
{
  ActPreIter(actBulk_);
  ActCompPnt(actBulk_);
  ActInterp(actBulk_);
  ActSpread(actBulk_);
  ActPostIter(actBulk_);
  actBulk_.scalar_.sync_host();
}

} // namespace nalu
} // namespace sierra

#endif // UNITTESTACTUATORNGP_H_
