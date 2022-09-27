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

#include <aero/actuator/ActuatorTypes.h>
#include <aero/actuator/ActuatorBulkFAST.h>
#include <aero/actuator/ActuatorFunctors.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

void ActFastCacheRelativeVelocities(ActuatorBulkFAST& actBulk);

struct ActFastUpdatePoints
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActFastUpdatePoints(ActuatorBulkFAST& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl points_;
  ActFixScalarInt offsets_;
  const int turbId_;
  fast::OpenFAST& fast_;
};

inline void
RunActFastUpdatePoints(ActuatorBulkFAST& actBulk)
{
  Kokkos::deep_copy(actBulk.pointCentroid_.view_host(), 0.0);
  actBulk.pointCentroid_.modify_host();
  Kokkos::parallel_for(
    "ActFastUpdatePoints", actBulk.local_range_policy(),
    ActFastUpdatePoints(actBulk));
  actuator_utils::reduce_view_on_host(actBulk.pointCentroid_.view_host());
}

struct ActFastAssignVel
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActFastAssignVel(ActuatorBulkFAST& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl velocity_;
  ActFixScalarInt offset_;
  const int turbId_;
  fast::OpenFAST& fast_;
};

struct ActFastComputeForce
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActFastComputeForce(ActuatorBulkFAST& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl force_;
  ActFixScalarInt offset_;
  const int turbId_;
  fast::OpenFAST& fast_;
};

inline void
RunActFastComputeForce(ActuatorBulkFAST& actBulk)
{
  Kokkos::deep_copy(actBulk.actuatorForce_.view_host(), 0.0);
  actBulk.actuatorForce_.modify_host();
  Kokkos::parallel_for(
    "ActFastComputeForce", actBulk.local_range_policy(),
    ActFastComputeForce(actBulk));
  actuator_utils::reduce_view_on_host(actBulk.actuatorForce_.view_host());
}

struct ActFastSetUpThrustCalc
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActFastSetUpThrustCalc(ActuatorBulkFAST& actBulk);

  void operator()(int index) const;

  ActuatorBulkFAST& actBulk_;
};

struct ActFastStashOrientationVectors
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActFastStashOrientationVectors(ActuatorBulkFAST& actBulk);

  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixTensorDbl orientation_;
  ActFixScalarInt offset_;
  const int turbId_;
  fast::OpenFAST& fast_;
};

inline void
RunActFastStashOrientVecs(ActuatorBulkFAST& actBulk)
{
  Kokkos::deep_copy(actBulk.orientationTensor_.view_host(), 0.0);
  actBulk.orientationTensor_.modify_host();
  Kokkos::parallel_for(
    "ActFastStashOrientations", actBulk.local_range_policy(),
    ActFastStashOrientationVectors(actBulk));
  actuator_utils::reduce_view_on_host(actBulk.orientationTensor_.view_host());
}

struct ActFastComputeThrustInnerLoop
{

  ActFastComputeThrustInnerLoop(ActuatorBulkFAST& actBulk) : actBulk_(actBulk)
  {
  }
  void operator()(
    const uint64_t pointId,
    const double* nodeCoords,
    double* sourceTerm,
    const double dualNodalVolume,
    const double scvIp) const;
  void preloop() {}

  ActuatorBulkFAST& actBulk_;
};

struct ActFastSpreadForceWhProjInnerLoop
{
  ActFastSpreadForceWhProjInnerLoop(ActuatorBulkFAST& actBulk)
    : actBulk_(actBulk)
  {
  }

  void operator()(
    const uint64_t pointId,
    const double* nodeCoords,
    double* sourceTerm,
    const double dualNodalVolume,
    const double scvIp) const;
  void preloop();

  ActuatorBulkFAST& actBulk_;
};

using ActFastComputeThrust = GenericLoopOverCoarseSearchResults<
  ActuatorBulkFAST,
  ActFastComputeThrustInnerLoop>;

using ActFastSpreadForceWhProjection = GenericLoopOverCoarseSearchResults<
  ActuatorBulkFAST,
  ActFastSpreadForceWhProjInnerLoop>;

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORFUNCTORSFAST_H_ */
