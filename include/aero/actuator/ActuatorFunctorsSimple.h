// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORFUNCTORSSIMPLE_H_
#define ACTUATORFUNCTORSSIMPLE_H_

#include <aero/actuator/ActuatorTypes.h>
#include <aero/actuator/ActuatorBulkSimple.h>
#include <aero/actuator/ActuatorFunctors.h>
#include <NaluParsedTypes.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

struct InterpActuatorDensity
{
  using execution_space = ActuatorFixedExecutionSpace;

  InterpActuatorDensity(
    ActuatorBulkSimple& actBulk, stk::mesh::BulkData& stkBulk);

  void operator()(int index) const;

  ActuatorBulkSimple& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  VectorFieldType* coordinates_;
  ScalarFieldType* density_;
};

// Things to calculate lift, drag, and AOA based on 2D airfoil theory
namespace AirfoilTheory2D {
// Can use this namespace to add functions relevant to 2D airfoil
// theory
void calculate_alpha(
  double ws[],
  const double zeroalphadir[],
  const double spanDir[],
  const double chodrNormalDir[],
  double twist,
  double ws2Da[],
  double& alpha);
} // namespace AirfoilTheory2D

#ifdef ENABLE_ACTSIMPLE_PTMOTION
struct ActSimpleUpdatePoints
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActSimpleUpdatePoints(
    ActuatorBulkSimple& actBulk, int numpoints, double p1[], double p2[]);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl points_;
  ActFixScalarInt offsets_;
  const int turbId_;
  const int numpoints_;
  double p1_[3]; // Start position of blade
  double p2_[3]; // End position of blade
};
#endif

void ActSimpleWriteToFile(
  ActuatorBulkSimple& actBulk, const ActuatorMetaSimple& actMeta);

struct ActSimpleAssignVel
{
  using execution_space = ActuatorFixedExecutionSpace;

  ActSimpleAssignVel(ActuatorBulkSimple& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl velocity_;
  ActFixScalarDbl density_;
  ActFixVectorDbl points_;
  ActFixScalarInt offset_;
  const int debug_output_;
  const int turbId_;
  std::vector<double> p1_;
  std::vector<double> p2_;
};

void ActSimpleComputeRelativeVelocity(
  ActuatorBulkSimple& actBulk, const ActuatorMetaSimple& actMeta);

void ActSimpleComputeForce(
  ActuatorBulkSimple& actBulk, const ActuatorMetaSimple& actMeta);

struct ActSimpleComputeThrustInnerLoop
{

  ActSimpleComputeThrustInnerLoop(ActuatorBulkSimple& actBulk)
    : actBulk_(actBulk)
  {
  }
  void operator()(
    const uint64_t pointId,
    const double* nodeCoords,
    double* sourceTerm,
    const double dualNodalVolume,
    const double scvIp) const;
  void preloop() {}

  ActuatorBulkSimple& actBulk_;
};

struct ActSimpleSpreadForceWhProjInnerLoop
{
  ActSimpleSpreadForceWhProjInnerLoop(ActuatorBulkSimple& actBulk)
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

  ActuatorBulkSimple& actBulk_;
};

using ActSimpleComputeThrust = GenericLoopOverCoarseSearchResults<
  ActuatorBulkSimple,
  ActSimpleComputeThrustInnerLoop>;

using ActSimpleSpreadForceWhProjection = GenericLoopOverCoarseSearchResults<
  ActuatorBulkSimple,
  ActSimpleSpreadForceWhProjInnerLoop>;

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORFUNCTORSSIMPLE_H_ */
