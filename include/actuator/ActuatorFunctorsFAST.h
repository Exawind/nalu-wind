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
#include <actuator/ActuatorFunctors.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

namespace actfast {
// tags
struct ComputeLocations{};
struct AssignVelocities{};
struct ComputeForces{};
}

// typedefs
using ActFastUpdatePoints = ActuatorFunctor<ActuatorBulkFAST, actfast::ComputeLocations, Kokkos::DefaultHostExecutionSpace>;
using ActFastAssignVel = ActuatorFunctor<ActuatorBulkFAST, actfast::AssignVelocities, Kokkos::DefaultHostExecutionSpace>;
using ActFastComputeForce = ActuatorFunctor<ActuatorBulkFAST,actfast::ComputeForces, Kokkos::DefaultHostExecutionSpace>;

// declarations
struct
ActFastZero{
  using execution_space=ActuatorExecutionSpace;

  ActFastZero(ActuatorBulkFAST& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorMemSpace> helper_;
  ActVectorDbl vel_;
  ActVectorDbl force_;
  ActVectorDbl point_;

};

template <>
ActFastUpdatePoints::ActuatorFunctor(ActuatorBulkFAST& actBulk);

template<>
void ActFastUpdatePoints::operator()(const int& index) const;

template<>
ActFastAssignVel::ActuatorFunctor(ActuatorBulkFAST& actBulk);

template<>
void ActFastAssignVel::operator()(const int& index) const;

template<>
ActFastComputeForce::ActuatorFunctor(ActuatorBulkFAST& actBulk);

template<>
void ActFastComputeForce::operator()(const int& index) const;

struct ActFastSetUpThrustCalc{
  using execution_space = ActuatorFixedExecutionSpace;

  ActFastSetUpThrustCalc(ActuatorBulkFAST& actBulk);

  void operator()(int index) const;

  ActuatorBulkFAST& actBulk_;
};

struct ActFastComputeThrust{
  using execution_space = ActuatorFixedExecutionSpace;

  ActFastComputeThrust(ActuatorBulkFAST& actBulk, stk::mesh::BulkData& stkBulk);

  void operator()(int index) const;

  ActuatorBulkFAST& actBulk_;
  stk::mesh::BulkData& stkBulk_;
};

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORFUNCTORSFAST_H_ */
