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
struct ZeroArrays{};
struct ComputeLocations{};
struct AssignVelocities{};
struct ComputeForces{};
struct ComputeThrust{};
}

// typedefs
using ActFastZero = ActuatorFunctor<ActuatorBulkFAST, actfast::ZeroArrays, ActuatorExecutionSpace>;
using ActFastUpdatePoints = ActuatorFunctor<ActuatorBulkFAST, actfast::ComputeLocations, Kokkos::DefaultHostExecutionSpace>;
using ActFastAssignVel = ActuatorFunctor<ActuatorBulkFAST, actfast::AssignVelocities, Kokkos::DefaultHostExecutionSpace>;
using ActFastComputeForce = ActuatorFunctor<ActuatorBulkFAST,actfast::ComputeForces, Kokkos::DefaultHostExecutionSpace>;

// declarations
template<>
ActFastZero::ActuatorFunctor(ActuatorBulkFAST& actBulk);

template<>
void ActFastZero::operator()(const int& index) const;

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

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORFUNCTORSFAST_H_ */
