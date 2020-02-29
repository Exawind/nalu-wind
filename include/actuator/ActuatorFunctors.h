// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORFUNCTORS_H_
#define ACTUATORFUNCTORS_H_

#include <actuator/ActuatorNGP.h>
#include <actuator/ActuatorBulk.h>

namespace sierra
{
namespace nalu
{

namespace actgeneral{
struct InterpolateVelocities{};
struct SpreadForce{};
}

using InterpolateActVel = ActuatorFunctor<
  ActuatorBulk,
  actgeneral::InterpolateVelocities,
  ActuatorFixedExecutionSpace>;

using SpreadActForce = ActuatorFunctor<ActuatorBulk, actgeneral::SpreadForce, ActuatorExecutionSpace>;

//functor should loop over local actuator points
template<>
InterpolateActVel::ActuatorFunctor(ActuatorBulk& actBulk);

template<>
void InterpolateActVel::operator()(const int& index) const;

// functor should loop over coarse search results
template<>
SpreadActForce::ActuatorFunctor(ActuatorBulk& actBulk);

template<>
void SpreadActForce::operator()(const int& index) const;

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORFUNCTORS_H_ */
