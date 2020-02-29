// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorFunctorsFAST.h>
#include <actuator/UtilitiesActuator.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {


template<>
ActFastZero::ActuatorFunctor(ActuatorBulkFAST& actBulk) : actBulk_(actBulk){
  touch_dual_view(actBulk_.velocity_);
  touch_dual_view(actBulk_.actuatorForce_);
}

template<>
void
ActFastZero::operator()(const int& index) const{
  auto vel = get_local_view(actBulk_.velocity_);
  auto force = get_local_view(actBulk_.actuatorForce_);
  for(int i =0; i<3; i++){
    vel(index, i)=0.0;
    force(index, i)=0.0;
  }
}

template <>
ActFastUpdatePoints::ActuatorFunctor(ActuatorBulkFAST& actBulk)
  : actBulk_(actBulk)
{
  touch_dual_view(actBulk_.pointCentroid_);
}

template <>
void
ActFastUpdatePoints::operator()(const int& index) const
{
  fast::OpenFAST& FAST = actBulk_.openFast_;
  auto points = get_local_view(actBulk_.pointCentroid_);
  auto offsets = get_local_view(actBulk_.turbIdOffset_);

  ThrowAssert(actBulk_.localTurbineId_>=0);
  // compute location
  std::vector<double> tempCoords(3, 0.0);
  auto rank = actBulk_.localTurbineId_;
  FAST.getForceNodeCoordinates(tempCoords, rank, rank);
  for (int i = 0; i < 3; i++) {
    points(index, i) = tempCoords[i];
  }
}

template<>
ActFastAssignVel::ActuatorFunctor(ActuatorBulkFAST& actBulk):actBulk_(actBulk){}

template<>
void ActFastAssignVel::operator ()(const int& index) const{
  auto vel = get_local_view(actBulk_.velocity_);
  auto offset = get_local_view(actBulk_.turbIdOffset_);

  const int localId = index - offset(actBulk_.localTurbineId_);

  std::vector<double> pointVel {vel(index,0), vel(index,1), vel(index,2)};

  actBulk_.openFast_.setVelocityForceNode(pointVel, localId, actBulk_.localTurbineId_);
}

template<>
ActFastComputeForce::ActuatorFunctor(ActuatorBulkFAST& actBulk):actBulk_(actBulk){
  touch_dual_view(actBulk_.actuatorForce_);
}

template<>
void ActFastComputeForce::operator()(const int& index) const{
  auto force = get_local_view(actBulk_.actuatorForce_);
  auto offset = get_local_view(actBulk_.turbIdOffset_);

  std::vector<double> pointForce(3);

  const int localId = index - offset(actBulk_.localTurbineId_);

  actBulk_.openFast_.getForce(pointForce, localId, actBulk_.localTurbineId_);

  for(int i = 0; i<3; i++){
    force(index,i) = pointForce[i];
  }

}




} /* namespace nalu */
} /* namespace sierra */
