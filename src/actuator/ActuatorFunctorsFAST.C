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
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <NaluEnv.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {


ActFastZero::ActFastZero(ActuatorBulkFAST& actBulk)
{
  vel_   = helper_.get_local_view(actBulk.velocity_     );
  force_ = helper_.get_local_view(actBulk.actuatorForce_);
  point_ = helper_.get_local_view(actBulk.pointCentroid_);

  helper_.touch_dual_view(actBulk.velocity_     );
  helper_.touch_dual_view(actBulk.actuatorForce_);
  helper_.touch_dual_view(actBulk.pointCentroid_);
}

void
ActFastZero::operator()(int index) const{
  for(int i =0; i<3; i++){
    vel_(index, i)=0.0;
    force_(index, i)=0.0;
    point_(index, i)=0.0;
  }
}

ActFastUpdatePoints::ActFastUpdatePoints(ActuatorBulkFAST& actBulk):
    points_(helper_.get_local_view(actBulk.pointCentroid_)),
    offsets_(helper_.get_local_view(actBulk.turbIdOffset_)),
    turbId_(actBulk.localTurbineId_),
    fast_(actBulk.openFast_)
{
  helper_.touch_dual_view(actBulk.pointCentroid_);
}

void
ActFastUpdatePoints::operator()(int index) const
{

  ThrowAssert(turbId_>=0);
  const int pointId = index - offsets_(turbId_);
  auto point = Kokkos::subview(points_, index, Kokkos::ALL);

  fast_.getForceNodeCoordinates(point.data(), pointId, turbId_);
}

ActFastAssignVel::ActFastAssignVel(ActuatorBulkFAST& actBulk):
  velocity_(helper_.get_local_view(actBulk.velocity_)),
  offset_(helper_.get_local_view(actBulk.turbIdOffset_)),
  turbId_(actBulk.localTurbineId_),
  fast_(actBulk.openFast_)
{}

void ActFastAssignVel::operator ()(int index) const{

  const int pointId = index - offset_(turbId_);
  auto vel = Kokkos::subview(velocity_, index, Kokkos::ALL);

  fast_.setVelocityForceNode(vel.data(), pointId, turbId_);
}

ActFastComputeForce::ActFastComputeForce(ActuatorBulkFAST& actBulk):
  force_(helper_.get_local_view(actBulk.actuatorForce_)),
  offset_(helper_.get_local_view(actBulk.turbIdOffset_)),
  turbId_(actBulk.localTurbineId_),
  fast_(actBulk.openFast_)
{
  helper_.touch_dual_view(actBulk.actuatorForce_);
}

void ActFastComputeForce::operator()(int index) const{

  auto pointForce = Kokkos::subview(force_, index, Kokkos::ALL);
  const int localId = index - offset_(turbId_);

  fast_.getForce(pointForce.data(), localId, turbId_);
}

ActFastSetUpThrustCalc::ActFastSetUpThrustCalc(ActuatorBulkFAST& actBulk):
    actBulk_(actBulk)
{}

void ActFastSetUpThrustCalc::operator ()(int index) const{
  auto hubLoc = Kokkos::subview(actBulk_.hubLocations_, index, Kokkos::ALL);
  auto hubOri = Kokkos::subview(actBulk_.hubOrientation_, index, Kokkos::ALL);
  auto thrust = Kokkos::subview(actBulk_.turbineThrust_, index, Kokkos::ALL);
  auto torque = Kokkos::subview(actBulk_.turbineTorque_, index, Kokkos::ALL);

  for(int i=0; i<3; i++){
    thrust(i)=0.0;
    torque(i)=0.0;
  }

  if(actBulk_.localTurbineId_ == index){
    actBulk_.openFast_.getHubPos(hubLoc.data(), index);
    actBulk_.openFast_.getHubShftDir(hubOri.data(), index);
  }
  else{
    for(int j=0; j<3; j++){
      hubLoc(j) = 0.0;
      hubOri(j) = 0.0;
    }
  }
}

void ActFastComputeThrustInnerLoop::operator()(const uint64_t pointId, const double* nodeCoords, double* sourceTerm, const double dual_vol, const double scvIp) const
{

  auto offsets = actBulk_.turbIdOffset_.view_host();

  //shouldn't thrust and torque contribs only come from blades?
  //probably not worth worrying about since this is just a debug calculation

  //determine turbine
  int turbId = 0;
  const int nPointId = static_cast<int>(pointId);
  for(;turbId<offsets.extent_int(0); turbId++){
    if(nPointId >= offsets(turbId)){
      break;
    }
  }

  auto hubLoc = Kokkos::subview(actBulk_.hubLocations_,turbId, Kokkos::ALL);
  auto hubOri = Kokkos::subview(actBulk_.hubOrientation_, turbId, Kokkos::ALL);
  auto thrust = Kokkos::subview(actBulk_.turbineThrust_, turbId, Kokkos::ALL);
  auto torque = Kokkos::subview(actBulk_.turbineTorque_, turbId, Kokkos::ALL);

  double r[3], rPerpShaft[3], forceTerm[3];

  for(int i=0; i<3; i++){
    forceTerm[i] = sourceTerm[i]*scvIp;
    r[i] = nodeCoords[i] - hubLoc(i);
    thrust(i) += forceTerm[i];
  }

  double rDotHubOri=0;
  for(int i=0; i<3; i++){
    rDotHubOri += r[i]*hubOri(i);
  }

  for(int i=0; i<3; i++){
    rPerpShaft[i] = r[i] - rDotHubOri*hubOri(i);
  }

  torque(0) += (rPerpShaft[1]*forceTerm[2] - rPerpShaft[2]*forceTerm[1]);
  torque(1) += (rPerpShaft[2]*forceTerm[0] - rPerpShaft[0]*forceTerm[2]);
  torque(2) += (rPerpShaft[0]*forceTerm[1] - rPerpShaft[1]*forceTerm[0]);

}

} /* namespace nalu */
} /* namespace sierra */
