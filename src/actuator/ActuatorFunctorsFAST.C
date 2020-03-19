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


//TODO(psakiev) move temporay allocaitons out of functors

template<>
ActFastZero::ActuatorFunctor(ActuatorBulkFAST& actBulk) : actBulk_(actBulk){
  touch_dual_view(actBulk_.velocity_);
  touch_dual_view(actBulk_.actuatorForce_);
  touch_dual_view(actBulk_.pointCentroid_);
}

template<>
void
ActFastZero::operator()(const int& index) const{
  auto vel = get_local_view(actBulk_.velocity_);
  auto force = get_local_view(actBulk_.actuatorForce_);
  auto point = get_local_view(actBulk_.pointCentroid_);
  for(int i =0; i<3; i++){
    vel(index, i)=0.0;
    force(index, i)=0.0;
    point(index, i)=0.0;
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
  const int myId = index - offsets(actBulk_.localTurbineId_);
  // compute location
  std::vector<double> tempCoords(3, 0.0);
  auto rank = actBulk_.localTurbineId_;
  FAST.getForceNodeCoordinates(tempCoords, myId, rank);
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

    std::vector<double> hubPos(3), hubShftDir(3);

    actBulk_.openFast_.getHubPos(hubPos, index);
    actBulk_.openFast_.getHubShftDir(hubShftDir, index);

    for(int j=0; j<3; j++){
      hubLoc(j) = hubPos[j];
      hubOri(j) = hubShftDir[j];
    }
  }
  else{
    for(int j=0; j<3; j++){
      hubLoc(j) = 0.0;
      hubOri(j) = 0.0;
    }
  }
}

ActFastComputeThrust::ActFastComputeThrust(ActuatorBulkFAST& actBulk, stk::mesh::BulkData& stkBulk):
    actBulk_(actBulk),stkBulk_(stkBulk)
{}

void ActFastComputeThrust::operator()(int index) const{

  const stk::mesh::MetaData& stkMeta = stkBulk_.mesh_meta_data();

  VectorFieldType* coordinates = stkMeta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");

  VectorFieldType* actuatorSource = stkMeta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "actuator_source");

 ScalarFieldType* dualNodalVolume = stkMeta.get_field<ScalarFieldType>(
                                      stk::topology::NODE_RANK, "dual_nodal_volume");

  auto offsets = actBulk_.turbIdOffset_.view_host();
  auto pointId = actBulk_.coarseSearchPointIds_.h_view(index);
  auto elemId = actBulk_.coarseSearchElemIds_.h_view(index);

  //determine turbine
  // TODO(psakiev) shouldn't thrust and torque contribs only come from blades?
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

  Kokkos::View<double[3], ActuatorFixedMemLayout, ActuatorFixedMemSpace> r("radius");
  Kokkos::View<double[3], ActuatorFixedMemLayout, ActuatorFixedMemSpace> rPerpShaft("radiusShift");
  Kokkos::View<double[3], ActuatorFixedMemLayout, ActuatorFixedMemSpace> forceTerm("forceTerm");

  //loop over elem's nodes and contribute source terms
  const stk::mesh::Entity elem = stkBulk_.get_entity(stk::topology::ELEMENT_RANK, elemId);
  const stk::topology& elemTopo = stkBulk_.bucket(elem).topology();
  MasterElement* meSCV = MasterElementRepo::get_volume_master_element(elemTopo);

  const int numScvIp = meSCV->num_integration_points();
  const unsigned numNodes = stkBulk_.num_nodes(elem);
  Kokkos::View<double*, ActuatorFixedMemLayout, ActuatorFixedMemSpace> scvElem("scvElem", numScvIp);
  Kokkos::View<double*[3], ActuatorFixedMemLayout, ActuatorFixedMemSpace> elemCoords("elemCoords", numNodes);
  stk::mesh::Entity const* elem_nod_rels = stkBulk_.begin_nodes(elem);

  for(unsigned i = 0; i<numNodes; i++){
    const double* coords = (double*) stk::mesh::field_data(*coordinates, elem_nod_rels[i]);
    for(int j=0; j<3; j++){
      elemCoords(i,j) = coords[j];
    }
  }

  double scvError =0.0;
  meSCV->determinant(1, elemCoords.data(), scvElem.data(), &scvError);

  for(unsigned iNode=0; iNode<numNodes; iNode++){
    stk::mesh::Entity node = elem_nod_rels[iNode];
    const double* nodeCoords =
        (double*) stk::mesh::field_data(*coordinates, node);
    const double dual_vol = *(double*)stk::mesh::field_data(*dualNodalVolume, node);
    double* sourceTerm = (double*) stk::mesh::field_data(*actuatorSource, node);

    for(int i=0; i<3; i++){
      // TODO(psakiev) I thought this should just be scvElem(iNode) since we are
      // integrating but that is ~20x too high
      forceTerm(i) = sourceTerm[i]*scvElem(iNode)/dual_vol;
      r(i) = nodeCoords[i] - hubLoc(i);
      thrust(i) += forceTerm(i);
    }

    double rDotHubOri=0;
    for(int i=0; i<3; i++){
      rDotHubOri += r(i)*hubOri(i);
    }

    for(int i=0; i<3; i++){
      rPerpShaft(i) = r(i) - rDotHubOri*hubOri(i);
    }

    torque(0) += (rPerpShaft(1)*forceTerm(2) - rPerpShaft(2)*forceTerm(1));
    torque(1) += (rPerpShaft(2)*forceTerm(0) - rPerpShaft(0)*forceTerm(2));
    torque(2) += (rPerpShaft(0)*forceTerm(1) - rPerpShaft(1)*forceTerm(0));
  }

}

} /* namespace nalu */
} /* namespace sierra */
