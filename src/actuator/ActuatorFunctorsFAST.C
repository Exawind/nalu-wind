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

    double hubPos[3], hubShftDir[3];

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

ActFastComputeThrust::ActFastComputeThrust(ActuatorBulkFAST& actBulk, stk::mesh::BulkData& stkBulk):
    actBulk_(actBulk),stkBulk_(stkBulk)
{}

//TODO(psakiev) fuse this with spread force to reduce loops over search
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
