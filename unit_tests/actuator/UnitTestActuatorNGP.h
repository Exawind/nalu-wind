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
#include <UnitTestUtils.h>

namespace sierra{
namespace nalu{
struct PreIter{};
struct ComputePointLocation{};
struct InterpolateValues{};
struct SpreadForces{};
struct PostIter{};

// host only examples
using ActPreIter = ActuatorFunctor<ActuatorBulk, PreIter             , Kokkos::DefaultHostExecutionSpace>;
using ActCompPnt = ActuatorFunctor<ActuatorBulk, ComputePointLocation, Kokkos::DefaultHostExecutionSpace>;
using ActInterp  = ActuatorFunctor<ActuatorBulk, InterpolateValues   , Kokkos::DefaultHostExecutionSpace>;
using ActSpread  = ActuatorFunctor<ActuatorBulk, SpreadForces        , Kokkos::DefaultHostExecutionSpace>;

template<>
ActPreIter::ActuatorFunctor(ActuatorBulk& bulk):actBulk_(bulk){
  //TODO(psakiev) it should probably be a feature of the bulk data
  // to recognize modification so users don't need to track this
  actBulk_.epsilon_.sync  <memory_space>();
  actBulk_.epsilon_.modify<memory_space>();
}

template<>
void ActPreIter::operator()(const int& index) const{
  auto epsilon = actBulk_.epsilon_.template view<memory_space>();
  epsilon(index,0) = index*3.0;
  epsilon(index,1) = index*6.0;
  epsilon(index,2) = index*9.0;
}

template<>
ActCompPnt::ActuatorFunctor(ActuatorBulk& bulk):actBulk_(bulk){
  actBulk_.pointCentroid_.sync  <memory_space>();
  actBulk_.pointCentroid_.modify<memory_space>();
}

template<>
void ActCompPnt::operator()(const int& index) const{
  auto points = actBulk_.pointCentroid_.template view<memory_space>();
  points(index,0) = index;
  points(index,1) = index*0.5;
  points(index,2) = index*0.25;
}

template<>
ActInterp::ActuatorFunctor(ActuatorBulk& bulk):actBulk_(bulk){
  actBulk_.velocity_.sync  <memory_space>();
  actBulk_.velocity_.modify<memory_space>();
}

template<>
void ActInterp::operator()(const int& index) const{
  auto velocity = actBulk_.velocity_.template view<memory_space>();
  velocity(index, 0) = index*2.5;
  velocity(index, 1) = index*5.0;
  velocity(index, 2) = index*7.5;
}

template<>
ActSpread::ActuatorFunctor(ActuatorBulk& bulk):actBulk_(bulk){
  actBulk_.actuatorForce_.sync  <memory_space>();
  actBulk_.actuatorForce_.modify<memory_space>();
}

template<>
void ActSpread::operator()(const int& index) const{
  auto force = actBulk_.actuatorForce_.template view<memory_space>();
  force(index, 0) = index*3.1;
  force(index, 1) = index*6.2;
  force(index, 2) = index*9.3;
}


using TestActuatorHostOnly = Actuator<ActuatorMeta, ActuatorBulk>;
template<>
void TestActuatorHostOnly::execute()
{
  const int nP = actBulk_.totalNumPoints_;

  Kokkos::parallel_for("actPreIter",      nP, ActPreIter(actBulk_));
  Kokkos::parallel_for("actCompPointLoc", nP, ActCompPnt(actBulk_));
  Kokkos::parallel_for("actInterpVals",   nP, ActInterp (actBulk_));
  Kokkos::parallel_for("actSpreadForce",  nP, ActSpread (actBulk_));
}

//Create a different bulk data that will allow execution on device and host
//for functors
struct ActuatorBulkMod : public ActuatorBulk{
  ActuatorBulkMod(ActuatorMeta meta):
    ActuatorBulk(meta),
    scalar_("scalar", totalNumPoints_){}
  ActScalarDblDv scalar_;
};
//-----------------------------------------------------------------
//host or device functor example
using ActPostIter= ActuatorFunctor<ActuatorBulkMod, PostIter, ActuatorExecutionSpace>;

template<>
ActPostIter::ActuatorFunctor(ActuatorBulkMod& bulk):actBulk_(bulk){
  actBulk_.scalar_.sync  <memory_space>();
  actBulk_.scalar_.modify<memory_space>();
}

template<>
void ActPostIter::operator()(const int& index) const{
  auto scalar = actBulk_.scalar_.template view<memory_space>();
  auto vel = actBulk_.velocity_.template  view<memory_space>();
  auto point = actBulk_.pointCentroid_.template view<memory_space>();
  scalar(index) = point(index, 0) * vel(index, 1);
}

using TestActuatorHostDev = Actuator<ActuatorMeta, ActuatorBulkMod>;
template<>
void TestActuatorHostDev::execute()
{
  const int nP = actBulk_.totalNumPoints_;
  Kokkos::parallel_for("actPreIter",      nP, ActPreIter (actBulk_));
  Kokkos::parallel_for("actCompPointLoc", nP, ActCompPnt (actBulk_));
  Kokkos::parallel_for("actInterpVals",   nP, ActInterp  (actBulk_));
  Kokkos::parallel_for("actSpreadForce",  nP, ActSpread  (actBulk_));
  Kokkos::parallel_for("actPostIter",     nP, ActPostIter(actBulk_));
  actBulk_.scalar_.sync_device();

}

//-----------------------------------------------------------------
struct SetPoints{};
struct ComputeForce{};
struct Interpolate{};

struct ActuatorBulkSearchAndInterp : public ActuatorBulk{
  ActuatorBulkSearchAndInterp(ActuatorMeta actMeta): ActuatorBulk(actMeta){}
};

using SetupActPoints = ActuatorFunctor<ActuatorBulkSearchAndInterp,SetPoints, ActuatorExecutionSpace>;
template<>
SetupActPoints::ActuatorFunctor(ActuatorBulkSearchAndInterp& actBulk):actBulk_(actBulk){
  actBulk_.pointCentroid_.sync  <memory_space>();
  actBulk_.searchRadius_.sync   <memory_space>();
  actBulk_.pointCentroid_.modify<memory_space>();
  actBulk_.searchRadius_.modify <memory_space>();
}

template<>
void SetupActPoints::operator()(const int& index) const{
  auto point = actBulk_.pointCentroid_.template view<memory_space>();
  auto radius = actBulk_.searchRadius_.template view<memory_space>();
  point(index, 0) = 1.0 + 1.5*index;
  point(index, 1) = 2.5;
  point(index, 2) = 2.5;
  radius(index) = 2.0;
}

using ComputeActuatorForce = ActuatorFunctor<ActuatorBulkSearchAndInterp, ComputeForce, ActuatorExecutionSpace>;
template<>
ComputeActuatorForce::ActuatorFunctor(ActuatorBulkSearchAndInterp& actBulk): actBulk_(actBulk){
  actBulk_.actuatorForce_.sync  <memory_space>();
  actBulk_.actuatorForce_.modify<memory_space>();
}

template<>
void ComputeActuatorForce::operator()(const int& index) const{
  auto force = actBulk_.actuatorForce_.template view<memory_space>();
  auto velocity = actBulk_.velocity_.template view<memory_space>();
  for(int j =0; j<3; j++){
    force(index,j) = 1.2 * velocity(index, j);
  }
}

using InterpolateActVel = ActuatorFunctor<ActuatorBulkSearchAndInterp,Interpolate, Kokkos::DefaultHostExecutionSpace>;
template<>
InterpolateActVel::ActuatorFunctor(ActuatorBulkSearchAndInterp& actBulk): actBulk_(actBulk){
  actBulk_.velocity_.sync<memory_space>();
  actBulk_.velocity_.modify<memory_space>();
}

template<>
void InterpolateActVel::operator()(const int& index) const {
  // create shorter alias could use h_view since restricted operation to host
  auto vel = actBulk_.velocity_.template view<memory_space>();
  auto localCoord = actBulk_.localCoords_;
  if(actBulk_.pointIsLocal_(index)){
    stk::mesh::BulkData& stkBulk = *(actBulk_.actuatorMeta_.stkBulk_);
    stk::mesh::Entity elem = stkBulk.get_entity(stk::topology::ELEMENT_RANK,actBulk_.elemContainingPoint_(index));
    const int nodesPerElem = stkBulk.num_nodes(elem);
    std::vector<double> ws_coordinates(3*nodesPerElem), ws_velocity(3*nodesPerElem);
    VectorFieldType* coordinates = stkBulk.mesh_meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
    VectorFieldType* velocity = stkBulk.mesh_meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
    actuator_utils::gather_field(3, &ws_coordinates[0], *coordinates, stkBulk.begin_nodes(elem), nodesPerElem);
    actuator_utils::gather_field_for_interp(3, &ws_velocity[0], *velocity, stkBulk.begin_nodes(elem), nodesPerElem);
    actuator_utils::interpolate_field(3, elem, stkBulk, &(localCoord(index,0)), &ws_velocity[0], &(vel(index,0)) );
  }

}

using TestActuatorSearchInterp = Actuator<ActuatorMeta, ActuatorBulkSearchAndInterp>;
template<>
// show how to interweave functions and ActuatorFunctors
void TestActuatorSearchInterp::execute()
{
  const int nP = actBulk_.totalNumPoints_;
  Kokkos::parallel_for("setPointLocations", nP, SetupActPoints(actBulk_));
  SearchForActuatorPoints(actBulk_); // TODO(psakiev) probably just make this a member function
  Kokkos::fence();
  Kokkos::parallel_for("interpVel", nP, InterpolateActVel(actBulk_));
  double* reducePointer = &(actBulk_.velocity_.h_view(0,0));
  MPI_Allreduce(reducePointer, reducePointer, nP*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  Kokkos::fence();
  Kokkos::parallel_for("computeActuatorForce", nP, ComputeActuatorForce(actBulk_));

}

} //namespace nalu
} //namespace sierra

#endif // UNITTESTACTUATORNGP_H_
