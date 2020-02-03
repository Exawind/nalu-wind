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

} //namespace nalu
} //namespace sierra

#endif // UNITTESTACTUATORNGP_H_
