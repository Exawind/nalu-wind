// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include<gtest/gtest.h>
#include<actuator/ActuatorNGP.h>
#include<actuator/ActuatorBulk.h>
#include<actuator/ActuatorInfo.h>

namespace sierra{
namespace nalu{
struct PreIter{};
struct ComputePointLocation{};
struct InterpolateValues{};
struct SpreadForces{};

using ActPreIter = ActuatorFunctor<ActuatorBulk, PreIter             , Kokkos::DefaultHostExecutionSpace>;
using ActCompPnt = ActuatorFunctor<ActuatorBulk, ComputePointLocation, Kokkos::DefaultHostExecutionSpace>;
using ActInterp  = ActuatorFunctor<ActuatorBulk, InterpolateValues   , Kokkos::DefaultHostExecutionSpace>;
using ActSpread  = ActuatorFunctor<ActuatorBulk, SpreadForces        , Kokkos::DefaultHostExecutionSpace>;

template<>
ActPreIter::ActuatorFunctor(ActuatorBulk& bulk):bulk_(bulk){
  bulk_.epsilon_.sync<memory_space>();
  bulk_.epsilon_.modify<memory_space>();
}

template<>
void ActPreIter::operator()(const int& index) const{
  ActVectorDbl epsilon = bulk_.epsilon_.template view<memory_space>();
  epsilon(index,0) = index*3.0;
  epsilon(index,1) = index*6.0;
  epsilon(index,2) = index*9.0;
}

template<>
ActCompPnt::ActuatorFunctor(ActuatorBulk& bulk):bulk_(bulk){
  bulk_.pointCentroid_.sync<memory_space>();
  bulk_.pointCentroid_.modify<memory_space>();
}

template<>
void ActCompPnt::operator()(const int& index) const{
  auto points = bulk_.pointCentroid_.template view<memory_space>();
  points(index,0) = index;
  points(index,1) = index*0.5;
  points(index,2) = index*0.25;
}

template<>
ActInterp::ActuatorFunctor(ActuatorBulk& bulk):bulk_(bulk){
  bulk_.velocity_.sync<memory_space>();
  bulk_.velocity_.modify<memory_space>();
}

template<>
void ActInterp::operator()(const int& index) const{
  auto velocity = bulk_.velocity_.template view<memory_space>();
  velocity(index, 0) = index*2.5;
  velocity(index, 1) = index*5.0;
  velocity(index, 2) = index*7.5;
}

template<>
ActSpread::ActuatorFunctor(ActuatorBulk& bulk):bulk_(bulk){
  bulk_.actuatorForce_.sync<memory_space>();
  bulk_.actuatorForce_.modify<memory_space>();
}

template<>
void ActSpread::operator()(const int& index) const{
  auto force = bulk_.actuatorForce_.template view<memory_space>();
  force(index, 0) = index*3.1;
  force(index, 1) = index*6.2;
  force(index, 2) = index*9.3;
}



template<>
void Actuator<ActuatorMeta, ActuatorBulk>::execute()
  {
    const int nP = actBulk_.totalNumPoints_;

    Kokkos::parallel_for("actPreIter",      nP, ActPreIter(actBulk_));
    Kokkos::parallel_for("actCompPointLoc", nP, ActCompPnt(actBulk_));
    Kokkos::parallel_for("actInterpVals",   nP, ActInterp (actBulk_));
    Kokkos::parallel_for("actSpreadForce",  nP, ActSpread (actBulk_));
  }

template<>
Actuator<ActuatorMeta, ActuatorBulk>::Actuator(ActuatorMeta meta):
  actBulk_(meta)
  {}

using TestActuator = Actuator<ActuatorMeta, ActuatorBulk>;
namespace{
TEST(ActuatorNGP, testExecutionOnHostOnly){
  ActuatorMeta meta(1);
  ActuatorInfoNGP infoTurb0;
  infoTurb0.turbineName_ = "Turbine0";
  infoTurb0.numPoints_ = 20;
  meta.add_turbine(0, infoTurb0);
  TestActuator actuator(meta);
  ASSERT_NO_THROW(actuator.execute());
  const ActuatorBulk& bulk = actuator.actuator_bulk();
  EXPECT_DOUBLE_EQ(3.0, bulk.epsilon_.h_view(1,0));
  EXPECT_DOUBLE_EQ(6.0, bulk.epsilon_.h_view(1,1));
  EXPECT_DOUBLE_EQ(9.0, bulk.epsilon_.h_view(1,2));

  EXPECT_DOUBLE_EQ(1.0, bulk.pointCentroid_.h_view(1,0));
  EXPECT_DOUBLE_EQ(0.5, bulk.pointCentroid_.h_view(1,1));
  EXPECT_DOUBLE_EQ(0.25, bulk.pointCentroid_.h_view(1,2));

  EXPECT_DOUBLE_EQ(2.5, bulk.velocity_.h_view(1,0));
  EXPECT_DOUBLE_EQ(5.0, bulk.velocity_.h_view(1,1));
  EXPECT_DOUBLE_EQ(7.5, bulk.velocity_.h_view(1,2));

  EXPECT_DOUBLE_EQ(3.1, bulk.actuatorForce_.h_view(1,0));
  EXPECT_DOUBLE_EQ(6.2, bulk.actuatorForce_.h_view(1,1));
  EXPECT_DOUBLE_EQ(9.3, bulk.actuatorForce_.h_view(1,2));
}

}

} //namespace nalu
} //namespace sierra
