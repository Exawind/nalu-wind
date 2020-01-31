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

template<>
void ActuatorPreIteration<ActuatorBulk>::operator()(const int& index) const{
  bulk_.epsilon_.h_view(index,0) = index*3.0;
  bulk_.epsilon_.h_view(index,1) = index*6.0;
  bulk_.epsilon_.h_view(index,2) = index*9.0;
}

template<>
void ActuatorComputePointLocation<ActuatorBulk>::operator()(const int& index) const{
  bulk_.pointCentroid_.h_view(index,0) = index;
  bulk_.pointCentroid_.h_view(index,1) = index*0.5;
  bulk_.pointCentroid_.h_view(index,2) = index*0.25;
}

template<>
void ActuatorInterpolateFieldValues<ActuatorBulk>::operator()(const int& index) const{
  bulk_.velocity_.d_view(index, 0) = index*2.5;
  bulk_.velocity_.d_view(index, 1) = index*5.0;
  bulk_.velocity_.d_view(index, 2) = index*7.5;
}

template<>
void ActuatorSpreadForces<ActuatorBulk>::operator()(const int& index) const{
  bulk_.actuatorForce_.d_view(index, 0) = index*3.1;
  bulk_.actuatorForce_.d_view(index, 1) = index*6.2;
  bulk_.actuatorForce_.d_view(index, 2) = index*9.3;
}

template<>
void Actuator<ActuatorMeta, ActuatorBulk>::execute()
  {
    //TODO(psakiev) set execution space i.e. range policy
    const int nP = actBulk_.totalNumPoints_;
    using range_policy_host = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>;
    using range_policy_device = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>;

    Kokkos::parallel_for("actPreIter",      range_policy_host(0,nP), preIteration_);
    actBulk_.epsilon_.modified_host();
    Kokkos::parallel_for("actCompPointLoc", range_policy_host(0,nP), computePointLocation_);
    actBulk_.pointCentroid_.modified_host();
    actBulk_.epsilon_.sync_device();
    actBulk_.pointCentroid_.sync_device();
    Kokkos::parallel_for("actInterpVals",   range_policy_device(0,nP), interpolateFieldValues_);
    actBulk_.velocity_.modified_device();
    Kokkos::parallel_for("actSpreadForce",  range_policy_device(0,nP), spreadForces_);
    actBulk_.actuatorForce_.modified_device();
    actBulk_.velocity_.sync_host();
    actBulk_.actuatorForce_.sync_host();
  }

template<>
Actuator<ActuatorMeta, ActuatorBulk>::Actuator(ActuatorMeta meta):
  actBulk_(meta),
  preIteration_(actBulk_),
  computePointLocation_(actBulk_),
  interpolateFieldValues_(actBulk_),
  spreadForces_(actBulk_),
  postIteration_(actBulk_)
  {}

using TestActuator = Actuator<ActuatorMeta, ActuatorBulk>;
namespace{
TEST(ActuatorNGP, testExecutionOneTurbine){
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
