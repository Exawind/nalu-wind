// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORSCALINGFLLC_H_
#define ACTUATORSCALINGFLLC_H_

#include <actuator/ActuatorBulk.h>
#include <actuator/ActuatorBulkSimple.h>
#include <stdexcept>

namespace sierra{
namespace nalu{
namespace FLLC{

template<typename range_type, typename helper_type>
void scale_lift_force(ActuatorBulk& actBulk, const ActuatorMeta& actMeta, range_type& rangePolicy, helper_type& helper)
{
  if (actMeta.actuatorType_==ActuatorType::ActLineSimpleNGP){
    auto actBulkSimple = dynamic_cast<ActuatorBulkSimple&>(actBulk);
    auto actMetaSimple = dynamic_cast<const ActuatorMetaSimple&>(actMeta);
    auto G = helper.get_local_view(actBulkSimple.liftForceDistribution_);
    auto rho = helper.get_local_view(actBulkSimple.density_);

    const int turbId = actBulkSimple.localTurbineId_;
    double dR = actMetaSimple.dR_.h_view(turbId);

    Kokkos::parallel_for("scale G", rangePolicy, KOKKOS_LAMBDA(int i){
      const double denom = rho(i)*dR;
      for(int j=0; j<3; ++j){
        G(i,j) /= denom;
      }
    });
  }
  else{
    throw std::runtime_error("Unsupported actuator type supplied to the fllc");
  }
}

}}}
#endif /* ACTUATORSCALINGFLLC_H_ */
