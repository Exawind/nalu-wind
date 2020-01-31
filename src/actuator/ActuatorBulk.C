// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorInfo.h>
#include <actuator/ActuatorBulk.h>

namespace sierra{
namespace nalu{

namespace{
  int extract_point_total(const ActuatorMeta& meta){
    int total = 0;
    int numActuators = meta.num_actuators();
    for(int i=0; i< numActuators; ++i){
      total+=meta.total_num_points(i);
    }
    return total;
  }
}

ActuatorMeta::ActuatorMeta(int numTurbines):
    numberOfActuators_(numTurbines),
    numPointsTotal_("numPointsTotal", numberOfActuators_)
{}

void ActuatorMeta::add_turbine(int turbineIndex, const ActuatorInfoNGP& info)
{
  numPointsTotal_.h_view(turbineIndex) = info.numPoints_;
}

ActuatorBulk::ActuatorBulk(ActuatorMeta meta):
    actuatorMeta_(meta),
    totalNumPoints_(extract_point_total(actuatorMeta_)),
    pointCentroid_("actPointCentroid", totalNumPoints_,3),
    velocity_("actVelocity", totalNumPoints_,3),
    actuatorForce_("actForce", totalNumPoints_),
    epsilon_("actEpsilon", totalNumPoints_,3)
{
}


}
}
