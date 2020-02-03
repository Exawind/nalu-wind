// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorBulk.h>
#include <actuator/ActuatorInfo.h>

namespace sierra{
namespace nalu{

ActuatorMeta::ActuatorMeta(int numTurbines):
    numberOfActuators_(numTurbines),
    numPointsTotal_(0),
    numPointsTurbine_("numPointsTurbine", numberOfActuators_)
{}

void ActuatorMeta::add_turbine(int turbineIndex, const ActuatorInfoNGP& info)
{
  numPointsTurbine_.h_view(turbineIndex) = info.numPoints_;
  numPointsTotal_+=info.numPoints_;
}

ActuatorBulk::ActuatorBulk(ActuatorMeta meta):
    actuatorMeta_(meta),
    totalNumPoints_(actuatorMeta_.num_points_total()),
    pointCentroid_("actPointCentroid", totalNumPoints_,3),
    velocity_("actVelocity", totalNumPoints_,3),
    actuatorForce_("actForce", totalNumPoints_,3),
    epsilon_("actEpsilon", totalNumPoints_,3)
{
}


}
}
