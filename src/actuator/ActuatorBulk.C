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

ActuatorMeta::ActuatorMeta(int numTurbines, ActuatorType actuatorType):
    numberOfActuators_(numTurbines),
    actuatorType_(actuatorType),
    numPointsTotal_(0),
    searchMethod_(stk::search::KDTREE),
    numPointsTurbine_("numPointsTurbine", numberOfActuators_),
    stkBulk_(nullptr)
{}

void ActuatorMeta::add_turbine(const ActuatorInfoNGP& info)
{
  numPointsTurbine_.h_view(info.turbineId_) = info.numPoints_;
  numPointsTotal_+=info.numPoints_;
}

ActuatorBulk::ActuatorBulk(ActuatorMeta meta):
    actuatorMeta_(meta),
    totalNumPoints_(actuatorMeta_.numPointsTotal_),
    pointCentroid_("actPointCentroid", totalNumPoints_),
    velocity_("actVelocity", totalNumPoints_),
    actuatorForce_("actForce", totalNumPoints_),
    epsilon_("actEpsilon", totalNumPoints_),
    searchRadius_("searchRadius", totalNumPoints_),
    localCoords_("localCoords", totalNumPoints_),
    pointIsLocal_("pointIsLocal", totalNumPoints_),
    elemContainingPoint_("elemContainPoint", totalNumPoints_)
{
}

void SearchForActuatorPoints(ActuatorBulk& actBulk){
  auto points = actBulk.pointCentroid_.template view<Kokkos::HostSpace>();
  auto radius = actBulk.searchRadius_.template view<Kokkos::HostSpace>();

  auto boundSpheres = CreateBoundingSpheres(points, radius);
  if(actBulk.actuatorMeta_.stkBulk_==nullptr){
    throw std::runtime_error("Stk search called on actuator meta data without stk::mesh::bulk data");
  }
  stk::mesh::BulkData& stkBulk = *(actBulk.actuatorMeta_.stkBulk_);
  auto elemBoxes = CreateElementBoxes(stkBulk, actBulk.actuatorMeta_.searchTargetNames_);
  actBulk.coarseSearchResults_ = ExecuteCoarseSearch(boundSpheres, elemBoxes, actBulk.actuatorMeta_.searchMethod_);
  actBulk.pointIsLocal_ = ExecuteFineSearch(
    stkBulk,
    actBulk.coarseSearchResults_,
    points,
    actBulk.elemContainingPoint_,
    actBulk.localCoords_);
}

}
}
