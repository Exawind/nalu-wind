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
    numPointsTurbine_("numPointsTurbine", numberOfActuators_)
{}

void ActuatorMeta::add_turbine(const ActuatorInfoNGP& info)
{
  numPointsTurbine_.h_view(info.turbineId_) = info.numPoints_;
  numPointsTotal_+=info.numPoints_;
}

ActuatorBulk::ActuatorBulk(const ActuatorMeta& meta, stk::mesh::BulkData& stkBulk):
    totalNumPoints_(meta.numPointsTotal_),
    pointCentroid_("actPointCentroid", totalNumPoints_),
    velocity_("actVelocity", totalNumPoints_),
    actuatorForce_("actForce", totalNumPoints_),
    epsilon_("actEpsilon", totalNumPoints_),
    searchRadius_("searchRadius", totalNumPoints_),
    stkBulk_(stkBulk),
    localCoords_("localCoords", totalNumPoints_),
    pointIsLocal_("pointIsLocal", totalNumPoints_),
    elemContainingPoint_("elemContainPoint", totalNumPoints_)
{
}

void ActuatorBulk::stk_search_act_pnts(const ActuatorMeta& actMeta){
  auto points = pointCentroid_.template view<Kokkos::HostSpace>();
  auto radius = searchRadius_.template view<Kokkos::HostSpace>();

  auto boundSpheres = CreateBoundingSpheres(points, radius);
  auto elemBoxes = CreateElementBoxes(stkBulk_, actMeta.searchTargetNames_);
  coarseSearchResults_ = ExecuteCoarseSearch(boundSpheres, elemBoxes, actMeta.searchMethod_);
  pointIsLocal_ = ExecuteFineSearch(
    stkBulk_,
    coarseSearchResults_,
    points,
    elemContainingPoint_,
    localCoords_);
}

}
}
