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
#include <actuator/UtilitiesActuator.h>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

ActuatorMeta::ActuatorMeta(int numTurbines, ActuatorType actuatorType)
  : numberOfActuators_(numTurbines),
    actuatorType_(actuatorType),
    numPointsTotal_(0),
    searchMethod_(stk::search::KDTREE),
    numPointsTurbine_("numPointsTurbine", numberOfActuators_)
{
}

void
ActuatorMeta::add_turbine(const ActuatorInfoNGP& info)
{
  numPointsTurbine_.h_view(info.turbineId_) = info.numPoints_;
  numPointsTotal_ += info.numPoints_;
}

ActuatorBulk::ActuatorBulk(const ActuatorMeta& actMeta)
  : turbIdOffset_("offsetsForTurbine", actMeta.numberOfActuators_),
    pointCentroid_("actPointCentroid", actMeta.numPointsTotal_),
    velocity_("actVelocity", actMeta.numPointsTotal_),
    actuatorForce_("actForce", actMeta.numPointsTotal_),
    epsilon_("actEpsilon", actMeta.numPointsTotal_),
    searchRadius_("searchRadius", actMeta.numPointsTotal_),
    coarseSearchPointIds_("coarseSearchPointIds", 0),
    coarseSearchElemIds_("coarseSearchElemIds", 0),
    localCoords_("localCoords", actMeta.numPointsTotal_),
    pointIsLocal_("pointIsLocal", actMeta.numPointsTotal_),
    localParallelRedundancy_("localParallelReundancy", actMeta.numPointsTotal_),
    elemContainingPoint_("elemContainPoint", actMeta.numPointsTotal_)
{
  compute_offsets(actMeta);
}

void
ActuatorBulk::compute_offsets(const ActuatorMeta& actMeta)
{
  turbIdOffset_.modify_host();

  const int numTurbs = actMeta.numberOfActuators_;

  for (int i = 1; i < numTurbs; ++i) {
    turbIdOffset_.h_view(i) =
      turbIdOffset_.h_view(i - 1) + actMeta.numPointsTurbine_.h_view(i - 1);
  }
}

void
ActuatorBulk::stk_search_act_pnts(
  const ActuatorMeta& actMeta, stk::mesh::BulkData& stkBulk)
{
  auto points = pointCentroid_.template view<ActuatorFixedMemSpace>();
  auto radius = searchRadius_.template view<ActuatorFixedMemSpace>();

  auto boundSpheres = CreateBoundingSpheres(points, radius);
  auto elemBoxes = CreateElementBoxes(stkBulk, actMeta.searchTargetNames_);

  ExecuteCoarseSearch(
    boundSpheres, elemBoxes, coarseSearchPointIds_, coarseSearchElemIds_,
    actMeta.searchMethod_);

  ExecuteFineSearch(
    stkBulk, coarseSearchPointIds_, coarseSearchElemIds_, points,
    elemContainingPoint_, localCoords_, pointIsLocal_,
    localParallelRedundancy_);

  actuator_utils::reduce_view_on_host(localParallelRedundancy_);
}

void
ActuatorBulk::zero_source_terms(stk::mesh::BulkData& stkBulk)
{

  const stk::mesh::MetaData& stkMeta = stkBulk.mesh_meta_data();

  VectorFieldType* actuatorSource = stkMeta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "actuator_source");
  ScalarFieldType* actuatorSourceLhs = stkMeta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "actuator_source_lhs");

  const double zero[3] = {0.0, 0.0, 0.0};

  stk::mesh::field_fill_component(zero, *actuatorSource);
  stk::mesh::field_fill(0.0, *actuatorSourceLhs);
}

void
ActuatorBulk::parallel_sum_source_term(stk::mesh::BulkData& stkBulk)
{

  const stk::mesh::MetaData& stkMeta = stkBulk.mesh_meta_data();
  VectorFieldType* actuatorSource = stkMeta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "actuator_source");

  stk::mesh::parallel_sum(stkBulk, {actuatorSource});
}

} // namespace nalu
} // namespace sierra
