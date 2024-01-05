// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorFunctors.h>
#include <aero/actuator/UtilitiesActuator.h>
#include <stk_mesh/base/BulkData.hpp>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

InterpActuatorVel::InterpActuatorVel(
  ActuatorBulk& actBulk, stk::mesh::BulkData& stkBulk)
  : actBulk_(actBulk),
    stkBulk_(stkBulk),
    coordinates_(stkBulk_.mesh_meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates")),
    velocity_(stkBulk_.mesh_meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity"))
{
  velocity_->sync_to_host();
  actBulk_.velocity_.sync_host();
  actBulk_.velocity_.modify_host();
}

void
InterpActuatorVel::operator()(int index) const
{
  auto vel = actBulk_.velocity_.view_host();
  auto localCoord = actBulk_.localCoords_;

  if (actBulk_.pointIsLocal_(index)) {

    stk::mesh::Entity elem = stkBulk_.get_entity(
      stk::topology::ELEMENT_RANK, actBulk_.elemContainingPoint_(index));

    const int nodesPerElem = stkBulk_.num_nodes(elem);

    // just allocate for largest expected size (hex27)
    double ws_coordinates[81], ws_velocity[81];

    actuator_utils::gather_field(
      3, &ws_coordinates[0], *coordinates_, stkBulk_.begin_nodes(elem),
      nodesPerElem);

    actuator_utils::gather_field_for_interp(
      3, &ws_velocity[0], *velocity_, stkBulk_.begin_nodes(elem), nodesPerElem);

    actuator_utils::interpolate_field(
      3, elem, stkBulk_, &(localCoord(index, 0)), &ws_velocity[0],
      &(vel(index, 0)));
    for (int i = 0; i < 3; i++) {
      vel(index, i) /= actBulk_.localParallelRedundancy_(index);
    }
  }
}

void
SpreadForceInnerLoop::preloop()
{
  actBulk_.actuatorForce_.sync_host();
}

void
SpreadForceInnerLoop::operator()(
  const uint64_t pointId,
  const double* nodeCoords,
  double* sourceTerm,
  const double dual_vol,
  const double scvIp) const
{

  auto pointCoords =
    Kokkos::subview(actBulk_.pointCentroid_.view_host(), pointId, Kokkos::ALL);

  auto pointForce =
    Kokkos::subview(actBulk_.actuatorForce_.view_host(), pointId, Kokkos::ALL);

  auto epsilon =
    Kokkos::subview(actBulk_.epsilon_.view_host(), pointId, Kokkos::ALL);

  double distance[3];
  double projectedForce[3];

  actuator_utils::compute_distance(
    3, nodeCoords, pointCoords.data(), &distance[0]);

  const double gauss =
    actuator_utils::Gaussian_projection(3, &distance[0], epsilon.data());

  for (int j = 0; j < 3; j++) {
    projectedForce[j] = gauss * pointForce(j);
  }

  for (int j = 0; j < 3; j++) {
    sourceTerm[j] += projectedForce[j] * scvIp / dual_vol;
  }
}

} /* namespace nalu */
} /* namespace sierra */
