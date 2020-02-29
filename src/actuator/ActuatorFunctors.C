// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorFunctors.h>
#include <actuator/UtilitiesActuator.h>
#include <FieldTypeDef.h>
#include <stk_mesh/base/BulkData.hpp>

namespace sierra
{
namespace nalu
{

template <>
InterpolateActVel::ActuatorFunctor(ActuatorBulk& actBulk)
  : actBulk_(actBulk)
{
  touch_dual_view(actBulk_.velocity_);
}

template <>
void
InterpolateActVel::operator()(const int& index) const
{
  auto vel = get_local_view(actBulk_.velocity_);
  auto localCoord = actBulk_.localCoords_;

  if (actBulk_.pointIsLocal_(index)) {
    const stk::mesh::BulkData& stkBulk = actBulk_.stkBulk_;

    stk::mesh::Entity elem = stkBulk.get_entity(
      stk::topology::ELEMENT_RANK, actBulk_.elemContainingPoint_(index));

    const int nodesPerElem = stkBulk.num_nodes(elem);

    std::vector<double> ws_coordinates(3 * nodesPerElem),
      ws_velocity(3 * nodesPerElem);

    VectorFieldType* coordinates =
      stkBulk.mesh_meta_data().get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates");

    VectorFieldType* velocity =
      stkBulk.mesh_meta_data().get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "velocity");

    actuator_utils::gather_field(
      3, &ws_coordinates[0], *coordinates, stkBulk.begin_nodes(elem),
      nodesPerElem);

    actuator_utils::gather_field_for_interp(
      3, &ws_velocity[0], *velocity, stkBulk.begin_nodes(elem), nodesPerElem);

    actuator_utils::interpolate_field(
      3, elem, stkBulk, &(localCoord(index, 0)), &ws_velocity[0],
      &(vel(index, 0)));
  }
}

template<>
SpreadActForce::ActuatorFunctor(ActuatorBulk& actBulk) : actBulk_(actBulk){
  touch_dual_view(actBulk_.actuatorForce_);
}

template<>
void SpreadActForce::operator ()(const int& index) const{

}

} /* namespace nalu */
} /* namespace sierra */
