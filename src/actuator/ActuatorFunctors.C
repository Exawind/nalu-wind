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
  actBulk_.coarseSearchElemIds_.template  sync<memory_space>();
  actBulk_.coarseSearchPointIds_.template sync<memory_space>();
}

template<>
void SpreadActForce::operator ()(const int& index) const{

  const stk::mesh::BulkData& stkBulk = actBulk_.stkBulk_;
  const stk::mesh::MetaData& stkMeta = stkBulk.mesh_meta_data();

  VectorFieldType* coordinates = stkMeta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");

  VectorFieldType* actuatorSource = stkMeta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "actuator_source");

  auto pointId = get_local_view(actBulk_.coarseSearchPointIds_)(index);
  auto elemId = get_local_view(actBulk_.coarseSearchElemIds_)(index);

  auto pointCoords = Kokkos::subview(
    get_local_view(actBulk_.pointCentroid_), pointId, Kokkos::ALL);

  auto pointForce = Kokkos::subview(
    get_local_view(actBulk_.actuatorForce_), pointId, Kokkos::ALL);

  auto epsilon = Kokkos::subview(
    get_local_view(actBulk_.epsilon_), pointId, Kokkos::ALL);

  Kokkos::View<double[3], ActuatorMemLayout, ActuatorMemSpace> distance("distance");
  Kokkos::View<double[3], ActuatorMemLayout, ActuatorMemSpace> projectedForce("projectedForce");

  //TODO ngpmesh
  const stk::mesh::Entity elem = stkBulk.get_entity(stk::topology::ELEMENT_RANK, elemId);
  stk::mesh::Entity const* elem_nod_rels = stkBulk.begin_nodes(elem);
  const unsigned numNodes = stkBulk.num_nodes(elem);

  for(unsigned iNode=0; iNode<numNodes; iNode++){
    stk::mesh::Entity node = elem_nod_rels[iNode];
    const double* nodeCoords =
        (double*) stk::mesh::field_data(*coordinates, node);

    actuator_utils::compute_distance(3, nodeCoords, pointCoords.data(), distance.data());

    const double gauss = actuator_utils::Gaussian_projection(3,distance.data(), epsilon.data());

    for(int j =0; j<3; j++){
      projectedForce(j) = gauss*pointForce(j);
    }

    double* sourceTerm = (double*) stk::mesh::field_data(*actuatorSource, node);
    for(int j=0; j<3; j++){
      sourceTerm[j] += projectedForce(j);
    }

  // TODO(psakiev) thrust contribution
  }

}

} /* namespace nalu */
} /* namespace sierra */
