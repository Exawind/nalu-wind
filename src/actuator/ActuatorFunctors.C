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
#include <stk_mesh/base/BulkData.hpp>
#include <FieldTypeDef.h>

namespace sierra
{
namespace nalu
{

InterpActuatorVel::InterpActuatorVel(ActuatorBulk& actBulk, stk::mesh::BulkData& stkBulk):
    actBulk_(actBulk),
    stkBulk_(stkBulk),
    coordinates_(stkBulk_.mesh_meta_data().get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates")),
    velocity_(stkBulk_.mesh_meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity"))
{
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
    double ws_coordinates[81],
      ws_velocity[81];

    actuator_utils::gather_field(
      3, &ws_coordinates[0], *coordinates_, stkBulk_.begin_nodes(elem),
      nodesPerElem);

    actuator_utils::gather_field_for_interp(
      3, &ws_velocity[0], *velocity_, stkBulk_.begin_nodes(elem), nodesPerElem);

    actuator_utils::interpolate_field(
      3, elem, stkBulk_, &(localCoord(index, 0)), &ws_velocity[0],
      &(vel(index, 0)));
    for(int i=0; i<3; i++){
      vel(index,i)/=actBulk_.localParallelRedundancy_(index);
    }
  }
}

SpreadActuatorForce::SpreadActuatorForce(ActuatorBulk& actBulk, stk::mesh::BulkData& stkBulk):
    actBulk_(actBulk),
    stkBulk_(stkBulk),
    coordinates_(stkBulk_.mesh_meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates")),
    actuatorSource_(stkBulk_.mesh_meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuator_source")),
    dualNodalVolume_(stkBulk_.mesh_meta_data().get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume"))
{
  actBulk_.actuatorForce_.sync_host();
  actBulk_.actuatorForce_.modify_host();
  actBulk_.coarseSearchElemIds_.sync_host();
  actBulk_.coarseSearchPointIds_.sync_host();
}

void SpreadActuatorForce::operator ()(int index) const{

  auto pointId = actBulk_.coarseSearchPointIds_.h_view(index);
  auto elemId = actBulk_.coarseSearchElemIds_.h_view(index);

  auto pointCoords = Kokkos::subview(
    actBulk_.pointCentroid_.view_host(), pointId, Kokkos::ALL);

  auto pointForce = Kokkos::subview(
    actBulk_.actuatorForce_.view_host(), pointId, Kokkos::ALL);

  auto epsilon = Kokkos::subview(
    actBulk_.epsilon_.view_host(), pointId, Kokkos::ALL);

  const stk::mesh::Entity elem = stkBulk_.get_entity(stk::topology::ELEMENT_RANK, elemId);
  const stk::topology& elemTopo = stkBulk_.bucket(elem).topology();
  MasterElement* meSCV = MasterElementRepo::get_volume_master_element(elemTopo);

  const unsigned numNodes = stkBulk_.num_nodes(elem);

  // just allocate for largest expected size (hex27)
  double scvElem[216];
  double elemCoords[81];
  double distance[3];
  double projectedForce[3];

  stk::mesh::Entity const* elem_nod_rels = stkBulk_.begin_nodes(elem);

  for(unsigned i = 0; i<numNodes; i++){
    const double* coords = (double*) stk::mesh::field_data(*coordinates_, elem_nod_rels[i]);
    for(int j=0; j<3; j++){
      elemCoords[j+i*3] = coords[j];
    }
  }

  double scvError =0.0;
  meSCV->determinant(1, &elemCoords[0], &scvElem[0], &scvError);

  for(unsigned iNode=0; iNode<numNodes; iNode++){
    stk::mesh::Entity node = elem_nod_rels[iNode];
    const double* nodeCoords =
        (double*) stk::mesh::field_data(*coordinates_, node);
    const double dual_vol = *(double*)stk::mesh::field_data(*dualNodalVolume_, node);

    // TODO(psakiev) cache distance in search so we can modify it for projected distance in an openfast loop
    actuator_utils::compute_distance(3, nodeCoords, &pointCoords[0], &distance[0]);

    const double gauss = actuator_utils::Gaussian_projection(3, &distance[0], epsilon.data());

    for(int j =0; j<3; j++){
      projectedForce[j] = gauss*pointForce(j);
    }

    double* sourceTerm = (double*) stk::mesh::field_data(*actuatorSource_, node);
    for(int j=0; j<3; j++){
      sourceTerm[j] += projectedForce[j]*scvElem[iNode]/dual_vol;
    }
  }

}

} /* namespace nalu */
} /* namespace sierra */
