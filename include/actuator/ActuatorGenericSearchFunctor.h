// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#ifndef ACTUATORGENERICSEARCHFUNCTOR_H_
#define ACTUATORGENERICSEARCHFUNCTOR_H_

#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Parallel.hpp>
#include <actuator/UtilitiesActuator.h>
#include <actuator/ActuatorTypes.h>
#include <stk_mesh/base/BulkData.hpp>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

// Template function for looping over coarse search results
// Specific operations on the elements are done by supplying
// an inner loop functor to operate on the element data
template <typename ActuatorBulk, typename functor>
void GenericLoopOverCoarseSearchResults(
  ActuatorBulk& actBulk,
  stk::mesh::BulkData& stkBulk,
  functor innerLoopFunctor)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  VectorFieldType* coordinates=stkBulk.mesh_meta_data().template get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* actuatorSource=stkBulk.mesh_meta_data().template get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuator_source");
  ScalarFieldType* dualNodalVolume=stkBulk.mesh_meta_data().template get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume");
  helper_.touch_dual_view(actBulk.coarseSearchElemIds_);
  helper_.touch_dual_view(actBulk.coarseSearchPointIds_);
  innerLoopFunctor.preloop();

  auto pointIds = helper_.get_local_view(actBulk.coarseSearchPointIds_);
  auto elemIds = helper_.get_local_view(actBulk.coarseSearchElemIds_);

  const int localSizeCoarseSearch = elemIds.extent_int(0);

  Kokkos::parallel_for("genericLoopOverCoarseSearch",
    Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0,localSizeCoarseSearch),
    [pointIds, elemIds, &stkBulk, coordinates, actuatorSource, dualNodalVolume, innerLoopFunctor](int index){
  auto pointId = pointIds(index);
  auto elemId = elemIds(index);

  const stk::mesh::Entity elem =
    stkBulk.get_entity(stk::topology::ELEMENT_RANK, elemId);
  const stk::topology& elemTopo = stkBulk.bucket(elem).topology();
  MasterElement* meSCV =
    MasterElementRepo::get_volume_master_element(elemTopo);

  const unsigned numNodes = stkBulk.num_nodes(elem);
  const int numIp = meSCV->num_integration_points();

  // just allocate for largest expected size (hex27)
  ThrowAssert(numIp<=216);
  ThrowAssert(numNodes<=27);
  double scvIp[216];
  double elemCoords[81];

  stk::mesh::Entity const* elem_nod_rels = stkBulk.begin_nodes(elem);

  for (unsigned i = 0; i < numNodes; i++) {
    const double* coords =
      (double*)stk::mesh::field_data(*coordinates, elem_nod_rels[i]);
    for (int j = 0; j < 3; j++) {
      elemCoords[j + i * 3] = coords[j];
    }
  }

  double scvError = 0.0;
  meSCV->determinant(1, &elemCoords[0], &scvIp[0], &scvError);

  const auto* ipNodeMap = meSCV->ipNodeMap();

  for (int nIp = 0; nIp < numIp; nIp++) {
    const int nodeIndex = ipNodeMap[nIp];
    stk::mesh::Entity node = elem_nod_rels[nodeIndex];
    const double* nodeCoords =
      (double*)stk::mesh::field_data(*coordinates, node);
    const double dual_vol =
      *(double*)stk::mesh::field_data(*dualNodalVolume, node);
    double* sourceTerm =
      (double*)stk::mesh::field_data(*actuatorSource, node);

    // anything else that is required should be stashed on the functor
    // during functor construction i.e. ActuatorBulk fields, flags, ActuatorMeta fields, etc.
    innerLoopFunctor(pointId, nodeCoords, sourceTerm, dual_vol, scvIp[nIp]);
  }});
}

// specialization for when the inner loop functor 
// only requires ActuatorBulk in its constructor (most cases)
template <typename ActuatorBulk, typename functor>
inline void  GenericLoopOverCoarseSearchResults(
    ActuatorBulk& actBulk, stk::mesh::BulkData& stkBulk)
  {
    GenericLoopOverCoarseSearchResults(actBulk, stkBulk, functor(actBulk));
  }


} // namespace nalu
} // namespace sierra

#endif /* ACTUATORGENERICSEARCHFUNCTOR_H_ */
