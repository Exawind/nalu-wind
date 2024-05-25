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

#include <aero/actuator/UtilitiesActuator.h>
#include <aero/actuator/ActuatorTypes.h>
#include <stk_mesh/base/BulkData.hpp>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {


//TODO: Should this be inherited somehow from GenericLoopOverCoarseSearchResults functor?
//TODO: the actuator meta is needed below. Do we have access to it?
template <typename ActuatorBulk, typename functor>
  //coarse search actuatorbulk.c L96
struct GenericLoopOverCoarseTurbineSearchResults
{
  using execution_space = ActuatorFixedExecutionSpace;

  // ctor if functor only requires actBulk for constructor
  GenericLoopOverCoarseTurbineSearchResults(
    ActuatorBulk& actBulk, stk::mesh::BulkData& stkBulk)
    : actBulk_(actBulk),
      stkBulk_(stkBulk),
      coordinates_(stkBulk_.mesh_meta_data().template get_field<double>(
        stk::topology::NODE_RANK, "coordinates")),
      actuatorSource_(stkBulk_.mesh_meta_data().template get_field<double>(
        stk::topology::NODE_RANK, "actuator_source")),
      dualNodalVolume_(stkBulk_.mesh_meta_data().template get_field<double>(
        stk::topology::NODE_RANK, "dual_nodal_volume")),
      innerLoopFunctor_(actBulk)
  {
    actBulk_.coarseSearchElemIds_.sync_host();
    actBulk_.coarseSearchPointIds_.sync_host();
    innerLoopFunctor_.preloop();
  }

  // ctor for functor constructor taking multiple args
  GenericLoopOverCoarseTurbineSearchResults(
    ActuatorBulk& actBulk,
    stk::mesh::BulkData& stkBulk,
    functor innerLoopFunctor)
    : actBulk_(actBulk),
      stkBulk_(stkBulk),
      coordinates_(stkBulk_.mesh_meta_data().template get_field<double>(
        stk::topology::NODE_RANK, "coordinates")),
      actuatorSource_(stkBulk_.mesh_meta_data().template get_field<double>(
        stk::topology::NODE_RANK, "actuator_source")),
      dualNodalVolume_(stkBulk_.mesh_meta_data().template get_field<double>(
        stk::topology::NODE_RANK, "dual_nodal_volume")),
      innerLoopFunctor_(innerLoopFunctor)
  {
    actBulk_.coarseSearchElemIds_.sync_host();
    actBulk_.coarseSearchPointIds_.sync_host();
    innerLoopFunctor_.preloop();
  }

  // see ActuatorExecutorFASTSngp.C line 58
  // TODO: should the index be loop over actuator points or loop over turbine points. I think it would be
  // more efficient to loop over all turbine points and then over each actuator point so index should be 
  // turbine points. 
  //
  // might be able to replace operator with individual cases
  void operator()(int index) const
  {
    auto pointId = actBulk_.coarseSearchPointIds_.h_view(index);
    auto elemId = actBulk_.coarseSearchElemIds_.h_view(index);

    // get element topology 
    const stk::mesh::Entity elem =
      stkBulk_.get_entity(stk::topology::ELEMENT_RANK, elemId);
    const stk::topology& elemTopo = stkBulk_.bucket(elem).topology();
    MasterElement* meSCV =
      MasterElementRepo::get_volume_master_element_on_host(elemTopo);

    // element number of nodes and integration points
    const unsigned numNodes = stkBulk_.num_nodes(elem);
    const int numIp = meSCV->num_integration_points();

    // just allocate for largest expected size (hex27)
    STK_ThrowAssert(numIp <= 216);
    STK_ThrowAssert(numNodes <= 27);

    double scvip[216];
    double elemcoords[27 * 3];
    sierra::nalu::SharedMemView<double*> scvIp(&scvip[0], 216);
    sierra::nalu::SharedMemView<double**> elemCoords(&elemcoords[0], 27, 3);

    stk::mesh::Entity const* elem_nod_rels = stkBulk_.begin_nodes(elem);

    // get element coordinates
    for (unsigned i = 0; i < numNodes; i++) {
      const double* coords =
        stk::mesh::field_data(*coordinates_, elem_nod_rels[i]);
      for (int j = 0; j < 3; j++) {
        elemCoords(i, j) = coords[j];
      }
    }

    meSCV->determinant(elemCoords, scvIp);

    // relationship of element nodes to integration points
    const auto* ipNodeMap = meSCV->ipNodeMap();

    // loop over integration points
    for (int nIp = 0; nIp < numIp; nIp++) {
      const int nodeIndex = ipNodeMap[nIp];
      stk::mesh::Entity node = elem_nod_rels[nodeIndex];
      const double* nodeCoords = stk::mesh::field_data(*coordinates_, node);
      const double dual_vol = *stk::mesh::field_data(*dualNodalVolume_, node);
      double* sourceTerm = stk::mesh::field_data(*actuatorSource_, node);

      //TODO: Need to get access to actMeta for this to work
      //This loop should also be parallelize I would think.. 
      //Option 1:
      //use kokkoss size function to get numPointsTotal from the size of pointCentroids. Cannot use meta here
      for (int actPtInd = 0; actPtInd < actBulk_.pointCentroids_.size(); actPtInd ++){
        //auto pointCoords = Kokkos::subview(actBulk_.pointCentroid_.view_host(), actPtInd, Kokkos::ALL);
      //Option 2:
      //Kokkos::parallel_for("GenericActPtLoop", HostRangePolicy(0, actMeta.numPointsTotal_), [&](int actPtInd) {
      //auto point = pointCentroid_.h_view(actPtInd)  don't need point just index
      //TODO: Is actPtInd the right index?
      //TODO: How can we avoid all the zero integration values if actPtId is not associated with element box? 
      //      Or is this OK because we spread the force to the entire disk
      innerLoopFunctor_(actPtInd, nodeCoords, sourceTerm, dual_vol, scvIp[nIp]);
      //
    }
  }

  ActuatorBulk& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  VectorFieldType* coordinates_;
  VectorFieldType* actuatorSource_;
  ScalarFieldType* dualNodalVolume_;
  functor innerLoopFunctor_;
};

} // namespace nalu
} // namespace sierra

#endif /* ACTUATORGENERICSEARCHFUNCTOR_H_ */
