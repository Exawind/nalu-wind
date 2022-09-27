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

template <typename ActuatorBulk, typename functor>
struct GenericLoopOverCoarseSearchResults
{
  using execution_space = ActuatorFixedExecutionSpace;

  // ctor if functor only requires actBulk for constructor
  GenericLoopOverCoarseSearchResults(
    ActuatorBulk& actBulk, stk::mesh::BulkData& stkBulk)
    : actBulk_(actBulk),
      stkBulk_(stkBulk),
      coordinates_(
        stkBulk_.mesh_meta_data().template get_field<VectorFieldType>(
          stk::topology::NODE_RANK, "coordinates")),
      actuatorSource_(
        stkBulk_.mesh_meta_data().template get_field<VectorFieldType>(
          stk::topology::NODE_RANK, "actuator_source")),
      dualNodalVolume_(
        stkBulk_.mesh_meta_data().template get_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "dual_nodal_volume")),
      innerLoopFunctor_(actBulk)
  {
    actBulk_.coarseSearchElemIds_.sync_host();
    actBulk_.coarseSearchPointIds_.sync_host();
    innerLoopFunctor_.preloop();
  }

  // ctor for functor constructor taking multiple args
  GenericLoopOverCoarseSearchResults(
    ActuatorBulk& actBulk,
    stk::mesh::BulkData& stkBulk,
    functor innerLoopFunctor)
    : actBulk_(actBulk),
      stkBulk_(stkBulk),
      coordinates_(
        stkBulk_.mesh_meta_data().template get_field<VectorFieldType>(
          stk::topology::NODE_RANK, "coordinates")),
      actuatorSource_(
        stkBulk_.mesh_meta_data().template get_field<VectorFieldType>(
          stk::topology::NODE_RANK, "actuator_source")),
      dualNodalVolume_(
        stkBulk_.mesh_meta_data().template get_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "dual_nodal_volume")),
      innerLoopFunctor_(innerLoopFunctor)
  {
    actBulk_.coarseSearchElemIds_.sync_host();
    actBulk_.coarseSearchPointIds_.sync_host();
    innerLoopFunctor_.preloop();
  }

  void operator()(int index) const
  {
    auto pointId = actBulk_.coarseSearchPointIds_.h_view(index);
    auto elemId = actBulk_.coarseSearchElemIds_.h_view(index);

    const stk::mesh::Entity elem =
      stkBulk_.get_entity(stk::topology::ELEMENT_RANK, elemId);
    const stk::topology& elemTopo = stkBulk_.bucket(elem).topology();
    MasterElement* meSCV =
      MasterElementRepo::get_volume_master_element(elemTopo);

    const unsigned numNodes = stkBulk_.num_nodes(elem);
    const int numIp = meSCV->num_integration_points();

    // just allocate for largest expected size (hex27)
    ThrowAssert(numIp <= 216);
    ThrowAssert(numNodes <= 27);

    double scvip[216];
    double elemcoords[27 * 3];
    sierra::nalu::SharedMemView<double*> scvIp(&scvip[0], 216);
    sierra::nalu::SharedMemView<double**> elemCoords(&elemcoords[0], 27, 3);

    stk::mesh::Entity const* elem_nod_rels = stkBulk_.begin_nodes(elem);

    for (unsigned i = 0; i < numNodes; i++) {
      const double* coords =
        (double*)stk::mesh::field_data(*coordinates_, elem_nod_rels[i]);
      for (int j = 0; j < 3; j++) {
        elemCoords(i, j) = coords[j];
      }
    }

    meSCV->determinant(elemCoords, scvIp);

    const auto* ipNodeMap = meSCV->ipNodeMap();

    for (int nIp = 0; nIp < numIp; nIp++) {
      const int nodeIndex = ipNodeMap[nIp];
      stk::mesh::Entity node = elem_nod_rels[nodeIndex];
      const double* nodeCoords =
        (double*)stk::mesh::field_data(*coordinates_, node);
      const double dual_vol =
        *(double*)stk::mesh::field_data(*dualNodalVolume_, node);
      double* sourceTerm =
        (double*)stk::mesh::field_data(*actuatorSource_, node);

      // anything else that is required should be stashed on the functor
      // during functor construction i.e. ActuatorBulk, flags, ActuatorMeta,
      // etc.
      innerLoopFunctor_(pointId, nodeCoords, sourceTerm, dual_vol, scvIp[nIp]);
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
