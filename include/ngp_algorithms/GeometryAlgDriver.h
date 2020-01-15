// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef GEOMETRYALGDRIVER_H
#define GEOMETRYALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class Realm;

/** Compute geometry fields
 *
 *  This class coordinates the computation of the following fields:
 *    - dual_nodal_volume
 *    - element_volume
 *    - edge_area_vector
 *    - exposed_area_vector
 *    - assembled_wall_area_wf
 *    - assembled_wall_normal_distance
 *
 *  While the volume computation happens at every invocation of the execute()
 *  method, the remaining fields are only computed if the user has requested
 *  certain options in the input file. The design follows a driver/algorithm
 *  pattern as the actual NGP computation of the fields for elements/edges/faces
 *  etc. are done using algorithms that are templated on the element topologies.
 *  See GeometryInteriorAlg and GeometryBoundaryAlg for more details of the
 *  exact computations.
 *
 *  \sa GeometryInteriorAlg, GeometryBoundaryAlg
 */
class GeometryAlgDriver : public NgpAlgDriver
{
public:
  GeometryAlgDriver(Realm&);

  virtual ~GeometryAlgDriver() = default;

  //! Reset fields before calling algorithms
  virtual void pre_work() override;

  //! Synchronize fields after algorithms have done their work
  virtual void post_work() override;

  /** Register wall function geometry calculation algorithm
   *
   *  Need a specialization here to track whether the user has requested wall functions
   */
  template<template <typename> class FaceElemAlg, class... Args>
  void register_wall_func_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const stk::topology elemTopo,
    const std::string& algSuffix,
    Args&&... args)
  {
    register_face_elem_algorithm<FaceElemAlg>(
      algType, part, elemTopo, algSuffix, std::forward<Args>(args)...);
    hasWallFunc_ = true;
  }

private:
  //! Flag to track whether wall functions are active
  bool hasWallFunc_{false};
};

}  // nalu
}  // sierra


#endif /* GEOMETRYALGDRIVER_H */
