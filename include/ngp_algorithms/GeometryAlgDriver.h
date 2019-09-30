/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef GEOMETRYALGDRIVER_H
#define GEOMETRYALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class Realm;

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
  bool hasWallFunc_{false};
};

}  // nalu
}  // sierra


#endif /* GEOMETRYALGDRIVER_H */
