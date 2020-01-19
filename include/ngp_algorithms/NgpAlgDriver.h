// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef NGPALGDRIVER_H
#define NGPALGDRIVER_H

#include <map>
#include <memory>
#include <string>

#include "Algorithm.h"
#include "AlgTraits.h"
#include "BuildTemplates.h"
#include "Enums.h"
#include "nalu_make_unique.h"
#include "NaluEnv.h"
#include "ngp_utils/NgpCreateElemInstance.h"

namespace sierra {
namespace nalu {

class Realm;

class NgpAlgDriver
{
public:
  NgpAlgDriver(Realm&);

  virtual ~NgpAlgDriver() = default;

  //! Tasks to be executed before executing the registered algorithms
  virtual void pre_work();

  //! Tasks to be performed after executing registered algorithms
  virtual void post_work();

  /** Execute all the algorithms registered to this driver
   *
   */
  virtual void execute();

  /** Register an edge algorithm
   *
   *  Currently only interior algorithms can be edge algorithms
   *
   *  @param algType Type of algorithm being registered (e.g., INLET, WALL, OPEN)
   *  @param part A valid part that over which this algorithm is applied algorithm
   *
   */
  template <typename EdgeAlg, class... Args>
  void register_edge_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const std::string& algSuffix,
    Args&&... args)
  {
    const std::string algName = unique_name(
      algType, "edge", algSuffix);

    register_algorithm_impl<EdgeAlg>(part, algName, std::forward<Args>(args)...);
  }

  /** Register a legacy algorithm
   *
   *  @param algType Type of algorithm being registered (e.g., INLET, WALL, OPEN)
   *  @param part A valid part that over which this algorithm is applied algorithm
   *
   */
  template <typename Algorithm, class... Args>
  void register_legacy_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const std::string& algSuffix,
    Args&&... args)
  {
    const std::string entityName = "algorithm";
    const std::string algName = unique_name(
      algType, entityName, algSuffix);

    register_algorithm_impl<Algorithm>(part, algName, std::forward<Args>(args)...);
  }

  /** Register an element algorithm
   *
   *  @param algType Type of algorithm being registered (e.g., INLET, WALL, OPEN)
   *  @param part A valid part that over which this algorithm is applied
   *
   */
  template<template <typename> class ElemAlg, class... Args>
  void register_elem_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const std::string& algSuffix,
    Args&&... args)
  {
    const auto topo = part->topology();
    const std::string entityName = "elem_" + topo.name();

    const std::string algName =
      unique_name(algType, entityName, algSuffix);

    const auto it = algMap_.find(algName);
    if (it == algMap_.end()) {
      algMap_[algName].reset(
        nalu_ngp::create_elem_algorithm<Algorithm, ElemAlg>(topo, realm_, part, std::forward<Args>(args)...));
      NaluEnv::self().naluOutputP0()
        << "Created algorithm = " << algName << std::endl;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  /** Register an face algorithm
   *
   *  @param algType Type of algorithm being registered (e.g., INLET, WALL, * OPEN)
   *  @param part A valid part that over which this algorithm is applied
   *
   */
  template <template <typename> class FaceAlg, class... Args>
  void register_face_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const std::string& algSuffix,
    Args&&... args)
  {
    const auto topo = part->topology();
    const std::string entityName = "face_" + topo.name();

    const std::string algName =
      unique_name(algType, entityName, algSuffix);

    const auto it = algMap_.find(algName);
    if (it == algMap_.end()) {
      algMap_[algName].reset(
        nalu_ngp::create_face_algorithm<Algorithm, FaceAlg>(topo, realm_, part, std::forward<Args>(args)...));
      NaluEnv::self().naluOutputP0()
        << "Created algorithm = " << algName << std::endl;
    } else {
      it->second->partVec_.push_back(part);
    }
  }

  template <template <typename> class FaceElemAlg, class... Args>
  void register_face_elem_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const stk::topology elemTopo,
    const std::string& algSuffix,
    Args&&... args)
  {
    const auto topo = part->topology();
    const std::string entityName = "face_" + topo.name() + "_" + elemTopo.name();

    const std::string algName =
      unique_name(algType, entityName, algSuffix);

    const auto it = algMap_.find(algName);
    if (it == algMap_.end()) {
      algMap_[algName].reset(
        nalu_ngp::create_face_elem_algorithm<Algorithm, FaceElemAlg>(topo, elemTopo, realm_, part,
          std::forward<Args>(args)...));
      NaluEnv::self().naluOutputP0()
        << "Created algorithm = " << algName << std::endl;
    } else {
      it->second->partVec_.push_back(part);
    }
  }

protected:
  template <typename NaluAlg, class... Args>
  void register_algorithm_impl(
    stk::mesh::Part* part,
    const std::string& algName,
    Args&&... args)
  {
    const auto it = algMap_.find(algName);
    if (it == algMap_.end()) {
      algMap_[algName].reset(
        new NaluAlg(realm_, part, std::forward<Args>(args)...));
      NaluEnv::self().naluOutputP0()
        << "Created algorithm = " << algName << std::endl;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  //! Return a unique name for the algorithms being registered
  std::string unique_name(AlgorithmType algType,
                          std::string entityType,
                          std::string algName);

  //! Algorithms registered
  std::map<std::string, std::unique_ptr<Algorithm>> algMap_;

  Realm& realm_;
};


}  // nalu
}  // sierra


#endif /* NGPALGDRIVER_H */
