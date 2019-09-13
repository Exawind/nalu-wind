/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
#include "element_promotion/ElementDescription.h"

namespace sierra {
namespace nalu {

class Realm;

inline bool is_ngp_element(const stk::topology topo)
{
  if (topo.is_super_topology()) return false;

  bool isNGP = false;
  switch (topo.value()) {
  case stk::topology::HEX_8:
  case stk::topology::TET_4:
  case stk::topology::PYRAMID_5:
  case stk::topology::WEDGE_6:
  case stk::topology::QUAD_4_2D:
    isNGP = true;
    break;

  case stk::topology::TRI_3_2D:
  case stk::topology::HEX_27:
  case stk::topology::QUAD_9_2D:
    isNGP = false;
    break;

  default:
    throw std::logic_error("Invalid element topology provided");
  }

  return isNGP;
}

inline bool is_ngp_face(const stk::topology topo)
{
  if (topo.is_super_topology()) return false;

  bool isNGP = false;
  switch (topo.value()) {
  case stk::topology::QUAD_4:
    isNGP = true;
    break;

  case stk::topology::TRI_3:
  case stk::topology::LINE_2:
  case stk::topology::QUAD_9:
  case stk::topology::LINE_3:
    isNGP = false;
    break;

  default:
    throw std::logic_error("Invalid face topology provided");
  }

  return isNGP;
}

template<template <typename> class T, int order, typename... Args>
Algorithm* create_ho_elem_algorithm(
  const int dimension,
  Args&&... args)
{
  if (dimension == 2)
    return new T<AlgTraitsQuadGL_2D<order>>(std::forward<Args>(args)...);

  return new T<AlgTraitsHexGL<order>>(std::forward<Args>(args)...);
}

template<template <typename> class T, typename... Args>
Algorithm* create_elem_algorithm(
  const int dimension,
  const stk::topology topo,
  Args&&... args)
{
  if (!topo.is_super_topology()) {
    switch (topo.value()) {
    case stk::topology::HEX_8:
      return new T<AlgTraitsHex8>(std::forward<Args>(args)...);
    case stk::topology::HEX_27:
      return new T<AlgTraitsHex27>(std::forward<Args>(args)...);
    case stk::topology::TET_4:
      return new T<AlgTraitsTet4>(std::forward<Args>(args)...);
    case stk::topology::PYRAMID_5:
      return new T<AlgTraitsPyr5>(std::forward<Args>(args)...);
    case stk::topology::WEDGE_6:
      return new T<AlgTraitsWed6>(std::forward<Args>(args)...);
    case stk::topology::QUAD_4_2D:
      return new T<AlgTraitsQuad4_2D>(std::forward<Args>(args)...);
    case stk::topology::QUAD_9_2D:
      return new T<AlgTraitsQuad9_2D>(std::forward<Args>(args)...);
    case stk::topology::TRI_3_2D:
      return new T<AlgTraitsTri3_2D>(std::forward<Args>(args)...);
    default:
      return nullptr;
    }
  } else {
    int poly_order = poly_order_from_topology(dimension, topo);
    switch (poly_order) {
    case 2:
      return create_ho_elem_algorithm<T, 2>(
        dimension, std::forward<Args>(args)...);
    case 3:
      return create_ho_elem_algorithm<T, 3>(
        dimension, std::forward<Args>(args)...);
    case 4:
      return create_ho_elem_algorithm<T, 4>(
        dimension, std::forward<Args>(args)...);
    case USER_POLY_ORDER:
      return create_ho_elem_algorithm<T, USER_POLY_ORDER>(
        dimension, std::forward<Args>(args)...);
    default:
      ThrowRequireMsg(
        false, "Polynomial order" + std::to_string(poly_order) +
                 "is not supported by default.  "
                 "Specify USER_POLY_ORDER and recompile to run.");
      return nullptr;
    }
  }
}

template<template <typename> class T, typename... Args>
Algorithm* create_face_algorithm(
  const int dimension,
  const stk::topology topo,
  Args&&... args)
{
  if (!topo.is_super_topology()) {
    switch (topo.value()) {
    case stk::topology::QUAD_4:
      return new T<AlgTraitsQuad4>(std::forward<Args>(args)...);
    case stk::topology::QUAD_9:
      return new T<AlgTraitsQuad9>(std::forward<Args>(args)...);
    case stk::topology::TRI_3:
      return new T<AlgTraitsTri3>(std::forward<Args>(args)...);
    case stk::topology::LINE_2:
      return new T<AlgTraitsEdge_2D>(std::forward<Args>(args)...);
    case stk::topology::LINE_3:
      return new T<AlgTraitsEdge3_2D>(std::forward<Args>(args)...);
    default:
      return nullptr;
    }
  } else {
    int poly_order = poly_order_from_topology(dimension, topo);
    if (dimension == 2) {
      switch (poly_order) {
      case 2:
        return new T<AlgTraitsEdgeGL<2>>(std::forward<Args>(args)...);
      case 3:
        return new T<AlgTraitsEdgeGL<2>>(std::forward<Args>(args)...);
      case 4:
        return new T<AlgTraitsEdgeGL<2>>(std::forward<Args>(args)...);
      case USER_POLY_ORDER:
        return new T<AlgTraitsEdgeGL<USER_POLY_ORDER>>(
          std::forward<Args>(args)...);
      default:
        return nullptr;
      }
    } else {
      switch (poly_order) {
      case 2:
        return new T<AlgTraitsQuadGL<2>>(std::forward<Args>(args)...);
      case 3:
        return new T<AlgTraitsQuadGL<3>>(std::forward<Args>(args)...);
      case 4:
        return new T<AlgTraitsQuadGL<4>>(std::forward<Args>(args)...);
      case USER_POLY_ORDER:
        return new T<AlgTraitsQuadGL<USER_POLY_ORDER>>(
          std::forward<Args>(args)...);
      default:
        return nullptr;
      }
    }
  }
}

template<template <typename> class T, typename... Args>
Algorithm* create_face_elem_algorithm(
  const int dimension,
  const stk::topology faceTopo,
  const stk::topology elemTopo,
  Args&&... args)
{
  if (dimension == 2) {
    throw std::runtime_error("NGP face_elem algorithm not implemented");
  }

  switch (faceTopo) {
  case stk::topology::QUAD_4:
    switch (elemTopo) {
    case stk::topology::HEX_8:
      return new T<AlgTraitsQuad4Hex8>(std::forward<Args>(args)...);

    default:
      throw std::runtime_error("NGP face_elem algorithm not implemented");
    }

  default:
    throw std::runtime_error("NGP face_elem algorithm not implemented");
  }
}

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
        create_elem_algorithm<ElemAlg>(
          nDim_, topo, realm_, part, std::forward<Args>(args)...));
      NaluEnv::self().naluOutputP0()
        << "Created algorithm = " << algName << std::endl;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  template<
    template <typename> class NGPAlg,
    typename LegacyAlg,
    class... Args>
  void register_elem_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const std::string& algSuffix,
    Args&&... args)
  {
    if (!is_ngp_element(part->topology()))
      register_legacy_algorithm<LegacyAlg>(
        algType, part, algSuffix, std::forward<Args>(args)...);
    else
      register_elem_algorithm<NGPAlg>(
        algType, part, algSuffix, std::forward<Args>(args)...);
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
        create_face_algorithm<FaceAlg>(
          nDim_, topo, realm_, part, std::forward<Args>(args)...));
      NaluEnv::self().naluOutputP0()
        << "Created algorithm = " << algName << std::endl;
    } else {
      it->second->partVec_.push_back(part);
    }
  }

  template<
    template <typename> class NGPAlg,
    typename LegacyAlg,
    class... Args>
  void register_face_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const std::string& algSuffix,
    Args&&... args)
  {
    if (!is_ngp_face(part->topology()))
      register_legacy_algorithm<LegacyAlg>(
        algType, part, algSuffix, std::forward<Args>(args)...);
    else
      register_face_algorithm<NGPAlg>(
        algType, part, algSuffix, std::forward<Args>(args)...);
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

  const int nDim_;
};


}  // nalu
}  // sierra


#endif /* NGPALGDRIVER_H */
