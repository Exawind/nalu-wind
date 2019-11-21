// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


/** \file NgpCreateElemInstance.h
 *  Utilities to create MasterElement templated instances of element algorithms
 *
 *  The Nalu-Wind NGP design requires element algorithms to be specialized on
 *  MasterElement types using AlgTraits (e.g., AlgTraitsHex8). This header
 *  contains utility functions that ease the creation of specialized instances
 *  for element, face, face-element pair based on topology of the part being
 *  processed. Most functions here will return a pointer to the base type (e.g.,
 *  Algorithm, Kernel, etc.).
 */

#ifndef NGPCREATEELEMINSTANCE_H
#define NGPCREATEELEMINSTANCE_H

#include "AlgTraits.h"
#include "BuildTemplates.h"
#include "element_promotion/ElementDescription.h"

namespace sierra {
namespace nalu {
namespace nalu_ngp {

/** Has the MasterElement for the topolgy fully transitioned to NGP?
 *
 *  A helpful query method to track progress of MasterElement conversions during
 *  instantiation of the algorithm. Useful for choosing legacy algorithms during
 *  transition period.
 *
 *  @param topo Part topology
 *  @return True if the master element is NGP ready, false otherwise.
 */
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

/** Has the sideset MasterElement for the topolgy fully transitioned to NGP?
 *
 *  A helpful query method to track progress of MasterElement conversions during
 *  instantiation of the algorithm. Useful for choosing legacy algorithms during
 *  transition period.
 *
 *  @param topo Part topology
 *  @return True if the master element is NGP ready, false otherwise.
 */
inline bool is_ngp_face(const stk::topology topo)
{
  if (topo.is_super_topology()) return false;

  bool isNGP = false;
  switch (topo.value()) {
  case stk::topology::QUAD_4:
  case stk::topology::TRI_3:
    isNGP = true;
    break;

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

/** Create a higher-order element Algorithm/Kernel
 *
 *  Create a specialized instance of a templated algorithm (or any element based
 *  operation class) and return a pointer to the base class of that algorithm.
 *
 *  @param dimension Dimensionality of the problem
 *  @param args Arguments necessary for the constructor of the algorithm
 *  @return Pointer to the base class of the newly created algorithm
 */
template<typename BaseType, template <typename> class T, int order, typename... Args>
BaseType* create_ho_elem_algorithm(
  const int dimension,
  Args&&... args)
{
  if (dimension == 2)
    return new T<AlgTraitsQuadGL_2D<order>>(std::forward<Args>(args)...);

  return new T<AlgTraitsHexGL<order>>(std::forward<Args>(args)...);
}

/** Create an interior element algorithm for the given topology
 */
template<typename BaseType, template <typename> class T, typename... Args>
BaseType* create_elem_algorithm(
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
      return create_ho_elem_algorithm<BaseType, T, 2>(
        dimension, std::forward<Args>(args)...);
    case 3:
      return create_ho_elem_algorithm<BaseType, T, 3>(
        dimension, std::forward<Args>(args)...);
    case 4:
      return create_ho_elem_algorithm<BaseType, T, 4>(
        dimension, std::forward<Args>(args)...);
    case USER_POLY_ORDER:
      return create_ho_elem_algorithm<BaseType, T, USER_POLY_ORDER>(
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

/** Create a boundary algorithm for a given topology
 */
template<typename BaseType, template <typename> class T, typename... Args>
BaseType* create_face_algorithm(
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

/** Create a boundary algorithm that depends on both the face topology and the
 *  connected element topology.
 */
template<typename BaseType, template <typename> class T, typename... Args>
BaseType* create_face_elem_algorithm(
  const int,
  const stk::topology faceTopo,
  const stk::topology elemTopo,
  Args&&... args)
{
  switch (faceTopo) {
  case stk::topology::QUAD_4:
    switch (elemTopo) {
    case stk::topology::HEX_8:
      return new T<AlgTraitsQuad4Hex8>(std::forward<Args>(args)...);
    case stk::topology::PYRAMID_5:
      return new T<AlgTraitsQuad4Pyr5>(std::forward<Args>(args)...);
    case stk::topology::WEDGE_6:
      return new T<AlgTraitsQuad4Wed6>(std::forward<Args>(args)...);

    default:
      throw std::runtime_error("NGP face_elem algorithm not implemented for QUAD_4 and "
                               + elemTopo.name() + " pair.");
    }

  case stk::topology::TRI_3:
    switch(elemTopo) {
    case stk::topology::TET_4:
      return new T<AlgTraitsTri3Tet4>(std::forward<Args>(args)...);
    case stk::topology::PYRAMID_5:
      return new T<AlgTraitsTri3Pyr5>(std::forward<Args>(args)...);
    case stk::topology::WEDGE_6:
      return new T<AlgTraitsTri3Wed6>(std::forward<Args>(args)...);
    default :
      throw std::runtime_error("NGP face_elem algorithm is not implemented for TRI_3 and "
                               + elemTopo.name() + " pair.");
    }

  case stk::topology::LINE_2:
    switch (elemTopo) {
    case stk::topology::QUAD_4_2D:
      return new T<AlgTraitsEdge2DQuad42D>(std::forward<Args>(args)...);
    default:
      throw std::runtime_error("NGP face_elem algorithm is not implemented for EDGE_2D and "
                               + elemTopo.name() + " pair.");
    }

  default:
    throw std::runtime_error("NGP face_elem algorithm not implemented " + faceTopo.name());
  }
}

}  // nalu_ngp
}  // nalu
}  // sierra

#endif /* NGPCREATEELEMINSTANCE_H */
