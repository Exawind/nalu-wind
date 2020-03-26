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

  return true;
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

  return true;
}

/** Create an interior element algorithm for the given topology
 */
template<typename BaseType, template <typename> class T, typename... Args>
BaseType* create_elem_algorithm(
  const stk::topology topo,
  Args&&... args)
{
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
      throw std::runtime_error(
        "NGP elem algorithm not implemented for " + topo.name());
  }
}

/** Create a boundary algorithm for a given topology
 */
template<typename BaseType, template <typename> class T, typename... Args>
BaseType* create_face_algorithm(
  const stk::topology topo,
  Args&&... args)
{
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
      throw std::runtime_error(
        "NGP face algorithm not implemented for " + topo.name());
  }
}

/** Create a boundary algorithm that depends on both the face topology and the
 *  connected element topology.
 */
template<typename BaseType, template <typename> class T, typename... Args>
BaseType* create_face_elem_algorithm(
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
    case stk::topology::TRI_3_2D:
      return new T<AlgTraitsEdge2DTri32D>(std::forward<Args>(args)...);
    default:
      throw std::runtime_error("NGP face_elem algorithm is not implemented for EDGE_2D and "
                               + elemTopo.name() + " pair.");
    }

  case stk::topology::QUAD_9:
    return new T<AlgTraitsQuad9Hex27>(std::forward<Args>(args)...);

  default:
    throw std::runtime_error("NGP face_elem algorithm not implemented " + faceTopo.name());
  }
}

}  // nalu_ngp
}  // nalu
}  // sierra

#endif /* NGPCREATEELEMINSTANCE_H */
