/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NGPTYPES_H
#define NGPTYPES_H

/** \file
 *  \brief Nalu-Wind custom types for NGP execution
 */

#include "ngp_utils/NgpMeshInfo.h"
#include "SimdInterface.h"

#include <memory>

namespace sierra {
namespace nalu {

template<typename T1, typename T2, typename T3>
class ScratchViews;

namespace nalu_ngp {

/** Lightweight data structure holding information of the entity in mesh loops
 *
 *  The bucket loop wrappers provide this object to the lambda function for use
 *  during iterations.
 *
 *  Depending on the bucket-loop, entity could be one of elem, edge, or face.
 */
template<typename Mesh = ngp::Mesh>
struct EntityInfo
{
  //! Bucket information
  typename Mesh::MeshIndex meshIdx;

  //! The entity being handled
  stk::mesh::Entity entity;

  //! Nodes comprising the entity
  typename Mesh::ConnectedNodes entityNodes;
};

/** Lightweight data structure for face/elem pair information in mesh loops
 *
 *  The bucket loop wrappers provide this object to the lambda function. The
 *  connected nodes are always for the parent element that owns the boundary
 *  face.
 */
template<typename Mesh = ngp::Mesh>
struct BcFaceElemInfo
{
  typename Mesh::MeshIndex meshIdx;

  //! The face being handled
  stk::mesh::Entity face;

  //! The element connected to the face
  stk::mesh::Entity elem;

  typename Mesh::ConnectedNodes faceNodes;

  //! Nodes connected to the parent element (elem)
  typename Mesh::ConnectedNodes elemNodes;

  //! Index of the face within the element
  int faceOrdinal;
};


/** Traits for dealing with STK NGP meshes
 */
template<typename Mesh=ngp::Mesh>
struct NGPMeshTraits
{
  //! SIMD data type used in element data structures
  using DblType = ::sierra::nalu::DoubleType;

  //! Default scalar type for field data
  using FieldScalarType = double;

  using TeamPolicy =
    Kokkos::TeamPolicy<typename Mesh::MeshExecSpace, ngp::ScheduleType>;
  using TeamHandleType = typename TeamPolicy::member_type;
  using ShmemType = typename Mesh::MeshExecSpace::scratch_memory_space;
  using MeshIndex = typename Mesh::MeshIndex;

  using ScratchViewsType = ScratchViews<DblType, TeamHandleType, ShmemType>;
  using EntityInfo = EntityInfo<Mesh>;
  using BcFaceElemInfo = BcFaceElemInfo<Mesh>;
};


}  // nalu_ngp
}  // nalu
}  // sierra


#endif /* NGPTYPES_H */
