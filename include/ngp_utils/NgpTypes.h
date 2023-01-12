// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPTYPES_H
#define NGPTYPES_H

/** \file
 *  \brief Nalu-Wind custom types for NGP execution
 */

#include "ngp_utils/NgpMeshInfo.h"
#include "stk_mesh/base/NgpMesh.hpp"
#include "KokkosInterface.h"
#include "SimdInterface.h"

#include <memory>

namespace sierra {
namespace nalu {

template <typename T1, typename T2, typename T3>
class ScratchViews;

namespace nalu_ngp {

//! maximum number of dimensions for arrays
constexpr int NDimMax = 3;

/** Lightweight data structure holding information of the entity in mesh loops
 *
 *  The bucket loop wrappers provide this object to the lambda function for use
 *  during iterations.
 *
 *  Depending on the bucket-loop, entity could be one of elem, edge, or face.
 */
template <typename Mesh = stk::mesh::NgpMesh>
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
template <typename Mesh = stk::mesh::NgpMesh>
struct BcFaceElemInfo
{
  typename Mesh::MeshIndex meshIdx;

  //! The face being handled
  stk::mesh::Entity entity;

  //! The element connected to the face
  stk::mesh::Entity elem;

  typename Mesh::ConnectedNodes entityNodes;

  //! Nodes connected to the parent element (elem)
  typename Mesh::ConnectedNodes elemNodes;

  //! Index of the face within the element
  int faceOrdinal;
};

/** Traits for dealing with STK NGP meshes
 */
template <typename Mesh = stk::mesh::NgpMesh>
struct NGPMeshTraits
{
  //! SIMD data type used in element data structures
  using DblType = ::sierra::nalu::DoubleType;

  //! Default scalar type for field data
  using FieldScalarType = double;

#if defined(KOKKOS_ENABLE_HIP)
  using TeamPolicy = Kokkos::TeamPolicy<
    typename Mesh::MeshExecSpace,
    Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
    stk::ngp::ScheduleType>;
#else
  using TeamPolicy =
    Kokkos::TeamPolicy<typename Mesh::MeshExecSpace, stk::ngp::ScheduleType>;
#endif
  using TeamHandleType = typename TeamPolicy::member_type;
  using ShmemType = typename Mesh::MeshExecSpace::scratch_memory_space;
  using MeshIndex = typename Mesh::MeshIndex;

  using ScratchViewsType = ScratchViews<DblType, TeamHandleType, ShmemType>;
  using EntityInfoType = EntityInfo<Mesh>;
  using BcFaceElemInfoType = BcFaceElemInfo<Mesh>;
};

} // namespace nalu_ngp
} // namespace nalu
} // namespace sierra

#endif /* NGPTYPES_H */
