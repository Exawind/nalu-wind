// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPSCRATCHDATA_H
#define NGPSCRATCHDATA_H

/** \file
 *  \brief Element connectivity and ScratchData holders
 */

#include "ngp_utils/NgpTypes.h"
#include "SimdInterface.h"
#include "ElemDataRequestsGPU.h"
#include <memory>

namespace sierra {
namespace nalu {

namespace nalu_ngp {

/** Element connectivity and gathered scratch data for element
 *
 *  This class holds the element connectivity information and the gathered
 *  element data (ScratchViews) within element loops.
 *
 */
template <typename Mesh>
struct ElemSimdData
{
  using MeshTraits = NGPMeshTraits<Mesh>;
  using TeamHandleType = typename MeshTraits::TeamHandleType;
  using ShmemType = typename MeshTraits::ShmemType;
  using EntityInfoType = EntityInfo<Mesh>;

  KOKKOS_INLINE_FUNCTION
  ElemSimdData(
    const TeamHandleType& team,
    unsigned ndim,
    unsigned nodesPerElem,
    const ElemDataRequestsGPU& dataReq)
    : simdScrView(team, ndim, nodesPerElem, dataReq)
  {
#if defined(KOKKOS_ENABLE_GPU)
    scrView[0] = &simdScrView;
#else
    for (int si = 0; si < simdLen; ++si) {
      scrView[si].reset(new ScratchViews<double, TeamHandleType, ShmemType>(
        team, ndim, nodesPerElem, dataReq));
    }
#endif

    simdScrView.fill_static_meviews(dataReq);
  }

  KOKKOS_DEFAULTED_FUNCTION ~ElemSimdData() = default;

  KOKKOS_INLINE_FUNCTION
  const EntityInfoType* info() const { return elemInfo; }

  //! Gathered element data (always in SIMD datatype)
  ScratchViews<DoubleType, TeamHandleType, ShmemType> simdScrView;

  /** Non-SIMD gathered element data
   *
   *  When executing on GPUs we dont
   */
#if defined(KOKKOS_ENABLE_GPU)
  ScratchViews<DoubleType, TeamHandleType, ShmemType>* scrView[1];
#else
  std::unique_ptr<ScratchViews<double, TeamHandleType, ShmemType>>
    scrView[simdLen];
#endif

  //! Element connectivity info for each element within the SIMD group
  EntityInfoType elemInfo[simdLen];

  //! Number of SIMD elements in this batch
  int numSimdElems;
};

template <typename Mesh>
struct FaceElemSimdData
{
  using MeshTraits = NGPMeshTraits<Mesh>;
  using TeamHandleType = typename MeshTraits::TeamHandleType;
  using ShmemType = typename MeshTraits::ShmemType;
  using EntityInfoType = BcFaceElemInfo<Mesh>;

  KOKKOS_INLINE_FUNCTION
  FaceElemSimdData(
    const TeamHandleType& team,
    unsigned ndim,
    unsigned nodesPerFace,
    unsigned nodesPerElem,
    const ElemDataRequestsGPU& faceDataReqs,
    const ElemDataRequestsGPU& elemDataReqs)
    : simdFaceView(team, ndim, nodesPerFace, faceDataReqs),
      simdElemView(team, ndim, nodesPerElem, elemDataReqs)
  {
#if defined(KOKKOS_ENABLE_GPU)
    scrFaceView[0] = &simdFaceView;
    scrElemView[0] = &simdElemView;
#else
    for (int si = 0; si < simdLen; ++si) {
      scrFaceView[si].reset(new ScratchViews<double, TeamHandleType, ShmemType>(
        team, ndim, nodesPerFace, faceDataReqs));
      scrElemView[si].reset(new ScratchViews<double, TeamHandleType, ShmemType>(
        team, ndim, nodesPerElem, elemDataReqs));
    }
#endif

    simdFaceView.fill_static_meviews(faceDataReqs);
    simdElemView.fill_static_meviews(elemDataReqs);
  }

  KOKKOS_DEFAULTED_FUNCTION ~FaceElemSimdData() = default;

  KOKKOS_INLINE_FUNCTION
  const EntityInfoType* info() const { return faceInfo; }

  ScratchViews<DoubleType, TeamHandleType, ShmemType> simdFaceView;
  ScratchViews<DoubleType, TeamHandleType, ShmemType> simdElemView;

  /** Non-SIMD gathered element data
   *
   *  When executing on GPUs we dont
   */
#if defined(KOKKOS_ENABLE_GPU)
  ScratchViews<DoubleType, TeamHandleType, ShmemType>* scrFaceView[1];
  ScratchViews<DoubleType, TeamHandleType, ShmemType>* scrElemView[1];
#else
  std::unique_ptr<ScratchViews<double, TeamHandleType, ShmemType>>
    scrFaceView[simdLen];
  std::unique_ptr<ScratchViews<double, TeamHandleType, ShmemType>>
    scrElemView[simdLen];
#endif

  EntityInfoType faceInfo[simdLen];

  int faceOrd;

  //! Number of SIMD face/elem pairs in this batch
  int numSimdElems;
};

} // namespace nalu_ngp
} // namespace nalu
} // namespace sierra

#endif /* NGPSCRATCHDATA_H */
