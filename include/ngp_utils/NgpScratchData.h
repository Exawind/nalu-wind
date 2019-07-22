/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
template<typename Mesh>
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
#ifdef KOKKOS_ENABLE_CUDA
    scrView[0] = &simdScrView;
#else
    for (int si=0; si < simdLen; ++si) {
      scrView[si].reset(
        new ScratchViews<double, TeamHandleType, ShmemType>(
          team, ndim, nodesPerElem, dataReq));
    }
#endif

    simdScrView.fill_static_meviews(dataReq);
  }

  KOKKOS_FUNCTION ~ElemSimdData() = default;

  //! Gathered element data (always in SIMD datatype)
  ScratchViews<DoubleType, TeamHandleType, ShmemType> simdScrView;

  /** Non-SIMD gathered element data
   *
   *  When executing on GPUs we dont
   */
#ifdef KOKKOS_ENABLE_CUDA
  ScratchViews<DoubleType, TeamHandleType, ShmemType>* scrView[1];
#else
  std::unique_ptr<ScratchViews<double, TeamHandleType, ShmemType>> scrView[simdLen];
#endif

  //! Element connectivity info for each element within the SIMD group
  EntityInfoType elemInfo[simdLen];

  //! Number of SIMD elements in this batch
  int numSimdElems;
};

}  // nalu_ngp
}  // nalu
}  // sierra



#endif /* NGPSCRATCHDATA_H */
