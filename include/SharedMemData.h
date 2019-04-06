/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef SharedMemData_h
#define SharedMemData_h

#include <stk_mesh/base/BulkData.hpp>

#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <memory>

namespace sierra{
namespace nalu{

template<typename TEAMHANDLETYPE, typename SHMEM>
struct SharedMemData {
    KOKKOS_FUNCTION
    SharedMemData(const TEAMHANDLETYPE& team,
         unsigned nDim,
         const ElemDataRequestsGPU& dataNeededByKernels,
         unsigned nodesPerEntity,
         unsigned rhsSize)
     : simdPrereqData(team, nDim, nodesPerEntity, dataNeededByKernels)
    {
#ifndef KOKKOS_ENABLE_CUDA
        for(int simdIndex=0; simdIndex<simdLen; ++simdIndex) {
          prereqData[simdIndex] = std::unique_ptr<ScratchViews<double,TEAMHANDLETYPE,SHMEM> >(new ScratchViews<double,TEAMHANDLETYPE,SHMEM>(team, nDim, nodesPerEntity, dataNeededByKernels));
        }
#else
        prereqData[0] = &simdPrereqData;
#endif
        simdrhs = get_shmem_view_1D<DoubleType,TEAMHANDLETYPE,SHMEM>(team, rhsSize);
        simdlhs = get_shmem_view_2D<DoubleType,TEAMHANDLETYPE,SHMEM>(team, rhsSize, rhsSize);
        rhs = get_shmem_view_1D<double,TEAMHANDLETYPE,SHMEM>(team, rhsSize);
        lhs = get_shmem_view_2D<double,TEAMHANDLETYPE,SHMEM>(team, rhsSize, rhsSize);

        scratchIds = get_shmem_view_1D<int,TEAMHANDLETYPE,SHMEM>(team, rhsSize);
        sortPermutation = get_shmem_view_1D<int,TEAMHANDLETYPE,SHMEM>(team, rhsSize);
    }

    KOKKOS_FUNCTION
    ~SharedMemData() = default;

    ngp::Mesh::ConnectedNodes ngpElemNodes[simdLen];
    int numSimdElems;
#ifdef KOKKOS_ENABLE_CUDA
    ScratchViews<DoubleType,TEAMHANDLETYPE,SHMEM>* prereqData[1];
#else
    std::unique_ptr<ScratchViews<double,TEAMHANDLETYPE,SHMEM>> prereqData[simdLen];
#endif
    ScratchViews<DoubleType,TEAMHANDLETYPE,SHMEM> simdPrereqData;
    SharedMemView<DoubleType*,SHMEM> simdrhs;
    SharedMemView<DoubleType**,SHMEM> simdlhs;
    SharedMemView<double*,SHMEM> rhs;
    SharedMemView<double**,SHMEM> lhs;

    SharedMemView<int*,SHMEM> scratchIds;
    SharedMemView<int*,SHMEM> sortPermutation;
};

template<typename TEAMHANDLETYPE, typename SHMEM>
struct SharedMemData_FaceElem {
    SharedMemData_FaceElem(const TEAMHANDLETYPE& team,
         unsigned nDim,
         const ElemDataRequestsGPU& faceDataNeeded,
         const ElemDataRequestsGPU& elemDataNeeded,
         const ScratchMeInfo& meElemInfo,
         unsigned rhsSize)
     : simdFaceViews(team, nDim, meElemInfo.nodesPerFace_, faceDataNeeded),
       simdElemViews(team, nDim, meElemInfo, elemDataNeeded)
    {
        for(int simdIndex=0; simdIndex<simdLen; ++simdIndex) {
          faceViews[simdIndex] = std::unique_ptr<ScratchViews<double,TEAMHANDLETYPE,SHMEM> >(new ScratchViews<double,TEAMHANDLETYPE,SHMEM>(team, nDim, meElemInfo.nodesPerFace_, faceDataNeeded));
          elemViews[simdIndex] = std::unique_ptr<ScratchViews<double,TEAMHANDLETYPE,SHMEM> >(new ScratchViews<double,TEAMHANDLETYPE,SHMEM>(team, nDim, meElemInfo, elemDataNeeded));
        }
        simdrhs = get_shmem_view_1D<DoubleType,TEAMHANDLETYPE,SHMEM>(team, rhsSize);
        simdlhs = get_shmem_view_2D<DoubleType,TEAMHANDLETYPE,SHMEM>(team, rhsSize, rhsSize);
        rhs = get_shmem_view_1D<double,TEAMHANDLETYPE,SHMEM>(team, rhsSize);
        lhs = get_shmem_view_2D<double,TEAMHANDLETYPE,SHMEM>(team, rhsSize, rhsSize);

        scratchIds = get_shmem_view_1D<int,TEAMHANDLETYPE,SHMEM>(team, rhsSize);
        sortPermutation = get_shmem_view_1D<int,TEAMHANDLETYPE,SHMEM>(team, rhsSize);
    }

    ngp::Mesh::ConnectedNodes ngpConnectedNodes[simdLen];
    int numSimdFaces;
    int elemFaceOrdinal;
    std::unique_ptr<ScratchViews<double,TEAMHANDLETYPE,SHMEM>> faceViews[simdLen];
    std::unique_ptr<ScratchViews<double,TEAMHANDLETYPE,SHMEM>> elemViews[simdLen];
    ScratchViews<DoubleType,TEAMHANDLETYPE,SHMEM> simdFaceViews;
    ScratchViews<DoubleType,TEAMHANDLETYPE,SHMEM> simdElemViews;
    SharedMemView<DoubleType*,SHMEM> simdrhs;
    SharedMemView<DoubleType**,SHMEM> simdlhs;
    SharedMemView<double*,SHMEM> rhs;
    SharedMemView<double**,SHMEM> lhs;

    SharedMemView<int*,SHMEM> scratchIds;
    SharedMemView<int*,SHMEM> sortPermutation;
};

} // namespace nalu
} // namespace Sierra

#endif
