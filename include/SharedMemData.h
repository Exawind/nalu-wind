// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SharedMemData_h
#define SharedMemData_h

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/NgpMesh.hpp>

#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <memory>

namespace sierra {
namespace nalu {

template <typename TEAMHANDLETYPE, typename SHMEM>
struct SharedMemData
{
  KOKKOS_FUNCTION
  SharedMemData(
    const TEAMHANDLETYPE& team,
    unsigned nDim,
    const ElemDataRequestsGPU& dataNeededByKernels,
    unsigned nodesPerEntity,
    unsigned rhsSize)
    : simdPrereqData(team, nDim, nodesPerEntity, dataNeededByKernels)
  {
#if !defined(KOKKOS_ENABLE_GPU)
    for (int simdIndex = 0; simdIndex < simdLen; ++simdIndex) {
      prereqData[simdIndex] =
        std::unique_ptr<ScratchViews<double, TEAMHANDLETYPE, SHMEM>>(
          new ScratchViews<double, TEAMHANDLETYPE, SHMEM>(
            team, nDim, nodesPerEntity, dataNeededByKernels));
    }
#else
    prereqData[0] = &simdPrereqData;
#endif
    simdrhs =
      get_shmem_view_1D<DoubleType, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    simdlhs = get_shmem_view_2D<DoubleType, TEAMHANDLETYPE, SHMEM>(
      team, rhsSize, rhsSize);
    rhs = get_shmem_view_1D<double, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    lhs =
      get_shmem_view_2D<double, TEAMHANDLETYPE, SHMEM>(team, rhsSize, rhsSize);

    scratchIds = get_shmem_view_1D<int, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    sortPermutation =
      get_shmem_view_1D<int, TEAMHANDLETYPE, SHMEM>(team, rhsSize);

    simdPrereqData.fill_static_meviews(dataNeededByKernels);
  }

  KOKKOS_DEFAULTED_FUNCTION
  ~SharedMemData() = default;

  stk::mesh::NgpMesh::ConnectedNodes ngpElemNodes[simdLen];
  int numSimdElems;
#if defined(KOKKOS_ENABLE_GPU)
  ScratchViews<DoubleType, TEAMHANDLETYPE, SHMEM>* prereqData[1];
#else
  std::unique_ptr<ScratchViews<double, TEAMHANDLETYPE, SHMEM>>
    prereqData[simdLen];
#endif
  ScratchViews<DoubleType, TEAMHANDLETYPE, SHMEM> simdPrereqData;
  SharedMemView<DoubleType*, SHMEM> simdrhs;
  SharedMemView<DoubleType**, SHMEM> simdlhs;
  SharedMemView<double*, SHMEM> rhs;
  SharedMemView<double**, SHMEM> lhs;

  SharedMemView<int*, SHMEM> scratchIds;
  SharedMemView<int*, SHMEM> sortPermutation;
};

template <typename TEAMHANDLETYPE, typename SHMEM>
struct SharedMemData_FaceElem
{
  KOKKOS_FUNCTION
  SharedMemData_FaceElem(
    const TEAMHANDLETYPE& team,
    unsigned nDim,
    const ElemDataRequestsGPU& faceDataNeeded,
    const ElemDataRequestsGPU& elemDataNeeded,
    unsigned nodesPerFace,
    unsigned nodesPerElem,
    unsigned rhsSize)
    : simdFaceViews(team, nDim, nodesPerFace, faceDataNeeded),
      simdElemViews(team, nDim, nodesPerElem, elemDataNeeded)
  {
#if !defined(KOKKOS_ENABLE_GPU)
    for (int simdIndex = 0; simdIndex < simdLen; ++simdIndex) {
      faceViews[simdIndex] =
        std::unique_ptr<ScratchViews<double, TEAMHANDLETYPE, SHMEM>>(
          new ScratchViews<double, TEAMHANDLETYPE, SHMEM>(
            team, nDim, nodesPerFace, faceDataNeeded));
      elemViews[simdIndex] =
        std::unique_ptr<ScratchViews<double, TEAMHANDLETYPE, SHMEM>>(
          new ScratchViews<double, TEAMHANDLETYPE, SHMEM>(
            team, nDim, nodesPerElem, elemDataNeeded));
    }
#else
    faceViews[0] = &simdFaceViews;
    elemViews[0] = &simdElemViews;
#endif
    simdrhs =
      get_shmem_view_1D<DoubleType, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    simdlhs = get_shmem_view_2D<DoubleType, TEAMHANDLETYPE, SHMEM>(
      team, rhsSize, rhsSize);
    rhs = get_shmem_view_1D<double, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    lhs =
      get_shmem_view_2D<double, TEAMHANDLETYPE, SHMEM>(team, rhsSize, rhsSize);

    scratchIds = get_shmem_view_1D<int, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    sortPermutation =
      get_shmem_view_1D<int, TEAMHANDLETYPE, SHMEM>(team, rhsSize);

    simdFaceViews.fill_static_meviews(faceDataNeeded);
    simdElemViews.fill_static_meviews(elemDataNeeded);
  }

  KOKKOS_DEFAULTED_FUNCTION
  ~SharedMemData_FaceElem() = default;

  stk::mesh::NgpMesh::ConnectedNodes ngpConnectedNodes[simdLen];
  int numSimdFaces;
  int elemFaceOrdinal;
#if defined(KOKKOS_ENABLE_GPU)
  ScratchViews<DoubleType, TEAMHANDLETYPE, SHMEM>* faceViews[1];
  ScratchViews<DoubleType, TEAMHANDLETYPE, SHMEM>* elemViews[1];
#else
  std::unique_ptr<ScratchViews<double, TEAMHANDLETYPE, SHMEM>>
    faceViews[simdLen];
  std::unique_ptr<ScratchViews<double, TEAMHANDLETYPE, SHMEM>>
    elemViews[simdLen];
#endif
  ScratchViews<DoubleType, TEAMHANDLETYPE, SHMEM> simdFaceViews;
  ScratchViews<DoubleType, TEAMHANDLETYPE, SHMEM> simdElemViews;
  SharedMemView<DoubleType*, SHMEM> simdrhs;
  SharedMemView<DoubleType**, SHMEM> simdlhs;
  SharedMemView<double*, SHMEM> rhs;
  SharedMemView<double**, SHMEM> lhs;

  SharedMemView<int*, SHMEM> scratchIds;
  SharedMemView<int*, SHMEM> sortPermutation;
};

template <typename TEAMHANDLETYPE, typename SHMEM>
struct SharedMemData_Edge
{
  KOKKOS_FUNCTION
  SharedMemData_Edge(const TEAMHANDLETYPE& team, unsigned rhsSize)
  {
    rhs = get_shmem_view_1D<double, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    lhs =
      get_shmem_view_2D<double, TEAMHANDLETYPE, SHMEM>(team, rhsSize, rhsSize);
    scratchIds = get_shmem_view_1D<int, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    sortPermutation =
      get_shmem_view_1D<int, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
  }

  stk::mesh::NgpMesh::ConnectedNodes ngpElemNodes;
  SharedMemView<double*, SHMEM> rhs;
  SharedMemView<double**, SHMEM> lhs;

  SharedMemView<int*, SHMEM> scratchIds;
  SharedMemView<int*, SHMEM> sortPermutation;
};
} // namespace nalu
} // namespace sierra

#endif
