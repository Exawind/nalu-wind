// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPLOOPUTILS_H
#define NGPLOOPUTILS_H

#include <type_traits>

#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpScratchData.h"
#include "ngp_utils/NgpMEUtils.h"
#include "CopyAndInterleave.h"
#include "ElemDataRequests.h"
#include "ElemDataRequestsGPU.h"
#include "ScratchViews.h"

#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Ngp.hpp"

namespace sierra {
namespace nalu {

namespace nalu_ngp {
namespace impl {

/** Return a Kokkos TeamPolicy object after setting the appropriate scratch
 * memory sizes
 *
 * @param sz The number of STK buckets we are looping over
 * @param bytes_per_team Bytes to be allocated at team parallelism level
 * @param bytes_per_thread Bytes to be allocated at thread level
 * @return Kokkos TeamPolicy instance for use with Kokkos::parallel_for
 */
template <typename TeamPolicy>
inline TeamPolicy
ngp_mesh_team_policy(
  const size_t sz, const size_t bytes_per_team, const size_t bytes_per_thread)
{
  TeamPolicy policy(sz, Kokkos::AUTO);
  return policy.set_scratch_size(
    1, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread));
}

/** Estimate the bytes required per thread to store ScratchViews for
 *  element data.
 *
 *  @param ndim Spatial dimension
 *  @param dataReq Element data requests object
 *  @return bytes_per_thread
 */
template <typename T, typename DataReqType>
inline int
ngp_calc_thread_shmem_size(
  int ndim, const DataReqType& dataReq, const ElemReqType reqType)
{
  int preReqSize = get_num_bytes_pre_req_data<T>(dataReq, ndim, reqType);
  int mdvSize = MultiDimViews<T>::bytes_needed(
    dataReq.get_total_num_fields(),
    count_needed_field_views(dataReq.get_host_fields()));

#if !defined(KOKKOS_ENABLE_GPU)
  // On host account for extra data to store the SIMD and non-SIMD versions of
  // the ScratchViews
  return (preReqSize + mdvSize) * 2 * simdLen;
#else
  return (preReqSize + mdvSize);
#endif
}

/** Estimate the bytes required per thread for face/element ScratchViews
 *
 *  @param ndim Spatial dimension
 *  @param Face data requests object
 *  @param Element data requests object (this is the parent element of the face)
 *  @param MasterElementInfo object that contains nodesPerElem/nodesPerFace etc.
 *  @return bytes_per_thread
 */
template <typename T, typename DataReqType>
inline int
ngp_calc_thread_shmem_size(
  int ndim, const DataReqType& faceDataReq, const DataReqType& elemDataReq)
{
  const int faceMemSize =
    ngp_calc_thread_shmem_size<T>(ndim, faceDataReq, ElemReqType::FACE);
  const int elemMemSize =
    ngp_calc_thread_shmem_size<T>(ndim, elemDataReq, ElemReqType::ELEM);

  return (faceMemSize + elemMemSize);
}

} // namespace impl

/** Execute the given functor for all entities in a Kokkos parallel loop
 *
 *  The functor is called with one argument MeshIndex, a struct containing a
 *  pointer to the NGP bucket and the index into the bucket array for this
 *  entity.
 *
 *. @param algName User-defined name for the parallel for loop
 *  @param mesh A STK NGP mesh instance
 *  @param rank Rank for the loop (node, elem, face, etc.)
 *  @param sel  STK mesh selector to choose buckets for looping
 *  @param algorithm A functor that will be executed for each entity
 */
template <typename Mesh, typename AlgFunctor>
void
run_entity_algorithm(
  const std::string& algName,
  const Mesh& mesh,
  const stk::topology::rank_t rank,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm)
{
  using Traits = NGPMeshTraits<Mesh>;
  using TeamPolicy = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex = typename Traits::MeshIndex;

  const auto& buckets = mesh.get_bucket_ids(rank, sel);
  auto team_exec = TeamPolicy(buckets.size(), Kokkos::AUTO);

  Kokkos::parallel_for(
    algName, team_exec, KOKKOS_LAMBDA(const TeamHandleType& team) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = mesh.get_bucket(rank, bktId);

      const size_t bktLen = bkt.size();
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, bktLen), [&](const size_t& bktIndex) {
          MeshIndex meshIdx{&bkt, static_cast<unsigned>(bktIndex)};
          algorithm(meshIdx);
        });
    });
}

/** Execute the given functor for all entities and perform global reduction
 *
 *  The functor is called with two argument MeshIndex, a struct containing a
 *  pointer to the NGP bucket and the index into the bucket array for this
 *  entity, and accumulator for reduction.
 *
 *. @param algName User-defined name for the parallel_reduce loop
 *  @param mesh A STK NGP mesh instance
 *  @param rank Rank for the loop (node, elem, face, etc.)
 *  @param sel  STK mesh selector to choose buckets for looping
 *  @param algorithm A functor that will be executed for each entity
 *  @param reduceVal A scalar value that has the reduced value
 */
template <typename Mesh, typename AlgFunctor, typename ReducerType>
void
run_entity_par_reduce(
  const std::string& algName,
  const Mesh& mesh,
  const stk::topology::rank_t rank,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm,
  ReducerType& reduceVal,
  typename std::enable_if<std::is_arithmetic<ReducerType>::value, int>::type* =
    nullptr)
{
  using Traits = NGPMeshTraits<Mesh>;
  using TeamPolicy = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex = typename Traits::MeshIndex;

  const auto& buckets = mesh.get_bucket_ids(rank, sel);
  auto team_exec = TeamPolicy(buckets.size(), Kokkos::AUTO);

  Kokkos::parallel_reduce(
    algName, team_exec,
    KOKKOS_LAMBDA(const TeamHandleType& team, ReducerType& teamVal) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = mesh.get_bucket(rank, bktId);

      ReducerType bktVal = 0.0;
      const size_t bktLen = bkt.size();
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, bktLen),
        [&](const size_t& bktIndex, ReducerType& threadVal) {
          MeshIndex meshIdx{&bkt, static_cast<unsigned>(bktIndex)};
          algorithm(meshIdx, threadVal);
        },
        bktVal);

      Kokkos::single(Kokkos::PerTeam(team), [&]() { teamVal += bktVal; });
    },
    reduceVal);
}

template <typename Mesh, typename AlgFunctor, typename ReducerType>
void
run_entity_par_reduce(
  const std::string& algName,
  const Mesh& mesh,
  const stk::topology::rank_t rank,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm,
  ReducerType& reduceVal,
  typename std::enable_if<!std::is_arithmetic<ReducerType>::value, int>::type* =
    nullptr)
{
  using Traits = NGPMeshTraits<Mesh>;
  using TeamPolicy = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex = typename Traits::MeshIndex;
  using value_type = typename ReducerType::value_type;

  const auto& buckets = mesh.get_bucket_ids(rank, sel);
  auto team_exec = TeamPolicy(buckets.size(), Kokkos::AUTO);

  Kokkos::parallel_reduce(
    algName, team_exec,
    KOKKOS_LAMBDA(const TeamHandleType& team, value_type& teamVal) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = mesh.get_bucket(rank, bktId);

      value_type bktVal;
      const size_t bktLen = bkt.size();
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, bktLen),
        [&](const size_t& bktIndex, value_type& threadVal) {
          MeshIndex meshIdx{&bkt, static_cast<unsigned>(bktIndex)};
          algorithm(meshIdx, threadVal);
        },
        ReducerType(bktVal));

      Kokkos::single(
        Kokkos::PerTeam(team), [&]() { reduceVal.join(teamVal, bktVal); });
    },
    reduceVal);
}

/** Execute the given functor for all edges in a Kokkos parallel loop
 *
 *  The functor is called with one argument MeshIndex, a struct containing a
 *  pointer to the NGP bucket and the index into the bucket array for this
 *  entity.
 *
 *. @param algName User-defined name for the edge parallel loop
 *  @param mesh A STK NGP mesh instance
 *  @param sel  STK mesh selector to choose buckets for looping
 *  @param algorithm A functor that will be executed for each entity
 */
template <typename Mesh, typename AlgFunctor>
inline void
run_edge_algorithm(
  const std::string& algName,
  const Mesh& mesh,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm)
{
  static constexpr stk::topology::rank_t rank = stk::topology::EDGE_RANK;
  using Traits = NGPMeshTraits<Mesh>;
  using MeshIndex = typename Traits::MeshIndex;

  run_entity_algorithm(
    algName, mesh, rank, sel, KOKKOS_LAMBDA(MeshIndex & meshIdx) {
      algorithm(EntityInfo<Mesh>{
        meshIdx, (*meshIdx.bucket)[meshIdx.bucketOrd],
        mesh.get_nodes(meshIdx)});
    });
}

/** Execute the given functor for all elements in a Kokkos parallel loop
 *
 *  The functor is called with one argument MeshIndex, a struct containing a
 *  pointer to the NGP bucket and the index into the bucket array for this
 *  entity.
 *
 *  Note that a rank is still passed as an argument to allow looping over both
 *  element and side/face ranks with the same function.
 *
 *. @param algName User-defined name for the elem parallel loop
 *  @param mesh A STK NGP mesh instance
 *  @param sel  STK mesh selector to choose buckets for looping
 *  @param algorithm A functor that will be executed for each entity
 */
template <typename Mesh, typename AlgFunctor>
inline void
run_elem_algorithm(
  const std::string& algName,
  const Mesh& mesh,
  const stk::topology::rank_t rank,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm)
{
  using Traits = NGPMeshTraits<Mesh>;
  using MeshIndex = typename Traits::MeshIndex;

  run_entity_algorithm(
    algName, mesh, rank, sel, KOKKOS_LAMBDA(MeshIndex & meshIdx) {
      algorithm(EntityInfo<Mesh>{
        meshIdx, (*meshIdx.bucket)[meshIdx.bucketOrd],
        mesh.get_nodes(meshIdx)});
    });
}

/** Gather element data in ScratchViews and execute functor over elements
 *
 *  The functor is called with an instance of ElemSimdData<Mesh> that contains
 *  an EntityInfo describing the element connectivity data, and a ScratchViews
 *  instance populated with all the data requested for a particular element
 *  through ElemDataRequests.
 *
 *  In addition to gather of element data, this function also handles the
 *  appropriate interleaving for SIMD data structures where appropriate.
 *
 *  @param meshInfo The MeshInfo object containing STK and NGP instances
 *  @param rank ELEM or side_rank()
 *  @param dataReqs Instance contaning element data to be added to ScratchViews
 *  @param sel STK mesh selector to choose buckets for looping
 *  @param algorithm The functor to be executed on each element
 */
template <
  typename Mesh,
  typename FieldManager,
  typename DataReqType,
  typename AlgFunctor>
void
run_elem_algorithm(
  const std::string algName,
  const MeshInfo<Mesh, FieldManager>& meshInfo,
  const stk::topology::rank_t rank,
  const DataReqType& dataReqs,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm)
{
  using Traits = NGPMeshTraits<Mesh>;
  using TeamPolicy = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex = typename Traits::MeshIndex;

  const auto& ndim = meshInfo.ndim();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  ElemDataRequestsGPU dataReqNGP(fieldMgr, dataReqs, meshInfo.num_fields());

  const int nodesPerElement = nodes_per_entity(dataReqNGP);
  NGP_ThrowRequire(nodesPerElement != 0);

  const auto reqType =
    (rank == stk::topology::ELEM_RANK) ? ElemReqType::ELEM : ElemReqType::FACE;
  const int bytes_per_team = 0;
  const int bytes_per_thread =
    impl::ngp_calc_thread_shmem_size<sierra::nalu::DoubleType>(
      ndim, dataReqNGP, reqType);

  const auto& buckets = ngpMesh.get_bucket_ids(rank, sel);
  auto team_exec = impl::ngp_mesh_team_policy<TeamPolicy>(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    algName, team_exec, KOKKOS_LAMBDA(const TeamHandleType& team) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = ngpMesh.get_bucket(rank, bktId);

      ElemSimdData<Mesh> elemData(team, ndim, nodesPerElement, dataReqNGP);

      const size_t bktLen = bkt.size();
      const size_t simdBktLen = get_num_simd_groups(bktLen);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, simdBktLen), [&](const size_t& bktIndex) {
          int nSimdElems = get_length_of_next_simd_group(bktIndex, bktLen);
          elemData.numSimdElems = nSimdElems;

          for (int is = 0; is < nSimdElems; ++is) {
            const unsigned bktOrd = bktIndex * simdLen + is;
            MeshIndex meshIdx{&bkt, bktOrd};
            const auto& elem = bkt[bktOrd];
            elemData.elemInfo[is] =
              EntityInfo<Mesh>{meshIdx, elem, ngpMesh.get_nodes(meshIdx)};

            fill_pre_req_data(
              dataReqNGP, ngpMesh, rank, elem, *elemData.scrView[is]);
          }

#if !defined(KOKKOS_ENABLE_GPU)
          copy_and_interleave(
            elemData.scrView, nSimdElems, elemData.simdScrView);
#endif

          fill_master_element_views(dataReqNGP, elemData.simdScrView);
          algorithm(elemData);
        });
    });
}

/** Gather element data in ScratchViews and execute a reduction over elements
 *
 *  The reduce functor is called with an instance of ElemSimdData<Mesh> that
 * contains an EntityInfo describing the element connectivity data, and a
 * ScratchViews instance populated with all the data requested for a particular
 * element through ElemDataRequests.
 *
 *  In addition to gather of element data, this function also handles the
 *  appropriate interleaving for SIMD data structures where appropriate.
 *
 *  @param meshInfo The MeshInfo object containing STK and NGP instances
 *  @param rank ELEM or side_rank()
 *  @param dataReqs Instance contaning element data to be added to ScratchViews
 *  @param sel STK mesh selector to choose buckets for looping
 *  @param algorithm The reduce functor to be executed on each element
 *  @param reduceVal A Kokkos reducer type
 */
template <
  typename Mesh,
  typename FieldManager,
  typename DataReqType,
  typename AlgFunctor,
  typename ReducerType>
void
run_elem_par_reduce(
  const std::string& algName,
  const MeshInfo<Mesh, FieldManager>& meshInfo,
  const stk::topology::rank_t rank,
  const DataReqType& dataReqs,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm,
  ReducerType& reduceVal)
{
  using Traits = NGPMeshTraits<Mesh>;
  using TeamPolicy = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex = typename Traits::MeshIndex;
  using ReducerValueType = typename ReducerType::value_type;

  const auto& ndim = meshInfo.ndim();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  ElemDataRequestsGPU dataReqNGP(fieldMgr, dataReqs, meshInfo.num_fields());

  const int nodesPerElement = nodes_per_entity(dataReqNGP);
  NGP_ThrowRequire(nodesPerElement != 0);

  const auto reqType =
    (rank == stk::topology::ELEM_RANK) ? ElemReqType::ELEM : ElemReqType::FACE;
  const int bytes_per_team = 0;
  const int bytes_per_thread =
    impl::ngp_calc_thread_shmem_size<sierra::nalu::DoubleType>(
      ndim, dataReqNGP, reqType);

  const auto& buckets = ngpMesh.get_bucket_ids(rank, sel);
  auto team_exec = impl::ngp_mesh_team_policy<TeamPolicy>(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_reduce(
    algName, team_exec,
    KOKKOS_LAMBDA(const TeamHandleType& team, ReducerValueType& teamVal) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = ngpMesh.get_bucket(rank, bktId);

      ElemSimdData<Mesh> elemData(team, ndim, nodesPerElement, dataReqNGP);

      const size_t bktLen = bkt.size();
      const size_t simdBktLen = get_num_simd_groups(bktLen);

      ReducerValueType bktVal;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, simdBktLen),
        [&](const size_t& bktIndex, ReducerValueType& threadVal) {
          int nSimdElems = get_length_of_next_simd_group(bktIndex, bktLen);
          elemData.numSimdElems = nSimdElems;

          for (int is = 0; is < nSimdElems; ++is) {
            const unsigned bktOrd = bktIndex * simdLen + is;
            MeshIndex meshIdx{&bkt, bktOrd};
            const auto& elem = bkt[bktOrd];
            elemData.elemInfo[is] =
              EntityInfo<Mesh>{meshIdx, elem, ngpMesh.get_nodes(meshIdx)};

            fill_pre_req_data(
              dataReqNGP, ngpMesh, rank, elem, *elemData.scrView[is]);
          }

#if !defined(KOKKOS_ENABLE_GPU)
          copy_and_interleave(
            elemData.scrView, nSimdElems, elemData.simdScrView);
#endif

          fill_master_element_views(dataReqNGP, elemData.simdScrView);
          algorithm(elemData, threadVal);
        },
        ReducerType(bktVal));

      Kokkos::single(
        Kokkos::PerTeam(team), [&]() { reduceVal.join(teamVal, bktVal); });
    },
    reduceVal);
}

template <
  typename Mesh,
  typename FieldManager,
  typename DataReqType,
  typename AlgFunctor>
void
run_face_elem_algorithm(
  const std::string& algName,
  const MeshInfo<Mesh, FieldManager>& meshInfo,
  const DataReqType& faceDataReqs,
  const DataReqType& elemDataReqs,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm)
{
  static constexpr stk::topology::rank_t elemRank = stk::topology::ELEMENT_RANK;
  const stk::topology::rank_t sideRank = meshInfo.meta().side_rank();
  using Traits = NGPMeshTraits<Mesh>;
  using TeamPolicy = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex = typename Traits::MeshIndex;

  const auto& ndim = meshInfo.ndim();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto& numFields = meshInfo.num_fields();

  ElemDataRequestsGPU faceDataNGP(fieldMgr, faceDataReqs, numFields);
  ElemDataRequestsGPU elemDataNGP(fieldMgr, elemDataReqs, numFields);

  const int nodesPerElement = nodes_per_entity(elemDataNGP);
  const int nodesPerFace = nodes_per_entity(faceDataNGP, METype::FACE);
  NGP_ThrowRequire(nodesPerElement != 0);
  NGP_ThrowRequire(nodesPerFace != 0);

  const int bytes_per_team = 0;
  const int bytes_per_thread =
    impl::ngp_calc_thread_shmem_size<sierra::nalu::DoubleType>(
      ndim, faceDataNGP, elemDataNGP);

  const auto& buckets = ngpMesh.get_bucket_ids(sideRank, sel);
  auto team_exec = impl::ngp_mesh_team_policy<TeamPolicy>(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    algName, team_exec, KOKKOS_LAMBDA(const TeamHandleType& team) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = ngpMesh.get_bucket(sideRank, bktId);

      FaceElemSimdData<Mesh> faceElemData(
        team, ndim, nodesPerFace, nodesPerElement, faceDataNGP, elemDataNGP);

      const size_t bktLen = bkt.size();
      const size_t simdBktLen = get_num_simd_groups(bktLen);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, simdBktLen), [&](const size_t& bktIndex) {
          size_t nSimdFaces = get_length_of_next_simd_group(bktIndex, bktLen);
          size_t nFacesProcessed = 0;

          // Only group face/elem pairs in SIMD that have the same face ordinals
          do {
            int elemFaceOrd = -1;
            int simdFaceIdx = 0;

            while ((nFacesProcessed + simdFaceIdx) < nSimdFaces) {
              const auto& bktOrd =
                bktIndex * simdLen + nFacesProcessed + simdFaceIdx;
              const auto& face = bkt[bktOrd];
              const auto faceIdx = ngpMesh.fast_mesh_index(face);
              int faceOrd = ngpMesh.get_element_ordinals(sideRank, faceIdx)[0];

              // If we have one or more faces processed, then does the current
              // candidate face the same face ordinal as the ones we have
              // processed so far
              if ((elemFaceOrd >= 0) && (faceOrd != elemFaceOrd))
                break;

              const auto elems = ngpMesh.get_elements(sideRank, faceIdx);
              MeshIndex meshIdx{&bkt, static_cast<unsigned>(bktOrd)};
              const auto elem = elems[0];
              const auto elemIdx = ngpMesh.fast_mesh_index(elem);
              faceElemData.faceInfo[simdFaceIdx] = BcFaceElemInfo<Mesh>{
                meshIdx,
                face,
                elem,
                ngpMesh.get_nodes(sideRank, faceIdx),
                ngpMesh.get_nodes(elemRank, elemIdx),
                faceOrd};

              fill_pre_req_data(
                faceDataNGP, ngpMesh, sideRank, face,
                *faceElemData.scrFaceView[simdFaceIdx]);
              fill_pre_req_data(
                elemDataNGP, ngpMesh, elemRank, elem,
                *faceElemData.scrElemView[simdFaceIdx]);

              elemFaceOrd = faceOrd;
              ++simdFaceIdx;
            }
            faceElemData.faceOrd = elemFaceOrd;
            faceElemData.numSimdElems = simdFaceIdx;
            nFacesProcessed += simdFaceIdx;

#if !defined(KOKKOS_ENABLE_GPU)
            copy_and_interleave(
              faceElemData.scrFaceView, faceElemData.numSimdElems,
              faceElemData.simdFaceView);
            copy_and_interleave(
              faceElemData.scrElemView, faceElemData.numSimdElems,
              faceElemData.simdElemView);
#endif
            fill_master_element_views(
              faceDataNGP, faceElemData.simdFaceView, elemFaceOrd);
            fill_master_element_views(
              elemDataNGP, faceElemData.simdElemView, elemFaceOrd);

            algorithm(faceElemData);
          } while (nFacesProcessed < nSimdFaces);
        });
    });
}

template <
  typename Mesh,
  typename FieldManager,
  typename DataReqType,
  typename AlgFunctor,
  typename ReducerType>
void
run_face_elem_par_reduce(
  const std::string& algName,
  const MeshInfo<Mesh, FieldManager>& meshInfo,
  const DataReqType& faceDataReqs,
  const DataReqType& elemDataReqs,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm,
  ReducerType& reduceVal)
{
  static constexpr stk::topology::rank_t elemRank = stk::topology::ELEMENT_RANK;
  const stk::topology::rank_t sideRank = meshInfo.meta().side_rank();
  using Traits = NGPMeshTraits<Mesh>;
  using TeamPolicy = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex = typename Traits::MeshIndex;
  using ReducerValueType = typename ReducerType::value_type;

  const auto& ndim = meshInfo.ndim();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto& numFields = meshInfo.num_fields();

  ElemDataRequestsGPU faceDataNGP(fieldMgr, faceDataReqs, numFields);
  ElemDataRequestsGPU elemDataNGP(fieldMgr, elemDataReqs, numFields);

  const int nodesPerElement = nodes_per_entity(elemDataNGP);
  const int nodesPerFace = nodes_per_entity(faceDataNGP, METype::FACE);
  NGP_ThrowRequire(nodesPerElement != 0);
  NGP_ThrowRequire(nodesPerFace != 0);

  const int bytes_per_team = 0;
  const int bytes_per_thread =
    impl::ngp_calc_thread_shmem_size<sierra::nalu::DoubleType>(
      ndim, faceDataNGP, elemDataNGP);

  const auto& buckets = ngpMesh.get_bucket_ids(sideRank, sel);
  auto team_exec = impl::ngp_mesh_team_policy<TeamPolicy>(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_reduce(
    algName, team_exec,
    KOKKOS_LAMBDA(const TeamHandleType& team, ReducerValueType& teamVal) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = ngpMesh.get_bucket(sideRank, bktId);

      FaceElemSimdData<Mesh> faceElemData(
        team, ndim, nodesPerFace, nodesPerElement, faceDataNGP, elemDataNGP);

      const size_t bktLen = bkt.size();
      const size_t simdBktLen = get_num_simd_groups(bktLen);

      ReducerValueType bktVal;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, simdBktLen),
        [&](const size_t& bktIndex, ReducerValueType& threadVal) {
          size_t nSimdFaces = get_length_of_next_simd_group(bktIndex, bktLen);
          size_t nFacesProcessed = 0;

          // Only group face/elem pairs in SIMD that have the same face ordinals
          do {
            int elemFaceOrd = -1;
            int simdFaceIdx = 0;

            while ((nFacesProcessed + simdFaceIdx) < nSimdFaces) {
              const auto& bktOrd =
                bktIndex * simdLen + nFacesProcessed + simdFaceIdx;
              const auto& face = bkt[bktOrd];
              const auto faceIdx = ngpMesh.fast_mesh_index(face);
              int faceOrd = ngpMesh.get_element_ordinals(sideRank, faceIdx)[0];

              // If we have one or more faces processed, then does the current
              // candidate face the same face ordinal as the ones we have
              // processed so far
              if ((elemFaceOrd >= 0) && (faceOrd != elemFaceOrd))
                break;

              const auto elems = ngpMesh.get_elements(sideRank, faceIdx);
              MeshIndex meshIdx{&bkt, static_cast<unsigned>(bktOrd)};
              const auto elem = elems[0];
              const auto elemIdx = ngpMesh.fast_mesh_index(elem);
              faceElemData.faceInfo[simdFaceIdx] = BcFaceElemInfo<Mesh>{
                meshIdx,
                face,
                elem,
                ngpMesh.get_nodes(sideRank, faceIdx),
                ngpMesh.get_nodes(elemRank, elemIdx),
                faceOrd};

              fill_pre_req_data(
                faceDataNGP, ngpMesh, sideRank, face,
                *faceElemData.scrFaceView[simdFaceIdx]);
              fill_pre_req_data(
                elemDataNGP, ngpMesh, elemRank, elem,
                *faceElemData.scrElemView[simdFaceIdx]);

              elemFaceOrd = faceOrd;
              ++simdFaceIdx;
            }
            faceElemData.faceOrd = elemFaceOrd;
            faceElemData.numSimdElems = simdFaceIdx;
            nFacesProcessed += simdFaceIdx;

#if !defined(KOKKOS_ENABLE_GPU)
            copy_and_interleave(
              faceElemData.scrFaceView, faceElemData.numSimdElems,
              faceElemData.simdFaceView);
            copy_and_interleave(
              faceElemData.scrElemView, faceElemData.numSimdElems,
              faceElemData.simdElemView);
#endif
            fill_master_element_views(
              faceDataNGP, faceElemData.simdFaceView, elemFaceOrd);
            fill_master_element_views(
              elemDataNGP, faceElemData.simdElemView, elemFaceOrd);

            algorithm(faceElemData, threadVal);
          } while (nFacesProcessed < nSimdFaces);
        },
        ReducerType(bktVal));

      Kokkos::single(
        Kokkos::PerTeam(team), [&]() { reduceVal.join(teamVal, bktVal); });
    },
    reduceVal);
}

template <
  typename Mesh,
  typename FieldManager,
  typename DataReqType,
  typename AlgFunctor>
void
run_face_elem_algorithm_nosimd(
  const MeshInfo<Mesh, FieldManager>& meshInfo,
  const DataReqType& faceDataReqs,
  const DataReqType& elemDataReqs,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm)
{
  static constexpr stk::topology::rank_t sideRank = meshInfo.meta().side_rank();
  static constexpr stk::topology::rank_t elemRank = stk::topology::ELEMENT_RANK;
  using Traits = NGPMeshTraits<Mesh>;
  using TeamPolicy = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex = typename Traits::MeshIndex;

  const auto& ndim = meshInfo.ndim();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto& numFields = meshInfo.num_fields();

  ElemDataRequestsGPU faceDataNGP(fieldMgr, faceDataReqs, numFields);
  ElemDataRequestsGPU elemDataNGP(fieldMgr, elemDataReqs, numFields);

  const int nodesPerElement = nodes_per_entity(elemDataNGP);
  const int nodesPerFace = nodes_per_entity(faceDataNGP, METype::FACE);
  NGP_ThrowRequire(nodesPerElement != 0);
  NGP_ThrowRequire(nodesPerFace != 0);

  const int bytes_per_team = 0;
  const int bytes_per_thread =
    impl::ngp_calc_thread_shmem_size<double>(ndim, faceDataNGP, elemDataNGP);

  const auto& buckets = ngpMesh.get_bucket_ids(sideRank, sel);
  auto team_exec = impl::ngp_mesh_team_policy<TeamPolicy>(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const TeamHandleType& team) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = ngpMesh.get_bucket(sideRank, bktId);

      typename Traits::ScratchViewsType faceViews(
        team, ndim, nodesPerFace, faceDataNGP);
      typename Traits::ScratchViewsType elemViews(
        team, ndim, nodesPerElement, elemDataNGP);
      faceViews.fill_static_meviews(faceDataNGP);
      elemViews.fill_static_meviews(elemDataNGP);

      const size_t bktLen = bkt.size();
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, bktLen), [&](const size_t& bktIndex) {
          MeshIndex meshIdx{&bkt, static_cast<unsigned>(bktIndex)};
          const auto face = bkt[bktIndex];
          const auto faceIdx = ngpMesh.fast_mesh_index(face);
          const auto elements = ngpMesh.get_elements(sideRank, faceIdx);

          NGP_ThrowAssert(elements.size() == 1);
          const auto faceOrd =
            ngpMesh.get_element_ordinals(sideRank, faceIdx)[0];
          const auto elem = elements[0];
          const auto elemIdx = ngpMesh.fast_mesh_index(elem);

          // Fill up face data
          fill_pre_req_data(faceDataNGP, ngpMesh, sideRank, face, faceViews);
          fill_master_element_views(faceDataNGP, faceViews);
          // Fill up element data
          fill_pre_req_data(elemDataNGP, ngpMesh, elemRank, elem, elemViews);
          fill_master_element_views(elemDataNGP, elemViews);

          const BcFaceElemInfo<Mesh> faceinfo{
            meshIdx, face, elem, ngpMesh.get_nodes(elemRank, elemIdx), faceOrd};

          algorithm(faceinfo, elemViews, faceViews);
        });
    });
}

} // namespace nalu_ngp
} // namespace nalu
} // namespace sierra

#endif /* NGPLOOPUTILS_H */
