// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssembleFaceElemSolverAlgorithm_h
#define AssembleFaceElemSolverAlgorithm_h

#include <SolverAlgorithm.h>
#include <ElemDataRequests.h>
#include <Realm.h>
#include <ScratchViews.h>
#include <SimdInterface.h>
#include <SharedMemData.h>
#include <CopyAndInterleave.h>
#include <stk_mesh/base/NgpMesh.hpp>
#include <ngp_utils/NgpFieldManager.h>
#include <ngp_utils/NgpMEUtils.h>

namespace stk {
namespace mesh {
class Part;
}
} // namespace stk

namespace sierra {
namespace nalu {

class AssembleFaceElemSolverAlgorithm : public SolverAlgorithm
{
public:
  AssembleFaceElemSolverAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    EquationSystem* eqSystem,
    unsigned nodesPerFace,
    unsigned nodesPerElem);
  virtual ~AssembleFaceElemSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  template <typename LambdaFunction>
  void
  run_face_elem_algorithm(stk::mesh::BulkData& bulk, LambdaFunction lamdbaFunc)
  {
    int nDim = bulk.mesh_meta_data().spatial_dimension();
    int totalNumFields = bulk.mesh_meta_data().get_fields().size();

    // Register face ME instance in elemdata also to obtain face integration
    // points
    if (elemDataNeeded_.get_cvfem_face_me() == nullptr)
      elemDataNeeded_.add_cvfem_face_me(faceDataNeeded_.get_cvfem_face_me());

    int rhsSize = nodesPerElem_ * numDof_, lhsSize = rhsSize * rhsSize,
        scratchIdsSize = rhsSize;

    const stk::mesh::NgpMesh& ngpMesh = realm_.ngp_mesh();
    const nalu_ngp::FieldManager& fieldMgr = realm_.ngp_field_manager();
    ElemDataRequestsGPU faceDataNGP(fieldMgr, faceDataNeeded_, totalNumFields);
    ElemDataRequestsGPU elemDataNGP(fieldMgr, elemDataNeeded_, totalNumFields);

    const int bytes_per_team = 0;
    const int bytes_per_thread = calculate_shared_mem_bytes_per_thread(
      lhsSize, rhsSize, scratchIdsSize, nDim, faceDataNGP, elemDataNGP);

    const auto nodesPerFace = nodesPerFace_;
    const auto nodesPerElem = nodesPerElem_;
    stk::mesh::Selector s_locally_owned_union =
      bulk.mesh_meta_data().locally_owned_part() &
      stk::mesh::selectUnion(partVec_);
    stk::mesh::EntityRank sideRank = bulk.mesh_meta_data().side_rank();
    const auto& buckets =
      stk::mesh::get_bucket_ids(bulk, sideRank, s_locally_owned_union);

    auto team_exec = sierra::nalu::get_device_team_policy(
      buckets.size(), bytes_per_team, bytes_per_thread);
    Kokkos::parallel_for(
      team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
        auto bktId = buckets.device_get(team.league_rank());
        auto& b = ngpMesh.get_bucket(sideRank, bktId);

#if !defined(KOKKOS_ENABLE_GPU)
        ThrowAssertMsg(
          b.topology().num_nodes() == (unsigned)nodesPerFace_,
          "AssembleFaceElemSolverAlgorithm expected nodesPerEntity_ = "
            << nodesPerFace_
            << ", but b.topology().num_nodes() = " << b.topology().num_nodes());
#endif

        SharedMemData_FaceElem<DeviceTeamHandleType, DeviceShmem> smdata(
          team, nDim, faceDataNGP, elemDataNGP, nodesPerFace, nodesPerElem,
          rhsSize);

        const size_t bucketLen = b.size();
        const size_t simdBucketLen =
          sierra::nalu::get_num_simd_groups(bucketLen);

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, simdBucketLen),
          [&](const size_t& bktIndex) {
            size_t simdGroupLen =
              sierra::nalu::get_length_of_next_simd_group(bktIndex, bucketLen);
            size_t numFacesProcessed = 0;
            do {
              int elemFaceOrdinal = -1;
              int simdFaceIndex = 0;
              while ((numFacesProcessed + simdFaceIndex) < simdGroupLen) {
                stk::mesh::Entity face =
                  b[bktIndex * simdLen + numFacesProcessed + simdFaceIndex];
                const auto ngpFaceIndex = ngpMesh.fast_mesh_index(face);
                // ThrowAssertMsg(
                //   bulk.num_elements(face) == 1,
                //   "Expecting just 1 element attached to face!");
                int thisElemFaceOrdinal =
                  ngpMesh.get_element_ordinals(sideRank, ngpFaceIndex)[0];

                if (
                  elemFaceOrdinal >= 0 &&
                  thisElemFaceOrdinal != elemFaceOrdinal) {
                  break;
                }

                const auto& elems =
                  ngpMesh.get_elements(sideRank, ngpFaceIndex);
                const auto elemIndex = ngpMesh.fast_mesh_index(elems[0]);

                smdata.ngpConnectedNodes[simdFaceIndex] =
                  ngpMesh.get_nodes(stk::topology::ELEMENT_RANK, elemIndex);
                smdata.elemFaceOrdinal = thisElemFaceOrdinal;
                elemFaceOrdinal = thisElemFaceOrdinal;
                sierra::nalu::fill_pre_req_data(
                  faceDataNGP, ngpMesh, sideRank, face,
                  *smdata.faceViews[simdFaceIndex]);
                sierra::nalu::fill_pre_req_data(
                  elemDataNGP, ngpMesh, stk::topology::ELEMENT_RANK, elems[0],
                  *smdata.elemViews[simdFaceIndex]);
                ++simdFaceIndex;
              }
              smdata.numSimdFaces = simdFaceIndex;
              numFacesProcessed += simdFaceIndex;

#if !defined(KOKKOS_ENABLE_GPU)
              // No need to interleave on GPUs
              copy_and_interleave(
                smdata.faceViews, smdata.numSimdFaces, smdata.simdFaceViews);
              copy_and_interleave(
                smdata.elemViews, smdata.numSimdFaces, smdata.simdElemViews);
#endif

              fill_master_element_views(
                faceDataNGP, smdata.simdFaceViews, smdata.elemFaceOrdinal);
              fill_master_element_views(
                elemDataNGP, smdata.simdElemViews, smdata.elemFaceOrdinal);

              lamdbaFunc(smdata);
            } while (numFacesProcessed < simdGroupLen);
          });
      });
  }

  ElemDataRequests faceDataNeeded_;
  ElemDataRequests elemDataNeeded_;
  double diagRelaxFactor_{1.0};
  unsigned numDof_;
  unsigned nodesPerFace_;
  unsigned nodesPerElem_;
  int rhsSize_;
};

} // namespace nalu
} // namespace sierra

#endif
