/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleFaceElemSolverAlgorithm_h
#define AssembleFaceElemSolverAlgorithm_h

#include <SolverAlgorithm.h>
#include <ElemDataRequests.h>
#include <Realm.h>
#include <ScratchViews.h>
#include <SimdInterface.h>
#include <SharedMemData.h>
#include <CopyAndInterleave.h>

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra{
namespace nalu{

class AssembleFaceElemSolverAlgorithm : public SolverAlgorithm
{
public:
  AssembleFaceElemSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    unsigned nodesPerFace,
    unsigned nodesPerElem,
    bool interleaveMeViews = true);
  virtual ~AssembleFaceElemSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  template<typename LambdaFunction>
  void run_face_elem_algorithm(stk::mesh::BulkData& bulk, LambdaFunction lamdbaFunc)
  {
      int nDim = bulk.mesh_meta_data().spatial_dimension();
      int totalNumFields = bulk.mesh_meta_data().get_fields().size();

      sierra::nalu::MasterElement* meFC = faceDataNeeded_.get_cvfem_face_me();
      sierra::nalu::MasterElement* meSCS = faceDataNeeded_.get_cvfem_surface_me();
      sierra::nalu::MasterElement* meSCV = faceDataNeeded_.get_cvfem_volume_me();

      sierra::nalu::ScratchMeInfo meElemInfo;
      meElemInfo.nodalGatherSize_ = nodesPerElem_;
      meElemInfo.nodesPerFace_ = nodesPerFace_;
      meElemInfo.nodesPerElement_ = nodesPerElem_;
      meElemInfo.numFaceIp_ = meFC != nullptr ? meFC->numIntPoints_ : 0;
      meElemInfo.numScsIp_ = meSCS != nullptr ? meSCS->numIntPoints_ : 0;
      meElemInfo.numScvIp_ = meSCV != nullptr ? meSCV->numIntPoints_ : 0;

      int rhsSize = meElemInfo.nodalGatherSize_*numDof_, lhsSize = rhsSize*rhsSize, scratchIdsSize = rhsSize;

      const ngp::Mesh& ngpMesh = realm_.ngp_mesh();
      const ngp::FieldManager& fieldMgr = realm_.ngp_field_manager();
      ElemDataRequestsGPU faceDataNGP(fieldMgr, faceDataNeeded_, totalNumFields);
      ElemDataRequestsGPU elemDataNGP(fieldMgr, elemDataNeeded_, totalNumFields);

      const int bytes_per_team = 0;
      const int bytes_per_thread = calculate_shared_mem_bytes_per_thread(
        lhsSize, rhsSize, scratchIdsSize, nDim, faceDataNGP, elemDataNGP,
        meElemInfo);

      const bool interleaveMeViews = false;

      stk::mesh::Selector s_locally_owned_union =
        bulk.mesh_meta_data().locally_owned_part() &
        stk::mesh::selectUnion(partVec_);
      stk::mesh::EntityRank sideRank = bulk.mesh_meta_data().side_rank();
      const auto& buckets = ngp::get_bucket_ids(bulk, sideRank, s_locally_owned_union);

      auto team_exec = sierra::nalu::get_device_team_policy(
        buckets.size(), bytes_per_team, bytes_per_thread);
      Kokkos::parallel_for(
        team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
          auto bktId = buckets.device_get(team.league_rank());
          auto& b = ngpMesh.get_bucket(sideRank, bktId);

#ifndef KOKKOS_ENABLE_CUDA
          ThrowAssertMsg(
            b.topology().num_nodes() == (unsigned)nodesPerFace_,
            "AssembleFaceElemSolverAlgorithm expected nodesPerEntity_ = "
              << nodesPerFace_ << ", but b.topology().num_nodes() = "
              << b.topology().num_nodes());
#endif

          SharedMemData_FaceElem<DeviceTeamHandleType, DeviceShmem> smdata(
            team, nDim, faceDataNGP, elemDataNGP, meElemInfo, rhsSize);

          const size_t bucketLen = b.size();
          const size_t simdBucketLen =
            sierra::nalu::get_num_simd_groups(bucketLen);

          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, simdBucketLen),
            [&](const size_t& bktIndex) {
              size_t simdGroupLen = sierra::nalu::get_length_of_next_simd_group(
                bktIndex, bucketLen);
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

                  if (elemFaceOrdinal >= 0 && thisElemFaceOrdinal != elemFaceOrdinal) {
                    break;
                  }

                  const auto& elems = ngpMesh.get_elements(sideRank, ngpFaceIndex);
                  const auto elemIndex = ngpMesh.fast_mesh_index(elems[0]);

                  smdata.ngpConnectedNodes[simdFaceIndex] =
                    ngpMesh.get_nodes(stk::topology::ELEMENT_RANK, elemIndex);
                  smdata.elemFaceOrdinal = thisElemFaceOrdinal;
                  elemFaceOrdinal = thisElemFaceOrdinal;
                  sierra::nalu::fill_pre_req_data(
                    faceDataNGP, ngpMesh, sideRank, face,
                    *smdata.faceViews[simdFaceIndex], interleaveMeViews);
                  sierra::nalu::fill_pre_req_data(
                    elemDataNGP, ngpMesh, stk::topology::ELEMENT_RANK, elems[0],
                    *smdata.elemViews[simdFaceIndex], interleaveMeViews);
                  ++simdFaceIndex;
                }
                smdata.numSimdFaces = simdFaceIndex;
                numFacesProcessed += simdFaceIndex;

#ifndef KOKKOS_ENABLE_CUDA
                // When we GPU-ize AssembleFaceElemSolverAlgorithm, 'lambdaFunc'
                // below will need to operate on the non-simd smdata.faceViews
                // etc... since we aren't going to copy_and_interleave. We will
                // probably want to make smdata.simdFaceViews be a
                // pointer/reference to smdata.faceViews[0] in some way...
                copy_and_interleave(
                  smdata.faceViews, smdata.numSimdFaces, smdata.simdFaceViews,
                  interleaveMeViews);
                copy_and_interleave(
                  smdata.elemViews, smdata.numSimdFaces, smdata.simdElemViews,
                  interleaveMeViews);

                fill_master_element_views(
                  faceDataNGP, smdata.simdFaceViews, smdata.elemFaceOrdinal);
                fill_master_element_views(
                  elemDataNGP, smdata.simdElemViews, smdata.elemFaceOrdinal);
//for now this simply isn't ready for GPU.
#endif

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
  const bool interleaveMEViews_;
};

} // namespace nalu
} // namespace Sierra

#endif

