/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleElemSolverAlgorithm_h
#define AssembleElemSolverAlgorithm_h

#include<Realm.h>
#include<SolverAlgorithm.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>
#include<ScratchViews.h>
#include <SharedMemData.h>
#include<CopyAndInterleave.h>
#include<FieldTypeDef.h>

namespace stk {
namespace mesh {
class Part;
class Topology;
}
}

namespace sierra{
namespace nalu{

class MasterElement;

class AssembleElemSolverAlgorithm : public SolverAlgorithm
{
public:
  AssembleElemSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    stk::mesh::EntityRank entityRank,
    unsigned nodesPerEntity,
    bool interleaveMeViews = true);
  virtual ~AssembleElemSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  template<typename LambdaFunction>
  void run_algorithm(stk::mesh::BulkData& bulk_data, LambdaFunction lambdaFunc)
  {
    stk::mesh::MetaData& meta_data = bulk_data.mesh_meta_data();
    const int nDim = meta_data.spatial_dimension();
    const int lhsSize = rhsSize_*rhsSize_;
    const int scratchIdsSize = rhsSize_;

    const ngp::Mesh& ngpMesh = realm_.ngp_mesh();
    const ngp::FieldManager& fieldMgr = realm_.ngp_field_manager();
    ElemDataRequestsGPU dataNeededNGP(
      fieldMgr, dataNeededByKernels_, meta_data.get_fields().size());

    const int bytes_per_team = 0;
    const int bytes_per_thread = calculate_shared_mem_bytes_per_thread(
      lhsSize, rhsSize_, scratchIdsSize, meta_data.spatial_dimension(),
      dataNeededNGP);
    stk::mesh::Selector elemSelector = meta_data.locally_owned_part() &
                                       stk::mesh::selectUnion(partVec_) &
                                       !realm_.get_inactive_selector();

    const auto& elem_buckets =
      ngp::get_bucket_ids(bulk_data, entityRank_, elemSelector);

    // Create local copies of class data
    const auto entityRank = entityRank_;
    const auto nodesPerEntity = nodesPerEntity_;
    const auto rhsSize = rhsSize_;
    const auto interleaveMEViews = interleaveMEViews_;

    auto team_exec = sierra::nalu::get_device_team_policy(
      elem_buckets.size(), bytes_per_team, bytes_per_thread);
    Kokkos::parallel_for(
      team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
        auto bktId = elem_buckets.device_get(team.league_rank());
        auto& b = ngpMesh.get_bucket(entityRank, bktId);

#ifndef KOKKOS_ENABLE_CUDA
        ThrowAssertMsg(
          b.topology().num_nodes() == (unsigned)nodesPerEntity_,
          "AssembleElemSolverAlgorithm expected nodesPerEntity_ = "
            << nodesPerEntity_
            << ", but b.topology().num_nodes() = " << b.topology().num_nodes());
#endif

        SharedMemData<DeviceTeamHandleType, DeviceShmem> smdata(
          team, nDim, dataNeededNGP, nodesPerEntity, rhsSize);

        const size_t bucketLen = b.size();
        const size_t simdBucketLen = get_num_simd_groups(bucketLen);

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, simdBucketLen), [&](const size_t& bktIndex) {
            int numSimdElems =
              get_length_of_next_simd_group(bktIndex, bucketLen);
            smdata.numSimdElems = numSimdElems;

            for (int simdElemIndex = 0; simdElemIndex < numSimdElems; ++simdElemIndex) {
              stk::mesh::Entity element = b[bktIndex * simdLen + simdElemIndex];
              const auto elemIndex = ngpMesh.fast_mesh_index(element);
              smdata.ngpElemNodes[simdElemIndex] =
                ngpMesh.get_nodes(entityRank, elemIndex);
              fill_pre_req_data(
                dataNeededNGP, ngpMesh, entityRank, element,
                *smdata.prereqData[simdElemIndex], interleaveMEViews);
            }

#ifndef KOKKOS_ENABLE_CUDA
            // When we GPU-ize AssembleElemSolverAlgorithm, 'lambdaFunc' below
            // will need to operate on smdata.prereqData[0] since we aren't going
            // to copy_and_interleave. We will probably want to make
            // smdata.simdPrereqData to be a pointer/reference to
            // smdata.prereqData[0] in some way...
            copy_and_interleave(
              smdata.prereqData, numSimdElems, smdata.simdPrereqData,
              interleaveMEViews_);
            if (!interleaveMEViews_) {
              fill_master_element_views(dataNeededNGP, smdata.simdPrereqData);
            }
//for now this simply isn't ready for GPU.
#endif

            lambdaFunc(smdata);
          });
      });
  }

  ElemDataRequests dataNeededByKernels_;
  stk::mesh::EntityRank entityRank_;

  //! Relaxation factor to be applied to the diagonal term
  double diagRelaxFactor_{1.0};
  unsigned nodesPerEntity_;
  int rhsSize_;
  const bool interleaveMEViews_;
};

} // namespace nalu
} // namespace Sierra

#endif

