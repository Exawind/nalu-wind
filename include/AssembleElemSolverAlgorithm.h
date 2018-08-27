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
#include<ElemDataRequests.h>
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
    const int lhsSize = rhsSize_*rhsSize_;
    const int scratchIdsSize = rhsSize_;

   const int bytes_per_team = 0;
   const int bytes_per_thread = calculate_shared_mem_bytes_per_thread(lhsSize, rhsSize_, scratchIdsSize,
                                                                    meta_data.spatial_dimension(), dataNeededByKernels_);
   stk::mesh::Selector elemSelector =
           meta_data.locally_owned_part()
         & stk::mesh::selectUnion(partVec_)
         & !realm_.get_inactive_selector();
 
   stk::mesh::BucketVector const& elem_buckets =
           realm_.get_buckets(entityRank_, elemSelector );
 
   ngp::Mesh ngpMesh(bulk_data);
   int totalNumFields = meta_data.get_fields().size();

   auto team_exec = sierra::nalu::get_device_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);
   Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team)
   {
     stk::mesh::Bucket & b = *elem_buckets[team.league_rank()];
 
     ThrowAssertMsg(b.topology().num_nodes() == (unsigned)nodesPerEntity_,
                    "AssembleElemSolverAlgorithm expected nodesPerEntity_ = "
                    <<nodesPerEntity_<<", but b.topology().num_nodes() = "<<b.topology().num_nodes());
 
     SharedMemData smdata(team, ngpMesh, totalNumFields, dataNeededByKernels_, nodesPerEntity_, rhsSize_);

     const size_t bucketLen   = b.size();
     const size_t simdBucketLen = get_num_simd_groups(bucketLen);
 
     Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&](const size_t& bktIndex)
     {
       int numSimdElems = get_length_of_next_simd_group(bktIndex, bucketLen);
       smdata.numSimdElems = numSimdElems;
 
       for(int simdElemIndex=0; simdElemIndex<numSimdElems; ++simdElemIndex) {
         stk::mesh::Entity entity = b[bktIndex*simdLen + simdElemIndex];
         stk::mesh::FastMeshIndex entityIndex = ngpMesh.fast_mesh_index(entity);
         smdata.elemNodes[simdElemIndex] = ngpMesh.get_nodes(entityRank_, entityIndex);
         fill_pre_req_data(dataNeededByKernels_, ngpMesh, entityRank_, entity,
                           *smdata.prereqData[simdElemIndex], interleaveMEViews_);
       }
 
       copy_and_interleave(smdata.prereqData, numSimdElems, smdata.simdPrereqData, interleaveMEViews_);
 
       if (!interleaveMEViews_) {
         fill_master_element_views(dataNeededByKernels_, smdata.simdPrereqData);
       }

       lambdaFunc(smdata);
     });
   });
  }

  ElemDataRequests dataNeededByKernels_;
  stk::mesh::EntityRank entityRank_;
  unsigned nodesPerEntity_;
  int rhsSize_;
  const bool interleaveMEViews_;
};

} // namespace nalu
} // namespace Sierra

#endif

