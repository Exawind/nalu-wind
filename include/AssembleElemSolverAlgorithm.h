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
#include<ElemDataRequestsNGP.h>
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

   ElemDataRequestsNGP dataNeededNGP(dataNeededByKernels_, meta_data.get_fields().size());

   const int bytes_per_team = 0;
   const int bytes_per_thread = calculate_shared_mem_bytes_per_thread(lhsSize, rhsSize_, scratchIdsSize,
                                                                    meta_data.spatial_dimension(), dataNeededNGP);
   stk::mesh::Selector elemSelector =
           meta_data.locally_owned_part()
         & stk::mesh::selectUnion(partVec_)
         & !realm_.get_inactive_selector();
 
   stk::mesh::BucketVector const& elem_buckets =
           realm_.get_buckets(entityRank_, elemSelector );
 
   auto team_exec = sierra::nalu::get_host_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);
   Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team)
   {
     stk::mesh::Bucket & b = *elem_buckets[team.league_rank()];
 
     ThrowAssertMsg(b.topology().num_nodes() == (unsigned)nodesPerEntity_,
                    "AssembleElemSolverAlgorithm expected nodesPerEntity_ = "
                    <<nodesPerEntity_<<", but b.topology().num_nodes() = "<<b.topology().num_nodes());
 
     SharedMemData<TeamHandleType,HostShmem> smdata(team, meta_data.spatial_dimension(), dataNeededNGP, nodesPerEntity_, rhsSize_);

     const size_t bucketLen   = b.size();
     const size_t simdBucketLen = get_num_simd_groups(bucketLen);
 
     Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&](const size_t& bktIndex)
     {
       int numSimdElems = get_length_of_next_simd_group(bktIndex, bucketLen);
       smdata.numSimdElems = numSimdElems;
 
       for(int simdElemIndex=0; simdElemIndex<numSimdElems; ++simdElemIndex) {
         stk::mesh::Entity element = b[bktIndex*simdLen + simdElemIndex];
         smdata.elemNodes[simdElemIndex] = bulk_data.begin_nodes(element);
         fill_pre_req_data(dataNeededNGP, bulk_data, element,
                           *smdata.prereqData[simdElemIndex], interleaveMEViews_);
       }
 
#ifndef KOKKOS_ENABLE_CUDA
//When we GPU-ize AssembleElemSolverAlgorithm, 'lambdaFunc' below will need to operate
//on smdata.prereqData[0] since we aren't going to copy_and_interleave. We will probably
//want to make smdata.simdPrereqData to be a pointer/reference to smdata.prereqData[0] in some way...
       copy_and_interleave(smdata.prereqData, numSimdElems, smdata.simdPrereqData, interleaveMEViews_);
//for now this simply isn't ready for GPU.
#endif
 
       if (!interleaveMEViews_) {
         fill_master_element_views(dataNeededNGP, smdata.simdPrereqData);
       }

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

