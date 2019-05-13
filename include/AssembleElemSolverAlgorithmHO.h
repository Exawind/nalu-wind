/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleElemSolverAlgorithmHO_h
#define AssembleElemSolverAlgorithmHO_h

#include <Realm.h>
#include <SolverAlgorithm.h>
#include <ElemDataRequests.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <CopyAndInterleave.h>
#include <FieldTypeDef.h>

#include "ScratchViewsHO.h"


namespace stk {
namespace mesh {
class Part;
class Topology;
}
}

namespace sierra{
namespace nalu{

class MasterElement;


inline std::array<stk::mesh::Entity, simdLen>
load_simd_elems(const stk::mesh::Bucket& b, int bktIndex, int /* bucketLen */, int numSimdElems)
{
  std::array<stk::mesh::Entity, simdLen> simdElems;
  for (int simdElemIndex = 0; simdElemIndex < numSimdElems; ++simdElemIndex) {
    simdElems[simdElemIndex] = b[bktIndex*simdLen + simdElemIndex];
  }
  return simdElems;
}

struct FieldGatherer
{
public:
  // Gather fields with ordinals permuted into a tensor product order

  FieldGatherer(int n1D, const Kokkos::View<int***>& defaultPermutation) :
    n1D_(n1D), defaultPermutation_(defaultPermutation) {};

  void gather_elem_node_field_3D(const stk::mesh::FieldBase& field,
    const std::array<const stk::mesh::Entity*, simdLen>& elemNodes,
    int numSimdElems,
    SharedMemView<DoubleType***>& shmemView) const
  {
    for (int k = 0; k < n1D_; ++k) {
      for (int j = 0; j < n1D_; ++j) {
        for (int i = 0; i < n1D_; ++i) {
          const int permutationIndex = defaultPermutation_(k,j,i);
          for (int simdIndex = 0; simdIndex < numSimdElems; ++simdIndex) {
            stk::simd::set_data(shmemView(k, j, i), simdIndex,
              *static_cast<const double*>(stk::mesh::field_data(field, elemNodes[simdIndex][permutationIndex])));
          }
        }
      }
    }
  }

  void gather_elem_node_field_3D(const stk::mesh::FieldBase& field,
    const std::array<const stk::mesh::Entity*, simdLen>& elemNodes,
    int numSimdElems,
    SharedMemView<DoubleType****>& shmemView) const
  {
    for (int k = 0; k < n1D_; ++k) {
      for (int j = 0; j < n1D_; ++j) {
        for (int i = 0; i < n1D_; ++i) {
          const int permutationIndex = defaultPermutation_(k,j,i);
          for (int simdIndex = 0; simdIndex < numSimdElems; ++simdIndex) {
            const double* dataPtr = static_cast<const double*>(stk::mesh::field_data(field,
              elemNodes[simdIndex][permutationIndex]));
            stk::simd::set_data(shmemView(k, j, i, 0), simdIndex, dataPtr[0]);
            stk::simd::set_data(shmemView(k, j, i, 1), simdIndex, dataPtr[1]);
            stk::simd::set_data(shmemView(k, j, i, 2), simdIndex, dataPtr[2]);
          }
        }
      }
    }
  }

  const int n1D_;
private:
  const Kokkos::View<int***> defaultPermutation_;
};

inline void fill_pre_req_data(
  const FieldGatherer& gatherer,
  ElemDataRequests& dataNeeded,
  const stk::mesh::BulkData& bulkData,
  const std::array<stk::mesh::Entity, simdLen>& elems,
  int numSimdElems,
  ScratchViewsHO<DoubleType>& prereqData)
{
  for (int simdIndex = 0; simdIndex < numSimdElems; ++simdIndex) {
    prereqData.elemNodes[simdIndex] = bulkData.begin_nodes(elems[simdIndex]);
  }
  prereqData.numSimdElems = numSimdElems;

  const FieldSet& neededFields = dataNeeded.get_fields();
  for(const FieldInfo& fieldInfo : neededFields) {
    stk::mesh::EntityRank fieldEntityRank = fieldInfo.field->entity_rank();
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;
    if (fieldEntityRank == stk::topology::NODE_RANK ) {
      if (scalarsDim1 == 1u) {
        auto shmemView3D = prereqData.get_scratch_view<SharedMemView<DoubleType***>>(*fieldInfo.field,
            gatherer.n1D_, gatherer.n1D_, gatherer.n1D_);
        gatherer.gather_elem_node_field_3D(*fieldInfo.field, prereqData.elemNodes, prereqData.numSimdElems,
          shmemView3D);
      }
      else {
        auto shmemView4D = prereqData.get_scratch_view<SharedMemView<DoubleType****>>(*fieldInfo.field,
            gatherer.n1D_, gatherer.n1D_, gatherer.n1D_, 3);
        gatherer.gather_elem_node_field_3D(*fieldInfo.field, prereqData.elemNodes, prereqData.numSimdElems,
          shmemView4D);
      }
    }
  }
}

struct SharedMemDataHO {
  SharedMemDataHO(const sierra::nalu::TeamHandleType& team,
    const stk::mesh::BulkData& bulk,
    const ElemDataRequests& dataNeededByKernels,
    int order, int dim,
    unsigned rhsSize)
  : simdPrereqData(team, bulk, order, dim, dataNeededByKernels)
  {
    simdrhs = get_shmem_view_1D<DoubleType>(team, rhsSize);
    simdlhs = get_shmem_view_2D<DoubleType>(team, rhsSize, rhsSize);
    rhs = get_shmem_view_1D<double>(team, rhsSize);
    lhs = get_shmem_view_2D<double>(team, rhsSize, rhsSize);

    scratchIds = get_shmem_view_1D<int>(team, rhsSize);
    sortPermutation = get_shmem_view_1D<int>(team, rhsSize);
  }

  ScratchViewsHO<DoubleType> simdPrereqData;

  SharedMemView<DoubleType*> simdrhs;
  SharedMemView<DoubleType**> simdlhs;
  SharedMemView<double*> rhs;
  SharedMemView<double**> lhs;

  SharedMemView<int*> scratchIds;
  SharedMemView<int*> sortPermutation;
};

inline int num_scalars_pre_req_data_HO(int nodesPerEntity, const ElemDataRequests& elemDataNeeded)
{
  int numScalars = 0;
  const FieldSet& neededFields = elemDataNeeded.get_fields();
  for(const FieldInfo& fieldInfo : neededFields) {
    stk::mesh::EntityRank fieldEntityRank = fieldInfo.field->entity_rank();
    unsigned scalarsPerEntity = fieldInfo.scalarsDim1;
    unsigned entitiesPerElem = fieldEntityRank==stk::topology::NODE_RANK ? nodesPerEntity : 1;
    ThrowRequire(entitiesPerElem > 0);
    if (fieldInfo.scalarsDim2 > 1) {
      scalarsPerEntity *= fieldInfo.scalarsDim2;
    }
    numScalars += entitiesPerElem*scalarsPerEntity;
  }
  return numScalars;
}

inline int calculate_shared_mem_bytes_per_thread_HO(int lhsSize, int rhsSize, int scratchIdsSize, int order, int dim, const ElemDataRequests& elemDataNeeded)
{
  int bytes_per_thread = ((rhsSize + lhsSize) * sizeof(double) + (2 * scratchIdsSize) * sizeof(int));
  bytes_per_thread += sizeof(double) * num_scalars_pre_req_data_HO(std::pow(order+1,dim), elemDataNeeded);
  bytes_per_thread *= 2*simdLen;
  return bytes_per_thread;
}


class AssembleElemSolverAlgorithmHO : public SolverAlgorithm
{
public:
  AssembleElemSolverAlgorithmHO(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    stk::mesh::EntityRank entityRank,
    unsigned nodesPerEntity);
  virtual ~AssembleElemSolverAlgorithmHO() = default;
  virtual void initialize_connectivity();
  virtual void execute();

  template<typename LambdaFunction>
  void run_algorithm(stk::mesh::BulkData& bulk_data, LambdaFunction lambdaFunc)
  {
    const stk::mesh::MetaData& meta_data = bulk_data.mesh_meta_data();
    const int lhsSize = rhsSize_*rhsSize_;
    const int scratchIdsSize = rhsSize_;

   const int bytes_per_team = 0;
   const int bytes_per_thread = calculate_shared_mem_bytes_per_thread_HO(lhsSize, rhsSize_, scratchIdsSize,
     polyOrder_, meta_data.spatial_dimension(), dataNeededByKernels_);
   stk::mesh::Selector elemSelector =
           meta_data.locally_owned_part()
         & stk::mesh::selectUnion(partVec_)
         & !realm_.get_inactive_selector();
 
   stk::mesh::BucketVector const& elem_buckets = realm_.get_buckets(entityRank_, elemSelector);
 
   auto team_exec = sierra::nalu::get_host_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);
   Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team)
   {
     stk::mesh::Bucket & b = *elem_buckets[team.league_rank()];
 
     ThrowAssertMsg(b.topology().num_nodes() == (unsigned)nodesPerEntity_,
                    "AssembleElemSolverAlgorithmHO expected nodesPerEntity_ = "
                    <<nodesPerEntity_<<", but b.topology().num_nodes() = "<<b.topology().num_nodes());
 
     SharedMemDataHO smdata(team, bulk_data, dataNeededByKernels_, polyOrder_, dim_, rhsSize_);
     const size_t bucketLen = b.size();
     const size_t simdBucketLen = get_num_simd_groups(bucketLen);
 
     Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&](const size_t& bktIndex)
     {
       const int numSimdElems = get_length_of_next_simd_group(bktIndex, bucketLen);
       const auto simdElems = load_simd_elems(b, bktIndex, bucketLen, numSimdElems);
       fill_pre_req_data(gatherer_, dataNeededByKernels_, bulk_data, simdElems, numSimdElems, smdata.simdPrereqData);
       lambdaFunc(smdata);
     });
   });
  }

  const int dim_;
  const int polyOrder_;
  const int ndof_;

  const stk::mesh::EntityRank entityRank_;
  const int nodesPerEntity_;
  const int rhsSize_;
  const int lhsSize_;

  const Kokkos::View<int***> defaultPermutation_;
  Kokkos::View<int*> vecDefaultPermutation_;
  const FieldGatherer gatherer_;

  ElemDataRequests dataNeededByKernels_;
};

} // namespace nalu
} // namespace Sierra

#endif

