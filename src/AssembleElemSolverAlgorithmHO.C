/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleElemSolverAlgorithmHO.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <SolverAlgorithm.h>
#include <TimeIntegrator.h>
#include <kernel/Kernel.h>
#include <element_promotion/NodeMapMaker.h>
#include <element_promotion/ElementDescription.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk topo
#include <stk_topology/topology.hpp>

#include <ScratchViewsHO.h>
#include <CopyAndInterleave.h>

namespace sierra{
namespace nalu{

AssembleElemSolverAlgorithmHO::AssembleElemSolverAlgorithmHO(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  stk::mesh::EntityRank entityRank,
  unsigned nodesPerEntity)
  : SolverAlgorithm(realm, part, eqSystem),
    dim_(realm.spatialDimension_),
    polyOrder_(poly_order_from_topology(dim_, part->topology())),
    ndof_(eqSystem->linsys_->numDof()),
    entityRank_(entityRank),
    nodesPerEntity_(nodesPerEntity),
    rhsSize_(nodesPerEntity * ndof_),
    lhsSize_(rhsSize_*rhsSize_),
    defaultPermutation_(make_node_map_hex(polyOrder_, part->topology().is_super_topology())),
    gatherer_(polyOrder_+1, defaultPermutation_),
    dataNeededByKernels_(realm.meta_data())
{
  vecDefaultPermutation_ = Kokkos::View<int*>("inverse_permutation", rhsSize_);
  for (int j = 0; j < nodesPerEntity_; ++j) {
    const int permuted_j = defaultPermutation_.data()[j];
    for (int d = 0; d < ndof_; ++d) {
      vecDefaultPermutation_[j * ndof_ + d] = permuted_j * ndof_ + d;
    }
  }
}
//--------------------------------------------------------------------------
void
AssembleElemSolverAlgorithmHO::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}
//--------------------------------------------------------------------------
namespace {
void extract_permuted_vector_lanes(
  const SharedMemView<DoubleType**>& simdlhs,
  const SharedMemView<DoubleType*>& simdrhs,
  const Kokkos::View<int*>& vecDefaultPermutation,
  int simdIndex,
  SharedMemView<double**>& lhs,
  SharedMemView<double*>& rhs)
{
  // extract vector lanes + permute ordinals back from the tensor-product ordering to the
  // topology-based ordering used by the linear system.
  const int rhsSize = rhs.size();
  for (int j = 0; j < rhsSize; ++j) {
    const int permuted_j = vecDefaultPermutation[j];
    rhs[permuted_j] = stk::simd::get_data(simdrhs[j], simdIndex);
    double* lhs_row = &lhs(permuted_j, 0);
    for (int i = 0; i < rhsSize; ++i) {
      lhs_row[vecDefaultPermutation[i]] = stk::simd::get_data(simdlhs(j,i), simdIndex);
    }
  }
}
}
//--------------------------------------------------------------------------
void
AssembleElemSolverAlgorithmHO::execute()
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  // set any data
  const size_t activeKernelsSize = activeKernels_.size();
  for ( size_t i = 0; i < activeKernelsSize; ++i )
    activeKernels_[i]->setup(*realm_.timeIntegrator_);

  run_algorithm(bulk_data, [&](SharedMemDataHO& smdata)
  {
    set_zero(smdata.simdrhs.data(), rhsSize_);
    set_zero(smdata.simdlhs.data(), lhsSize_);

    // call supplemental; gathers happen inside the elem_execute method
    for ( size_t i = 0; i < activeKernelsSize; ++i )
      activeKernels_[i]->execute( smdata.simdlhs, smdata.simdrhs, smdata.simdPrereqData );

    for(int simdElemIndex=0; simdElemIndex< smdata.simdPrereqData.numSimdElems; ++simdElemIndex) {
      extract_permuted_vector_lanes(smdata.simdlhs, smdata.simdrhs, vecDefaultPermutation_, simdElemIndex,
        smdata.lhs, smdata.rhs);

      apply_coeff(nodesPerEntity_, smdata.simdPrereqData.elemNodes[simdElemIndex],
        smdata.scratchIds, smdata.sortPermutation, smdata.rhs, smdata.lhs, __FILE__);
    }
  });
}

} // namespace nalu
} // namespace Sierra
