/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleElemSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>
#include <master_element/MasterElement.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <TimeIntegrator.h>

#include <kernel/Kernel.h>
#include <NGPInstance.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk topo
#include <stk_topology/topology.hpp>

#include <KokkosInterface.h>
#include <ScratchViews.h>
#include <CopyAndInterleave.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleElemSolverAlgorithm - add LHS/RHS for element-based contribution
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleElemSolverAlgorithm::AssembleElemSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  stk::mesh::EntityRank entityRank,
  unsigned nodesPerEntity)
  : SolverAlgorithm(realm, part, eqSystem),
    dataNeededByKernels_(realm.meta_data()),
    entityRank_(entityRank),
    nodesPerEntity_(nodesPerEntity),
    rhsSize_(nodesPerEntity*eqSystem->linsys_->numDof())
{
  if (eqSystem->dofName_ != "pressure") {
    diagRelaxFactor_ = realm.solutionOptions_->get_relaxation_factor(
      eqSystem->dofName_);
  }
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleElemSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleElemSolverAlgorithm::execute()
{
  const size_t numKernels = activeKernels_.size();
  for ( size_t i = 0; i < numKernels; ++i )
    activeKernels_[i]->setup(*realm_.timeIntegrator_);

  auto ngpKernels = nalu_ngp::create_ngp_view<Kernel>(activeKernels_);

#ifdef KOKKOS_ENABLE_CUDA
  CoeffApplier* coeffApplier = eqSystem_->linsys_->get_coeff_applier();
  CoeffApplier* deviceCoeffApplier = coeffApplier->device_pointer();

  double diagRelaxFactor = diagRelaxFactor_;
  int rhsSize = rhsSize_;
  unsigned nodesPerEntity = nodesPerEntity_;
#endif

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(SharedMemData<DeviceTeamHandleType, DeviceShmem> & smdata) {
      set_zero(smdata.simdrhs.data(), smdata.simdrhs.size());
      set_zero(smdata.simdlhs.data(), smdata.simdlhs.size());
      for (size_t i=0; i < numKernels; i++) {
        Kernel* kernel = ngpKernels(i);
        kernel->execute(smdata.simdlhs, smdata.simdrhs, smdata.simdPrereqData);
      }

#ifndef KOKKOS_ENABLE_CUDA
      for(int simdElemIndex=0; simdElemIndex<smdata.numSimdElems; ++simdElemIndex) {
        extract_vector_lane(smdata.simdrhs, simdElemIndex, smdata.rhs);
        extract_vector_lane(smdata.simdlhs, simdElemIndex, smdata.lhs);
        for (int ir=0; ir < rhsSize_; ++ir)
          smdata.lhs(ir, ir) /= diagRelaxFactor_;
        apply_coeff(nodesPerEntity_, smdata.ngpElemNodes[simdElemIndex],
                    smdata.scratchIds, smdata.sortPermutation, smdata.rhs, smdata.lhs, __FILE__);
      }
#else
      extract_vector_lane(smdata.simdrhs, 0, smdata.rhs);
      extract_vector_lane(smdata.simdlhs, 0, smdata.lhs);
        for (int ir=0; ir < rhsSize; ++ir)
          smdata.lhs(ir, ir) /= diagRelaxFactor;
        (*deviceCoeffApplier)(nodesPerEntity, smdata.ngpElemNodes[0],
                    smdata.scratchIds, smdata.sortPermutation, smdata.rhs, smdata.lhs, __FILE__);
#endif
    });

#ifdef KOKKOS_ENABLE_CUDA
  coeffApplier->free_device_pointer();
  delete coeffApplier;
#endif
}

} // namespace nalu
} // namespace Sierra
