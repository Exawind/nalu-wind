// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

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

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// AssembleElemSolverAlgorithm - add LHS/RHS for element-based contribution
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleElemSolverAlgorithm::AssembleElemSolverAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  stk::mesh::EntityRank entityRank,
  unsigned nodesPerEntity)
  : SolverAlgorithm(realm, part, eqSystem),
    dataNeededByKernels_(realm.meta_data()),
    entityRank_(entityRank),
    nodesPerEntity_(nodesPerEntity),
    rhsSize_(nodesPerEntity * eqSystem->linsys_->numDof())
{
  if (eqSystem->dofName_ != "pressure") {
    diagRelaxFactor_ =
      realm.solutionOptions_->get_relaxation_factor(eqSystem->dofName_);
  }
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleElemSolverAlgorithm::initialize_connectivity()
{
  if (entityRank_ == stk::topology::ELEM_RANK)
    eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
  else if (entityRank_ == realm_.meta_data().side_rank())
    eqSystem_->linsys_->buildFaceToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleElemSolverAlgorithm::execute()
{
  const size_t numKernels = activeKernels_.size();
  for (size_t i = 0; i < numKernels; ++i)
    activeKernels_[i]->setup(*realm_.timeIntegrator_);

  auto ngpKernels = nalu_ngp::create_ngp_view<Kernel>(activeKernels_);
  auto coeffApplier = coeff_applier();

  double diagRelaxFactor = diagRelaxFactor_;
  int rhsSize = rhsSize_;
  unsigned nodesPerEntity = nodesPerEntity_;

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(SharedMemData<DeviceTeamHandleType, DeviceShmem> & smdata) {
      set_vals(smdata.simdrhs, 0.0);
      set_vals(smdata.simdlhs, 0.0);
      for (size_t i = 0; i < numKernels; i++) {
        Kernel* kernel = ngpKernels(i);
        kernel->execute(smdata.simdlhs, smdata.simdrhs, smdata.simdPrereqData);
      }

#if defined(KOKKOS_ENABLE_GPU)
      const int simdElemIndex = 0;
#else
      for (int simdElemIndex = 0; simdElemIndex < smdata.numSimdElems;
           ++simdElemIndex)
#endif
      {
        extract_vector_lane(smdata.simdrhs, simdElemIndex, smdata.rhs);
        extract_vector_lane(smdata.simdlhs, simdElemIndex, smdata.lhs);
        for (int ir = 0; ir < rhsSize; ++ir)
          smdata.lhs(ir, ir) /= diagRelaxFactor;
        coeffApplier(
          nodesPerEntity, smdata.ngpElemNodes[simdElemIndex], smdata.scratchIds,
          smdata.sortPermutation, smdata.rhs, smdata.lhs, __FILE__);
      }
    });
  coeffApplier.free_coeff_applier();
}

} // namespace nalu
} // namespace sierra
