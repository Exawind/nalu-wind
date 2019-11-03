/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleFaceElemSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>
#include <master_element/MasterElement.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <TimeIntegrator.h>

// kernel
#include <kernel/Kernel.h>

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
#include <SimdInterface.h>
#include <ScratchViews.h>
#include <CopyAndInterleave.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleFaceElemSolverAlgorithm - add LHS/RHS for element-based contribution
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleFaceElemSolverAlgorithm::AssembleFaceElemSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  unsigned nodesPerFace,
  unsigned nodesPerElem)
  : SolverAlgorithm(realm, part, eqSystem),
    faceDataNeeded_(realm.meta_data()),
    elemDataNeeded_(realm.meta_data()),
    numDof_(eqSystem->linsys_->numDof()),
    nodesPerFace_(nodesPerFace),
    nodesPerElem_(nodesPerElem),
    rhsSize_(nodesPerFace*eqSystem->linsys_->numDof())
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
AssembleFaceElemSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleFaceElemSolverAlgorithm::execute()
{
  for (auto kernel : activeKernels_) {
    kernel->setup(*realm_.timeIntegrator_);
  }

  auto ngpKernels = nalu_ngp::create_ngp_view<Kernel>(activeKernels_);
  const size_t numKernels = activeKernels_.size();

  const unsigned nodesPerEntity = nodesPerElem_;
  const unsigned numDof = numDof_;
  CoeffApplier* deviceCoeffApplier = eqSystem_->linsys_->get_coeff_applier();
  double diagRelaxFactor = diagRelaxFactor_;

  run_face_elem_algorithm(realm_.bulk_data(),
    KOKKOS_LAMBDA(sierra::nalu::SharedMemData_FaceElem<DeviceTeamHandleType,DeviceShmem> &smdata)
    {
        set_zero(smdata.simdrhs.data(), smdata.simdrhs.size());
        set_zero(smdata.simdlhs.data(), smdata.simdlhs.size());

        for (size_t i=0; i<numKernels; ++i) {
          Kernel* kernel = ngpKernels(i);
          kernel->execute( smdata.simdlhs, smdata.simdrhs, smdata.simdFaceViews, smdata.simdElemViews, smdata.elemFaceOrdinal );
        }

#ifdef KOKKOS_ENABLE_CUDA
        extract_vector_lane(smdata.simdrhs, 0, smdata.rhs);
        extract_vector_lane(smdata.simdlhs, 0, smdata.lhs);
        for (unsigned ir=0; ir < nodesPerEntity*numDof; ++ir)
          smdata.lhs(ir, ir) /= diagRelaxFactor;
        (*deviceCoeffApplier)(nodesPerEntity, smdata.ngpConnectedNodes[0],
                    smdata.scratchIds, smdata.sortPermutation, smdata.rhs, smdata.lhs, __FILE__);
#else
        for(int simdIndex=0; simdIndex<smdata.numSimdFaces; ++simdIndex) {
          extract_vector_lane(smdata.simdrhs, simdIndex, smdata.rhs);
          extract_vector_lane(smdata.simdlhs, simdIndex, smdata.lhs);
          for (unsigned ir=0; ir < nodesPerElem_*numDof; ++ir)
            smdata.lhs(ir, ir) /= diagRelaxFactor;
          (*deviceCoeffApplier)(nodesPerEntity, smdata.ngpConnectedNodes[simdIndex],
                      smdata.scratchIds, smdata.sortPermutation, smdata.rhs, smdata.lhs, __FILE__);
        }
#endif
    });
}

} // namespace nalu
} // namespace Sierra

