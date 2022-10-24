// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

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

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// AssembleFaceElemSolverAlgorithm - add LHS/RHS for element-based contribution
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleFaceElemSolverAlgorithm::AssembleFaceElemSolverAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  unsigned nodesPerFace,
  unsigned nodesPerElem)
  : SolverAlgorithm(realm, part, eqSystem),
    faceDataNeeded_(realm.meta_data()),
    elemDataNeeded_(realm.meta_data()),
    numDof_(eqSystem->linsys_->numDof()),
    nodesPerFace_(nodesPerFace),
    nodesPerElem_(nodesPerElem),
    rhsSize_(nodesPerFace * eqSystem->linsys_->numDof())
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
  auto coeffApplier = coeff_applier();

  const unsigned nodesPerEntity = nodesPerElem_;
  const unsigned numDof = numDof_;
  double diagRelaxFactor = diagRelaxFactor_;

  run_face_elem_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      sierra::nalu::SharedMemData_FaceElem<DeviceTeamHandleType, DeviceShmem> &
      smdata) {
      set_vals(smdata.simdrhs, DoubleType(0.0));
      set_vals(smdata.simdlhs, DoubleType(0.0));

      for (size_t i = 0; i < numKernels; ++i) {
        Kernel* kernel = ngpKernels(i);
        kernel->execute(
          smdata.simdlhs, smdata.simdrhs, smdata.simdFaceViews,
          smdata.simdElemViews, smdata.elemFaceOrdinal);
      }
#if defined(KOKKOS_ENABLE_GPU)
      const int simdIndex = 0;
#else
      for (int simdIndex = 0; simdIndex < smdata.numSimdFaces; ++simdIndex)
#endif
      {
        extract_vector_lane(smdata.simdrhs, simdIndex, smdata.rhs);
        extract_vector_lane(smdata.simdlhs, simdIndex, smdata.lhs);
        for (unsigned ir = 0; ir < nodesPerEntity * numDof; ++ir)
          smdata.lhs(ir, ir) /= diagRelaxFactor;

        coeffApplier(
          nodesPerEntity, smdata.ngpConnectedNodes[simdIndex],
          smdata.scratchIds, smdata.sortPermutation, smdata.rhs, smdata.lhs,
          __FILE__);
      }
    });
  coeffApplier.free_coeff_applier();
}

} // namespace nalu
} // namespace sierra
