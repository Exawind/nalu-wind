/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "overset/AssembleOversetDecoupledAlgorithm.h"
#include "overset/OversetManager.h"
#include "overset/OversetInfo.h"
#include "EquationSystem.h"
#include "LinearSystem.h"
#include "NaluEnv.h"
#include "Realm.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {

AssembleOversetDecoupledAlgorithm::AssembleOversetDecoupledAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  stk::mesh::FieldBase* field
) : OversetConstraintBase(realm, part, eqSystem, field)
{}

void AssembleOversetDecoupledAlgorithm::execute()
{
  // Reset LHS/RHS for the overset constraint rows
  prepare_constraints();

  const auto& fringeNodes = realm_.oversetManager_->fringeNodes_;
  const int numDof = eqSystem_->linsys_->numDof();
  eqSystem_->linsys_->resetRows(fringeNodes, 0, numDof, 1.0, 0.0);

#if 0
  const int rank = NaluEnv::self().parallel_rank();
  const int nDof = eqSystem_->linsys_->numDof();
  const auto& bulk = realm_.bulk_data();

  // RHS size is the same as number of DOFs
  std::vector<double> lhs(nDof * nDof, 0.0);
  std::vector<double> rhs(nDof, 0.0);
  std::vector<int> scratchIds(nDof);
  std::vector<double> scratchVals(nDof);
  std::vector<stk::mesh::Entity> connected_nodes(1);

  // Set diagonal values to 1.0
  for (int i = 0; i < nDof; ++i)
    lhs[i * nDof + i] = 1.0;

  for (auto* oinfo: realm_.oversetManager_->oversetInfoVec_) {
    auto node = oinfo->orphanNode_;
    const int nodeRank = bulk.parallel_owner_rank(node);

    // Only process nodes owned by this MPI rank
    if (rank != nodeRank) continue;

    // Assign the row for the fringe node
    connected_nodes[0] = node;

    // Don't use apply_coeff here as it checks for overset logic
    eqSystem_->linsys_->sumInto(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);
  }
#endif
}

}  // nalu
}  // sierra
