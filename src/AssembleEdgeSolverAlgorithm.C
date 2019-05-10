/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "AssembleEdgeSolverAlgorithm.h"
#include "EquationSystem.h"
#include "LinearSystem.h"
#include "Realm.h"

namespace sierra {
namespace nalu {

AssembleEdgeSolverAlgorithm::AssembleEdgeSolverAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem
) : SolverAlgorithm(realm, part, eqSystem),
    dataNeeded_(realm.meta_data()),
    rhsSize_(nodesPerEntity_ * eqSystem->linsys_->numDof())
{}

void
AssembleEdgeSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildEdgeToNodeGraph(partVec_);
}

}  // nalu
}  // sierra
