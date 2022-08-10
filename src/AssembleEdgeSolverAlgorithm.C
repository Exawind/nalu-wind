// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "AssembleEdgeSolverAlgorithm.h"
#include "EquationSystem.h"
#include "LinearSystem.h"
#include "Realm.h"

namespace sierra {
namespace nalu {

AssembleEdgeSolverAlgorithm::AssembleEdgeSolverAlgorithm(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    dataNeeded_(realm.meta_data()),
    rhsSize_(nodesPerEntity_ * eqSystem->linsys_->numDof())
{
}

void
AssembleEdgeSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildEdgeToNodeGraph(partVec_);
}

} // namespace nalu
} // namespace sierra
