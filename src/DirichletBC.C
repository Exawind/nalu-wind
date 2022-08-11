// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <DirichletBC.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>

namespace sierra {
namespace nalu {

DirichletBC::DirichletBC(
  Realm& realm,
  EquationSystem* eqSystem,
  stk::mesh::Part* part,
  stk::mesh::FieldBase* field,
  stk::mesh::FieldBase* bcValues,
  const unsigned beginPos,
  const unsigned endPos)
  : SolverAlgorithm(realm, part, eqSystem),
    field_(field),
    bcValues_(bcValues),
    beginPos_(beginPos),
    endPos_(endPos)
{
}

void
DirichletBC::initialize_connectivity()
{
  eqSystem_->linsys_->buildDirichletNodeGraph(partVec_);
}

void
DirichletBC::execute()
{

  eqSystem_->linsys_->applyDirichletBCs(
    field_, bcValues_, partVec_, beginPos_, endPos_);
}

} // namespace nalu
} // namespace sierra
