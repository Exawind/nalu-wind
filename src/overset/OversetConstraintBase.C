// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "overset/OversetConstraintBase.h"
#include "overset/OversetManager.h"

#include "EquationSystem.h"
#include "LinearSystem.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

OversetConstraintBase::OversetConstraintBase(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  stk::mesh::FieldBase* fieldQ)
  : SolverAlgorithm(realm, part, eqSystem),
    fieldQ_(fieldQ),
    dualNodalVolume_(realm.meta_data().get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume"))
{}

void
OversetConstraintBase::initialize_connectivity()
{
  eqSystem_->linsys_->buildOversetNodeGraph(partVec_);
}

void
OversetConstraintBase::prepare_constraints()
{
  const auto& holeRows = realm_.oversetManager_->holeNodes_;
  const int& numDof = eqSystem_->linsys_->numDof();

  // Reset existing entries and zero out the entire row
  eqSystem_->linsys_->resetRows(holeRows, 0, numDof, 1.0, 0.0);
}

}  // nalu
}  // sierra
