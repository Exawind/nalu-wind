/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "overset/OversetConstraintBase.h"

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
  const int sysStart = 0;
  const int sysEnd = eqSystem_->linsys_->numDof();

  eqSystem_->linsys_->prepareConstraints(sysStart, sysEnd);

  reset_hole_rows();
}

void
OversetConstraintBase::reset_hole_rows()
{
  const auto& holeRows = realm_.oversetManager_->holeNodes_;
  const int& numDof = eqSystem_->linsys_->numDof();

  // Reset existing entries and zero out the entire row
  eqSystem_->linsys_->resetRows(holeRows, 0, numDof, 1.0, 0.0);
}

}  // nalu
}  // sierra
