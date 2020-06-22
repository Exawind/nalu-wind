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

  const int numDof = eqSystem_->linsys_->numDof();
  const auto& fringeNodes = realm_.oversetManager_->ngpFringeNodes_;
  auto* coeffApplier = eqSystem_->linsys_->get_coeff_applier();
  Kokkos::parallel_for(
    fringeNodes.size(), KOKKOS_LAMBDA(const size_t& i) {
      coeffApplier->resetRows(1, &fringeNodes(i), 0, numDof, 1.0, 0.0);
    });
}

}  // nalu
}  // sierra
