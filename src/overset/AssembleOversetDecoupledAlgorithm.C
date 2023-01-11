// Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

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
  stk::mesh::FieldBase* field)
  : OversetConstraintBase(realm, part, eqSystem, field)
{
}

void
AssembleOversetDecoupledAlgorithm::execute()
{
  // Reset LHS/RHS for the overset constraint rows
  prepare_constraints();

  const int numDof = eqSystem_->linsys_->numDof();
  const auto& fringeNodes = realm_.oversetManager_->ngpFringeNodes_;
  auto* coeffApplier = eqSystem_->linsys_->get_coeff_applier();
  Kokkos::parallel_for(
    DeviceRangePolicy(0, fringeNodes.size()), KOKKOS_LAMBDA(const size_t& i) {
      coeffApplier->resetRows(1, &fringeNodes(i), 0, numDof, 1.0, 0.0);
    });

  eqSystem_->linsys_->free_coeff_applier(coeffApplier);
}

} // namespace nalu
} // namespace sierra
