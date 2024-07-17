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
    dualNodalVolume_(realm.meta_data().get_field<double>(
      stk::topology::NODE_RANK, "dual_nodal_volume"))
{
}

void
OversetConstraintBase::initialize_connectivity()
{
  eqSystem_->linsys_->buildOversetNodeGraph(partVec_);
}

void
OversetConstraintBase::prepare_constraints()
{
  const int& numDof = eqSystem_->linsys_->numDof();
  const auto& holeRows = realm_.oversetManager_->ngpHoleNodes_;

  auto* coeffApplier = eqSystem_->linsys_->get_coeff_applier();
  Kokkos::parallel_for(
    DeviceRangePolicy(0, holeRows.size()), KOKKOS_LAMBDA(const size_t& i) {
      coeffApplier->resetRows(1, &holeRows(i), 0, numDof, 1.0, 0.0);
    });

  eqSystem_->linsys_->free_coeff_applier(coeffApplier);
}

} // namespace nalu
} // namespace sierra
