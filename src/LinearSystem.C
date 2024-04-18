// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <LinearSystem.h>
#include <EquationSystem.h>
#include <Realm.h>
#include <Simulation.h>
#include <LinearSolver.h>
#include <master_element/MasterElement.h>
#include <NaluEnv.h>

#ifdef NALU_USES_HYPRE
#include "HypreLinearSystem.h"
#include "HypreUVWLinearSystem.h"
#endif

#ifdef NALU_USES_TRILINOS_SOLVERS
#include <TpetraLinearSystem.h>
#include <TpetraSegregatedLinearSystem.h>
#endif

#include <stk_util/parallel/Parallel.hpp>

#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Part.hpp>
#include "stk_mesh/base/NgpMesh.hpp"
#include <stk_topology/topology.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include "stk_mesh/base/NgpFieldParallel.hpp"

#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_FancyOStream.hpp>

#include <sstream>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// LinearSystem - base class linear system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
LinearSystem::LinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver)
  : realm_(realm),
    eqSys_(eqSys),
    inConstruction_(false),
    numDof_(numDof),
    eqSysName_(eqSys->name_),
    linearSolver_(linearSolver),
    linearSolveIterations_(0),
    nonLinearResidual_(0.0),
    linearResidual_(0.0),
    firstNonLinearResidual_(1.0e8),
    scaledNonLinearResidual_(1.0e8),
    recomputePreconditioner_(true),
    reusePreconditioner_(false),
    provideOutput_(true)
{
  // nothing to do
}

void
LinearSystem::zero_timer_precond()
{
  linearSolver_->zero_timer_precond();
}

double
LinearSystem::get_timer_precond()
{
  return linearSolver_->get_timer_precond();
}

bool
LinearSystem::debug()
{
  if (NaluEnv::self().debug())
    return true;
  return false;
}

bool
LinearSystem::useSegregatedSolver() const
{
  return linearSolver_ ? linearSolver_->getConfig()->useSegregatedSolver()
                       : false;
}

const LinearSolverConfig&
LinearSystem::config() const
{
  STK_ThrowAssert(linearSolver_ != nullptr);
  return *(linearSolver_->getConfig());
}

// static method
LinearSystem*
LinearSystem::create(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* solver)
{
  switch (solver->getType()) {
#ifdef NALU_USES_TRILINOS_SOLVERS
  case PT_TPETRA:
    return new TpetraLinearSystem(realm, numDof, eqSys, solver);
// Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
    break;
#endif

  case PT_TPETRA_SEGREGATED:
    return new TpetraSegregatedLinearSystem(realm, numDof, eqSys, solver);
// Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
    break;
#endif
#endif // NALU_USES_TRILINOS_SOLVERS

#ifdef NALU_USES_HYPRE
  case PT_HYPRE:
    realm.hypreIsActive_ = true;
    return new HypreLinearSystem(realm, numDof, eqSys, solver);

  case PT_HYPRE_SEGREGATED:
    realm.hypreIsActive_ = true;
    return new HypreUVWLinearSystem(realm, numDof, eqSys, solver);
#endif

  case PT_END:
  default:
    throw std::logic_error("create lin sys");
  }
// Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
  return 0;
#endif
}

void
LinearSystem::sync_field(const stk::mesh::FieldBase* field)
{
  const auto& fieldMgr = realm_.ngp_field_manager();
  const std::vector<NGPDoubleFieldType*> ngpFields{
    &fieldMgr.get_field<double>(field->mesh_meta_data_ordinal())};

  stk::mesh::copy_owned_to_shared(realm_.bulk_data(), ngpFields);
}

} // namespace nalu
} // namespace sierra
