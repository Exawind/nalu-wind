/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <LinearSystem.h>
#include <TpetraLinearSystem.h>
#include <EquationSystem.h>
#include <Realm.h>
#include <Simulation.h>
#include <LinearSolver.h>
#include <master_element/MasterElement.h>

#ifdef NALU_USES_HYPRE
#include "HypreLinearSystem.h"
#include "HypreUVWLinearSystem.h"
#endif

#include <stk_util/parallel/Parallel.hpp>

#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_topology/topology.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_FancyOStream.hpp>

#include <sstream>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// LinearSystem - base class linear system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
LinearSystem::LinearSystem(
  Realm &realm,
  const unsigned numDof,
  EquationSystem *eqSys,
  LinearSolver *linearSolver)
  : realm_(realm),
    eqSys_(eqSys),
    inConstruction_(false),
    writeCounter_(0),
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

void LinearSystem::zero_timer_precond()
{
  linearSolver_->zero_timer_precond();  
}

double LinearSystem::get_timer_precond() 
{
  return linearSolver_->get_timer_precond();
}

bool LinearSystem::debug()
{
  if (linearSolver_ && linearSolver_->root() && linearSolver_->root()->debug()) return true;
  return false;
}

// static method
LinearSystem *LinearSystem::create(Realm& realm, const unsigned numDof, EquationSystem *eqSys, LinearSolver *solver)
{
  switch(solver->getType()) {
  case PT_TPETRA:
    return new TpetraLinearSystem(realm, numDof, eqSys, solver);
    break;

#ifdef NALU_USES_HYPRE
  case PT_HYPRE:
    realm.hypreIsActive_ = true;
    return new HypreLinearSystem(realm, numDof, eqSys, solver);
    break;

  case PT_HYPRE_SEGREGATED:
    realm.hypreIsActive_ = true;
    return new HypreUVWLinearSystem(realm, numDof, eqSys, solver);
#endif

  case PT_END:
  default:
    throw std::logic_error("create lin sys");
  }
  return 0;
}

void LinearSystem::sync_field(const stk::mesh::FieldBase *field)
{
  std::vector< const stk::mesh::FieldBase *> fields(1,field);
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  stk::mesh::copy_owned_to_shared( bulkData, fields);
}

} // namespace nalu
} // namespace Sierra
