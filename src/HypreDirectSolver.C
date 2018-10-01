#ifdef NALU_USES_HYPRE

#include "HypreDirectSolver.h"
#include "XSDKHypreInterface.h"
#include "NaluEnv.h"

namespace sierra {
namespace nalu {

namespace {
// This anonymous namespace contains wrapper methods to HYPRE solver creation
// methods. It hides around the fact that some solvers require an MPI
// communicator while others do not. This allows HypreDirectSolver::CreateSolver
// methods to assign pointers using the same function signature.
//
// Note that this section has been modeled after xSDK Trilinos package. See
// <https://github.com/trilinos/xSDKTrilinos/blob/master/hypre/src/Ifpack2_Hypre.hpp>
// for more details

HypreIntType Hypre_BoomerAMGCreate(MPI_Comm, HYPRE_Solver* solver)
{ return HYPRE_BoomerAMGCreate(solver); }

HypreIntType Hypre_ParaSailsCreate(MPI_Comm comm, HYPRE_Solver* solver)
{ return HYPRE_ParaSailsCreate(comm, solver); }

HypreIntType Hypre_EuclidCreate(MPI_Comm comm, HYPRE_Solver* solver)
{ return HYPRE_EuclidCreate(comm, solver); }

HypreIntType Hypre_AMSCreate(MPI_Comm, HYPRE_Solver *solver)
{ return HYPRE_AMSCreate(solver);}

HypreIntType Hypre_ParCSRHybridCreate(MPI_Comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRHybridCreate(solver);}

HypreIntType Hypre_ParCSRPCGCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRPCGCreate(comm, solver);}

HypreIntType Hypre_ParCSRGMRESCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRGMRESCreate(comm, solver);}

#ifdef HYPRE_COGMRES
HypreIntType Hypre_ParCSRCOGMRESCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRCOGMRESCreate(comm, solver);}
#endif

HypreIntType Hypre_ParCSRFlexGMRESCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRFlexGMRESCreate(comm, solver);}

HypreIntType Hypre_ParCSRLGMRESCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRLGMRESCreate(comm, solver);}

HypreIntType Hypre_ParCSRBiCGSTABCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRBiCGSTABCreate(comm, solver);}
}

HypreDirectSolver::HypreDirectSolver(
  std::string name,
  HypreLinearSolverConfig* config,
  LinearSolvers* linearSolvers
) : LinearSolver(name, linearSolvers, config)
{}

HypreDirectSolver::~HypreDirectSolver()
{
  destroyLinearSolver();
}

int
HypreDirectSolver::solve(
  int& numIterations,
  double& finalResidualNorm,
  bool isFinalOuterIter)
{
  // Initialize the solver on first entry
  double time = -NaluEnv::self().nalu_time();
  if (!isInitialized_ || config_->recomputePreconditioner())
    initSolver();
  time += NaluEnv::self().nalu_time();
  timerPrecond_ = time;

  numIterations = 0;
  finalResidualNorm = 0.0;

  // Can use the return value from solverSolvePtr_. However, Hypre seems to
  // return a non-zero value and that causes spurious error message output in
  // Nalu.
  int status = 0;

  if (isFinalOuterIter)
    solverSetTolPtr_(solver_, config_->finalTolerance());
  else
    solverSetTolPtr_(solver_, config_->tolerance());

  // Solve the system Ax = b
  solverSolvePtr_(solver_, parMat_, parRhs_, parSln_);

  // Extract linear num. iterations and linear residual. Unlike the TPetra
  // interface, Hypre returns the relative residual norm and not the final
  // absolute residual.
  HypreIntType numIters;
  solverNumItersPtr_(solver_, &numIters);
  solverFinalResidualNormPtr_(solver_, &finalResidualNorm);
  numIterations = numIters;

  return status;
}

void
HypreDirectSolver::destroyLinearSolver()
{
  if (isSolverSetup_) solverDestroyPtr_(solver_);
  isSolverSetup_ = false;

  if (isPrecondSetup_) precondDestroyPtr_(precond_);
  isPrecondSetup_ = false;
}

void
HypreDirectSolver::initSolver()
{
  namespace Hypre = Ifpack2::Hypre;

  auto plist = config_->paramsPrecond();

  solverType_ = plist->get("Solver", Hypre::GMRES);
  usePrecond_ = plist->get("SetPreconditioner", false);
  if (usePrecond_)
    precondType_ = plist->get("Preconditioner", Hypre::BoomerAMG);

  Hypre::Hypre_Chooser chooser = plist->get("SolveOrPrecondition", Hypre::Solver);
  if (chooser != Hypre::Solver)
    throw std::runtime_error(
      "HypreDirectSolver::initParameters: Invalid option provided for Hypre Solver");

  // Everything checks out... create the solver and preconditioner
  createSolver();
  if (usePrecond_) createPrecond();

  // Apply user configuration parameters to solver and precondtioner
  int numFuncs = plist->get("NumFunctions", 0);
  if (numFuncs > 0) {
    Teuchos::RCP<Ifpack2::FunctionParameter>* params =
      plist->get<Teuchos::RCP<Ifpack2::FunctionParameter>*>("Functions");

    for (int i=0; i < numFuncs; i++) {
      params[i]->CallFunction(solver_, precond_);
    }
  }

  if (usePrecond_)
    solverPrecondPtr_(solver_, precondSolvePtr_, precondSetupPtr_, precond_);

  setupSolver();

  isInitialized_ = true;
}

void
HypreDirectSolver::setupSolver()
{
  // We are always using HYPRE solver
  solverSetupPtr_(solver_, parMat_, parRhs_, parSln_);
}

void
HypreDirectSolver::createSolver()
{
  namespace Hypre = Ifpack2::Hypre;

  if (isSolverSetup_) {
    solverDestroyPtr_(solver_);
    isSolverSetup_ = false;
  }

  switch(solverType_) {
  case Hypre::BoomerAMG:
    solverCreatePtr_ = &Hypre_BoomerAMGCreate;
    solverDestroyPtr_ = &HYPRE_BoomerAMGDestroy;
    solverSetupPtr_ = &HYPRE_BoomerAMGSetup;
    solverPrecondPtr_ = nullptr;
    solverSolvePtr_ = &HYPRE_BoomerAMGSolve;
    solverSetTolPtr_ = &HYPRE_BoomerAMGSetTol;
    solverNumItersPtr_ = &HYPRE_BoomerAMGGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_BoomerAMGGetFinalRelativeResidualNorm;
    break;

  case Hypre::GMRES:
    solverCreatePtr_ = &Hypre_ParCSRGMRESCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRGMRESDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRGMRESSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRGMRESSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRGMRESSolve;
    solverSetTolPtr_ = &HYPRE_ParCSRGMRESSetTol;
    solverNumItersPtr_ = &HYPRE_GMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_GMRESGetFinalRelativeResidualNorm;
    break;

#ifdef HYPRE_COGMRES
  case Hypre::COGMRES:
    solverCreatePtr_ = &Hypre_ParCSRCOGMRESCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRCOGMRESDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRCOGMRESSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRCOGMRESSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRCOGMRESSolve;
    solverSetTolPtr_ = &HYPRE_ParCSRCOGMRESSetTol;
    solverNumItersPtr_ = &HYPRE_COGMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_COGMRESGetFinalRelativeResidualNorm;
    break;
#endif

  case Hypre::FlexGMRES:
    solverCreatePtr_ = &Hypre_ParCSRFlexGMRESCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRFlexGMRESDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRFlexGMRESSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRFlexGMRESSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRFlexGMRESSolve;
    solverSetTolPtr_ = &HYPRE_ParCSRFlexGMRESSetTol;
    solverNumItersPtr_ = &HYPRE_FlexGMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_FlexGMRESGetFinalRelativeResidualNorm;
    break;

  case Hypre::LGMRES:
    solverCreatePtr_ = &Hypre_ParCSRLGMRESCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRLGMRESDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRLGMRESSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRLGMRESSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRLGMRESSolve;
    solverSetTolPtr_ = &HYPRE_ParCSRLGMRESSetTol;
    solverNumItersPtr_ = &HYPRE_LGMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_LGMRESGetFinalRelativeResidualNorm;
    break;

  case Hypre::BiCGSTAB:
    solverCreatePtr_ = &Hypre_ParCSRBiCGSTABCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRBiCGSTABDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRBiCGSTABSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRBiCGSTABSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRBiCGSTABSolve;
    solverSetTolPtr_ = &HYPRE_ParCSRBiCGSTABSetTol;
    solverNumItersPtr_ = &HYPRE_BiCGSTABGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_BiCGSTABGetFinalRelativeResidualNorm;
    break;

  case Hypre::AMS:
    solverCreatePtr_ = &Hypre_AMSCreate;
    solverDestroyPtr_ = &HYPRE_AMSDestroy;
    solverSetupPtr_ = &HYPRE_AMSSetup;
    solverPrecondPtr_ = nullptr;
    solverSolvePtr_ = &HYPRE_AMSSolve;
    solverSetTolPtr_ = &HYPRE_AMSSetTol;
    solverNumItersPtr_ = &HYPRE_AMSGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_AMSGetFinalRelativeResidualNorm;
    break;

  case Hypre::PCG:
    solverCreatePtr_ = &Hypre_ParCSRPCGCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRPCGDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRPCGSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRPCGSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRPCGSolve;
    solverSetTolPtr_ = &HYPRE_ParCSRPCGSetTol;
    solverNumItersPtr_ = &HYPRE_PCGGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_PCGGetFinalRelativeResidualNorm;
    break;

  case Hypre::Hybrid:
    solverCreatePtr_ = &Hypre_ParCSRHybridCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRHybridDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRHybridSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRHybridSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRHybridSolve;
    solverSetTolPtr_ = &HYPRE_ParCSRHybridSetTol;
    solverNumItersPtr_ = &HYPRE_ParCSRHybridGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_ParCSRHybridGetFinalRelativeResidualNorm;
    break;

  default:
    solverCreatePtr_ = nullptr;
    break;
  }

  if (solverCreatePtr_ == nullptr)
    throw std::runtime_error("Error initializing HYPRE Solver");

  solverCreatePtr_(comm_, &solver_);
  isSolverSetup_ = true;
}

void
HypreDirectSolver::createPrecond()
{
  namespace Hypre = Ifpack2::Hypre;

  if (isPrecondSetup_) {
    precondDestroyPtr_(precond_);
    isPrecondSetup_ = false;
  }

  switch(precondType_) {
  case Hypre::BoomerAMG:
    precondCreatePtr_ = &Hypre_BoomerAMGCreate;
    precondDestroyPtr_ = &HYPRE_BoomerAMGDestroy;
    precondSetupPtr_ = &HYPRE_BoomerAMGSetup;
    precondSolvePtr_ = &HYPRE_BoomerAMGSolve;
    break;

  case Hypre::Euclid:
    precondCreatePtr_ = &Hypre_EuclidCreate;
    precondDestroyPtr_ = &HYPRE_EuclidDestroy;
    precondSetupPtr_ = &HYPRE_EuclidSetup;
    precondSolvePtr_ = &HYPRE_EuclidSolve;
    break;

  case Hypre::ParaSails:
    precondCreatePtr_ = &Hypre_ParaSailsCreate;
    precondDestroyPtr_ = &HYPRE_ParaSailsDestroy;
    precondSetupPtr_ = &HYPRE_ParaSailsSetup;
    precondSolvePtr_ = &HYPRE_ParaSailsSolve;
    break;

  case Hypre::AMS:
    precondCreatePtr_ = &Hypre_AMSCreate;
    precondDestroyPtr_ = &HYPRE_AMSDestroy;
    precondSetupPtr_ = &HYPRE_AMSSetup;
    precondSolvePtr_ = &HYPRE_AMSSolve;
    break;

  default:
    precondCreatePtr_ = nullptr;
    break;
  }

  if (precondCreatePtr_ == nullptr)
    throw std::runtime_error("Error initializing HYPRE Preconditioner");

  precondCreatePtr_(comm_, &precond_);
  isPrecondSetup_ = true;
}

}  // nalu
}  // sierra

#endif
