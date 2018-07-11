/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "HypreUVWSolver.h"
#include "XSDKHypreInterface.h"
#include "NaluEnv.h"

namespace sierra {
namespace nalu {

HypreUVWSolver::HypreUVWSolver(
  std::string name,
  HypreLinearSolverConfig* config,
  LinearSolvers* linearSolvers
) : HypreDirectSolver(name, config, linearSolvers),
    parRhsU_(3),
    parSlnU_(3)
{}

HypreUVWSolver::~HypreUVWSolver()
{}

int
HypreUVWSolver::solve(
  int dim,
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
  solverSolvePtr_(solver_, parMat_, parRhsU_[dim], parSlnU_[dim]);

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
HypreUVWSolver::setupSolver()
{
  solverSetupPtr_(solver_, parMat_, parRhsU_[0], parSlnU_[0]);
}

}  // nalu
}  // sierra
