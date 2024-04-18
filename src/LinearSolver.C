// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <LinearSolver.h>
#include <LinearSolvers.h>

#include <NaluEnv.h>
#include <LinearSolverTypes.h>

#include <stk_util/util/ReportHandler.hpp>

#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterXMLFileReader.hpp>
#include <Kokkos_Core.hpp>

#ifdef NALU_USES_TRILINOS_SOLVERS

// Tpetra support
#include <BelosLinearProblem.hpp>
#include <BelosMultiVecTraits.hpp>
#include <BelosOperatorTraits.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosSolverManager.hpp>
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

#include <Ifpack2_Factory.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>

#include <MueLu_CreateTpetraPreconditioner.hpp>

#endif // NALU_USES_TRILINOS_SOLVERS

#include <iostream>

namespace sierra {
namespace nalu {

#ifdef NALU_USES_TRILINOS_SOLVERS

TpetraLinearSolver::TpetraLinearSolver(
  std::string solverName,
  TpetraLinearSolverConfig* config,
  const Teuchos::RCP<Teuchos::ParameterList> params,
  const Teuchos::RCP<Teuchos::ParameterList> paramsPrecond,
  LinearSolvers* linearSolvers)
  : LinearSolver(solverName, linearSolvers, config),
    params_(params),
    paramsPrecond_(paramsPrecond),
    preconditionerType_(config->preconditioner_type())
{
  activateMueLu_ = config->use_MueLu();
}

TpetraLinearSolver::~TpetraLinearSolver() { destroyLinearSolver(); }

void
TpetraLinearSolver::setSystemObjects(
  Teuchos::RCP<LinSys::Matrix> matrix, Teuchos::RCP<LinSys::MultiVector> rhs)
{
  STK_ThrowRequire(!matrix.is_null());
  STK_ThrowRequire(!rhs.is_null());

  matrix_ = matrix;
  rhs_ = rhs;
}

void
TpetraLinearSolver::setupLinearSolver(
  Teuchos::RCP<LinSys::MultiVector> sln,
  Teuchos::RCP<LinSys::Matrix> matrix,
  Teuchos::RCP<LinSys::MultiVector> rhs,
  Teuchos::RCP<LinSys::MultiVector> coords)
{

  setSystemObjects(matrix, rhs);
  problem_ = Teuchos::RCP<LinSys::LinearProblem>(
    new LinSys::LinearProblem(matrix_, sln, rhs_));

  if (activateMueLu_) {
    coords_ = coords;
    // Inject coordinates into the parameter list for use within MueLu
    auto& userParamList = paramsPrecond_->sublist("user data");
    userParamList.set("Coordinates", coords_);
  } else {
    Ifpack2::Factory factory;
    preconditioner_ = factory.create(
      preconditionerType_,
      Teuchos::rcp_const_cast<const LinSys::Matrix>(matrix_), 0);
    preconditioner_->setParameters(*paramsPrecond_);

    // delay initialization for some preconditioners
    if ("RILUK" != preconditionerType_) {
      preconditioner_->initialize();
    }
    problem_->setRightPrec(preconditioner_);

    // create the solver, e.g., gmres, cg, tfqmr, bicgstab
    LinSys::SolverFactory sFactory;
    solver_ = sFactory.create(config_->get_method(), params_);
    solver_->setProblem(problem_);
  }
}

void
TpetraLinearSolver::destroyLinearSolver()
{
  problem_ = Teuchos::null;
  preconditioner_ = Teuchos::null;
  solver_ = Teuchos::null;
  coords_ = Teuchos::null;
  if (activateMueLu_)
    mueluPreconditioner_ = Teuchos::null;
}

void
TpetraLinearSolver::setMueLu()
{
  TpetraLinearSolverConfig* config =
    reinterpret_cast<TpetraLinearSolverConfig*>(config_);

  if (
    solver_ != Teuchos::null && !recomputePreconditioner_ &&
    !reusePreconditioner_)
    return;

  {
    Teuchos::RCP<Teuchos::Time> tm =
      Teuchos::TimeMonitor::getNewTimer("nalu MueLu preconditioner setup");
    Teuchos::TimeMonitor timeMon(*tm);

    if (recomputePreconditioner_ || mueluPreconditioner_ == Teuchos::null) {
      mueluPreconditioner_ = MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(
        Teuchos::RCP<Tpetra::Operator<SC, LO, GO, NO>>(matrix_),
        *paramsPrecond_);
    } else if (reusePreconditioner_) {
      MueLu::ReuseTpetraPreconditioner(matrix_, *mueluPreconditioner_);
    }
    if (config->getSummarizeMueluTimer())
      Teuchos::TimeMonitor::summarize(
        std::cout, false, true, false, Teuchos::Union);
  }

  problem_->setRightPrec(mueluPreconditioner_);

  // create the solver, e.g., gmres, cg, tfqmr, bicgstab
  LinSys::SolverFactory sFactory;
  solver_ = sFactory.create(config->get_method(), params_);
  solver_->setProblem(problem_);
}

int
TpetraLinearSolver::residual_norm(
  int whichNorm, Teuchos::RCP<LinSys::MultiVector> sln, double& norm)
{
  const size_t numVecs = sln->getNumVectors();
  LinSys::MultiVector resid(rhs_->getMap(), numVecs);
  STK_ThrowRequire(!(sln.is_null() || rhs_.is_null()));

  if (matrix_->isFillActive()) {
    // FIXME
    //! matrix_->fillComplete(map_, map_);
    throw std::runtime_error("residual_norm");
  }
  matrix_->apply(*sln, resid);

  resid.update(-1.0, *rhs_, 1.0);

  norm = 0.0;
  Teuchos::Array<double> mv_norm(numVecs);
  if (whichNorm == 0) {
    resid.normInf(mv_norm());
    for (size_t vecIdx = 0; vecIdx < numVecs; ++vecIdx) {
      norm =
        (norm < std::abs(mv_norm[vecIdx]) ? std::abs(mv_norm[vecIdx]) : norm);
    }
  } else if (whichNorm == 1) {
    resid.norm1(mv_norm);
    for (size_t vecIdx = 0; vecIdx < numVecs; ++vecIdx) {
      norm += std::abs(mv_norm[vecIdx]);
    }
  } else if (whichNorm == 2) {
    resid.norm2(mv_norm);
    for (size_t vecIdx = 0; vecIdx < numVecs; ++vecIdx) {
      norm += mv_norm[vecIdx] * mv_norm[vecIdx];
    }
    norm = std::sqrt(norm);
  } else {
    return 1;
  }

  return 0;
}

int
TpetraLinearSolver::solve(
  Teuchos::RCP<LinSys::MultiVector> sln,
  int& iters,
  double& finalResidNrm,
  bool isFinalOuterIter)
{
  STK_ThrowRequire(!sln.is_null());

  const int status = 0;
  int whichNorm = 2;
  finalResidNrm = 0.0;

  double time = -NaluEnv::self().nalu_time();
  if (activateMueLu_) {
    setMueLu();
  } else {
    if ("RILUK" == preconditionerType_) {
      preconditioner_->initialize();
    }
    preconditioner_->compute();
  }
  time += NaluEnv::self().nalu_time();

  // Update preconditioner timer for this timestep; actual summing over
  // timesteps is handled in EquationSystem::assemble_and_solve
  timerPrecond_ = time;

  Teuchos::RCP<Teuchos::ParameterList> params(
    Teuchos::rcp(new Teuchos::ParameterList));
  if (isFinalOuterIter) {
    params->set("Convergence Tolerance", config_->finalTolerance());
  } else {
    params->set("Convergence Tolerance", config_->tolerance());
  }

  solver_->setParameters(params);

  problem_->setProblem();
  solver_->solve();

  iters = solver_->getNumIters();
  residual_norm(whichNorm, sln, finalResidNrm);

  return status;
}

#endif // NALU_USES_TRILINOS_SOLVERS

} // namespace nalu
} // namespace sierra
