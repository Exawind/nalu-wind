// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <LinearSolverConfig.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <yaml-cpp/yaml.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#ifdef NALU_USES_TRILINOS_SOLVERS
#include <BelosTypes.hpp>
#endif

#include <ostream>

namespace sierra {
namespace nalu {

LinearSolverConfig::LinearSolverConfig()
  : params_(Teuchos::rcp(new Teuchos::ParameterList)),
    paramsPrecond_(Teuchos::rcp(new Teuchos::ParameterList))
{
}

#ifdef NALU_USES_TRILINOS_SOLVERS

TpetraLinearSolverConfig::TpetraLinearSolverConfig() : LinearSolverConfig() {}

TpetraLinearSolverConfig::~TpetraLinearSolverConfig() {}

void
TpetraLinearSolverConfig::load(const YAML::Node& node)
{
  name_ = node["name"].as<std::string>();
  method_ = node["method"].as<std::string>();
  get_if_present(node, "preconditioner", precond_, std::string("default"));
  solverType_ = "tpetra";

  double tol;
  int max_iterations, kspace, output_level;

  get_if_present(node, "tolerance", tolerance_, 1.e-4);
  get_if_present(node, "final_tolerance", finalTolerance_, tolerance_);
  get_if_present(node, "max_iterations", max_iterations, 50);
  get_if_present(node, "kspace", kspace, 50);
  get_if_present(node, "output_level", output_level, 0);

  tol = tolerance_;

  // Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::params();
  if (method_ == "sstep_gmres") {
    method_ = "TPETRA GMRES S-STEP";

    int step_size;
    get_if_present(node, "krylov_step_size", step_size, step_size);
    params_->set("Step Size", step_size);

    bool ritz_on_fly = true;
    params_->set("Compute Ritz Values on Fly", ritz_on_fly);

    bool useCholQR2 = true;
    params_->set("CholeskyQR2", useCholQR2);
  }
  params_->set("Convergence Tolerance", tol);
  params_->set("Maximum Iterations", max_iterations);
  if (output_level > 0) {
    params_->set(
      "Verbosity", Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
    params_->set("Output Style", Belos::Brief);
  }

  params_->set("Output Frequency", output_level);
  Teuchos::RCP<std::ostream> belosOutputStream =
    Teuchos::rcpFromRef(NaluEnv::self().naluOutputP0());
  params_->set("Output Stream", belosOutputStream);
  params_->set("Num Blocks", kspace);
  params_->set("Maximum Restarts", std::max(1, max_iterations / kspace));
  std::string orthoType = "ICGS";
  params_->set("Orthogonalization", orthoType);
  params_->set(
    "Implicit Residual Scaling", "Norm of Preconditioned Initial Residual");

  if (precond_ == "sgs") {
    preconditionerType_ = "RELAXATION";
    paramsPrecond_->set("relaxation: type", "Symmetric Gauss-Seidel");
    paramsPrecond_->set("relaxation: sweeps", 1);
  } else if (precond_ == "mt_sgs") {
    preconditionerType_ = "RELAXATION";
    paramsPrecond_->set("relaxation: type", "MT Symmetric Gauss-Seidel");
    paramsPrecond_->set("relaxation: sweeps", 1);
  } else if (precond_ == "sgs2") {
    preconditionerType_ = "RELAXATION";
    paramsPrecond_->set("relaxation: type", "Two-stage Symmetric Gauss-Seidel");
    paramsPrecond_->set("relaxation: sweeps", 1);

    int inner_iterations;
    get_if_present(node, "inner_iterations", inner_iterations, 1);
    paramsPrecond_->set("relaxation: inner sweeps", inner_iterations);
  } else if (precond_ == "jacobi" || precond_ == "default") {
    preconditionerType_ = "RELAXATION";
    paramsPrecond_->set("relaxation: type", "Jacobi");
    paramsPrecond_->set("relaxation: sweeps", 1);
  } else if (precond_ == "ilut") {
    preconditionerType_ = "ILUT";
  } else if (precond_ == "riluk") {
    preconditionerType_ = "RILUK";
  } else if (precond_ == "muelu") {
    muelu_xml_file_ = std::string("milestone.xml");
    get_if_present(
      node, "muelu_xml_file_name", muelu_xml_file_, muelu_xml_file_);
    paramsPrecond_->set("xml parameter file", muelu_xml_file_);
    useMueLu_ = true;
  } else {
    throw std::runtime_error("invalid linear solver preconditioner specified ");
  }

  params_->set("Solver Name", method_);

  get_if_present(
    node, "write_matrix_files", writeMatrixFiles_, writeMatrixFiles_);
  get_if_present(
    node, "summarize_muelu_timer", summarizeMueluTimer_, summarizeMueluTimer_);

  get_if_present(
    node, "recompute_preconditioner", recomputePreconditioner_,
    recomputePreconditioner_);
  get_if_present(
    node, "reuse_preconditioner", reusePreconditioner_, reusePreconditioner_);
  get_if_present(
    node, "segregated_solver", useSegregatedSolver_, useSegregatedSolver_);
  get_if_present(
    node, "reuse_linear_system", reuseLinSysIfPossible_,
    reuseLinSysIfPossible_);
}

#endif // NALU_USES_TRILINOS_SOLVERS

} // namespace nalu
} // namespace sierra
