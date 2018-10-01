/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifdef NALU_USES_HYPRE

#include <LinearSolverConfig.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <yaml-cpp/yaml.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <BelosTypes.hpp>

#include <Ifpack2_Preconditioner.hpp>
#include "XSDKHypreInterface.h"

#include <ostream>

namespace sierra {
namespace nalu {

HypreLinearSolverConfig::HypreLinearSolverConfig()
  : LinearSolverConfig()
{}

void
HypreLinearSolverConfig::load(const YAML::Node& node)
{
  const std::string hypre_check("hypre_");
  name_ = node["name"].as<std::string>() ;
  method_ = node["method"].as<std::string>() ;
  get_if_present(node, "preconditioner", precond_, std::string("none"));
  solverType_ = node["type"].as<std::string>();

  get_if_present(node, "tolerance", tolerance_, 1.0e-4);
  get_if_present(node, "final_tolerance", finalTolerance_, tolerance_);
  get_if_present(node, "max_iterations", maxIterations_, maxIterations_);
  get_if_present(node, "output_level", outputLevel_, outputLevel_);
  get_if_present(node, "kspace", kspace_, kspace_);

  get_if_present(node, "write_matrix_files", writeMatrixFiles_, writeMatrixFiles_);

  get_if_present(node, "recompute_preconditioner",
                 recomputePreconditioner_, recomputePreconditioner_);
  get_if_present(node, "reuse_preconditioner",
                 reusePreconditioner_, reusePreconditioner_);
  get_if_present(node, "segregated_solver", useSegregatedSolver_, useSegregatedSolver_);

  if (node["absolute_tolerance"]) {
    hasAbsTol_ = true;
    absTol_ = node["absolute_tolerance"].as<double>();
  }

  isHypreSolver_ = (method_.compare(0, hypre_check.length(), hypre_check) == 0);

  if ( (precond_ == "none") && !isHypreSolver_)
    throw std::runtime_error("Invalid combination of Hypre preconditioner and solver method specified.");

  if (method_ == "hypre_boomerAMG") {
    boomerAMG_solver_config(node);
  }
  else {
    funcParams_.clear();

    // Configure preconditioner (must always be Hypre)
    configure_hypre_preconditioner(node);

    if (isHypreSolver_) {
      paramsPrecond_->set("SolveOrPrecondition", Ifpack2::Hypre::Solver);
      configure_hypre_solver(node);
    } else {
      throw std::runtime_error("Non-hypre solver option not supported yet");
    }

    // Assign hypre config function calls
    int numFunctions = funcParams_.size();
    paramsPrecond_->set("NumFunctions", numFunctions);
    paramsPrecond_->set<Teuchos::RCP<Ifpack2::FunctionParameter>*>(
      "Functions", funcParams_.data());
  }
}

void
HypreLinearSolverConfig::boomerAMG_solver_config(const YAML::Node& node)
{
  get_if_present(node, "bamg_coarsen_type", bamgCoarsenType_, bamgCoarsenType_);
  get_if_present(node, "bamg_cycle_type", bamgCycleType_, bamgCycleType_);
  get_if_present(node, "bamg_relax_type", bamgRelaxType_, bamgRelaxType_);
  get_if_present(node, "bamg_relax_order", bamgRelaxOrder_, bamgRelaxOrder_);
  get_if_present(node, "bamg_num_sweeps", bamgNumSweeps_, bamgNumSweeps_);
  get_if_present(node, "bamg_max_levels", bamgMaxLevels_, bamgMaxLevels_);
  get_if_present(node, "bamg_strong_threshold", bamgStrongThreshold_, bamgStrongThreshold_);

  // Setup AMG parameters
  funcParams_.resize(10);
  funcParams_[0] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetPrintLevel,
    outputLevel_)); // print AMG solution info
  funcParams_[1] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetCoarsenType,
    bamgCoarsenType_)); // Falgout coarsening
  funcParams_[2] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetCycleType,
    bamgCycleType_));
  funcParams_[3] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetRelaxType,
    bamgRelaxType_));
  funcParams_[4] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetRelaxOrder,
    bamgRelaxOrder_));
  funcParams_[5] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetNumSweeps,
    bamgNumSweeps_));
  funcParams_[6] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetMaxLevels,
    bamgMaxLevels_));
  funcParams_[7] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetStrongThreshold,
    bamgStrongThreshold_));
  funcParams_[8] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetTol,
    tolerance_)); // Conv tolerance
  funcParams_[9] = Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BoomerAMGSetMaxIter,
    maxIterations_)); // Maximum number of iterations

  // Populate the parameter list for configuration
  paramsPrecond_->set("SolveOrPrecondition", Ifpack2::Hypre::Solver);
  paramsPrecond_->set("Solver", Ifpack2::Hypre::BoomerAMG);
  paramsPrecond_->set("Preconditioner", Ifpack2::Hypre::BoomerAMG);
  paramsPrecond_->set("SetPreconditioner", false);

  int numFunctions = funcParams_.size();
  paramsPrecond_->set("NumFunctions", numFunctions);
  paramsPrecond_->set<Teuchos::RCP<Ifpack2::FunctionParameter>*>("Functions", funcParams_.data());
}

void
HypreLinearSolverConfig::boomerAMG_precond_config(const YAML::Node& node)
{
  int output_level = 0;

  get_if_present(node, "bamg_coarsen_type", bamgCoarsenType_, bamgCoarsenType_);
  get_if_present(node, "bamg_cycle_type", bamgCycleType_, bamgCycleType_);
  get_if_present(node, "bamg_relax_type", bamgRelaxType_, bamgRelaxType_);
  get_if_present(node, "bamg_relax_order", bamgRelaxOrder_, bamgRelaxOrder_);
  get_if_present(node, "bamg_num_sweeps", bamgNumSweeps_, bamgNumSweeps_);
  get_if_present(node, "bamg_max_levels", bamgMaxLevels_, bamgMaxLevels_);
  get_if_present(node, "bamg_strong_threshold", bamgStrongThreshold_, bamgStrongThreshold_);
  get_if_present(node, "bamg_output_level", output_level, output_level);

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetPrintLevel,
    output_level))); // print AMG solution info
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetLogging,
    output_level))); // print AMG solution info
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetCoarsenType,
    bamgCoarsenType_))); // Falgout coarsening
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetCycleType,
    bamgCycleType_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetRelaxType,
    bamgRelaxType_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetRelaxOrder, bamgRelaxOrder_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetNumSweeps,
    bamgNumSweeps_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetMaxLevels,
    bamgMaxLevels_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetStrongThreshold,
    bamgStrongThreshold_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetTol,
    0.0))); // Conv tolerance
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetMaxIter,
    1))); // Maximum number of iterations

  if (node["bamg_non_galerkin_tol"]) {
    double non_galerkin_tol = node["bamg_non_galerkin_tol"].as<double>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetNonGalerkinTol,
      non_galerkin_tol)));

    if (node["bamg_non_galerkin_level_tols"]) {
      auto& ngnode = node["bamg_non_galerkin_level_tols"];
      std::vector<int> levels = ngnode["levels"].as<std::vector<int>>();
      std::vector<double> tol = ngnode["tolerances"].as<std::vector<double>>();

      if (levels.size() != tol.size())
        throw std::runtime_error("Hypre Config:: Invalid bamg_non_galerkin_level_tols");

      for (size_t i = 0; i < levels.size(); i++) {
        funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
          Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetLevelNonGalerkinTol, tol[i],
          levels[i])));
      }
    }
  }

  if (node["bamg_variant"]) {
    int int_value = node["bamg_variant"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetVariant, int_value)));
  }

  if (node["bamg_keep_transpose"]) {
    int int_value = node["bamg_keep_transpose"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetKeepTranspose, int_value)));
  }

  if (node["bamg_interp_type"]) {
    int int_value = node["bamg_interp_type"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetInterpType, int_value)));
  }

  if (node["bamg_smooth_type"]) {
    int smooth_type = node["bamg_smooth_type"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetSmoothType,
      smooth_type)));

    // Process Euclid smoother parameters
    if (smooth_type == 9) {
      if (node["bamg_euclid_file"]) {
        bamgEuclidFile_ = node["bamg_euclid_file"].as<std::string>();
        funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
          Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetEuclidFile,
          const_cast<char*>(bamgEuclidFile_.c_str()))));
      }

      if (node["bamg_smooth_num_levels"]) {
        int int_value = node["bamg_smooth_num_levels"].as<int>();
        funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
          Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetSmoothNumLevels,
          int_value)));
      }
      if (node["bamg_smooth_num_sweeps"]) {
        int int_value = node["bamg_smooth_num_sweeps"].as<int>();
        funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
          Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetSmoothNumSweeps,
          int_value)));
      }
    }
  }

  if (node["bamg_min_coarse_size"]) {
    int int_value = node["bamg_min_coarse_size"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetMinCoarseSize, int_value)));
  }
  if (node["bamg_max_coarse_size"]) {
    int int_value = node["bamg_max_coarse_size"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetMaxCoarseSize, int_value)));
  }
  if (node["bamg_pmax_elmts"]) {
    int int_value = node["bamg_pmax_elmts"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetPMaxElmts, int_value)));
  }
  if (node["bamg_agg_num_levels"]) {
    int int_value = node["bamg_agg_num_levels"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetAggNumLevels, int_value)));
  }
  if (node["bamg_agg_interp_type"]) {
    int int_value = node["bamg_agg_interp_type"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetAggInterpType, int_value)));
  }
  if (node["bamg_agg_pmax_elmts"]) {
    int int_value = node["bamg_agg_pmax_elmts"].as<int>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetAggPMaxElmts, int_value)));
  }
  if (node["bamg_trunc_factor"]) {
    double float_value = node["bamg_trunc_factor"].as<double>();
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Prec, &HYPRE_BoomerAMGSetTruncFactor, float_value)));
  }

  paramsPrecond_->set("Preconditioner", Ifpack2::Hypre::BoomerAMG);
  paramsPrecond_->set("SetPreconditioner", true);
}

void
HypreLinearSolverConfig::euclid_precond_config(const YAML::Node& node)
{
  // Defaults are based on recommendations from HYPRE Ref. Manual
  int level = 1;    // Assume 3-D problem, set to 4-8 for 2-D
  int bj = 0;       // 0 - PILU; 1 - Block Jacobi
  int eu_stats = 0; // write out euclid stats
  int eu_mem = 0;   // print memory diagnostic
  std::string eu_filename = "none";

  get_if_present(node, "euclid_levels", level, level);
  get_if_present(node, "euclid_use_block_jacobi", bj, bj);
  get_if_present(node, "euclid_stats", eu_stats, eu_stats);
  get_if_present(node, "euclid_mem", eu_mem, eu_mem);
  get_if_present(node, "euclid_filename", eu_filename, eu_filename);

  funcParams_.push_back(
    Teuchos::rcp(new Ifpack2::FunctionParameter(
                   Ifpack2::Hypre::Prec, &HYPRE_EuclidSetLevel, level)));
  funcParams_.push_back(
    Teuchos::rcp(new Ifpack2::FunctionParameter(
                   Ifpack2::Hypre::Prec, &HYPRE_EuclidSetBJ, bj)));
  funcParams_.push_back(
    Teuchos::rcp(new Ifpack2::FunctionParameter(
                   Ifpack2::Hypre::Prec, &HYPRE_EuclidSetStats, eu_stats)));
  funcParams_.push_back(
    Teuchos::rcp(new Ifpack2::FunctionParameter(
                   Ifpack2::Hypre::Prec, &HYPRE_EuclidSetMem, eu_mem)));

  paramsPrecond_->set("Preconditioner", Ifpack2::Hypre::Euclid);
  paramsPrecond_->set("SetPreconditioner", true);
}

void
HypreLinearSolverConfig::hypre_gmres_solver_config(const YAML::Node& node)
{
  int logLevel = 1;
  get_if_present(node, "log_level", logLevel, logLevel);

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_GMRESSetKDim, kspace_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_GMRESSetMaxIter, maxIterations_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_GMRESSetTol, tolerance_)));

  if (hasAbsTol_) {
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Solver, &HYPRE_GMRESSetAbsoluteTol, absTol_)));
  }

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_GMRESSetPrintLevel, outputLevel_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_GMRESSetLogging, logLevel)));

  paramsPrecond_->set("Solver", Ifpack2::Hypre::GMRES);
}

void
HypreLinearSolverConfig::hypre_cogmres_solver_config(const YAML::Node& node)
{
#ifdef HYPRE_COGMRES
  int logLevel = 1;
  get_if_present(node, "log_level", logLevel, logLevel);

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_COGMRESSetKDim, kspace_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_COGMRESSetMaxIter, maxIterations_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_COGMRESSetTol, tolerance_)));

  if (hasAbsTol_) {
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Solver, &HYPRE_COGMRESSetAbsoluteTol, absTol_)));
  }

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_COGMRESSetPrintLevel, outputLevel_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_COGMRESSetLogging, logLevel)));
  paramsPrecond_->set("Solver", Ifpack2::Hypre::COGMRES);
#endif
}

void
HypreLinearSolverConfig::hypre_flexgmres_solver_config(const YAML::Node& node)
{
  int logLevel = 1;
  get_if_present(node, "log_level", logLevel, logLevel);

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_FlexGMRESSetKDim, kspace_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_FlexGMRESSetMaxIter, maxIterations_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_FlexGMRESSetTol, tolerance_)));

  if (hasAbsTol_) {
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Solver, &HYPRE_FlexGMRESSetAbsoluteTol, absTol_)));
  }

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_FlexGMRESSetPrintLevel, outputLevel_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_FlexGMRESSetLogging, logLevel)));
  paramsPrecond_->set("Solver", Ifpack2::Hypre::FlexGMRES);
}

void
HypreLinearSolverConfig::hypre_lgmres_solver_config(const YAML::Node& node)
{
  int logLevel = 1;
  int augDim = 2;
  get_if_present(node, "log_level", logLevel, logLevel);
  get_if_present(node, "lgmres_aug_dim", augDim, augDim);

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_LGMRESSetKDim, kspace_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_LGMRESSetAugDim, augDim)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_LGMRESSetMaxIter, maxIterations_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_LGMRESSetTol, tolerance_)));

  if (hasAbsTol_) {
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Solver, &HYPRE_LGMRESSetAbsoluteTol, absTol_)));
  }

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_LGMRESSetPrintLevel, outputLevel_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_LGMRESSetLogging, logLevel)));
  paramsPrecond_->set("Solver", Ifpack2::Hypre::LGMRES);
}

void
HypreLinearSolverConfig::hypre_bicgstab_solver_config(const YAML::Node& node)
{
  int logLevel = 1;
  get_if_present(node, "log_level", logLevel, logLevel);

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BiCGSTABSetMaxIter, maxIterations_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BiCGSTABSetTol, tolerance_)));

  if (hasAbsTol_) {
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Solver, &HYPRE_BiCGSTABSetAbsoluteTol, absTol_)));
  }

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BiCGSTABSetPrintLevel, outputLevel_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_BiCGSTABSetLogging, logLevel)));
  paramsPrecond_->set("Solver", Ifpack2::Hypre::BiCGSTAB);
}


void
HypreLinearSolverConfig::hypre_pcg_solver_config(const YAML::Node& node)
{
  int logLevel = 1;
  get_if_present(node, "log_level", logLevel, logLevel);

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_PCGSetMaxIter, maxIterations_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_PCGSetTol, tolerance_)));

  if (hasAbsTol_) {
    funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
      Ifpack2::Hypre::Solver, &HYPRE_PCGSetAbsoluteTol, absTol_)));
  }

  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_PCGSetPrintLevel, outputLevel_)));
  funcParams_.push_back(Teuchos::rcp(new Ifpack2::FunctionParameter(
    Ifpack2::Hypre::Solver, &HYPRE_PCGSetLogging, logLevel)));
  paramsPrecond_->set("Solver", Ifpack2::Hypre::PCG);
}

void
HypreLinearSolverConfig::configure_hypre_preconditioner
(
  const YAML::Node& node
)
{
  if (precond_ == "boomerAMG") {
    boomerAMG_precond_config(node);
  }
  else if (precond_ == "euclid") {
    euclid_precond_config(node);
  }
  else if (precond_ == "none") {
    paramsPrecond_->set("SetPreconditioner", false);
  }
  else {
    throw std::runtime_error("Invalid HYPRE preconditioner specified: " + precond_);
  }

}

void
HypreLinearSolverConfig::configure_hypre_solver
(
  const YAML::Node& node
)
{
  if (method_ == "hypre_gmres") {
    hypre_gmres_solver_config(node);
  }
  else if (method_ == "hypre_cogmres") {
#ifdef HYPRE_COGMRES
    hypre_cogmres_solver_config(node);
#else
    throw std::runtime_error("HYPRE version does not support COGMRES");
#endif
  }
  else if (method_ == "hypre_lgmres") {
    hypre_lgmres_solver_config(node);
  }
  else if (method_ == "hypre_flexgmres") {
    hypre_flexgmres_solver_config(node);
  }
  else if (method_ == "hypre_pcg") {
    hypre_pcg_solver_config(node);
  }
  else if (method_ == "hypre_bicgstab") {
    hypre_bicgstab_solver_config(node);
  }
  else {
    throw std::runtime_error("Invalid HYPRE solver specified: " + method_);
  }

}

} // nalu
} // sierra

#endif
