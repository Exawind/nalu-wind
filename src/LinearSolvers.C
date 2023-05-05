// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSolverConfig.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <Simulation.h>
#include <Teuchos_ParameterList.hpp>

#ifdef NALU_USES_HYPRE
#include "HypreDirectSolver.h"
#include "HypreUVWSolver.h"
#endif

#include <yaml-cpp/yaml.h>

namespace sierra {
namespace nalu {

LinearSolvers::LinearSolvers(Simulation& sim) : sim_(sim) {}
LinearSolvers::~LinearSolvers()
{
  for (SolverMap::const_iterator pos = solvers_.begin(); pos != solvers_.end();
       ++pos)
    delete pos->second;
#ifdef NALU_USES_TRILINOS_SOLVERS
  for (SolverTpetraConfigMap::const_iterator pos = solverTpetraConfig_.begin();
       pos != solverTpetraConfig_.end(); ++pos)
    delete pos->second;
#endif

#ifdef NALU_USES_HYPRE
  for (auto item : solverHypreConfig_) {
    delete (item.second);
  }
#endif
}

void
LinearSolvers::load(const YAML::Node& node)
{
  const YAML::Node nodes = node["linear_solvers"];
  if (nodes) {
    for (size_t inode = 0; inode < nodes.size(); ++inode) {
      const YAML::Node linear_solver_node = nodes[inode];
#ifdef NALU_USES_TRILINOS_SOLVERS
      std::string solver_type = "tpetra";
      // this used to be "tpetra" unconditionally.
      // now it is ifdef'd, but we are guaranteed that
      // if TRILINOS_SOLVERS is off, then HYPRE is on.
#else
      std::string solver_type = "hypre";
#endif
      get_if_present_no_default(linear_solver_node, "type", solver_type);
      // proceed with the single supported solver strategy
      if (solver_type == "tpetra") {
#ifdef NALU_USES_TRILINOS_SOLVERS
        TpetraLinearSolverConfig* linearSolverConfig =
          new TpetraLinearSolverConfig();
        linearSolverConfig->load(linear_solver_node);
        solverTpetraConfig_[linearSolverConfig->name()] = linearSolverConfig;
#else
        throw std::runtime_error(
          "Trilinos solver support must be enabled during compile time.");
#endif
      } else if (solver_type == "hypre") {
#ifdef NALU_USES_HYPRE
        HypreLinearSolverConfig* linSolverCfg = new HypreLinearSolverConfig();
        linSolverCfg->load(linear_solver_node);
        solverHypreConfig_[linSolverCfg->name()] = linSolverCfg;
#else
        throw std::runtime_error(
          "HYPRE support must be enabled during compile time.");
#endif
      } else if (solver_type == "epetra") {
        throw std::runtime_error("epetra solver_type has been deprecated");
      } else {
        throw std::runtime_error("unknown solver type");
      }
    }
  }
}

Teuchos::ParameterList
LinearSolvers::get_solver_configuration(std::string solverBlockName)
{
  auto it = solverTpetraConfig_.find(solverBlockName);
  if (it == solverTpetraConfig_.end()) {
    throw std::runtime_error(
      "solver name block not found; error in solver creation; check: " +
      solverBlockName);
  }
  return *it->second->params();
}

LinearSolver*
LinearSolvers::create_solver(
  std::string solverBlockName, const std::string realmName, EquationType theEQ)
{

  // provide unique name
  std::string solverName = EquationTypeMap[theEQ] + "_Solver";

  LinearSolver* theSolver = NULL;

  // check in tpetra map...
  bool foundT = false;
  SolverTpetraConfigMap::const_iterator iterT =
    solverTpetraConfig_.find(solverBlockName);
  if (iterT != solverTpetraConfig_.end()) {
#ifdef NALU_USES_TRILINOS_SOLVERS
    TpetraLinearSolverConfig* linearSolverConfig = (*iterT).second;
    foundT = true;
    theSolver = new TpetraLinearSolver(
      solverName, linearSolverConfig, linearSolverConfig->params(),
      linearSolverConfig->paramsPrecond(), this);
#else
    throw std::runtime_error(
      solverBlockName +
      " found but "
      "Trilinos solver support not enabled during compile time.");
#endif
  }
#ifdef NALU_USES_HYPRE
  else {
    auto hIter = solverHypreConfig_.find(solverBlockName);
    if (hIter != solverHypreConfig_.end()) {
      HypreLinearSolverConfig* cfg = hIter->second;
      foundT = true;
      if ((theEQ == EQ_MOMENTUM) && cfg->useSegregatedSolver())
        theSolver = new HypreUVWSolver(solverName, cfg, this);
      else
        theSolver = new HypreDirectSolver(solverName, cfg, this);
    }
  }
#endif

  // error check; none found
  if (!foundT) {
    throw std::runtime_error(
      "solver name block not found; error in solver creation; check: " +
      solverName);
  }

  // set it and return
  const std::string key = realmName + std::to_string(static_cast<int>(theEQ));
  solvers_[key] = theSolver;
  return theSolver;
}

LinearSolver*
LinearSolvers::reinitialize_solver(
  const std::string& solverBlockName,
  const std::string& realmName,
  const EquationType theEQ)
{
  const std::string key = realmName + std::to_string(static_cast<int>(theEQ));

  auto it = solvers_.find(key);
  if (it != solvers_.end()) {
    delete (it->second);
    solvers_.erase(it);
  }

  return create_solver(solverBlockName, realmName, theEQ);
}

} // namespace nalu
} // namespace sierra
