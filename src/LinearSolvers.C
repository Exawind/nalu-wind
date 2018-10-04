/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSolverConfig.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <Simulation.h>

#ifdef NALU_USES_HYPRE
#include "HypreDirectSolver.h"
#include "HypreUVWSolver.h"
#endif

#include <yaml-cpp/yaml.h>

namespace sierra{
namespace nalu{

LinearSolvers::LinearSolvers(Simulation& sim) : sim_(sim) {}
LinearSolvers::~LinearSolvers()
{
  for(SolverMap::const_iterator pos=solvers_.begin(); pos!=solvers_.end(); ++pos)
    delete pos->second;
  for(SolverTpetraConfigMap::const_iterator pos=solverTpetraConfig_.begin(); pos!=solverTpetraConfig_.end(); ++pos)
    delete pos->second;

#ifdef NALU_USES_HYPRE
  for (auto item: solverHypreConfig_) {
    delete (item.second);
  }
#endif
}

Simulation *LinearSolvers::root() { return &sim_; }
Simulation *LinearSolvers::parent() { return root(); }

void
LinearSolvers::load(const YAML::Node & node)
{
  const YAML::Node nodes = node["linear_solvers"];
  if ( nodes ) {
    for ( size_t inode = 0; inode <  nodes.size(); ++inode ) {
      const YAML::Node linear_solver_node = nodes[inode] ;
      std::string solver_type = "tpetra";
      get_if_present_no_default(linear_solver_node, "type", solver_type);      
      // proceed with the single supported solver strategy
      if (solver_type == "tpetra") {
        TpetraLinearSolverConfig * linearSolverConfig = new TpetraLinearSolverConfig();
        linearSolverConfig->load(linear_solver_node);
        solverTpetraConfig_[linearSolverConfig->name()] = linearSolverConfig; 
      }
      else if (solver_type == "hypre") {
#ifdef NALU_USES_HYPRE
        HypreLinearSolverConfig *linSolverCfg = new HypreLinearSolverConfig();
        linSolverCfg->load(linear_solver_node);
        solverHypreConfig_[linSolverCfg->name()] = linSolverCfg;
#else
        throw std::runtime_error("HYPRE support must be enabled during compile time.");
#endif
      }
      else if (solver_type == "epetra") {
        throw std::runtime_error("epetra solver_type has been deprecated");
      }
      else {
        throw std::runtime_error("unknown solver type");
      }
    }
  }
}

LinearSolver *
LinearSolvers::create_solver(
  std::string solverBlockName,
  EquationType theEQ )
{

  // provide unique name
  std::string solverName = EquationTypeMap[theEQ] + "_Solver";
  
  LinearSolver *theSolver = NULL;
  
  // check in tpetra map...
  bool foundT = false;
  SolverTpetraConfigMap::const_iterator iterT
    = solverTpetraConfig_.find(solverBlockName);
  if (iterT != solverTpetraConfig_.end()) {
    TpetraLinearSolverConfig *linearSolverConfig = (*iterT).second;
    foundT = true;
    theSolver = new TpetraLinearSolver(solverName,
                                       linearSolverConfig,
                                       linearSolverConfig->params(),
                                       linearSolverConfig->paramsPrecond(), this);
  }
#ifdef NALU_USES_HYPRE
  else {
    auto hIter = solverHypreConfig_.find(solverBlockName);
    if (hIter != solverHypreConfig_.end()) {
      HypreLinearSolverConfig *cfg = hIter->second;
      foundT = true;
      if ((theEQ == EQ_MOMENTUM) && cfg->useSegregatedSolver())
        theSolver = new HypreUVWSolver(solverName, cfg, this);
      else
        theSolver = new HypreDirectSolver(solverName, cfg, this);
    }
  }
#endif

  // error check; none found
  if ( !foundT ) {
    throw std::runtime_error("solver name block not found; error in solver creation; check: " + solverName);
  }

  // set it and return
  solvers_[theEQ] = theSolver;
  return theSolver;
}


} // namespace nalu
} // namespace Sierra
