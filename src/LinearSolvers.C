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

namespace sierra {
namespace nalu {

LinearSolvers::LinearSolvers(Simulation& sim) : sim_(sim) {}
LinearSolvers::~LinearSolvers()
{
  for (SolverMap::const_iterator pos = solvers_.begin(); pos != solvers_.end();
       ++pos)
    delete pos->second;
  for (SolverTpetraConfigMap::const_iterator pos = solverTpetraConfig_.begin();
       pos != solverTpetraConfig_.end(); ++pos)
    delete pos->second;

#ifdef NALU_USES_HYPRE
  for (auto item : solverHypreConfig_) {
    delete (item.second);
  }
#endif
}

Simulation*
LinearSolvers::root()
{
  return &sim_;
}
Simulation*
LinearSolvers::parent()
{
  return root();
}

void
LinearSolvers::load(const YAML::Node& node)
{
  register_presets();
  const YAML::Node nodes = node["linear_solvers"];
  if (nodes) {
    for (size_t inode = 0; inode < nodes.size(); ++inode) {
      YAML::Node linear_solver_node = nodes[inode];

      std::string preset_name;
      get_if_present(linear_solver_node, "solver_preset", preset_name, std::string("none"));
      std::transform(preset_name.begin(), preset_name.end(), preset_name.begin(), ::tolower);

      Teuchos::ParameterList presetParams;
      Teuchos::ParameterList presetParamsPrecond;
      if (preset_name != "none") {
        auto * preset_solver = PresetSolverRepo::getSolver(preset_name);
        preset_solver->populateParams(linear_solver_node, presetParamsPrecond);
      }
      std::string solver_type = "tpetra";
      get_if_present_no_default(linear_solver_node, "type", solver_type);
      // proceed with the single supported solver strategy
      if (solver_type == "tpetra") {
        TpetraLinearSolverConfig* linearSolverConfig =
          new TpetraLinearSolverConfig();
        linearSolverConfig->load(linear_solver_node, presetParamsPrecond);
        solverTpetraConfig_[linearSolverConfig->name()] = linearSolverConfig;
      } else if (solver_type == "hypre") {
#ifdef NALU_USES_HYPRE
        HypreLinearSolverConfig* linSolverCfg = new HypreLinearSolverConfig();
        linSolverCfg->load(linear_solver_node, presetParamsPrecond);
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
    TpetraLinearSolverConfig* linearSolverConfig = (*iterT).second;
    foundT = true;
    theSolver = new TpetraLinearSolver(
      solverName, linearSolverConfig, linearSolverConfig->params(),
      linearSolverConfig->paramsPrecond(), this);
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

void LinearSolvers::register_presets()
{
  if (PresetSolverRepo::getSolverMap().empty())
  {
    PresetSolverRepo::registerPreset(new ScalarTpetraPresetSolver);
    PresetSolverRepo::registerPreset(new EllipticTpetraPresetSolver);
    PresetSolverRepo::registerPreset(new MomentumHyprePresetSolver);
    PresetSolverRepo::registerPreset(new ScalarHyprePresetSolver);
    PresetSolverRepo::registerPreset(new EllipticHyprePresetSolver);
  }
}
void ScalarTpetraPresetSolver::populateParams(YAML::Node & node, Teuchos::ParameterList & /* Unused */) const
{
  YAML::Node myNode;
  myNode["name"] = node["name"];
  myNode["type"] = "tpetra";
  myNode["method"] = "gmres";
  myNode["preconditioner"] = "sgs";
  myNode["tolerance"] = "1e-6";
  myNode["max_iterations"] = "75";
  myNode["kspace"] = "75";
  myNode["output_level"] = "0";
  node = myNode;
  
  NaluEnv::self().naluOutputP0() << "Scalar Tpetra preset solver selected"<<std::endl;
  NaluEnv::self().naluOutputP0() << "Equivalent yaml input:"<<std::endl;
  NaluEnv::self().naluOutputP0() << myNode<<std::endl;
  NaluEnv::self().naluOutputP0() << "---------------------------"<<std::endl;
}
std::string ScalarTpetraPresetSolver::getName() const
{
  return "scalar_tpetra";
}

void EllipticTpetraPresetSolver::populateParams(YAML::Node & node, Teuchos::ParameterList & presetParamsPrecond) const
{
  YAML::Node myNode;
  myNode["name"] = node["name"];
  myNode["type"] = "tpetra";
  myNode["method"] = "gmres";
  myNode["preconditioner"] = "muelu";
  myNode["tolerance"] = "1e-6";
  myNode["max_iterations"] = "75";
  myNode["kspace"] = "75";
  myNode["output_level"] = "0";
  myNode["recompute_preconditioner"] = "no";
  node = myNode;

  presetParamsPrecond.set("verbosity", "none");
  presetParamsPrecond.set("coarse: max size", 1000);
  presetParamsPrecond.set("smoother: type", "CHEBYSHEV");
  auto & smoother_pl = presetParamsPrecond.sublist("smoother: params", false);
  smoother_pl.set("chebyshev: degree", 2);
  smoother_pl.set("chebyshev: ratio eigenvalue", 20.0);
  smoother_pl.set("chebyshev: min eigenvalue", 1.0);
  smoother_pl.set("chebyshev: zero starting solution", true);
  smoother_pl.set("chebyshev: eigenvalue max iterations", 15);
  presetParamsPrecond.set("aggregation: type", "uncoupled");
  presetParamsPrecond.set("aggregation: drop tol", 0.02);
  presetParamsPrecond.set("repartition: enable", true);
  presetParamsPrecond.set("repartition: min rows per proc", 1000);
  presetParamsPrecond.set("repartition: start level", 2);
  presetParamsPrecond.set("repartition: max imbalance", 1.327);
  presetParamsPrecond.set("repartition: partitioner", "zoltan2");

  NaluEnv::self().naluOutputP0() << "Elliptic Tpetra preset solver selected"<<std::endl;
  NaluEnv::self().naluOutputP0() << "Equivalent yaml input:"<<std::endl;
  NaluEnv::self().naluOutputP0() << myNode<<std::endl;
  NaluEnv::self().naluOutputP0() << "muelu_xml_file_name: elliptic_tpetra.xml"<<std::endl;
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "Contents of MueLu xml file:"<<std::endl;

  // Tried doing this with the xml writer, didn't like the output.
  NaluEnv::self().naluOutputP0() << "<ParameterList name=\"MueLu\">"<<std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"verbosity\"                        type=\"string\"   value=\"none\"/>" <<std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"coarse: max size\"                 type=\"int\"      value=\"1000\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"smoother: type\"                   type=\"string\"   value=\"CHEBYSHEV\"/>" <<std::endl;
  NaluEnv::self().naluOutputP0() << "<ParameterList    name=\"smoother: params\">"<<std::endl;
  NaluEnv::self().naluOutputP0() << "   <Parameter name=\"chebyshev: degree\"                    type=\"int\"      value=\"2\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "   <Parameter name=\"chebyshev: ratio eigenvalue\"          type=\"double\"   value=\"20\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "   <Parameter name=\"chebyshev: min eigenvalue\"            type=\"double\"   value=\"1.0\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "   <Parameter name=\"chebyshev: zero starting solution\"    type=\"bool\"     value=\"true\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "   <Parameter name=\"chebyshev: eigenvalue max iterations\" type=\"int\"      value=\"15\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "</ParameterList>"<<std::endl;
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"aggregation: type\"                type=\"string\"   value=\"uncoupled\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"aggregation: drop tol\"            type=\"double\"   value=\"0.02\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"repartition: enable\"              type=\"bool\"     value=\"true\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"repartition: min rows per proc\"   type=\"int\"      value=\"1000\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"repartition: start level\"         type=\"int\"      value=\"2\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"repartition: max imbalance\"       type=\"double\"   value=\"1.327\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "<Parameter        name=\"repartition: partitioner\"         type=\"string\"   value=\"zoltan2\"/>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "</ParameterList>"<<std::endl;
  NaluEnv::self().naluOutputP0() << "---------------------------"<<std::endl;
}
std::string EllipticTpetraPresetSolver::getName() const
{
  return "elliptic_tpetra";
}

void MomentumHyprePresetSolver::populateParams(YAML::Node & node, Teuchos::ParameterList & /* Unused */) const
{
  YAML::Node myNode;
  myNode["name"] = node["name"];
  myNode["type"] = "hypre";
  myNode["method"] = "hypre_gmres";
  myNode["preconditioner"] = "boomerAMG";
  myNode["tolerance"] = "1e-12";
  myNode["max_iterations"] = "200";
  myNode["kspace"] = "75";
  myNode["output_level"] = "0";
  myNode["segregated_solver"] = "no";
  myNode["write_matrix_files"] = "no";
  myNode["reuse_linear_system"] = "yes";
  myNode["recompute_preconditioner_frequency"] = 100;
  myNode["simple_hypre_matrix_assemble"] = "no";
  myNode["dump_hypre_matrix_stats"] = "no";
  myNode["write_preassembly_matrix_files"] = "no";
  myNode["bamg_max_levels"] = 1;
  myNode["bamg_relax_type"] = 12;
  myNode["bamg_num_sweeps"] = 2;
  myNode["bamg_relax_order"] = 0;
  node = myNode;

  NaluEnv::self().naluOutputP0() << "Momentum Hypre preset solver selected"<<std::endl;
  NaluEnv::self().naluOutputP0() << "Equivalent yaml input:"<<std::endl;
  NaluEnv::self().naluOutputP0() << myNode<<std::endl;
  NaluEnv::self().naluOutputP0() << "---------------------------"<<std::endl;
}
std::string MomentumHyprePresetSolver::getName() const
{
  return "momentum_hypre";
}

void ScalarHyprePresetSolver::populateParams(YAML::Node & node, Teuchos::ParameterList & /* Unused */) const
{
  YAML::Node myNode;
  myNode["name"] = node["name"];
  myNode["type"] = "hypre";
  myNode["method"] = "hypre_gmres";
  myNode["preconditioner"] = "boomerAMG";
  myNode["tolerance"] = "1e-12";
  myNode["max_iterations"] = "200";
  myNode["kspace"] = "75";
  myNode["output_level"] = "0";
  myNode["write_matrix_files"] = "no";
  myNode["reuse_linear_system"] = "yes";
  myNode["recompute_preconditioner_frequency"] = 100;
  myNode["simple_hypre_matrix_assemble"] = "no";
  myNode["dump_hypre_matrix_stats"] = "no";
  myNode["write_preassembly_matrix_files"] = "no";
  myNode["bamg_max_levels"] = 1;
  myNode["bamg_relax_type"] = 12;
  myNode["bamg_num_sweeps"] = 2;
  myNode["bamg_relax_order"] = 0;
  node = myNode;

  NaluEnv::self().naluOutputP0() << "Scalar Hypre preset solver selected"<<std::endl;
  NaluEnv::self().naluOutputP0() << "Equivalent yaml input:"<<std::endl;
  NaluEnv::self().naluOutputP0() << myNode<<std::endl;
  NaluEnv::self().naluOutputP0() << "---------------------------"<<std::endl;
}
std::string ScalarHyprePresetSolver::getName() const
{
  return "scalar_hypre";
}

void EllipticHyprePresetSolver::populateParams(YAML::Node & node, Teuchos::ParameterList & /* Unused */) const
{
  YAML::Node myNode;
  myNode["name"] = node["name"];
  myNode["type"] = "hypre";
  myNode["method"] = "hypre_gmres";
  myNode["preconditioner"] = "boomerAMG";
  myNode["tolerance"] = "1e-12";
  myNode["max_iterations"] = "200";
  myNode["kspace"] = "75";
  myNode["output_level"] = "0";
  myNode["write_matrix_files"] = "no";
  myNode["reuse_linear_system"] = "yes";
  myNode["recompute_preconditioner_frequency"] = 100;
  myNode["simple_hypre_matrix_assemble"] = "no";
  myNode["dump_hypre_matrix_stats"] = "no";
  myNode["write_preassembly_matrix_files"] = "no";
  myNode["bamg_max_levels"] = 7;
  myNode["bamg_coarsen_type"] = 8;
  myNode["bamg_interp_type"] = 6;
  myNode["bamg_relax_type"] = 11;
  myNode["bamg_num_up_sweeps"] = 1;
  myNode["bamg_num_down_sweeps"] = 2;
  myNode["bamg_num_coarse_sweeps"] = 1;
  myNode["bamg_cycle_type"] = 1;
  myNode["bamg_relax_order"] = 0;
  myNode["bamg_trunc_factor"] = 0.1;
  myNode["bamg_agg_num_levels"] = 2;
  myNode["bamg_agg_interp_type"] = 7;
  myNode["bamg_agg_pmax_elmts"] = 3;
  myNode["bamg_pmax_elmts"] = 3;
  myNode["bamg_strong_threshold"] = 0.25;
  node = myNode;

  NaluEnv::self().naluOutputP0() << "Elliptic Hypre preset solver selected"<<std::endl;
  NaluEnv::self().naluOutputP0() << "Equivalent yaml input:"<<std::endl;
  NaluEnv::self().naluOutputP0() << myNode<<std::endl;
  NaluEnv::self().naluOutputP0() << "---------------------------"<<std::endl;
}
std::string EllipticHyprePresetSolver::getName() const
{
  return "elliptic_hypre";
}

PresetSolverRepo::map_type PresetSolverRepo::presets_map_;

void PresetSolverRepo::registerPreset(PresetSolverBase * solver_type)
{
  auto type = solver_type->getName();
  if (presets_map_.find(type) == presets_map_.end())
  {
    presets_map_[type] = std::unique_ptr<PresetSolverBase>(solver_type);
  }
}

PresetSolverBase * PresetSolverRepo::getSolver(const std::string & type)
{
  auto iter = presets_map_.find(type);
  if (iter == presets_map_.end())
  {
    throw std::runtime_error(
          "Desired preset solver not found.");
  }
  return presets_map_[type].get();
}

const PresetSolverRepo::map_type & PresetSolverRepo::getSolverMap()
{
  return PresetSolverRepo::presets_map_;
}

} // namespace nalu
} // namespace sierra
