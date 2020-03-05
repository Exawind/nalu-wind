// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef EquationSystems_h
#define EquationSystems_h

#include <Enums.h>
#include "NaluParsedTypes.h"

// stk
namespace stk{
namespace mesh{
class Part;
class FieldBase;
}
}

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace YAML {
  class Node;
}

#include<vector>
#include<string>

namespace sierra{
namespace nalu{

class Realm;
class EquationSystem;
class Simulation;
class AlgorithmDriver;
class UpdateOversetFringeAlgorithmDriver;

typedef std::vector<EquationSystem *> EquationSystemVector;

/** A collection of Equations to be solved on a Realm
 *
 *  EquationSystems holds a vector of EquationSystem instances representing the
 *  equations that are being solved in a given Realm and is responsible for the
 *  management of the solve and update of the various field quantities in a
 *  given timestep.
 *
 *  \sa EquationSystems::solve_and_update
 */
class EquationSystems
{
 public:

  EquationSystems(
    Realm &realm);
  ~EquationSystems();

  void load(const YAML::Node & node);
  
  std::string get_solver_block_name(
    const std::string eqName) const;

  void breadboard();

  Simulation *root();
  Realm *parent();

  // ease of access methods to particular equation system
  size_t size() {return equationSystemVector_.size();}
  EquationSystem *operator[](int i) { return equationSystemVector_[i];}
  
  void register_nodal_fields(
    const std::vector<std::string> targetNames);

  void register_edge_fields(
    const std::vector<std::string> targetNames);

  void register_element_fields(
    const std::vector<std::string> targetNames);

  void register_interior_algorithm(
    const std::vector<std::string> targetNames);

  void register_wall_bc(
    const std::string targetName,
    const WallBoundaryConditionData &wallBCData);

  void register_inflow_bc(
    const std::string targetName,
    const InflowBoundaryConditionData &inflowBCData);

  void register_open_bc(
    const std::string targetName,
    const OpenBoundaryConditionData &openBCData);

  void register_symmetry_bc(
    const std::string targetName,
    const SymmetryBoundaryConditionData &symmetryBCData);

  void register_abltop_bc(
    const std::string targetName,
    const ABLTopBoundaryConditionData &ablTopBCData);

  void register_periodic_bc(
    const std::string targetNameMaster,
    const std::string targetNameSlave,
    const PeriodicBoundaryConditionData &periodicBCData);

  void register_overset_bc(
    const OversetBoundaryConditionData &oversetBCData);

  void register_non_conformal_bc(
    const NonConformalBoundaryConditionData &nonConformalBCData);

  void register_initial_condition_fcn(
    stk::mesh::Part *part,
    const UserFunctionInitialConditionData &fcnIC);

  void initialize();
  void reinitialize_linear_system();
  void populate_derived_quantities();
  void initial_work();

  /** Solve and update the state of all variables for a given timestep
   *
   *  This method is responsible for executing setup actions before calling
   *  solve, performing the actual solve, updating the solution, and performing
   *  post-solve actions after the solution has been updated. To provide
   *  sufficient granularity and control of this pre- and post- solve actions,
   *  the solve method uses the following series of steps:
   *
   *  ```
   *  // Perform tasks for this timestep before any Equation system is called
   *  pre_iter_work();
   *  // Iterate over all equation systems
   *  for (auto eqsys: equationSystems_) {
   *    eqsys->pre_iter_work();
   *    eqsys->solve_and_update();
   *    eqsys->post_iter_work();
   *  }
   *  // Perform tasks after all equation systems have updated
   *  post_iter_work();
   *  ```
   *
   *  Tasks that require to be performed before any equation system is solved
   *  for needs to be registered to preIterAlgDriver_ on EquationSystems,
   *  similiary for post-solve tasks. And actions to be performed immediately
   *  before individual equation system solves need to be registered in
   *  EquationSystem::preIterAlgDriver_.
   *
   *  \sa pre_iter_work(), post_iter_work(), EquationSystem::pre_iter_work(),
   *  \sa EquationSystem::post_iter_work()
   */
  bool solve_and_update();
  double provide_system_norm();
  double provide_mean_system_norm();

  void predict_state();
  void populate_boundary_data();
  void boundary_data_to_state_data();
  void provide_output();
  void dump_eq_time();
  void pre_timestep_work();
  void post_converged_work();
  void evaluate_properties();

  /** Perform necessary setup tasks that affect all EquationSystem instances at
   *  a given timestep.
   *
   *  \sa EquationSystems::solve_and_update()
   */
  void pre_iter_work();

  /** Perform necessary actions once all EquationSystem instances have been
   * updated for the prescribed number of _outer iterations_ at a given
   * timestep.
   *
   *  \sa EquationSystems::solve_and_update()
   */
  void post_iter_work();

  void post_external_data_transfer_work();


  void register_overset_field_update(stk::mesh::FieldBase*, int, int);

  bool all_systems_decoupled() const;

  Realm &realm_;
  std::string name_;
  int maxIterations_;

  EquationSystemVector equationSystemVector_;
  std::map<std::string, std::string> solverSpecMap_;

  /// A list of tasks to be performed before all EquationSystem::solve_and_update
  std::vector<AlgorithmDriver*> preIterAlgDriver_;

  /// A list of tasks to be performed after all EquationSystem::solve_and_update
  std::vector<AlgorithmDriver*> postIterAlgDriver_;

  std::unique_ptr<UpdateOversetFringeAlgorithmDriver> oversetUpdater_;

  /** Default number of overset coupling iterations
   *
   *  This parameter controls the global settings for _decoupled overset_
   *  simulations. User can override this for individual equations by specifying
   *  the values for the specific equation system.
   */
  int numOversetItersDefault_{1};

  /** Global flag indicating whether decoupled overset is used for all equation
   * systems in this Realm.
   *
   *  User can override this for individual equation systems by using the
   *  appropriate input options.
   */
  bool decoupledOversetGlobalFlag_{false};
};

} // namespace nalu
} // namespace Sierra

#endif
