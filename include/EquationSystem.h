/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef EquationSystem_h
#define EquationSystem_h

#include "KokkosInterface.h"
#include "NaluParsing.h"
#include "Realm.h"
#include "PecletFunction.h"

namespace stk{
struct topology;
namespace mesh{
class FieldBase;
class Part;
typedef std::vector<Part*> PartVector;
}
}

namespace sierra{
namespace nalu{

class Algorithm;
class AlgorithmDriver;
class AuxFunctionAlgorithm;
class SolverAlgorithmDriver;
class InitialCondition;
class EquationSystems;
class LinearSystem;
class PostProcessingData;

/** Base class representation of a PDE.
 *
 *  EquationSystem defines the API supported by all concrete implementations of
 *  PDEs for performing the following actions:
 *
 *    - Register computational fields
 *    - Register computational algorithms for interior domain and boundary conditions
 *    - Manage solve and update of the PDE for a given timestep
 */
class EquationSystem
{
public:

  EquationSystem(
    EquationSystems& eqSystems,
    const std::string name = "no_name",
    const std::string eqnTypeName = "no_eqn_type_name");
  virtual ~EquationSystem();

  void set_nodal_gradient(
   const std::string &dofName);

  virtual void initial_work();

  virtual void populate_derived_quantities() {}

  // base class with desired default no-op
  virtual void register_nodal_fields(
    stk::mesh::Part * /* part */) {}

  virtual void register_edge_fields(
    stk::mesh::Part * /* part */) {}

  virtual void register_element_fields(
    stk::mesh::Part * /* part */,
    const stk::topology & /* theTopo */) {}

  // since Equation systems hold other equations systems
  // defaults are provided for all methods below

  virtual void initialize() {}

  /** Assemble the LHS and RHS and perform linear solve for prescribed number of
   * iterations.
   *
   *  This method is invoked in EquationSystems::solve_and_update method as
   *  shown below
   *
   *  ```
   *  pre_iter_work();
   *  // Iterate over all equation systems
   *  for (auto eqsys: equationSystems_) {
   *    eqsys->pre_iter_work();
   *    eqsys->solve_and_update();             //<<<< Assemble and solve system
   *    eqsys->post_iter_work();
   *  }
   *  post_iter_work();
   *  ```
   *
   *  \sa EquationSystems::solve_and_update
   */
  virtual void solve_and_update() {}

  /** Perform setup tasks before entering the solve and update step.
   *
   *  This method is invoked in EquationSystems::solve_and_update method as
   *  shown below
   *
   *  ```
   *  pre_iter_work();
   *  // Iterate over all equation systems
   *  for (auto eqsys: equationSystems_) {
   *    eqsys->pre_iter_work();                //<<<< Pre-iteration setup
   *    eqsys->solve_and_update();
   *    eqsys->post_iter_work();
   *  }
   *  post_iter_work();
   *  ```
   *
   *  \sa EquationSystems::solve_and_update
   */
  virtual void pre_iter_work();

  /** Perform setup tasks after he solve and update step.
   *
   *  This method is invoked in EquationSystems::solve_and_update method as
   *  shown below
   *
   *  ```
   *  pre_iter_work();
   *  // Iterate over all equation systems
   *  for (auto eqsys: equationSystems_) {
   *    eqsys->pre_iter_work();
   *    eqsys->solve_and_update();
   *    eqsys->post_iter_work();                //<<<< Post-iteration actions
   *  }
   *  post_iter_work();
   *  ```
   *
   *  \sa EquationSystems::solve_and_update
   */
  virtual void post_iter_work();

  /** Deprecated post iteration work logic
   *
   *  \deprecated This method is there to support some tasks in
   *  EnthalpyEquationSystem that should be eventually moved to
   *  EquationSystems::post_iter_work.
   */
  virtual void post_iter_work_dep() {}
  virtual void assemble_and_solve(
    stk::mesh::FieldBase *deltaSolution);
  virtual void predict_state() {}
  virtual void register_interior_algorithm(
    stk::mesh::Part * /* part */) {}
  virtual void provide_output() {}
  virtual void pre_timestep_work();
  virtual void reinitialize_linear_system() {}
  virtual void post_adapt_work() {}
  virtual void dump_eq_time();
  virtual double provide_scaled_norm();
  virtual double provide_norm();
  virtual double provide_norm_increment();
  virtual bool system_is_converged();
  
  virtual void register_wall_bc(
    stk::mesh::Part * /* part */,
    const stk::topology & /* theTopo */,
    const WallBoundaryConditionData & /* wallBCData */) {}

  virtual void register_inflow_bc(
    stk::mesh::Part * /* part */,
    const stk::topology & /* theTopo */,
    const InflowBoundaryConditionData & /* inflowBCData */) {}

  virtual void register_open_bc(
    stk::mesh::Part * /* part */,
    const stk::topology & /* theTopo */,
    const OpenBoundaryConditionData & /* openBCData */) {}

  virtual void register_symmetry_bc(
    stk::mesh::Part * /* part */,
    const stk::topology & /* theTopo */,
    const SymmetryBoundaryConditionData & /* symmetryBCData */) {}

  virtual void register_abltop_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const ABLTopBoundaryConditionData &abltopBCData) {
      SymmetryBoundaryConditionData simData(abltopBCData.boundaryConditions_);
      register_symmetry_bc( part, theTopo, simData );
    }

  virtual void register_periodic_bc(
    stk::mesh::Part * /* partMaster */,
    stk::mesh::Part * /* partSlave */,
    const stk::topology & /* theTopoMaster */,
    const stk::topology & /* theTopoSlave */,
    const PeriodicBoundaryConditionData & /* periodicBCData */) {}
  
  virtual void register_non_conformal_bc(
    stk::mesh::Part * /* part */,
    const stk::topology & /* theTopo */) {}

  virtual void register_overset_bc() {}

  virtual void create_constraint_algorithm(
    stk::mesh::FieldBase *theField);

  virtual void register_surface_pp_algorithm(
    const PostProcessingData & /* theData */,
    stk::mesh::PartVector & /* partVector */) {}

  virtual void register_initial_condition_fcn(
    stk::mesh::Part * /* part */,
    const std::map<std::string, std::string> & /* theNames */,
    const std::map<std::string, std::vector<double> > & /* theParams */) {}

  // rip through the propertyAlg_
  virtual void evaluate_properties();

  // provide helper function for Peclet function
  template<typename T>
  PecletFunction<T>* create_peclet_function( const std::string dofName);

  virtual void load(const YAML::Node & node)
  {
    get_required(node, "name", userSuppliedName_);
    get_required(node, "max_iterations", maxIterations_);
    get_required(node, "convergence_tolerance", convergenceTolerance_);
  }

  Simulation *root();
  EquationSystems *parent();

  void report_invalid_supp_alg_names();
  void report_built_supp_alg_names();
  bool supp_alg_is_requested(std::string name);
  bool supp_alg_is_requested(std::vector<std::string>);

  bool nodal_src_is_requested();

  EquationSystems &equationSystems_;
  Realm &realm_;
  std::string name_;
  std::string userSuppliedName_;
  const std::string eqnTypeName_;
  int maxIterations_;
  double convergenceTolerance_;

  // driver that holds all solver algorithms
  SolverAlgorithmDriver *solverAlgDriver_;

  double timerAssemble_;
  double timerLoadComplete_;
  double timerSolve_;
  double timerMisc_;
  double timerInit_;
  double timerPrecond_;
  double avgLinearIterations_;
  double maxLinearIterations_;
  double minLinearIterations_;
  int nonLinearIterationCount_;
  bool reportLinearIterations_;
  bool firstTimeStepSolve_;
  bool edgeNodalGradient_;

  void update_iteration_statistics(
    const int & iters);
  
  bool bc_data_specified(
    const UserData&, std::string &name);
  
  UserDataType
  get_bc_data_type(
    const UserData&, std::string &name);
  
  std::string
  get_bc_function_name(
    const UserData&, std::string &name);
  
  std::vector<double>
  get_bc_function_params(
    const UserData&, std::string &name);

  std::vector<std::string>
  get_bc_function_string_params(
    const UserData&, std::string &name);

  virtual void post_converged_work() {}

  virtual void save_diagonal_term(
    const std::vector<stk::mesh::Entity>&,
    const std::vector<int>&,
    const std::vector<double>&
  ) {}

  virtual void save_diagonal_term(
    unsigned,
    const stk::mesh::Entity*,
    const SharedMemView<const double**>&
  ) {}

  std::vector<Algorithm *> bcDataAlg_;
  std::vector<Algorithm *> bcDataMapAlg_;
  std::vector<Algorithm *> copyStateAlg_;
  
  LinearSystem *linsys_;

  size_t num_graph_entries_;

  // vector of property algorithms
  std::vector<Algorithm *> propertyAlg_;

  /// List of tasks to be performed before each EquationSystem::solve_and_update
  std::vector<AlgorithmDriver *> preIterAlgDriver_;

  /// List of tasks to be performed after each EquationSystem::solve_and_update
  std::vector<AlgorithmDriver*> postIterAlgDriver_;

  std::string dofName_{"undefined"};

  bool extractDiagonal_{false};

  // owner equation system
  /*EquationSystem *ownerEqs_;*/

};

//--------------------------------------------------------------------------
//-------- create_peclet_function ------------------------------------------
//--------------------------------------------------------------------------
template<typename T>
PecletFunction<T>*
EquationSystem::create_peclet_function(
  const std::string dofName)
{
  PecletFunction<T> *pecletFunction = NULL;
  if ( "classic" == realm_.get_tanh_functional_form(dofName) ) {
    const T hybridFactor = realm_.get_hybrid_factor(dofName);
    const T A = 5.0;
    pecletFunction = new ClassicPecletFunction<T>(A, hybridFactor);
  }
  else {
    const T c1 = realm_.get_tanh_trans(dofName);
    const T c2 = realm_.get_tanh_width(dofName);
    pecletFunction = new TanhFunction<T>(c1, c2);
  }
  return pecletFunction;
}

} // namespace nalu
} // namespace Sierra

#endif
