// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef REALM_H
#define REALM_H

#include <Enums.h>
#include <FieldTypeDef.h>

#include <BoundaryConditions.h>
#include <InitialConditions.h>
#include <MaterialPropertys.h>
#include <EquationSystems.h>

#if defined(NALU_USES_PERCEPT)
#include <Teuchos_RCP.hpp>
#endif

#include <ngp_utils/NgpFieldManager.h>
#include "ngp_utils/NgpMeshInfo.h"

#include "stk_mesh/base/NgpMesh.hpp"

// standard c++
#include <map>
#include <string>
#include <vector>
#include <memory>

namespace stk {
namespace mesh {
class Part;
}
namespace io {
class StkMeshIoBroker;
}

namespace util {
class ParameterList;
}
} // namespace stk

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

class Algorithm;
class AlgorithmDriver;
class AuxFunctionAlgorithm;
class GeometryAlgDriver;

class NonConformalManager;
class ErrorIndicatorAlgorithmDriver;
class EquationSystems;
class OutputInfo;
class OversetManager;
class PostProcessingInfo;
class PeriodicManager;
class Realms;
class Simulation;
class SolutionOptions;
class TimeIntegrator;
class MasterElement;
class PropertyEvaluator;
class Transfer;
class MeshMotionAlg;
class MeshTransformationAlg;

class SolutionNormPostProcessing;
class SideWriterContainer;
class TurbulenceAveragingPostProcessing;
class DataProbePostProcessing;
class LidarLineOfSite;
class AeroContainer;
class ABLForcingAlgorithm;
class BdyLayerStatistics;

class TensorProductQuadratureRule;
class LagrangeBasis;
class PromotedElementIO;

/** Representation of a computational domain and physics equations solved on
 * this domain.
 *
 */
class Realm
{
public:
  using NgpMeshInfo =
    nalu_ngp::MeshInfo<stk::mesh::NgpMesh, nalu_ngp::FieldManager>;

  Realm(Realms&, const YAML::Node& node);
  virtual ~Realm();

  typedef size_t SizeType;

  virtual void load(const YAML::Node& node);
  void look_ahead_and_creation(const YAML::Node& node);
  void look_ahead_create_lidar(const YAML::Node& node);

  virtual void breadboard();

  virtual void initialize_prolog();
  virtual void initialize_epilog();

  Simulation* root() const;
  Simulation* root();
  Realms* parent() const;
  Realms* parent();

  bool debug() const;
  bool get_activate_memory_diagnostic();
  void provide_memory_summary();
  std::string convert_bytes(double bytes);

  void create_mesh();

  void setup_nodal_fields();
  void setup_edge_fields();
  void setup_element_fields();

  void setup_interior_algorithms();
  void setup_post_processing_algorithms();
  void setup_bc();
  void enforce_bc_on_exposed_faces();
  void setup_initial_conditions();
  void setup_property();
  void extract_universal_constant(
    const std::string name, double& value, const bool useDefault);
  void
  augment_property_map(PropertyIdentifier propID, ScalarFieldType* theField);

  void makeSureNodesHaveValidTopology();

  void initialize_global_variables();

  void rebalance_mesh();

  void balance_nodes();

  void create_output_mesh();
  void create_restart_mesh();
  void input_variables_from_mesh();

  void augment_output_variable_list(const std::string fieldName);

  void augment_restart_variable_list(std::string restartFieldName);

  void create_edges();
  void provide_entity_count();
  void delete_edges();
  void commit();

  void init_current_coordinates();

  std::string get_coordinates_name();
  bool has_mesh_motion() const;
  bool has_mesh_deformation() const;
  bool does_mesh_move() const;
  bool has_non_matching_boundary_face_alg() const;

  // overset boundary condition requires elemental field registration
  bool query_for_overset();

  void set_current_coordinates(stk::mesh::Part* targetPart);

  // non-conformal-like algorithm suppoer
  void initialize_non_conformal();
  void initialize_post_processing_algorithms();

  void compute_geometry();
  void compute_vrtm(const std::string& = "velocity");
  void compute_l2_scaling();
  void output_converged_results();
  void provide_output();
  void provide_restart_output();

  void register_interior_algorithm(stk::mesh::Part* part);

  void register_nodal_fields(stk::mesh::Part* part);

  void register_wall_bc(stk::mesh::Part* part, const stk::topology& theTopo);

  void register_inflow_bc(stk::mesh::Part* part, const stk::topology& theTopo);

  void register_open_bc(stk::mesh::Part* part, const stk::topology& theTopo);

  void
  register_symmetry_bc(stk::mesh::Part* part, const stk::topology& theTopo);

  void register_abltop_bc(stk::mesh::Part* part, const stk::topology& theTopo)
  {
    register_symmetry_bc(part, theTopo);
  }

  void register_periodic_bc(
    stk::mesh::Part* masterMeshPart,
    stk::mesh::Part* slaveMeshPart,
    const double& searchTolerance,
    const std::string& searchMethodName);

  void setup_non_conformal_bc(
    stk::mesh::PartVector currentPartVec,
    stk::mesh::PartVector opposingPartVec,
    const NonConformalBoundaryConditionData& nonConformalBCData);

  void register_non_conformal_bc(
    stk::mesh::Part* part, const stk::topology& theTopo);

  void register_overset_bc();

  void setup_overset_bc(const OversetBoundaryConditionData& oversetBCData);

  void periodic_field_update(
    stk::mesh::FieldBase* theField,
    const unsigned& sizeOfTheField,
    const bool& bypassFieldCheck = true) const;

  void periodic_field_max(
    stk::mesh::FieldBase* theField, const unsigned& sizeOfTheField) const;

  void periodic_delta_solution_update(
    stk::mesh::FieldBase* theField,
    const unsigned& sizeOfField,
    const bool& doCommunication = true) const;

  void periodic_max_field_update(
    stk::mesh::FieldBase* theField, const unsigned& sizeOfField) const;

  const stk::mesh::PartVector& get_slave_part_vector();

  void overset_field_update(
    stk::mesh::FieldBase* field,
    const unsigned nRows,
    const unsigned nCols,
    const bool doFinalSyncToDevice = true);

  virtual void populate_initial_condition();
  virtual void populate_boundary_data();
  virtual void boundary_data_to_state_data();
  virtual double populate_variables_from_input(const double currentTime);
  virtual void
  populate_external_variables_from_input(const double /* currentTime */)
  {
  }
  virtual double populate_restart(double& timeStepNm1, int& timeStepCount);
  virtual void populate_derived_quantities();
  virtual void evaluate_properties();
  virtual double compute_adaptive_time_step();
  virtual void swap_states();
  virtual void predict_state();
  virtual void update_geometry_due_to_mesh_motion();
  virtual void update_graph_connectivity_and_coordinates_due_to_mesh_motion();
  virtual void output_banner();
  virtual void advance_time_step();
  virtual void nonlinear_iterations(const int);
  virtual void initial_work();

  void set_global_id();

  /** Initialize the HYPRE global row IDs
   *
   *  \sa Realm::hypreGlobalId_
   */
  void set_hypre_global_id();

  /// check job for fitting in memory
  void check_job(bool get_node_count);

  void dump_simulation_time();
  double provide_mean_norm();

  double get_hybrid_factor(const std::string dofname);
  double get_alpha_factor(const std::string dofname);
  double get_alpha_upw_factor(const std::string dofname);
  double get_upw_factor(const std::string dofname);
  bool primitive_uses_limiter(const std::string dofname);
  double get_lam_schmidt(const std::string dofname);
  double get_lam_prandtl(const std::string dofname, bool& prProvided);
  double get_turb_schmidt(const std::string dofname);
  double get_turb_prandtl(const std::string dofname);
  bool get_noc_usage(const std::string dofname);
  bool get_shifted_grad_op(const std::string dofname);
  bool get_skew_symmetric(const std::string dofname);
  double get_divU();

  // tanh factor specifics
  std::string get_tanh_functional_form(const std::string dofname);
  double get_tanh_trans(const std::string dofname);
  double get_tanh_width(const std::string dofname);

  // consistent mass matrix for projected nodal gradient
  bool get_consistent_mass_matrix_png(const std::string dofname);

  // pressure poisson nuance
  double get_mdot_interp();
  bool get_cvfem_shifted_mdot();
  bool get_cvfem_reduced_sens_poisson();

  bool has_nc_gauss_labatto_quadrature();
  bool get_nc_alg_upwind_advection();
  bool get_nc_alg_include_pstab();
  bool get_nc_alg_current_normal();

  PropertyEvaluator* get_material_prop_eval(const PropertyIdentifier thePropID);

  bool is_turbulent();
  void is_turbulent(bool isIt);

  bool needs_enthalpy();
  void needs_enthalpy(bool needsEnthalpy);

  int number_of_states();

  std::string name();

  // redirection of stk::mesh::get_buckets to allow global selector
  //  to be applied, e.g., in adaptivity we need to avoid the parent
  //  elements
  stk::mesh::BucketVector const& get_buckets(
    stk::mesh::EntityRank rank, const stk::mesh::Selector& selector) const;

  // get aura, bulk and meta data
  bool get_activate_aura();
  stk::mesh::BulkData& bulk_data();
  const stk::mesh::BulkData& bulk_data() const;
  stk::mesh::MetaData& meta_data();
  const stk::mesh::MetaData& meta_data() const;

  inline NgpMeshInfo& mesh_info()
  {
    if ((meshModCount_ != bulkData_->synchronized_count()) || (!meshInfo_)) {
      meshModCount_ = bulkData_->synchronized_count();
      meshInfo_.reset(new NgpMeshInfo(*bulkData_));
    }
    return *meshInfo_;
  }

  inline const stk::mesh::NgpMesh& ngp_mesh() { return mesh_info().ngp_mesh(); }

  inline const nalu_ngp::FieldManager& ngp_field_manager()
  {
    return mesh_info().ngp_field_manager();
  }

  // inactive part
  stk::mesh::Selector get_inactive_selector();

  // push back equation to equation systems vector
  void push_equation_to_systems(EquationSystem* eqSystem);

  // provide all of the physics target names
  const std::vector<std::string>& get_physics_target_names();
  double get_tanh_blending(const std::string dofName);

  void output_lidar();

  Realms& realms_;

  std::string name_;
  std::string type_;
  std::string inputDBName_;
  unsigned spatialDimension_;

  bool realmUsesEdges_;
  int solveFrequency_;
  bool isTurbulent_;
  bool needsEnthalpy_;

  double l2Scaling_;

  // ioBroker, meta and bulk data
  std::shared_ptr<stk::mesh::BulkData> bulkData_;
  stk::io::StkMeshIoBroker* ioBroker_;
  std::unique_ptr<SideWriterContainer> sideWriters_;

  size_t resultsFileIndex_;
  size_t restartFileIndex_;

  // nalu field data
  GlobalIdFieldType* naluGlobalId_;

  // algorithm drivers managed by region
  std::unique_ptr<GeometryAlgDriver> geometryAlgDriver_;
  unsigned numInitialElements_;

  TimeIntegrator* timeIntegrator_;

  BoundaryConditionVector boundaryConditions_;
  InitialConditionVector initialConditions_;
  MaterialPropertys materialPropertys_;

  EquationSystems equationSystems_;

  double maxCourant_;
  double maxReynolds_;
  double targetCourant_;
  double timeStepChangeFactor_;
  int currentNonlinearIteration_;

  SolutionOptions* solutionOptions_;
  OutputInfo* outputInfo_;
  PostProcessingInfo* postProcessingInfo_;
  SolutionNormPostProcessing* solutionNormPostProcessing_;
  TurbulenceAveragingPostProcessing* turbulenceAveragingPostProcessing_;
  DataProbePostProcessing* dataProbePostProcessing_;
  std::unique_ptr<AeroContainer> aeroModels_;
  ABLForcingAlgorithm* ablForcingAlg_;
  BdyLayerStatistics* bdyLayerStats_{nullptr};
  std::unique_ptr<MeshMotionAlg> meshMotionAlg_;
  std::unique_ptr<MeshTransformationAlg> meshTransformationAlg_;
  std::vector<LidarLineOfSite> lidarLOS_;

  std::vector<Algorithm*> propertyAlg_;
  std::map<PropertyIdentifier, ScalarFieldType*> propertyMap_;
  std::vector<Algorithm*> initCondAlg_;

  SizeType nodeCount_;
  bool estimateMemoryOnly_;
  double availableMemoryPerCoreGB_;
  double timerActuator_{0};
  double timerCreateMesh_;
  double timerPopulateMesh_;
  double timerPopulateFieldData_;
  double timerOutputFields_;
  double timerCreateEdges_;
  double timerNonconformal_;
  double timerInitializeEqs_;
  double timerPropertyEval_;
  double timerTransferSearch_;
  double timerTransferExecute_;
  double timerSkinMesh_;
  double timerPromoteMesh_;
  double timerSortExposedFace_;

  NonConformalManager* nonConformalManager_;
  OversetManager* oversetManager_;
  bool hasNonConformal_;
  bool hasOverset_;
  bool isExternalOverset_{false};

  // three type of transfer operations
  bool hasMultiPhysicsTransfer_;
  bool hasInitializationTransfer_;
  bool hasIoTransfer_;
  bool hasExternalDataTransfer_;

  PeriodicManager* periodicManager_;
  bool hasPeriodic_;
  bool hasFluids_;

  // global parameter list
  std::unique_ptr<stk::util::ParameterList> globalParameters_;

  // part for all exposed surfaces in the mesh
  stk::mesh::Part* exposedBoundaryPart_;

  // part for new edges
  stk::mesh::Part* edgesPart_;

  // cheack that all exposed surfaces have a bc applied
  bool checkForMissingBcs_;

  // check if there are negative Jacobians
  bool checkJacobians_;

  // types of physics
  bool isothermalFlow_;
  bool uniformFlow_;

  // some post processing of entity counts
  bool provideEntityCount_;

  // automatic mesh decomposition; None, rib, rcb, multikl, etc.
  std::string autoDecompType_;

  // STK rebalance options
  bool rebalanceMesh_{false};

  std::string rebalanceMethod_;

  // allow aura to be optional
  bool activateAura_;

  // allow detailed output (memory) to be provided
  bool activateMemoryDiagnostic_;

  // sometimes restarts can be missing states or dofs
  bool supportInconsistentRestart_;

  bool doBalanceNodes_;
  struct BalanceNodeOptions
  {
    BalanceNodeOptions() : target(1.0), numIters(5){};

    double target;
    int numIters;
  };
  BalanceNodeOptions balanceNodeOptions_;

  // beginning wall time
  double wallTimeStart_;

  // mesh parts for all interior domains
  stk::mesh::PartVector interiorPartVec_;

  /** Vector holding side sets that have been registered with the boundary
   * conditions in the input file.
   *
   * The member is intended to for use in Realm::enforce_bc_on_exposed_faces to
   * check for "exposed surfaces" that might have not been assigned BCs in the
   * input file.
   *
   */
  stk::mesh::PartVector bcPartVec_;
  stk::mesh::PartVector oversetBCPartVec_;

  // empty part vector should it be required
  stk::mesh::PartVector emptyPartVector_;

  // base and promote mesh parts
  stk::mesh::PartVector basePartVector_;
  stk::mesh::PartVector superPartVector_;

  std::vector<Algorithm*> bcDataAlg_;

  // transfer information; three types
  std::vector<Transfer*> multiPhysicsTransferVec_;
  std::vector<Transfer*> initializationTransferVec_;
  std::vector<Transfer*> ioTransferVec_;
  std::vector<Transfer*> externalDataTransferVec_;
  void augment_transfer_vector(
    Transfer* transfer, const std::string transferObjective, Realm* toRealm);
  void process_multi_physics_transfer();
  void process_initialization_transfer();
  void process_io_transfer();
  void process_external_data_transfer();

  // process end of time step converged work
  void post_converged_work();

  // time information; calls through timeIntegrator
  double get_current_time();
  double get_time_step();
  double get_gamma1();
  double get_gamma2();
  double get_gamma3();
  int get_time_step_count() const;
  double get_time_step_from_file();
  bool get_is_fixed_time_step();
  bool get_is_terminate_based_on_time();
  double get_total_sim_time();
  int get_max_time_step_count();

  // restart
  bool restarted_simulation();
  bool support_inconsistent_restart();

  double get_stefan_boltzmann();
  double get_turb_model_constant(const TurbulenceModelConstant turbModelEnum);

  TurbulenceModel get_turbulence_model() const;

  // element promotion options
  bool doPromotion_; // conto
  unsigned promotionOrder_;

  // id for the input mesh
  size_t inputMeshIdx_;

  // save off the node
  const YAML::Node& node_;

  // tools
  std::unique_ptr<PromotedElementIO> promotionIO_; // mesh outputer
  std::vector<std::string> superTargetNames_;

  void setup_element_promotion(); // create super parts
  void promote_mesh();            // create new super element / sides on parts
  void create_promoted_output_mesh(); // method to create output of linear
                                      // subelements
  bool using_tensor_product_kernels() const;
  bool high_order_active() const { return doPromotion_; };

  std::string physics_part_name(std::string) const;
  std::vector<std::string> physics_part_names(std::vector<std::string>) const;

  // high order
  int polynomial_order() const;
  bool matrix_free() const;
  bool matrixFree_{false};

  Teuchos::ParameterList solver_parameters(std::string) const;

  stk::mesh::PartVector allPeriodicInteractingParts_;
  stk::mesh::PartVector allNonConformalInteractingParts_;

  bool isFinalOuterIter_{false};

  std::vector<stk::mesh::EntityId> hypreOffsets_;

  /** The starting index (global) of the HYPRE linear system in this MPI rank
   *
   *  Note that this is actually the offset into the linear system. This index
   *  must be adjusted accordingly to account for multiple degrees of freedom on
   *  a particular node. This is performed in sierra::nalu::HypreLinearSystem.
   */
  stk::mesh::EntityId hypreILower_;

  /** The ending index (global) of the HYPRE linear system in this MPI rank
   *
   *  Note that this is actually the offset into the linear system. This index
   *  must be adjusted accordingly to account for multiple degrees of freedom on
   *  a particular node. This is performed in sierra::nalu::HypreLinearSystem.
   */
  stk::mesh::EntityId hypreIUpper_;

  /** The total number of HYPRE nodes in the linear system
   *
   *  Note that this is not an MPI rank local quantity
   */
  stk::mesh::EntityId hypreNumNodes_;

  /** Global Row IDs for the HYPRE linear system
   *
   *  The HYPRE IDs are different from STK IDs and Realm::naluGlobalId_ because
   *  HYPRE expects contiguous IDs for matrix rows and further requires that the
   *  IDs be ordered across MPI ranks; i.e., startIdx (MPI_rank + 1) =
   *  endIdx(MPI_rank) + 1.
   */
  HypreIDFieldType* hypreGlobalId_{nullptr};
  TpetIDFieldType* tpetGlobalId_{nullptr};

  /** Flag indicating whether Hypre solver is being used for any of the equation
   * systems.
   */
  bool hypreIsActive_{false};

  std::vector<std::string>
  handle_all_element_part_alias(const std::vector<std::string>& names) const;

protected:
  std::unique_ptr<NgpMeshInfo> meshInfo_;

  unsigned meshModCount_{0};
  const std::string allElementPartAlias{"all_blocks"};
};

} // namespace nalu
} // namespace sierra

#endif /* REALM_H */
