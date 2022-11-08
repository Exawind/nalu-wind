// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Realm.h>
#include <Simulation.h>
#include <NaluEnv.h>
#include <stk_mesh/base/GetNgpField.hpp>

#include <AuxFunction.h>
#include <AuxFunctionAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <Enums.h>
#include <EntityExposedFaceSorter.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <LinearSolvers.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <MaterialPropertys.h>
#include <NaluParsing.h>
#include <NonConformalManager.h>
#include <NonConformalInfo.h>
#include <OutputInfo.h>
#include <PostProcessingInfo.h>
#include <PostProcessingData.h>
#include <PecletFunction.h>
#include <PeriodicManager.h>
#include <Realms.h>
#include <SolutionOptions.h>
#include <SideWriter.h>
#include <TimeIntegrator.h>

#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedElementIO.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/HexNElementDescription.h>
#include <master_element/QuadratureRule.h>

// mesh motion
#include <mesh_motion/MeshMotionAlg.h>
#include <mesh_motion/MeshTransformationAlg.h>

// overset
#include <overset/OversetManager.h>

#ifdef NALU_USES_TIOGA
#include <overset/OversetManagerTIOGA.h>
#endif

// post processing
#include <SolutionNormPostProcessing.h>
#include <TurbulenceAveragingPostProcessing.h>
#include <DataProbePostProcessing.h>
#include <wind_energy/BdyLayerStatistics.h>

// actuator line/fsi
#include <aero/AeroContainer.h>

#include <wind_energy/ABLForcingAlgorithm.h>
#include <wind_energy/SyntheticLidar.h>

// props; algs, evaluators and data
#include <property_evaluator/GenericPropAlgorithm.h>
#include <property_evaluator/InverseDualVolumePropAlgorithm.h>
#include <property_evaluator/InversePropAlgorithm.h>
#include <property_evaluator/TemperaturePropAlgorithm.h>
#include <property_evaluator/LinearPropAlgorithm.h>
#include <property_evaluator/ConstantPropertyEvaluator.h>
#include <property_evaluator/EnthalpyPropertyEvaluator.h>
#include <property_evaluator/IdealGasPropertyEvaluator.h>
#include <property_evaluator/PropertyEvaluator.h>
#include <property_evaluator/ReferencePropertyData.h>
#include <property_evaluator/SpecificHeatPropertyEvaluator.h>
#include <property_evaluator/SutherlandsPropertyEvaluator.h>
#include <property_evaluator/WaterPropertyEvaluator.h>
#include <property_evaluator/MaterialPropertyData.h>

// transfer
#include <xfer/Transfer.h>

#include "utils/StkHelpers.h"
#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldBLAS.h"
#include "ngp_utils/NgpFieldManager.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "ngp_algorithms/GeometryAlgDriver.h"
#include "ngp_algorithms/GeometryInteriorAlg.h"
#include "ngp_algorithms/GeometryBoundaryAlg.h"

#include "gcl/MeshVelocityAlg.h"
#include "gcl/MeshVelocityEdgeAlg.h"
#include "AlgTraits.h"

// stk_util
#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/environment/WallTime.hpp>
#include <stk_util/environment/perf_util.hpp>
#include <stk_util/environment/FileUtils.hpp>
#include <stk_util/util/ParameterList.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Comm.hpp>
#include <stk_mesh/base/CreateEdges.hpp>
#include <stk_mesh/base/SkinBoundary.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/GetNgpField.hpp>

// stk_io
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_io/IossBridge.hpp>
#include <stk_io/InputFile.hpp>
#include <Ioss_SubSystem.h>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_balance
#include <Zoltan2_config.h>
#include <stk_balance/balance.hpp>
#include <stk_balance/balanceUtils.hpp>

// Ioss for propertManager (io)
#include <Ioss_PropertyManager.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>
#include <NaluParsingHelper.h>

// basic c++
#include <map>
#include <cmath>
#include <limits>
#include <utility>
#include <stdint.h>

#ifdef NALU_USES_CATALYST
// catalyst visualization output
#include <Iovs_exodus_DatabaseIO.h>
#endif

#define USE_NALU_PERFORMANCE_TESTING_CALLGRIND 0
#if USE_NALU_PERFORMANCE_TESTING_CALLGRIND
#include "/usr/netpub/valgrind-3.8.1/include/valgrind/callgrind.h"
#endif

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// Realm - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Realm::Realm(Realms& realms, const YAML::Node& node)
  : realms_(realms),
    name_("na"),
    type_("multi_physics"),
    inputDBName_("input_unknown"),
    spatialDimension_(3u), // for convenience; can always get it from meta data
    realmUsesEdges_(true),
    solveFrequency_(1),
    isTurbulent_(false),
    needsEnthalpy_(false),
    l2Scaling_(1.0),
    ioBroker_(NULL),
    sideWriters_(new SideWriterContainer()),
    resultsFileIndex_(99),
    restartFileIndex_(99),
    numInitialElements_(0),
    timeIntegrator_(0),
    materialPropertys_(*this),
    equationSystems_(*this),
    maxCourant_(0.0),
    maxReynolds_(0.0),
    targetCourant_(1.0),
    timeStepChangeFactor_(1.25),
    currentNonlinearIteration_(1),
    solutionOptions_(new SolutionOptions()),
    outputInfo_(new OutputInfo()),
    postProcessingInfo_(new PostProcessingInfo()),
    solutionNormPostProcessing_(NULL),
    turbulenceAveragingPostProcessing_(NULL),
    dataProbePostProcessing_(NULL),
    ablForcingAlg_(NULL),
    nodeCount_(0),
    estimateMemoryOnly_(false),
    availableMemoryPerCoreGB_(0),
    timerCreateMesh_(0.0),
    timerPopulateMesh_(0.0),
    timerPopulateFieldData_(0.0),
    timerOutputFields_(0.0),
    timerCreateEdges_(0.0),
    timerNonconformal_(0.0),
    timerInitializeEqs_(0.0),
    timerPropertyEval_(0.0),
    timerTransferSearch_(0.0),
    timerTransferExecute_(0.0),
    timerSkinMesh_(0.0),
    timerPromoteMesh_(0.0),
    timerSortExposedFace_(0.0),
    nonConformalManager_(NULL),
    oversetManager_(NULL),
    hasNonConformal_(false),
    hasOverset_(false),
    hasMultiPhysicsTransfer_(false),
    hasInitializationTransfer_(false),
    hasIoTransfer_(false),
    hasExternalDataTransfer_(false),
    periodicManager_(NULL),
    hasPeriodic_(false),
    hasFluids_(false),
    globalParameters_(new stk::util::ParameterList()),
    exposedBoundaryPart_(0),
    edgesPart_(0),
    checkForMissingBcs_(false),
    checkJacobians_(false),
    isothermalFlow_(true),
    uniformFlow_(true),
    provideEntityCount_(false),
    autoDecompType_("None"),
    activateAura_(false),
    activateMemoryDiagnostic_(false),
    supportInconsistentRestart_(false),
    doBalanceNodes_(false),
    balanceNodeOptions_(),
    wallTimeStart_(stk::wall_time()),
    doPromotion_(false),
    promotionOrder_(1u),
    inputMeshIdx_(std::numeric_limits<size_t>::max()),
    node_(node)
{
  // deal with specialty options that live off of the realm;
  // choose to do this now rather than waiting for the load stage
  look_ahead_and_creation(node);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Realm::~Realm()
{
  meshInfo_.reset();
  delete ioBroker_;

  // prop algs
  std::vector<Algorithm*>::iterator ii;
  for (ii = initCondAlg_.begin(); ii != initCondAlg_.end(); ++ii)
    delete *ii;
  for (ii = propertyAlg_.begin(); ii != propertyAlg_.end(); ++ii) {
    delete *ii;
  }

  // any bc data
  for (auto ialg : bcDataAlg_)
    delete ialg;

  delete solutionOptions_;
  delete outputInfo_;
  delete postProcessingInfo_;

  // post processing-like objects
  if (NULL != solutionNormPostProcessing_)
    delete solutionNormPostProcessing_;

  if (NULL != turbulenceAveragingPostProcessing_)
    delete turbulenceAveragingPostProcessing_;

  if (NULL != dataProbePostProcessing_)
    delete dataProbePostProcessing_;

  // delete non-conformal related things
  if (NULL != nonConformalManager_)
    delete nonConformalManager_;

  // delete periodic related things
  if (NULL != periodicManager_)
    delete periodicManager_;

  // Delete abl forcing pointer
  if (NULL != ablForcingAlg_)
    delete ablForcingAlg_;

  if (nullptr != bdyLayerStats_)
    delete bdyLayerStats_;

  if (nullptr != oversetManager_)
    delete oversetManager_;

  MasterElementRepo::clear();
}

void
Realm::breadboard()
{
  geometryAlgDriver_.reset(new GeometryAlgDriver(*this));
  equationSystems_.breadboard();
}

bool
Realm::debug() const
{
  return root()->debug_;
}

//--------------------------------------------------------------------------
//-------- get_activate_memory_diagnostic ----------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_activate_memory_diagnostic()
{
  return activateMemoryDiagnostic_;
}

//--------------------------------------------------------------------------
//-------- provide_memory_summary ------------------------------------------
//--------------------------------------------------------------------------
void
Realm::provide_memory_summary()
{
  size_t now, hwm;
  stk::get_memory_usage(now, hwm);
  // min, max, sum
  size_t global_now[3] = {now, now, now};
  size_t global_hwm[3] = {hwm, hwm, hwm};

  stk::all_reduce(
    NaluEnv::self().parallel_comm(), stk::ReduceSum<1>(&global_now[2]));
  stk::all_reduce(
    NaluEnv::self().parallel_comm(), stk::ReduceMin<1>(&global_now[0]));
  stk::all_reduce(
    NaluEnv::self().parallel_comm(), stk::ReduceMax<1>(&global_now[1]));

  stk::all_reduce(
    NaluEnv::self().parallel_comm(), stk::ReduceSum<1>(&global_hwm[2]));
  stk::all_reduce(
    NaluEnv::self().parallel_comm(), stk::ReduceMin<1>(&global_hwm[0]));
  stk::all_reduce(
    NaluEnv::self().parallel_comm(), stk::ReduceMax<1>(&global_hwm[1]));

  NaluEnv::self().naluOutputP0() << "Memory Overview: " << std::endl;
  NaluEnv::self().naluOutputP0()
    << "nalu memory: total (over all cores) current/high-water mark= "
    << std::setw(15) << convert_bytes(global_now[2]) << std::setw(15)
    << convert_bytes(global_hwm[2]) << std::endl;

  NaluEnv::self().naluOutputP0()
    << "nalu memory:   min (over all cores) current/high-water mark= "
    << std::setw(15) << convert_bytes(global_now[0]) << std::setw(15)
    << convert_bytes(global_hwm[0]) << std::endl;

  NaluEnv::self().naluOutputP0()
    << "nalu memory:   max (over all cores) current/high-water mark= "
    << std::setw(15) << convert_bytes(global_now[1]) << std::setw(15)
    << convert_bytes(global_hwm[1]) << std::endl;
}

//--------------------------------------------------------------------------
//-------- convert_bytes ---------------------------------------------------
//--------------------------------------------------------------------------
std::string
Realm::convert_bytes(double bytes)
{
  const double K = 1024;
  const double M = K * 1024;
  const double G = M * 1024;

  std::ostringstream out;
  if (bytes < K) {
    out << bytes << " B";
  } else if (bytes < M) {
    bytes /= K;
    out << bytes << " K";
  } else if (bytes < G) {
    bytes /= M;
    out << bytes << " M";
  } else {
    bytes /= G;
    out << bytes << " G";
  }
  return out.str();
}

void
Realm::initialize_prolog()
{
  NaluEnv::self().naluOutputP0() << "Realm::initialize() Begin " << std::endl;

  if (doPromotion_) {
    setup_element_promotion();
  }
  // field registration
  setup_nodal_fields();
  setup_edge_fields();
  setup_element_fields();

  // property maps and evaluation algorithms
  setup_property();

  // interior algorithm creation
  setup_interior_algorithms();

  // create boundary conditions
  setup_bc();

  // post processing algorithm creation
  setup_post_processing_algorithms();

  // create initial conditions
  setup_initial_conditions();

  // set global variables that have not yet been set
  initialize_global_variables();

  // Populate_mesh fills in the entities (nodes/elements/etc) and
  // connectivities, but no field-data. Field-data is not allocated yet.
  NaluEnv::self().naluOutputP0()
    << "Realm::ioBroker_->populate_mesh() Begin" << std::endl;
  double time = -NaluEnv::self().nalu_time();
  ioBroker_->populate_mesh();
  time += NaluEnv::self().nalu_time();
  timerPopulateMesh_ += time;
  NaluEnv::self().naluOutputP0()
    << "Realm::ioBroker_->populate_mesh() End" << std::endl;

  // If we want to create all internal edges, we want to do it before
  // field-data is allocated because that allows better performance in
  // the create-edges code.
  if (realmUsesEdges_)
    create_edges();

  // create the nodes for possible data probe

  // output entity counts including max/min
  if (provideEntityCount_)
    provide_entity_count();

  // Now the mesh is fully populated, so we're ready to populate
  // field-data including coordinates, and attributes and/or distribution
  // factors if those exist on the input mesh file.
  NaluEnv::self().naluOutputP0()
    << "Realm::ioBroker_->populate_field_data() Begin" << std::endl;
  time = -NaluEnv::self().nalu_time();
  ioBroker_->populate_field_data();
  time += NaluEnv::self().nalu_time();
  timerPopulateFieldData_ += time;
  NaluEnv::self().naluOutputP0()
    << "Realm::ioBroker_->populate_field_data() End" << std::endl;

  // rebalance mesh using stk_balance
  if (rebalanceMesh_) {
    rebalance_mesh();
  }

  if (doBalanceNodes_) {
    balance_nodes();
  }

  if (doPromotion_) {
    promote_mesh();
    create_promoted_output_mesh();
  }

  // manage NaluGlobalId for linear system
  set_global_id();

  // check that all bcs are covering exposed surfaces
  if (checkForMissingBcs_)
    enforce_bc_on_exposed_faces();

  // output and restart files
  create_output_mesh();
  create_restart_mesh();

  // sort exposed faces only when using consolidated bc NGP approach
  if (solutionOptions_->useConsolidatedBcSolverAlg_) {
    const double timeSort = NaluEnv::self().nalu_time();
    bulkData_->sort_entities(EntityExposedFaceSorter());
    timerSortExposedFace_ += (NaluEnv::self().nalu_time() - timeSort);
  }

  // variables that may come from the initial mesh
  input_variables_from_mesh();

  populate_boundary_data();

  ScalarIntFieldType* iblank = meta_data().get_field<ScalarIntFieldType>(
    stk::topology::NODE_RANK, "iblank");
  stk::mesh::field_fill(1, *iblank);

  if (has_mesh_deformation() || solutionOptions_->meshMotion_)
    init_current_coordinates();

  if (hasPeriodic_)
    periodicManager_->build_constraints();

  if (solutionOptions_->meshTransformation_)
    meshTransformationAlg_->initialize(get_current_time());

  if (solutionOptions_->meshMotion_)
    meshMotionAlg_->initialize(get_current_time());

  compute_geometry();

  if (solutionOptions_->meshMotion_)
    meshMotionAlg_->post_compute_geometry();

  if (hasNonConformal_)
    initialize_non_conformal();
}

void
Realm::initialize_epilog()
{
  initialize_post_processing_algorithms();

  compute_l2_scaling();

  // Now that the inactive selectors have been processed; we are ready to setup
  // HYPRE IDs
  set_hypre_global_id();

  equationSystems_.initialize();

  // check job run size after mesh creation, linear system initialization
  check_job(false);

  NaluEnv::self().naluOutputP0() << "Realm::initialize() End " << std::endl;
}

//--------------------------------------------------------------------------
//-------- look_ahead_and_creation -----------------------------------------
//--------------------------------------------------------------------------
void
Realm::look_ahead_and_creation(const YAML::Node& node)
{
  // look for turbulence averaging
  std::vector<const YAML::Node*> foundTurbAveraging;
  NaluParsingHelper::find_nodes_given_key(
    "turbulence_averaging", node, foundTurbAveraging);
  if (foundTurbAveraging.size() > 0) {
    if (foundTurbAveraging.size() != 1)
      throw std::runtime_error(
        "look_ahead_and_create::error: Too many turbulence_averaging");
    turbulenceAveragingPostProcessing_ =
      new TurbulenceAveragingPostProcessing(*this, *foundTurbAveraging[0]);
  }

  // look for SolutionNormPostProcessing
  std::vector<const YAML::Node*> foundNormPP;
  NaluParsingHelper::find_nodes_given_key("solution_norm", node, foundNormPP);
  if (foundNormPP.size() > 0) {
    if (foundNormPP.size() != 1)
      throw std::runtime_error(
        "look_ahead_and_create::error: Too many Solution Norm blocks");
    solutionNormPostProcessing_ =
      new SolutionNormPostProcessing(*this, *foundNormPP[0]);
  }

  // look for DataProbe
  std::vector<const YAML::Node*> foundProbe;
  NaluParsingHelper::find_nodes_given_key("data_probes", node, foundProbe);
  if (foundProbe.size() > 0) {
    if (foundProbe.size() != 1) {
      throw std::runtime_error(
        "look_ahead_and_create::error: Too many data probe blocks");
    }
    auto& probe_block = *foundProbe.front();
    dataProbePostProcessing_ = new DataProbePostProcessing(*this, probe_block);
    look_ahead_create_lidar(probe_block["data_probes"]);
  }

  // Contains actuators and FSI data structures
  aeroModels_ = std::make_unique<AeroContainer>(node);

  // Boundary Layer Statistics post-processing
  if (node["boundary_layer_statistics"]) {
    const YAML::Node blStatNode = node["boundary_layer_statistics"];
    bdyLayerStats_ = new BdyLayerStatistics(*this, blStatNode);
  }

  // ABL Forcing parameters
  if (node["abl_forcing"]) {
    const YAML::Node ablNode = node["abl_forcing"];
    ablForcingAlg_ = new ABLForcingAlgorithm(*this, ablNode);
  }
}

void
Realm::look_ahead_create_lidar(const YAML::Node& node)
{
  if (!lidarLOS_) {
    lidarLOS_ = std::make_unique<LidarLOS>();
  }
  NaluEnv::self().naluOutputP0() << "LidarLineOfSite::load" << std::endl;
  lidarLOS_->load(node, dataProbePostProcessing_);
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::load(const YAML::Node& node)
{

  //======================================
  // realm commands first
  //======================================

  name_ = node["name"].as<std::string>();
  inputDBName_ = node["mesh"].as<std::string>();
  get_if_present(node, "type", type_, type_);

  // provide a high level banner
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0()
    << "Realm Options Review: " << name_ << std::endl;
  NaluEnv::self().naluOutputP0() << "===========================" << std::endl;

  get_if_present(node, "estimate_memory_only", estimateMemoryOnly_, false);
  get_if_present(
    node, "available_memory_per_core_GB", availableMemoryPerCoreGB_, 0.0);

  // exposed bc check
  get_if_present(
    node, "check_for_missing_bcs", checkForMissingBcs_, checkForMissingBcs_);

  // check for bad Jacobians in the mesh
  get_if_present(node, "check_jacobians", checkJacobians_, checkJacobians_);

  // entity count
  get_if_present(
    node, "provide_entity_count", provideEntityCount_, provideEntityCount_);

  // determine if edges are required and whether or not stk handles this
  get_if_present(node, "use_edges", realmUsesEdges_, realmUsesEdges_);

  get_if_present(node, "polynomial_order", promotionOrder_, promotionOrder_);
  if (promotionOrder_ > 2) {
    doPromotion_ = true;
  }

  if (promotionOrder_ >= 5) {
    throw std::runtime_error("Only polynomial orders 1-4 supported");
  }

  get_if_present(node, "matrix_free", matrixFree_, matrixFree_);
  if (polynomial_order() > 1 && !matrixFree_) {
    throw std::runtime_error("Polynomial orders > 1 must be matrix free");
  }

  if (matrixFree_) {
    NaluEnv::self().naluOutputP0()
      << "Warning: matrix free capability is experimental and only supports a "
         "limited set of use cases"
      << std::endl;
  }

  // let everyone know about core algorithm
  if (realmUsesEdges_) {
    NaluEnv::self().naluOutputP0()
      << "Edge-based scheme will be activated" << std::endl;
  } else if (matrixFree_) {
    NaluEnv::self().naluOutputP0()
      << "Matrix-free scheme will be activated" << std::endl;
  } else {
    throw std::runtime_error(
      "Realm: Nalu-Wind only supports edge-based or matrix-free schemes");
  }

  // how often is the realm solved..
  get_if_present(node, "solve_frequency", solveFrequency_, solveFrequency_);

  // automatic decomposition
  get_if_present(
    node, "automatic_decomposition_type", autoDecompType_, autoDecompType_);
  if ("None" != autoDecompType_) {
    NaluEnv::self().naluOutputP0()
      << "Warning: When using automatic_decomposition_type, one must have a "
         "serial file"
      << std::endl;
  }

  get_if_present(node, "rebalance_mesh", rebalanceMesh_, rebalanceMesh_);
  if (rebalanceMesh_) {
    get_required(node, "stk_rebalance_method", rebalanceMethod_);
    NaluEnv::self().naluOutputP0()
      << "Nalu will rebalance mesh using " << rebalanceMethod_ << std::endl;
  }

  // activate aura
  get_if_present(node, "activate_aura", activateAura_, activateAura_);
  if (activateAura_)
    NaluEnv::self().naluOutputP0()
      << "Nalu will activate aura ghosting" << std::endl;
  else
    NaluEnv::self().naluOutputP0()
      << "Nalu will deactivate aura ghosting" << std::endl;

  // memory diagnostic
  get_if_present(
    node, "activate_memory_diagnostic", activateMemoryDiagnostic_,
    activateMemoryDiagnostic_);
  if (activateMemoryDiagnostic_)
    NaluEnv::self().naluOutputP0()
      << "Nalu will activate detailed memory pulse" << std::endl;

  // allow for inconsistent restart (fields are missing)
  get_if_present(
    node, "support_inconsistent_multi_state_restart",
    supportInconsistentRestart_, supportInconsistentRestart_);

  // time step control
  const bool dtOptional = true;
  const YAML::Node y_time_step =
    expect_map(node, "time_step_control", dtOptional);
  if (y_time_step) {
    get_if_present(
      y_time_step, "target_courant", targetCourant_, targetCourant_);
    get_if_present(
      y_time_step, "time_step_change_factor", timeStepChangeFactor_,
      timeStepChangeFactor_);
  }

  get_if_present(node, "balance_nodes", doBalanceNodes_, doBalanceNodes_);
  get_if_present(
    node, "balance_nodes_iterations", balanceNodeOptions_.numIters,
    balanceNodeOptions_.numIters);
  get_if_present(
    node, "balance_nodes_target", balanceNodeOptions_.target,
    balanceNodeOptions_.target);
  if (node["balance_nodes_iterations"] || node["balance_nodes_target"]) {
    doBalanceNodes_ = true;
  }

  //======================================
  // now other commands/actions
  //======================================

  // load output first so we can check for serializing i/o
  outputInfo_->load(node);
  if (root()->serializedIOGroupSize_ == 0) {
    // only set from input file if command-line didn't set it
    root()->setSerializedIOGroupSize(outputInfo_->serializedIOGroupSize_);
  }

  // Parse catalyst input file if requested
  if (!outputInfo_->catalystFileName_.empty()) {
#ifdef NALU_USES_CATALYST
    int error = Iovs_exodus::DatabaseIO::parseCatalystFile(
      outputInfo_->catalystFileName_, outputInfo_->catalystParseJson_);
    if (error)
      throw std::runtime_error(
        "Catalyst file parse failed: " + outputInfo_->catalystFileName_);
#else
    throw std::runtime_error("Nalu-Wind not built with Catalyst support");
#endif
  }

  // solution options - loaded before create_mesh
  solutionOptions_->load(node);
  sideWriters_->load(node);

  // once we know the mesh name, we can open the meta data, and set spatial
  // dimension
  create_mesh();
  spatialDimension_ = meta_data().spatial_dimension();

  // post processing
  postProcessingInfo_->load(node);

  // boundary, init, material and equation systems "load"
  if (type_ == "multi_physics") {
    NaluEnv::self().naluOutputP0() << std::endl;
    NaluEnv::self().naluOutputP0()
      << "Boundary Condition Review: " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "===========================" << std::endl;
    boundaryConditions_ = BoundaryConditionCreator().create_bc_vector(node);
    NaluEnv::self().naluOutputP0() << std::endl;
    NaluEnv::self().naluOutputP0()
      << "Initial Condition Review:  " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "===========================" << std::endl;
    initialConditions_ =
      InitialConditionCreator(debug()).create_ic_vector(node);
    NaluEnv::self().naluOutputP0() << std::endl;
    NaluEnv::self().naluOutputP0()
      << "Material Prop Review:      " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "===========================" << std::endl;
    materialPropertys_.load(node);
    NaluEnv::self().naluOutputP0() << std::endl;
    NaluEnv::self().naluOutputP0()
      << "EqSys/options Review:      " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "===========================" << std::endl;
    equationSystems_.load(node);
  }

  // second set of options: mesh transformation... this means that the Realm
  // will expect to provide mesh transformation
  const YAML::Node meshTransformationNode =
    expect_sequence(node, "mesh_transformation", true);
  if (meshTransformationNode) {
    // mesh motion is active
    solutionOptions_->meshTransformation_ = true;

    // instantiate mesh transformation class once the mesh has been created
    meshTransformationAlg_.reset(
      new MeshTransformationAlg(*bulkData_, meshTransformationNode));
  }

  // second set of options: mesh motion... this means that the Realm will expect
  // to provide mesh motion
  const YAML::Node meshMotionNode = expect_sequence(node, "mesh_motion", true);
  if (meshMotionNode) {
    // has a user stated that mesh motion is external?
    if (solutionOptions_->externalMeshDeformation_) {
      NaluEnv::self().naluOutputP0()
        << "mesh motion set to external (will prevail over mesh motion "
           "specification)!"
        << std::endl;
    } else {
      // mesh motion is active
      solutionOptions_->meshMotion_ = true;

      // instantiate mesh motion class once the mesh has been created
      meshMotionAlg_.reset(new MeshMotionAlg(*bulkData_, meshMotionNode));
    }
  }

  // set number of nodes, check job run size
  check_job(true);
}

Simulation*
Realm::root()
{
  return parent()->root();
}
Simulation*
Realm::root() const
{
  return parent()->root();
}
Realms*
Realm::parent()
{
  return &realms_;
}
Realms*
Realm::parent() const
{
  return &realms_;
}

//--------------------------------------------------------------------------
//-------- setup_nodal_fields ----------------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_nodal_fields()
{
#ifdef NALU_USES_HYPRE
  hypreGlobalId_ = &(meta_data().declare_field<HypreIDFieldType>(
    stk::topology::NODE_RANK, "hypre_global_id"));
#endif
  tpetGlobalId_ = &(meta_data().declare_field<TpetIDFieldType>(
    stk::topology::NODE_RANK, "tpet_global_id"));

  // register global id and rank fields on all parts
  const stk::mesh::PartVector parts = meta_data().get_parts();
  for (size_t ipart = 0; ipart < parts.size(); ++ipart) {
    naluGlobalId_ = &(meta_data().declare_field<GlobalIdFieldType>(
      stk::topology::NODE_RANK, "nalu_global_id"));
    stk::mesh::put_field_on_mesh(*naluGlobalId_, *parts[ipart], nullptr);

#ifdef NALU_USES_HYPRE
    stk::mesh::put_field_on_mesh(*hypreGlobalId_, *parts[ipart], nullptr);
#endif
    stk::mesh::put_field_on_mesh(*tpetGlobalId_, *parts[ipart], nullptr);
    stk::mesh::field_fill(
      std::numeric_limits<LinSys::GlobalOrdinal>::max(), *tpetGlobalId_);
  }

  // loop over all material props targets and register nodal fields
  std::vector<std::string> targetNames = get_physics_target_names();
  equationSystems_.register_nodal_fields(targetNames);
}

//--------------------------------------------------------------------------
//-------- setup_edge_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_edge_fields()
{
  // loop over all material props targets and register edge fields
  std::vector<std::string> targetNames = get_physics_target_names();
  equationSystems_.register_edge_fields(targetNames);
}
//--------------------------------------------------------------------------
//-------- setup_element_fields --------------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_element_fields()
{
  // loop over all material props targets and register element fields
  std::vector<std::string> targetNames = get_physics_target_names();
  equationSystems_.register_element_fields(targetNames);

  const int numVolStates = does_mesh_move() ? number_of_states() : 1;

  if (has_mesh_deformation()) {
    const auto entityRank =
      realmUsesEdges_ ? stk::topology::EDGE_RANK : stk::topology::ELEM_RANK;
    const std::string fvm_fieldName =
      realmUsesEdges_ ? "edge_face_velocity_mag" : "face_velocity_mag";
    const std::string sv_fieldName =
      realmUsesEdges_ ? "edge_swept_face_volume" : "swept_face_volume";
    GenericFieldType* faceVelMag =
      &(meta_data().declare_field<GenericFieldType>(entityRank, fvm_fieldName));
    GenericFieldType* sweptFaceVolume =
      &(meta_data().declare_field<GenericFieldType>(
        entityRank, sv_fieldName, numVolStates));
    for (auto target : targetNames) {
      auto* targetPart = meta_data().get_part(target);
      auto fieldSize = 1;
      if (!realmUsesEdges_) {
        auto* meSCS =
          sierra::nalu::MasterElementRepo::get_surface_master_element(
            targetPart->topology());
        fieldSize = meSCS->num_integration_points();
      }
      stk::mesh::put_field_on_mesh(
        *faceVelMag, *targetPart, fieldSize, nullptr);
      stk::mesh::put_field_on_mesh(
        *sweptFaceVolume, *targetPart, fieldSize, nullptr);
    }
  }
}

//--------------------------------------------------------------------------
//-------- setup_interior_algorithms ---------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_interior_algorithms()
{
  if (has_mesh_deformation()) {
    const AlgorithmType algType = INTERIOR;
    stk::mesh::PartVector mmPartVec = meshMotionAlg_->get_partvec();
    for (auto p : mmPartVec) {
      if (p->topology() != stk::topology::HEX_8) {
        NaluEnv::self().naluOutputP0()
          << "Skipping registration of MeshVelocityEdgeAlg on part "
          << p->name()
          << ". GCL operations are currently only supported on HEX_8 "
             "elemeents.\n";
        continue;
      }
      if (realmUsesEdges_) {
        geometryAlgDriver_->register_elem_algorithm<MeshVelocityEdgeAlg>(
          algType, p, "mesh_vel");
      } else {
        geometryAlgDriver_->register_elem_algorithm<MeshVelocityAlg>(
          algType, p, "mesh_vel");
      }
    }
  }

  // loop over all material props targets and register interior algs
  std::vector<std::string> targetNames = get_physics_target_names();
  equationSystems_.register_interior_algorithm(targetNames);
}

//--------------------------------------------------------------------------
//-------- setup_post_processing_algorithms --------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_post_processing_algorithms()
{
  // get a pointer to the post processing data vector
  std::vector<PostProcessingData*>& ppDataVec = postProcessingInfo_->ppDataVec_;

  // iterate and set-up
  std::vector<PostProcessingData*>::const_iterator ii;
  for (ii = ppDataVec.begin(); ii != ppDataVec.end(); ++ii) {

    PostProcessingData& theData = *(*ii);
    // type
    std::string theType = theData.type_;
    NaluEnv::self().naluOutputP0()
      << "the post processing type is " << theType << std::endl;

    // output name
    std::string theFile = theData.outputFileName_;
    NaluEnv::self().naluOutputP0()
      << "the post processing file name: " << theFile << std::endl;

    // physics
    std::string thePhysics = theData.physics_;
    NaluEnv::self().naluOutputP0()
      << "the post processing physics name: " << thePhysics << std::endl;

    // target
    // map target names to physics parts
    theData.targetNames_ = physics_part_names(theData.targetNames_);

    const std::vector<std::string>& targetNames = theData.targetNames_;
    for (size_t in = 0; in < targetNames.size(); ++in)
      NaluEnv::self().naluOutputP0()
        << "Target name(s): " << targetNames[in] << std::endl;

    // params
    std::vector<double> parameters = theData.parameters_;
    for (size_t in = 0; in < parameters.size(); ++in)
      NaluEnv::self().naluOutputP0()
        << "Parameters used are: " << parameters[in] << std::endl;

    // call through to the Eqsys
    if (theType == "surface") {
      equationSystems_.register_surface_pp_algorithm(theData);
    } else {
      throw std::runtime_error(
        "Post Processing Error: only  surface-based is supported");
    }
  }

  // check for turbulence averaging fields
  if (
    NULL == turbulenceAveragingPostProcessing_ &&
    solutionOptions_->has_set_boussinesq_time_scale()) {

    turbulenceAveragingPostProcessing_ =
      new TurbulenceAveragingPostProcessing(*this, {});
  }

  if (NULL != turbulenceAveragingPostProcessing_)
    turbulenceAveragingPostProcessing_->setup();

  // check for data probes
  if (NULL != dataProbePostProcessing_) {
    dataProbePostProcessing_->setup();
  }

  if (aeroModels_->is_active())
    aeroModels_->setup(get_time_step_from_file(), bulk_data());

  // check for norm nodal fields
  if (NULL != solutionNormPostProcessing_)
    solutionNormPostProcessing_->setup();

  // Boundary layer statistics (MUST BE after turbulence averaging)
  if (nullptr != bdyLayerStats_)
    bdyLayerStats_->setup();
}

//--------------------------------------------------------------------------
//-------- setup_bc --------------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_bc()
{
  // loop over all bcs and register
  for (auto&& bc : boundaryConditions_) {
    std::string name = physics_part_name(bc->targetName_);

    switch (bc->theBcType_) {
    case WALL_BC:
      equationSystems_.register_wall_bc(
        name, *reinterpret_cast<const WallBoundaryConditionData*>(bc.get()));
      break;
    case INFLOW_BC:
      equationSystems_.register_inflow_bc(
        name, *reinterpret_cast<const InflowBoundaryConditionData*>(bc.get()));
      break;
    case OPEN_BC:
      equationSystems_.register_open_bc(
        name, *reinterpret_cast<const OpenBoundaryConditionData*>(bc.get()));
      break;
    case SYMMETRY_BC:
      equationSystems_.register_symmetry_bc(
        name,
        *reinterpret_cast<const SymmetryBoundaryConditionData*>(bc.get()));
      break;
    case ABLTOP_BC:
      equationSystems_.register_abltop_bc(
        name, *reinterpret_cast<const ABLTopBoundaryConditionData*>(bc.get()));
      break;
    case PERIODIC_BC: {
      ThrowAssert(
        reinterpret_cast<const PeriodicBoundaryConditionData*>(bc.get()) !=
        nullptr);
      const auto& pbc =
        (*reinterpret_cast<const PeriodicBoundaryConditionData*>(bc.get()));

      std::string masterName = physics_part_name(pbc.masterSlave_.master_);
      std::string slaveName = physics_part_name(pbc.masterSlave_.slave_);
      equationSystems_.register_periodic_bc(masterName, slaveName, pbc);
      break;
    }
    case NON_CONFORMAL_BC:
      equationSystems_.register_non_conformal_bc(
        *reinterpret_cast<const NonConformalBoundaryConditionData*>(bc.get()));
      break;
    case OVERSET_BC: {
      const OversetBoundaryConditionData& obc =
        reinterpret_cast<const OversetBoundaryConditionData&>(*bc);
      setup_overset_bc(obc);
      equationSystems_.register_overset_bc(obc);
      break;
    }
    default:
      throw std::runtime_error("unknown bc");
    }
  }
}

//--------------------------------------------------------------------------
//-------- enforce_bc_on_exposed_faces  ------------------------------------
//--------------------------------------------------------------------------
void
Realm::enforce_bc_on_exposed_faces()
{
  double start_time = NaluEnv::self().nalu_time();

  NaluEnv::self().naluOutputP0() << "Realm::skin_mesh(): Begin" << std::endl;

  // first, skin mesh and, therefore, populate
  stk::mesh::Selector activePart =
    meta_data().locally_owned_part() | meta_data().globally_shared_part();
  stk::mesh::PartVector partVec;
  partVec.push_back(exposedBoundaryPart_);
  stk::mesh::create_exposed_block_boundary_sides(
    *bulkData_, activePart, partVec);

  stk::mesh::Selector selectRule = stk::mesh::Selector(*exposedBoundaryPart_) &
                                   !stk::mesh::selectUnion(bcPartVec_);

  stk::mesh::BucketVector const& face_buckets =
    bulkData_->get_buckets(meta_data().side_rank(), selectRule);

  if (!face_buckets.empty()) {
    NaluEnv::self().naluOutputP0()
      << "Exposed surfaces found without a boundary condition applied"
      << std::endl;

    // proceed to show the problem faces
    for (stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
         ib != face_buckets.end(); ++ib) {
      stk::mesh::Bucket& b = **ib;
      const stk::mesh::Bucket::size_type length = b.size();
      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
        // extract the face
        stk::mesh::Entity face = b[k];

        // report the offending face id
        NaluEnv::self().naluOutput()
          << "Face Id: " << bulkData_->identifier(face)
          << " is not properly covered" << std::endl;

        // extract face nodes
        const stk::mesh::Entity* face_node_rels = bulkData_->begin_nodes(face);
        const unsigned numberOfNodes = bulkData_->num_nodes(face);
        NaluEnv::self().naluOutput()
          << " Number of nodes connected to this face is: " << numberOfNodes
          << std::endl;
        for (unsigned n = 0; n < numberOfNodes; ++n) {
          stk::mesh::Entity node = face_node_rels[n];
          NaluEnv::self().naluOutput()
            << " attached node Id: " << bulkData_->identifier(node)
            << std::endl;
        }

        // extract the element relations to report to the user and the number of
        // elements connected
        const stk::mesh::Entity* face_elem_rels =
          bulkData_->begin_elements(face);
        const unsigned numberOfElems = bulkData_->num_elements(face);
        NaluEnv::self().naluOutput()
          << " Number of elements connected to this face is: " << numberOfElems
          << std::endl;

        for (unsigned faceElem = 0; faceElem < numberOfElems; ++faceElem) {
          stk::mesh::Entity element = face_elem_rels[faceElem];
          NaluEnv::self().naluOutput()
            << " attached element Id: " << bulkData_->identifier(element)
            << std::endl;
        }
      }
    }
    throw std::runtime_error(
      "Realm::Error: Please aply bc to problematic exposed surfaces ");
  }

  const double end_time = NaluEnv::self().nalu_time();

  // set mesh reading
  timerSkinMesh_ = (end_time - start_time);

  NaluEnv::self().naluOutputP0() << "Realm::skin_mesh(): End" << std::endl;
}

//--------------------------------------------------------------------------
//-------- setup_initial_conditions ----------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_initial_conditions()
{
  // loop over all ics and register
  for (auto&& initCond : initialConditions_) {

    const std::vector<std::string> targetNames =
      handle_all_element_part_alias(initCond->targetNames_);

    for (size_t itarget = 0; itarget < targetNames.size(); ++itarget) {
      const std::string targetName = physics_part_name(targetNames[itarget]);

      // target need not be subsetted since nothing below will depend on topo
      stk::mesh::Part* targetPart = meta_data().get_part(targetName);
      if (!targetPart) {
        throw std::runtime_error(
          "Part: " + targetName +
          " in the initial_conditions target does not exist.");
      }

      switch (initCond->theIcType_) {

      case CONSTANT_UD: {
        const ConstantInitialConditionData& genIC =
          *reinterpret_cast<const ConstantInitialConditionData*>(
            initCond.get());
        ThrowAssert(genIC.data_.size() == genIC.fieldNames_.size());
        for (size_t ifield = 0; ifield < genIC.fieldNames_.size(); ++ifield) {

          std::vector<double> genSpec = genIC.data_[ifield];
          stk::mesh::FieldBase* field = stk::mesh::get_field_by_name(
            genIC.fieldNames_[ifield], meta_data());
          ThrowAssert(field);

          stk::mesh::FieldBase* fieldWithState =
            (field->number_of_states() > 1)
              ? field->field_state(stk::mesh::StateNP1)
              : field->field_state(stk::mesh::StateNone);

          std::vector<double> userGen = genSpec;
          ConstantAuxFunction* theGenFunc =
            new ConstantAuxFunction(0, genSpec.size(), userGen);
          AuxFunctionAlgorithm* auxGen = new AuxFunctionAlgorithm(
            *this, targetPart, fieldWithState, theGenFunc,
            stk::topology::NODE_RANK);
          initCondAlg_.push_back(auxGen);
        }
      } break;

      case FUNCTION_UD: {
        const UserFunctionInitialConditionData& fcnIC =
          *reinterpret_cast<const UserFunctionInitialConditionData*>(
            initCond.get());
        equationSystems_.register_initial_condition_fcn(targetPart, fcnIC);
      } break;

      case USER_SUB_UD:
        throw std::runtime_error(
          "Realm::setup_initial_conditions: USER_SUB not supported: ");

      case UserDataType_END:
        break;

      default:
        NaluEnv::self().naluOutputP0()
          << "Realm::setup_initial_conditions: unknown type: "
          << initCond->theIcType_ << std::endl;
        throw std::runtime_error(
          "Realm::setup_initial_conditions: unknown type:");
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- setup_property --------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_property()
{
  // loop over all target names
  const std::vector<std::string> targetNames = get_physics_target_names();
  for (size_t itarget = 0; itarget < targetNames.size(); ++itarget) {

    // target need not be subsetted since nothing below will depend on topo
    stk::mesh::Part* targetPart = meta_data().get_part(targetNames[itarget]);
    if (!targetPart) {
      throw std::runtime_error(
        "Part: " + targetNames[itarget] +
        " in the material_properties target does not exist.");
    }

    // loop over propertyMap
    std::map<PropertyIdentifier, ScalarFieldType*>::iterator ii;
    for (ii = propertyMap_.begin(); ii != propertyMap_.end(); ++ii) {

      // extract property id and field pointer
      PropertyIdentifier thePropId = (*ii).first;
      ScalarFieldType* thePropField = (*ii).second;

      // find the material property data object
      MaterialPropertyData* matData = NULL;
      std::map<PropertyIdentifier, MaterialPropertyData*>::iterator itf =
        materialPropertys_.propertyDataMap_.find(thePropId);
      if (itf == materialPropertys_.propertyDataMap_.end()) {
        // will need to throw
        NaluEnv::self().naluOutputP0()
          << "issue with property: " << PropertyIdentifierNames[thePropId]
          << std::endl;
        throw std::runtime_error("Please add property specification ");
      } else {
        matData = (*itf).second;
      }

      switch (matData->type_) {

      case CONSTANT_MAT: {

        // for constant specific heat, proceed in specialty code; create the
        // appropriate enthalpy evaluator
        if (thePropId == SPEC_HEAT_ID && needsEnthalpy_) {
          // extract reference temperature
          double tRef = 300.0;
          extract_universal_constant("reference_temperature", tRef, true);

          // set up evaluators required for all cases
          PropertyEvaluator* theCpPropEval = NULL;
          PropertyEvaluator* theEnthPropEval = NULL;

          // check for species-based cp
          if (matData->cpConstMap_.size() > 0.0) {
            if (uniformFlow_) {
              throw std::runtime_error(
                "uniform flow cp should simply use the single-valued constant");
            } else {
              // props computed based on local mass fractions, however, constant
              // per species k
              theCpPropEval = new SpecificHeatConstCpkPropertyEvaluator(
                matData->cpConstMap_, meta_data());
              theEnthPropEval = new EnthalpyConstCpkPropertyEvaluator(
                matData->cpConstMap_, matData->hfConstMap_, meta_data(), tRef);
            }

            // create the algorithm to compute Cp; EnthalpyEqs manages h
            // population, i.e., no alg required
            TemperaturePropAlgorithm* auxAlg = new TemperaturePropAlgorithm(
              *this, targetPart, thePropField, theCpPropEval);
            propertyAlg_.push_back(auxAlg);

          } else {
            // single constant value
            double specificHeatValue = matData->constValue_;
            theCpPropEval = new ConstantPropertyEvaluator(specificHeatValue);
            theEnthPropEval = new EnthalpyConstSpecHeatPropertyEvaluator(
              specificHeatValue, tRef);

            // set the default begin/end
            int theBegin = 0;
            int theEnd = 1;

            // create everything
            std::vector<double> userConstData(1);
            userConstData[0] = matData->constValue_;
            ConstantAuxFunction* theAuxFunc =
              new ConstantAuxFunction(theBegin, theEnd, userConstData);
            AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
              *this, targetPart, thePropField, theAuxFunc,
              stk::topology::NODE_RANK);
            propertyAlg_.push_back(auxAlg);
          }
          // push to prop eval
          materialPropertys_.propertyEvalMap_[SPEC_HEAT_ID] = theCpPropEval;
          materialPropertys_.propertyEvalMap_[ENTHALPY_ID] = theEnthPropEval;
        } else {

          // set the default begin/end
          int theBegin = 0;
          int theEnd = 1;

          // create everything
          std::vector<double> userConstData(1);
          userConstData[0] = matData->constValue_;
          ConstantAuxFunction* theAuxFunc =
            new ConstantAuxFunction(theBegin, theEnd, userConstData);
          AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
            *this, targetPart, thePropField, theAuxFunc,
            stk::topology::NODE_RANK);
          propertyAlg_.push_back(auxAlg);
        }
      } break;

      case MIXFRAC_MAT: {
        // extract the mixture fraction field
        ScalarFieldType* mixFrac = meta_data().get_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "mixture_fraction");

        // primary and secondary
        const double propPrim = matData->primary_;
        const double propSec = matData->secondary_;

        // density requires inverse weighting
        if (DENSITY_ID == thePropId) {
          // create the inverse mix frac property algorithm
          InversePropAlgorithm* auxAlg = new InversePropAlgorithm(
            *this, targetPart, thePropField, mixFrac, propPrim, propSec);
          propertyAlg_.push_back(auxAlg);
        } else {
          // all else need linear weighting
          LinearPropAlgorithm* auxAlg = new LinearPropAlgorithm(
            *this, targetPart, thePropField, mixFrac, propPrim, propSec);
          propertyAlg_.push_back(auxAlg);
        }
      } break;

      case POLYNOMIAL_MAT: {

        // switch on property id
        switch (thePropId) {

        case VISCOSITY_ID: {
          PropertyEvaluator* viscPropEval = NULL;

          if (isothermalFlow_) {

            // all props will use Tref
            double tRef = 0.0;
            extract_universal_constant("reference_temperature", tRef, true);

            if (uniformFlow_) {
              // props computed based on YkRef and Tref
              throw std::runtime_error(
                "Realm::setup_property: Sorry, polynomial visc Ykref and Tref "
                "is not supported");
            } else {
              // props computed based on Yk and Tref
              viscPropEval = new SutherlandsYkTrefPropertyEvaluator(
                matData->polynomialCoeffsMap_, meta_data(), tRef);
            }
            // create the GenericPropAlgorithm; push it back
            GenericPropAlgorithm* auxAlg = new GenericPropAlgorithm(
              *this, targetPart, thePropField, viscPropEval);
            propertyAlg_.push_back(auxAlg);
          } else {
            // all props will use [transported] T
            if (uniformFlow_) {
              // props computed based on YkRef and T
              viscPropEval = new SutherlandsPropertyEvaluator(
                materialPropertys_.referencePropertyDataMap_,
                matData->polynomialCoeffsMap_);
            } else {
              // props computed based on Yk and T
              viscPropEval = new SutherlandsYkPropertyEvaluator(
                matData->polynomialCoeffsMap_, meta_data());
            }
            // create the TemperaturePropAlgorithm; push it back
            TemperaturePropAlgorithm* auxAlg = new TemperaturePropAlgorithm(
              *this, targetPart, thePropField, viscPropEval);
            propertyAlg_.push_back(auxAlg);
          }

          // create the property alg and push to evalmap
          materialPropertys_.propertyEvalMap_[thePropId] = viscPropEval;

        } break;

        case ENTHALPY_ID: {
          NaluEnv::self().naluOutputP0()
            << "Enthalpy specification is not required as Cp is sufficient";
        } break;

        case SPEC_HEAT_ID: {
          // R
          double universalR = 8314.4621;
          extract_universal_constant(
            "universal_gas_constant", universalR, true);

          // create the property alg and push to evalmap
          PropertyEvaluator* theCpPropEval = NULL;
          PropertyEvaluator* theEnthPropEval = NULL;
          if (uniformFlow_) {
            // props computed based on reference values
            theCpPropEval = new SpecificHeatPropertyEvaluator(
              materialPropertys_.referencePropertyDataMap_,
              matData->lowPolynomialCoeffsMap_,
              matData->highPolynomialCoeffsMap_, universalR);
            theEnthPropEval = new EnthalpyPropertyEvaluator(
              materialPropertys_.referencePropertyDataMap_,
              matData->lowPolynomialCoeffsMap_,
              matData->highPolynomialCoeffsMap_, universalR);
          } else {
            // props computed based on transported Yk values
            theCpPropEval = new SpecificHeatTYkPropertyEvaluator(
              materialPropertys_.referencePropertyDataMap_,
              matData->lowPolynomialCoeffsMap_,
              matData->highPolynomialCoeffsMap_, universalR, meta_data());
            theEnthPropEval = new EnthalpyTYkPropertyEvaluator(
              materialPropertys_.referencePropertyDataMap_,
              matData->lowPolynomialCoeffsMap_,
              matData->highPolynomialCoeffsMap_, universalR, meta_data());
          }

          // create the algorithm to compute Cp; EnthalpyEqs manages h
          // population, i.e., no alg required
          TemperaturePropAlgorithm* auxAlg = new TemperaturePropAlgorithm(
            *this, targetPart, thePropField, theCpPropEval);
          propertyAlg_.push_back(auxAlg);

          // set property maps...
          materialPropertys_.propertyEvalMap_[SPEC_HEAT_ID] = theCpPropEval;
          materialPropertys_.propertyEvalMap_[ENTHALPY_ID] = theEnthPropEval;
        } break;

        default:
          throw std::runtime_error(
            "Realm::setup_property: polynomial supports mu, Cp and h:");
        }
      } break;

      case IDEAL_GAS_T_MAT:
      case IDEAL_GAS_T_P_MAT: {
        if (DENSITY_ID == thePropId) {

          // everyone will require R
          double universalR = 8314.4621;
          extract_universal_constant(
            "universal_gas_constant", universalR, true);

          // create the property evaluator
          PropertyEvaluator* rhoPropEval = NULL;
          if (uniformFlow_) {
            // load mw and reference species
            std::vector<std::pair<double, double>> mwMassFracVec;
            std::map<std::string, ReferencePropertyData*>::const_iterator itrp;
            for (itrp = materialPropertys_.referencePropertyDataMap_.begin();
                 itrp != materialPropertys_.referencePropertyDataMap_.end();
                 ++itrp) {
              ReferencePropertyData* propData = (*itrp).second;
              std::pair<double, double> thePair;
              thePair = std::make_pair(propData->mw_, propData->massFraction_);
              mwMassFracVec.push_back(thePair);
            }
            if (IDEAL_GAS_T_MAT == matData->type_) {
              double pRef = 101325.0;
              extract_universal_constant("reference_pressure", pRef, true);
              rhoPropEval =
                new IdealGasTPropertyEvaluator(pRef, universalR, mwMassFracVec);
            } else {
              rhoPropEval = new IdealGasTPPropertyEvaluator(
                universalR, mwMassFracVec, meta_data());
            }
          } else {
            // load mw
            std::vector<double> mwVec;
            std::map<std::string, ReferencePropertyData*>::const_iterator itrp;
            for (itrp = materialPropertys_.referencePropertyDataMap_.begin();
                 itrp != materialPropertys_.referencePropertyDataMap_.end();
                 ++itrp) {
              ReferencePropertyData* propData = (*itrp).second;
              mwVec.push_back(propData->mw_);
            }
            if (IDEAL_GAS_T_MAT == matData->type_) {
              double pRef = 101325.0;
              extract_universal_constant("reference_pressure", pRef, true);
              rhoPropEval = new IdealGasTYkPropertyEvaluator(
                pRef, universalR, mwVec, meta_data());
            } else {
              throw std::runtime_error("Realm::setup_property: ideal_gas_tp "
                                       "only supported for uniform flow:");
            }
          }

          // push back property evaluator to map
          materialPropertys_.propertyEvalMap_[thePropId] = rhoPropEval;

          // create the property algorithm
          TemperaturePropAlgorithm* auxAlg = new TemperaturePropAlgorithm(
            *this, targetPart, thePropField, rhoPropEval);
          propertyAlg_.push_back(auxAlg);
        } else {
          throw std::runtime_error(
            "Realm::setup_property: ideal_gas_t only supported for density:");
        }

      } break;

      case IDEAL_GAS_YK_MAT: {
        if (DENSITY_ID == thePropId) {

          // pRef, tRef and R
          double pRef = 101325.0;
          double tRef = 300.0;
          double universalR = 8314.4621;
          extract_universal_constant("reference_pressure", pRef, true);
          extract_universal_constant("reference_temperature", tRef, true);
          extract_universal_constant(
            "universal_gas_constant", universalR, true);

          // load mw
          std::vector<double> mwVec;
          std::map<std::string, ReferencePropertyData*>::const_iterator itrp;
          for (itrp = materialPropertys_.referencePropertyDataMap_.begin();
               itrp != materialPropertys_.referencePropertyDataMap_.end();
               ++itrp) {
            ReferencePropertyData* propData = (*itrp).second;
            mwVec.push_back(propData->mw_);
          }

          // create the property evaluator
          PropertyEvaluator* rhoPropEval = new IdealGasYkPropertyEvaluator(
            pRef, tRef, universalR, mwVec, meta_data());

          // push back property evaluator to map
          materialPropertys_.propertyEvalMap_[thePropId] = rhoPropEval;

          // create the property algorithm
          GenericPropAlgorithm* auxAlg = new GenericPropAlgorithm(
            *this, targetPart, thePropField, rhoPropEval);
          propertyAlg_.push_back(auxAlg);

        } else {
          throw std::runtime_error(
            "Realm::setup_property: ideal_gas_yk only supported for density:");
        }
      } break;

      case GEOMETRIC_MAT: {
        // propery is a function of inverse dual volume
        InverseDualVolumePropAlgorithm* auxAlg =
          new InverseDualVolumePropAlgorithm(*this, targetPart, thePropField);
        propertyAlg_.push_back(auxAlg);
      } break;

      case GENERIC: {
        // default property evaluator
        PropertyEvaluator* propEval = NULL;

        // extract the property evaluator name
        std::string propEvalName = matData->genericPropertyEvaluatorName_;

        if (propEvalName == "water_viscosity_T") {
          propEval = new WaterViscosityTPropertyEvaluator(meta_data());
        } else if (propEvalName == "water_density_T") {
          propEval = new WaterDensityTPropertyEvaluator(meta_data());
        } else if (propEvalName == "water_specific_heat_T") {
          propEval = new WaterSpecHeatTPropertyEvaluator(meta_data());
          // create the enthalpy prop evaluator and store
          WaterEnthalpyTPropertyEvaluator* theEnthPropEval =
            new WaterEnthalpyTPropertyEvaluator(meta_data());
          materialPropertys_.propertyEvalMap_[ENTHALPY_ID] = theEnthPropEval;
        } else if (propEvalName == "water_thermal_conductivity_T") {
          propEval = new WaterThermalCondTPropertyEvaluator(meta_data());
        } else {
          throw std::runtime_error(
            "Realm::setup_property: unknown GENERIC type: " + propEvalName);
        }

        // for now, all of the above are TempPropAlgs; push it back
        TemperaturePropAlgorithm* auxAlg = new TemperaturePropAlgorithm(
          *this, targetPart, thePropField, propEval);
        propertyAlg_.push_back(auxAlg);

        // push back property evaluator to map
        materialPropertys_.propertyEvalMap_[thePropId] = propEval;
      } break;

      case MaterialPropertyType_END:
        break;

      default:
        throw std::runtime_error("Realm::setup_property: unknown type:");
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- extract_universal_constant --------------------------------------
//--------------------------------------------------------------------------
void
Realm::extract_universal_constant(
  const std::string name, double& value, const bool useDefault)
{
  std::map<std::string, double>::iterator it =
    materialPropertys_.universalConstantMap_.find(name);
  if (it == materialPropertys_.universalConstantMap_.end()) {
    // not found
    if (useDefault) {
      NaluEnv::self().naluOutputP0()
        << "WARNING: Reference value for " << name << " not found "
        << " using " << value << std::endl;
    } else {
      throw std::runtime_error(
        "Realm::setup_property: reference value not found: " + name);
    }
  } else {
    value = (*it).second;
  }
}

//--------------------------------------------------------------------------
//-------- initialize_global_variables -------------------------------------
//--------------------------------------------------------------------------
void
Realm::initialize_global_variables()
{
  // other variables created on the fly during Eqs registration
  const bool needInOutput = false;
  const bool needInRestart = true;
  globalParameters_->set_param("timeStepNm1", 1.0, needInOutput, needInRestart);
  globalParameters_->set_param("timeStepCount", 1, needInOutput, needInRestart);

  // consider pushing this parameter to some higher level design
  if (NULL != turbulenceAveragingPostProcessing_)
    globalParameters_->set_param(
      "currentTimeFilter", 0.0, needInOutput, needInRestart);
}

//--------------------------------------------------------------------------
//-------- augment_property_map --------------------------------------------
//--------------------------------------------------------------------------
void
Realm::augment_property_map(
  PropertyIdentifier propID, ScalarFieldType* theField)
{
  propertyMap_[propID] = theField;
}

//--------------------------------------------------------------------------
//-------- makeSureNodesHaveValidTopology ----------------------------------
//--------------------------------------------------------------------------
void
Realm::makeSureNodesHaveValidTopology()
{
  // To make sure nodes have valid topology, we have to make sure they are in a
  // part that has NODE topology. So first, let's obtain the node topology part:
  const stk::mesh::Part& nodePart =
    bulkData_->mesh_meta_data().get_topology_root_part(stk::topology::NODE);
  stk::mesh::Selector nodesNotInNodePart =
    (!nodePart) & bulkData_->mesh_meta_data().locally_owned_part();

  // get all the nodes that are *NOT* in nodePart
  std::vector<stk::mesh::Entity> nodes_vector;
  stk::mesh::get_selected_entities(
    nodesNotInNodePart, bulkData_->buckets(stk::topology::NODE_RANK),
    nodes_vector);
  // now we require all nodes are in proper node part
  if (nodes_vector.size())
    std::cout << "nodes_vector= " << nodes_vector.size() << std::endl;
  ThrowRequire(0 == nodes_vector.size());
}

void
Realm::update_geometry_due_to_mesh_motion()
{
  // check for mesh motion
  if (solutionOptions_->meshMotion_) {

    meshMotionAlg_->execute(get_current_time());

    compute_geometry();

    meshMotionAlg_->post_compute_geometry();

    // and non-conformal algorithm
    if (hasNonConformal_)
      initialize_non_conformal();
  }
}

void
Realm::update_graph_connectivity_and_coordinates_due_to_mesh_motion()
{
  if (solutionOptions_->meshMotion_) {
    // Reset the stk::mesh::NgpMesh instance
    meshInfo_.reset(new typename Realm::NgpMeshInfo(*bulkData_));

    // now re-initialize linear system
    equationSystems_.reinitialize_linear_system();
  }

  // deal with non-topology changes, however, moving mesh
  if (has_mesh_deformation()) {
    // extract target parts for this physics
    if (solutionOptions_->externalMeshDeformation_) {
      std::vector<std::string> targetNames = get_physics_target_names();
      for (size_t itarget = 0; itarget < targetNames.size(); ++itarget) {
        stk::mesh::Part* targetPart =
          meta_data().get_part(targetNames[itarget]);
        set_current_coordinates(targetPart);
      }
    }
    compute_geometry();
  }

  // ask the equation system to do some work
  equationSystems_.pre_timestep_work();
}

//--------------------------------------------------------------------------
//-------- evaluate_properties ---------------------------------------------
//--------------------------------------------------------------------------
void
Realm::evaluate_properties()
{
  double start_time = NaluEnv::self().nalu_time();
  for (size_t k = 0; k < propertyAlg_.size(); ++k) {
    propertyAlg_[k]->execute();
  }
  equationSystems_.evaluate_properties();
  double end_time = NaluEnv::self().nalu_time();
  timerPropertyEval_ += (end_time - start_time);
}

//--------------------------------------------------------------------------
//-------- advance_time_step -----------------------------------------------
//--------------------------------------------------------------------------
void
Realm::advance_time_step()
{
  // leave if we do not need to solve
  const int timeStepCount = get_time_step_count();
  const bool advanceMe = (timeStepCount % solveFrequency_) == 0 ? true : false;
  if (!advanceMe)
    return;
  NaluEnv::self().naluOutputP0()
    << name_ << "::advance_time_step() " << std::endl;

  NaluEnv::self().naluOutputP0()
    << "NLI" << std::setw(8) << std::right << "Name" << std::setw(22)
    << std::right << "Linear Iter" << std::setw(16) << std::right
    << "Linear Res" << std::setw(16) << std::right << "NLinear Res"
    << std::setw(14) << std::right << "Scaled NLR" << std::endl;

  NaluEnv::self().naluOutputP0()
    << "---" << std::setw(8) << std::right << "----" << std::setw(22)
    << std::right << "-----------" << std::setw(16) << std::right
    << "----------" << std::setw(16) << std::right << "-----------"
    << std::setw(14) << std::right << "----------" << std::endl;

  // evaluate new geometry based on latest mesh motion geometry state (provided
  // that external is active)
  if (solutionOptions_->externalMeshDeformation_)
    compute_geometry();

  // evaluate properties based on latest state including boundary and and
  // possible xfer
  evaluate_properties();

  // compute velocity relative to mesh
  compute_vrtm();

  // check for  actuator; assemble the source terms for this step
  if (aeroModels_->is_active()) {
    const double start_time = NaluEnv::self().nalu_time();
    aeroModels_->execute(timerActuator_);
    const double end_time = NaluEnv::self().nalu_time();
    timerActuator_ += end_time - start_time;
  }
  // Check for ABL forcing; estimate source terms for this time step
  if (NULL != ablForcingAlg_) {
    ablForcingAlg_->execute();
  }

  nonlinear_iterations(equationSystems_.maxIterations_);
}

void
Realm::nonlinear_iterations(const int numNonLinearIterations)
{
  for (int i = 0; i < numNonLinearIterations; ++i) {
    currentNonlinearIteration_ = i + 1;
    NaluEnv::self().naluOutputP0()
      << currentNonlinearIteration_ << "/" << numNonLinearIterations
      << std::setw(29) << std::right << "Equation System Iteration"
      << std::endl;

    isFinalOuterIter_ = ((i + 1) == numNonLinearIterations);

    const bool isConverged = equationSystems_.solve_and_update();

    // evaluate properties based on latest np1 solution
    evaluate_properties();

    if (isConverged) {
      NaluEnv::self().naluOutputP0()
        << "norm convergence criteria met for all equation systems: "
        << std::endl;
      NaluEnv::self().naluOutputP0()
        << "max scaled norm is: " << equationSystems_.provide_system_norm()
        << std::endl;
      break;
    }
  }
}

//--------------------------------------------------------------------------
//-------- output_converged_results ----------------------------------------
//--------------------------------------------------------------------------
void
Realm::output_converged_results()
{
  provide_output();
  provide_restart_output();
}

//--------------------------------------------------------------------------
//-------- compute_adaptive_time_step --------------------------------------
//--------------------------------------------------------------------------
double
Realm::compute_adaptive_time_step()
{
  // extract current time
  const double dtN = get_time_step();

  // ratio of how off we are
  const double factorOff = targetCourant_ / maxCourant_;

  // scaling for dt and candidate
  const double dtScaling = (targetCourant_ < maxCourant_)
                             ? std::max(factorOff, 1.0 / timeStepChangeFactor_)
                             : std::min(factorOff, timeStepChangeFactor_);
  const double candidateDt = dtN * dtScaling;

  return candidateDt;
}

//--------------------------------------------------------------------------
//-------- commit ----------------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::commit()
{
  //====================================================
  // Commit the meta data
  //====================================================
  meta_data().commit();
}

//--------------------------------------------------------------------------
//-------- create_mesh() ---------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::create_mesh()
{
  double start_time = NaluEnv::self().nalu_time();

  NaluEnv::self().naluOutputP0() << "Realm::create_mesh(): Begin" << std::endl;
  stk::ParallelMachine pm = NaluEnv::self().parallel_comm();

  // news for mesh constructs
  stk::mesh::MeshBuilder meshBuilder(pm);
  meshBuilder.set_aura_option(
    activateAura_ ? stk::mesh::BulkData::AUTO_AURA
                  : stk::mesh::BulkData::NO_AUTO_AURA);
  bulkData_ = meshBuilder.create();
  ioBroker_ = new stk::io::StkMeshIoBroker(pm);
  ioBroker_->set_auto_load_distribution_factor_per_nodeset(false);
  ioBroker_->set_bulk_data(*bulkData_);

  // allow for automatic decomposition
  if (autoDecompType_ != "None")
    ioBroker_->property_add(
      Ioss::Property("DECOMPOSITION_METHOD", autoDecompType_));

  // Initialize meta data (from exodus file); can possibly be a restart file..
  inputMeshIdx_ = ioBroker_->add_mesh_database(
    inputDBName_,
    restarted_simulation() ? stk::io::READ_RESTART : stk::io::READ_MESH);
  ioBroker_->create_input_mesh();

  // declare an exposed part for later bc coverage check
  if (checkForMissingBcs_) {
    exposedBoundaryPart_ = &meta_data().declare_part(
      "exposed_boundary_part", meta_data().side_rank());
  }

  // declare a part to hold new edges
  if (realmUsesEdges_) {
    edgesPart_ =
      &meta_data().declare_part("create_edges_part", stk::topology::EDGE_RANK);
  }

  // set mesh creation
  const double end_time = NaluEnv::self().nalu_time();
  timerCreateMesh_ = (end_time - start_time);

  NaluEnv::self().naluOutputP0() << "Realm::create_mesh() End" << std::endl;
}

//--------------------------------------------------------------------------
//-------- create_output_mesh() --------------------------------------------
//--------------------------------------------------------------------------
void
Realm::create_output_mesh()
{
  sideWriters_->construct_writers(bulk_data());
  // exodus output file creation
  if (outputInfo_->hasOutputBlock_) {

    double start_time = NaluEnv::self().nalu_time();
    NaluEnv::self().naluOutputP0()
      << "Realm::create_output_mesh(): Begin" << std::endl;

    if (outputInfo_->outputFreq_ == 0)
      return;

    std::string oname = outputInfo_->outputDBName_;
    if (
      !outputInfo_->catalystFileName_.empty() ||
      !outputInfo_->paraviewScriptName_.empty()) {
#ifdef NALU_USES_CATALYST
      outputInfo_->outputPropertyManager_->add(Ioss::Property(
        "CATALYST_BLOCK_PARSE_JSON_STRING", outputInfo_->catalystParseJson_));
      std::string input_deck_name = "%B";
      stk::util::filename_substitution(input_deck_name);
      outputInfo_->outputPropertyManager_->add(Ioss::Property(
        "CATALYST_BLOCK_PARSE_INPUT_DECK_NAME", input_deck_name));

      if (!outputInfo_->paraviewScriptName_.empty())
        outputInfo_->outputPropertyManager_->add(Ioss::Property(
          "CATALYST_SCRIPT", outputInfo_->paraviewScriptName_.c_str()));

      outputInfo_->outputPropertyManager_->add(
        Ioss::Property("CATALYST_CREATE_SIDE_SETS", 1));

      resultsFileIndex_ = ioBroker_->create_output_mesh(
        oname, stk::io::WRITE_RESULTS, *outputInfo_->outputPropertyManager_,
        "catalyst");
#else
      throw std::runtime_error("Nalu-Wind not built with Catalyst support");
#endif
    } else {
      resultsFileIndex_ = ioBroker_->create_output_mesh(
        oname, stk::io::WRITE_RESULTS, *outputInfo_->outputPropertyManager_);
    }

    // Tell stk_io how to output element block nodal fields:
    // if 'true' passed to function, then output them as nodeset fields;
    // if 'false', then output as nodal fields (on all nodes of the mesh,
    // zero-filled) The option is provided since some
    // post-processing/visualization codes do not correctly handle nodeset
    // fields.
    ioBroker_->use_nodeset_for_part_nodes_fields(
      resultsFileIndex_, outputInfo_->outputNodeSet_);

    // FIXME: add_field can take user-defined output name, not just varName
    for (std::set<std::string>::iterator itorSet =
           outputInfo_->outputFieldNameSet_.begin();
         itorSet != outputInfo_->outputFieldNameSet_.end(); ++itorSet) {
      std::string varName = *itorSet;
      stk::mesh::FieldBase* theField =
        stk::mesh::get_field_by_name(varName, meta_data());
      if (NULL == theField) {
        NaluEnv::self().naluOutputP0()
          << " Sorry, no field by the name " << varName << std::endl;
      } else {
        // 'varName' is the name that will be written to the database
        // For now, just using the name of the stk field
        ioBroker_->add_field(resultsFileIndex_, *theField, varName);
      }
    }

    // set mesh creation
    const double end_time = NaluEnv::self().nalu_time();
    timerCreateMesh_ = (end_time - start_time);

    NaluEnv::self().naluOutputP0()
      << "Realm::create_output_mesh() End" << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- create_restart_mesh() --------------------------------------------
//--------------------------------------------------------------------------
void
Realm::create_restart_mesh()
{
  // exodus restart file creation
  if (outputInfo_->hasRestartBlock_) {

    if (outputInfo_->restartFreq_ == 0)
      return;

    restartFileIndex_ = ioBroker_->create_output_mesh(
      outputInfo_->restartDBName_, stk::io::WRITE_RESTART,
      *outputInfo_->restartPropertyManager_);

    // loop over restart variable field names supplied by Eqs
    for (std::set<std::string>::iterator itorSet =
           outputInfo_->restartFieldNameSet_.begin();
         itorSet != outputInfo_->restartFieldNameSet_.end(); ++itorSet) {
      std::string varName = *itorSet;
      stk::mesh::FieldBase* theField =
        stk::mesh::get_field_by_name(varName, meta_data());
      if (NULL == theField) {
        NaluEnv::self().naluOutputP0()
          << " Sorry, no field by the name " << varName << std::endl;
      } else {
        // add the field for a restart output
        ioBroker_->add_field(restartFileIndex_, *theField, varName);
        // if this is a restarted simulation, we will need input
        if (restarted_simulation())
          ioBroker_->add_input_field(stk::io::MeshField(*theField, varName));
      }
    }

    // now global params
    stk::util::ParameterMapType::const_iterator i = globalParameters_->begin();
    stk::util::ParameterMapType::const_iterator iend = globalParameters_->end();
    for (; i != iend; ++i) {
      std::string parameterName = (*i).first;
      stk::util::Parameter parameter = (*i).second;
      if (parameter.toRestartFile) {
        ioBroker_->add_global(restartFileIndex_, parameterName, parameter);
      }
    }

    // set max size for restart data base
    ioBroker_->get_output_io_region(restartFileIndex_)
      ->get_database()
      ->set_cycle_count(outputInfo_->restartMaxDataBaseStepSize_);
  }
}

//--------------------------------------------------------------------------
//-------- input_variables_from_mesh()
//--------------------------------------------
//--------------------------------------------------------------------------
void
Realm::input_variables_from_mesh()
{
  // no variables from an input mesh if this is a restart
  if (!restarted_simulation()) {

    // check whether to snap or interpolate data; all fields treated the same
    const stk::io::MeshField::TimeMatchOption fieldInterpOption =
      solutionOptions_->inputVariablesInterpolateInTime_
        ? stk::io::MeshField::LINEAR_INTERPOLATION
        : stk::io::MeshField::CLOSEST;

    // check for periodic cycling of data based on start time and periodic time;
    // scale time set to unity
    if (solutionOptions_->inputVariablesPeriodicTime_ > 0.0) {
      ioBroker_->get_mesh_database(inputMeshIdx_)
        .set_periodic_time(
          solutionOptions_->inputVariablesPeriodicTime_,
          solutionOptions_->inputVariablesRestorationTime_,
          stk::io::InputFile::CYCLIC)
        .set_scale_time(1.0);
    }

    std::map<std::string, std::string>::const_iterator iter;
    for (iter = solutionOptions_->inputVarFromFileMap_.begin();
         iter != solutionOptions_->inputVarFromFileMap_.end(); ++iter) {

      std::string varName = iter->first;
      std::string userName = iter->second;

      stk::mesh::FieldBase* theField =
        stk::mesh::get_field_by_name(varName, meta_data());
      if (NULL == theField) {
        NaluEnv::self().naluOutputP0()
          << " Sorry, no field by the name " << varName << std::endl;
      } else {
        ioBroker_->add_input_field(
          stk::io::MeshField(*theField, userName, fieldInterpOption));
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- augment_output_variable_list() ----------------------------------
//--------------------------------------------------------------------------
void
Realm::augment_output_variable_list(const std::string fieldName)
{
  outputInfo_->outputFieldNameSet_.insert(fieldName);
}

//--------------------------------------------------------------------------
//-------- augment_restart_variable_list -----------------------------------
//--------------------------------------------------------------------------
void
Realm::augment_restart_variable_list(std::string restartFieldName)
{
  outputInfo_->restartFieldNameSet_.insert(restartFieldName);
}

//--------------------------------------------------------------------------
//-------- create_edges -----------------------------------------------
//--------------------------------------------------------------------------
void
Realm::create_edges()
{
  NaluEnv::self().naluOutputP0()
    << "Realm::create_edges(): Nalu Realm: " << name_
    << " requires edge creation: Begin" << std::endl;

  static stk::diag::Timer timerCE_("CreateEdges", Simulation::rootTimer());
  stk::diag::TimeBlock tbCreateEdges_(timerCE_);

  double start_time = NaluEnv::self().nalu_time();
  stk::mesh::create_edges(*bulkData_, meta_data().universal_part(), edgesPart_);
  double stop_time = NaluEnv::self().nalu_time();

  // timer close-out
  const double total_edge_time = stop_time - start_time;
  timerCreateEdges_ += total_edge_time;
  NaluEnv::self().naluOutputP0()
    << "Realm::create_edges(): Nalu Realm: " << name_
    << " requires edge creation: End" << std::endl;
}

//--------------------------------------------------------------------------
//-------- provide_entity_count() ------------------------------------------
//--------------------------------------------------------------------------
void
Realm::provide_entity_count()
{

  std::vector<size_t> counts;
  std::vector<size_t> minCounts;
  std::vector<size_t> maxCounts;
  stk::mesh::comm_mesh_counts(*bulkData_, counts, minCounts, maxCounts);

  NaluEnv::self().naluOutputP0() << "===========================" << std::endl;
  NaluEnv::self().naluOutputP0()
    << "Realm::provide_entity_count:   " << std::endl
    << "nodes,    " << counts[0] << " min/max: " << minCounts[0] << "/"
    << maxCounts[0] << std::endl
    << "edges,    " << counts[1] << " min/max: " << minCounts[1] << "/"
    << maxCounts[1] << std::endl
    << "faces,    " << counts[2] << " min/max: " << minCounts[2] << "/"
    << maxCounts[2] << std::endl
    << "elements, " << counts[3] << " min/max: " << minCounts[3] << "/"
    << maxCounts[3] << std::endl;
  NaluEnv::self().naluOutputP0() << "===========================" << std::endl;
}

//--------------------------------------------------------------------------
//-------- delete_edges -----------------------------------------------
//--------------------------------------------------------------------------
void
Realm::delete_edges()
{
  if (debug()) {
    std::vector<size_t> counts;
    stk::mesh::comm_mesh_counts(*bulkData_, counts);

    NaluEnv::self().naluOutputP0()
      << "Realm::delete_edges: before delete_edges, mesh has  " << counts[0]
      << " nodes, " << counts[1] << " edges, " << counts[2] << " faces, "
      << counts[3] << " elements" << std::endl;
  }

  stk::mesh::BucketVector const& edge_buckets =
    bulkData_->get_buckets(stk::topology::EDGE_RANK, *edgesPart_);
  std::vector<stk::mesh::Entity> edges;
  stk::mesh::get_selected_entities(*edgesPart_, edge_buckets, edges);

  if (debug()) {
    size_t sz = edges.size(), g_sz = 0;
    stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &sz, &g_sz, 1);
    NaluEnv::self().naluOutputP0()
      << "P[" << bulkData_->parallel_rank()
      << "] Realm::delete_edges: edge list local size= " << sz
      << " global size= " << g_sz << std::endl;
  }

  // delete elem -> edge relations
  bulkData_->modification_begin();
  for (unsigned ii = 0; ii < edges.size(); ++ii) {
    while (true) {

      if (!bulkData_->is_valid(edges[ii]))
        throw std::runtime_error("bad edge 1");

      unsigned num_elems = bulkData_->num_elements(edges[ii]);

      if (!num_elems)
        break;

      stk::mesh::Entity const* const edge_elems =
        bulkData_->begin_elements(edges[ii]);
      stk::mesh::ConnectivityOrdinal const* edge_elem_ordinals =
        bulkData_->begin_element_ordinals(edges[ii]);

      stk::mesh::Entity to_rel = edge_elems[0];
      stk::mesh::RelationIdentifier to_id = edge_elem_ordinals[0];

      bool del = bulkData_->destroy_relation(to_rel, edges[ii], to_id);
      if (!del)
        throw std::runtime_error("delete_edges failed to delete up relation");
    }

    if (3 == meta_data().spatial_dimension()) {
      while (true) {

        if (!bulkData_->is_valid(edges[ii]))
          throw std::runtime_error("bad edge 1");

        unsigned num_faces = bulkData_->num_faces(edges[ii]);

        if (!num_faces)
          break;

        stk::mesh::Entity const* const edge_faces =
          bulkData_->begin_faces(edges[ii]);
        stk::mesh::ConnectivityOrdinal const* edge_face_ordinals =
          bulkData_->begin_face_ordinals(edges[ii]);

        stk::mesh::Entity to_rel = edge_faces[0];
        stk::mesh::RelationIdentifier to_id = edge_face_ordinals[0];

        bool del = bulkData_->destroy_relation(to_rel, edges[ii], to_id);
        if (!del)
          throw std::runtime_error(
            "delete_edges failed to delete up relation for face");
      }
    }
  }

  // now delete edges
  for (unsigned ii = 0; ii < edges.size(); ++ii) {

    if (
      bulkData_->is_valid(edges[ii]) && bulkData_->bucket(edges[ii]).owned()) {
      if (!bulkData_->destroy_entity(edges[ii])) {
        unsigned num_elems = bulkData_->num_elements(edges[ii]);
        NaluEnv::self().naluOutputP0()
          << "P[" << bulkData_->parallel_rank()
          << "] deleting edge num_elems= " << num_elems << std::endl;

        stk::mesh::EntityRank topRank = stk::topology::ELEMENT_RANK;
        for (stk::mesh::EntityRank irank = stk::topology::EDGE_RANK;
             irank <= topRank; ++irank) {
          unsigned nc = bulkData_->num_connectivity(edges[ii], irank);
          NaluEnv::self().naluOutputP0()
            << "P[" << bulkData_->parallel_rank() << "] deleting edge nc["
            << irank << "]= " << nc << std::endl;
        }

        throw std::runtime_error("delete_edges failed to delete edge");
      }
    }
  }
  bulkData_->modification_end();

  if (debug()) {
    std::vector<size_t> counts;
    stk::mesh::comm_mesh_counts(*bulkData_, counts);

    NaluEnv::self().naluOutputP0()
      << "P[" << bulkData_->parallel_rank() << "] "
      << "Realm::delete_edges: after delete_edges, mesh has  " << counts[0]
      << " nodes, " << counts[1] << " edges, " << counts[2] << " faces, "
      << counts[3] << " elements" << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- initialize_non_conformal ----------------------------------------
//--------------------------------------------------------------------------
void
Realm::initialize_non_conformal()
{
  nonConformalManager_->initialize();
}

//--------------------------------------------------------------------------
//-------- initialize_post_processing_algorithms ---------------------------
//--------------------------------------------------------------------------
void
Realm::initialize_post_processing_algorithms()
{
  // check for data probes
  if (NULL != dataProbePostProcessing_)
    dataProbePostProcessing_->initialize();

  if (NULL != ablForcingAlg_) {
    ablForcingAlg_->initialize();
  }

  if (aeroModels_->is_active())
    aeroModels_->init(bulk_data());

  if (lidarLOS_) {
    lidarLOS_->set_time_for_all(get_current_time());
  }
}

//--------------------------------------------------------------------------
//-------- get_coordinates_name ---------------------------------------------
//--------------------------------------------------------------------------
std::string
Realm::get_coordinates_name()
{
  return (
    (solutionOptions_->meshMotion_ ||
     solutionOptions_->externalMeshDeformation_)
      ? "current_coordinates"
      : "coordinates");
}

//--------------------------------------------------------------------------
//-------- has_mesh_motion -------------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::has_mesh_motion() const
{
  return solutionOptions_->meshMotion_;
}

//--------------------------------------------------------------------------
//-------- has_mesh_deformation --------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::has_mesh_deformation() const
{
  if (meshMotionAlg_) {
    return meshMotionAlg_->is_deforming() ||
           solutionOptions_->externalMeshDeformation_;
  } else
    return solutionOptions_->externalMeshDeformation_;
}

//--------------------------------------------------------------------------
//-------- does_mesh_move --------------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::does_mesh_move() const
{
  return has_mesh_motion() || has_mesh_deformation();
}

//--------------------------------------------------------------------------
//-------- has_non_matching_boundary_face_alg ------------------------------
//--------------------------------------------------------------------------
bool
Realm::has_non_matching_boundary_face_alg() const
{
  return hasNonConformal_ | hasOverset_;
}

//--------------------------------------------------------------------------
//-------- query_for_overset -----------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::query_for_overset()
{
  for (auto&& bc : boundaryConditions_) {
    switch (bc->theBcType_) {
    case OVERSET_BC:
      hasOverset_ = true;
      break;
    default:
      hasOverset_ = false;
    }
  }
  return hasOverset_;
}

//--------------------------------------------------------------------------
//-------- set_current_coordinates -----------------------------------------
//--------------------------------------------------------------------------
void
Realm::set_current_coordinates(stk::mesh::Part* targetPart)
{
  const int nDim = meta_data().spatial_dimension();

  VectorFieldType* modelCoords = meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* currentCoords = meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates");
  VectorFieldType* displacement = meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement");

  stk::mesh::Selector s_all_nodes = stk::mesh::Selector(*targetPart);

  stk::mesh::BucketVector const& node_buckets =
    bulkData_->get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();
    const double* mCoords = stk::mesh::field_data(*modelCoords, b);
    double* cCoords = stk::mesh::field_data(*currentCoords, b);
    const double* dx = stk::mesh::field_data(*displacement, b);
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const int offSet = k * nDim;
      for (int j = 0; j < nDim; ++j)
        cCoords[offSet + j] = mCoords[offSet + j] + dx[offSet + j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- compute_geometry ------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::compute_geometry()
{
  // interior and boundary
  geometryAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- compute_vrtm ----------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::compute_vrtm(const std::string& velName)
{
  if (
    !solutionOptions_->meshMotion_ &&
    !solutionOptions_->externalMeshDeformation_)
    return;

  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using MeshIndex = Traits::MeshIndex;

  const int nDim = meta_data().spatial_dimension();
  const auto& ngpMesh = ngp_mesh();
  const auto& fieldMgr = ngp_field_manager();
  const auto vel =
    fieldMgr.get_field<double>(get_field_ordinal(meta_data(), velName));
  const auto meshVel =
    fieldMgr.get_field<double>(get_field_ordinal(meta_data(), "mesh_velocity"));
  auto vrtm = fieldMgr.get_field<double>(
    get_field_ordinal(meta_data(), velName + "_rtm"));

  auto* vrtm_field = meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, velName + "_rtm");
  const stk::mesh::Selector sel =
    (meta_data().locally_owned_part() | meta_data().globally_shared_part()) &
    stk::mesh::selectField(*vrtm_field);
  nalu_ngp::run_entity_algorithm(
    "compute_vrtm", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      for (int d = 0; d < nDim; ++d)
        vrtm.get(mi, d) = vel.get(mi, d) - meshVel.get(mi, d);
    });

  vrtm.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- init_current_coordinates -----------------------------------------
//--------------------------------------------------------------------------
void
Realm::init_current_coordinates()
{
  const int nDim = meta_data().spatial_dimension();

  VectorFieldType* modelCoords = meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* currentCoords = meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates");
  VectorFieldType* displacement = meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement");

  stk::mesh::Selector s_all_nodes =
    (meta_data().locally_owned_part() | meta_data().globally_shared_part()) &
    stk::mesh::selectField(*currentCoords);

  stk::mesh::BucketVector const& node_buckets =
    bulkData_->get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();
    const double* mCoords = stk::mesh::field_data(*modelCoords, b);
    double* cCoords = stk::mesh::field_data(*currentCoords, b);
    double* dx = stk::mesh::field_data(*displacement, b);
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const int offSet = k * nDim;
      for (int j = 0; j < nDim; ++j) {
        dx[offSet + j] = 0.0; // RESTART...
        cCoords[offSet + j] = mCoords[offSet + j];
      }
    }
  }

  // sync fields to device
  currentCoords->modify_on_host();
  currentCoords->sync_to_device();

  displacement->modify_on_host();
  displacement->sync_to_device();
}

//--------------------------------------------------------------------------
//-------- compute_l2_scaling ----------------------------------------------
//--------------------------------------------------------------------------
void
Realm::compute_l2_scaling()
{
  // loop over all material propertys  and save off part vector
  stk::mesh::PartVector partVec;
  const std::vector<std::string> targetNames = get_physics_target_names();
  for (size_t itarget = 0; itarget < targetNames.size(); ++itarget) {
    // target need not be subsetted since nothing below will depend on topo
    stk::mesh::Part* targetPart = meta_data().get_part(targetNames[itarget]);
    partVec.push_back(targetPart);
  }

  size_t totalNodes = 0;

  // selector for all locally owned nodes
  stk::mesh::Selector s_locally_owned_union =
    meta_data().locally_owned_part() & stk::mesh::selectUnion(partVec);

  stk::mesh::BucketVector const& node_bucket =
    bulkData_->get_buckets(stk::topology::NODE_RANK, s_locally_owned_union);

  for (stk::mesh::BucketVector::const_iterator ib = node_bucket.begin();
       ib != node_bucket.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();
    totalNodes += length;
  }

  // Parallel assembly of total nodes
  size_t g_totalNodes = 0;
  stk::all_reduce_sum(
    NaluEnv::self().parallel_comm(), &totalNodes, &g_totalNodes, 1);

  l2Scaling_ = 1.0 / std::sqrt(g_totalNodes);
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
Realm::register_nodal_fields(stk::mesh::Part* part)
{
  // register high level common fields
  const int nDim = meta_data().spatial_dimension();

  // Declare volume/area_vector fields
  const int numVolStates = does_mesh_move() ? number_of_states() : 1;
  auto& dualNodalVol = meta_data().declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume", numVolStates);
  stk::mesh::put_field_on_mesh(dualNodalVol, *part, 1, nullptr);
  if (numVolStates > 1)
    augment_restart_variable_list("dual_nodal_volume");
  auto& elemVol = meta_data().declare_field<ScalarFieldType>(
    stk::topology::ELEM_RANK, "element_volume");
  stk::mesh::put_field_on_mesh(elemVol, *part, 1, nullptr);

  if (realmUsesEdges_) {
    auto& edgeAreaVec = meta_data().declare_field<VectorFieldType>(
      stk::topology::EDGE_RANK, "edge_area_vector");
    stk::mesh::put_field_on_mesh(
      edgeAreaVec, *part, meta_data().spatial_dimension(), nullptr);
  }

  // mesh motion/deformation is high level
  // clang-format off
  if ( does_mesh_move()) {
    VectorFieldType *displacement = &(meta_data().declare_field<VectorFieldType>(stk::topology::NODE_RANK, "mesh_displacement",numVolStates));
    stk::mesh::put_field_on_mesh(*displacement, *part, nDim, nullptr);
    augment_restart_variable_list("mesh_displacement");
    VectorFieldType *currentCoords = &(meta_data().declare_field<VectorFieldType>(stk::topology::NODE_RANK, "current_coordinates"));
    stk::mesh::put_field_on_mesh(*currentCoords, *part, nDim, nullptr);
    augment_restart_variable_list("current_coordinates");
    VectorFieldType *meshVelocity = &(meta_data().declare_field<VectorFieldType>(stk::topology::NODE_RANK, "mesh_velocity"));
    stk::mesh::put_field_on_mesh(*meshVelocity, *part, nDim, nullptr);
    augment_restart_variable_list("mesh_velocity");
    VectorFieldType *velocityRTM = &(meta_data().declare_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm"));
    stk::mesh::put_field_on_mesh(*velocityRTM, *part, nDim, nullptr);
    if(has_mesh_deformation()){
      ScalarFieldType *divV = &(meta_data().declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_velocity"));
      stk::mesh::put_field_on_mesh(*divV, *part, nullptr);
    }
  }
  // clang-format on

  ScalarIntFieldType& iblank = meta_data().declare_field<ScalarIntFieldType>(
    stk::topology::NODE_RANK, "iblank");
  stk::mesh::put_field_on_mesh(iblank, *part, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
Realm::register_interior_algorithm(stk::mesh::Part* part)
{
  if (matrixFree_) {
    return;
  }

  const AlgorithmType algType = INTERIOR;
  geometryAlgDriver_->register_elem_algorithm<GeometryInteriorAlg>(
    algType, part, "geometry");

  // Track parts that are registered to interior algorithms
  interiorPartVec_.push_back(part);
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::register_wall_bc(stk::mesh::Part* part, const stk::topology& theTopo)
{
  if (matrixFree_) {
    return;
  }

  //====================================================
  // Register face (boundary condition) data
  //====================================================

  // push back the part for book keeping and, later, skin mesh
  bcPartVec_.push_back(part);

  const int nDim = meta_data().spatial_dimension();

  // register fields
  MasterElement* meFC = MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsIp = meFC->num_integration_points();

  GenericFieldType* exposedAreaVec_ =
    &(meta_data().declare_field<GenericFieldType>(
      static_cast<stk::topology::rank_t>(meta_data().side_rank()),
      "exposed_area_vector"));
  stk::mesh::put_field_on_mesh(
    *exposedAreaVec_, *part, nDim * numScsIp, nullptr);

  const AlgorithmType algType = BOUNDARY;
  geometryAlgDriver_->register_face_algorithm<GeometryBoundaryAlg>(
    algType, part, "geometry");
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::register_inflow_bc(stk::mesh::Part* part, const stk::topology& theTopo)
{
  if (matrixFree_) {
    return;
  }

  //====================================================
  // Register face (boundary condition) data
  //====================================================

  // push back the part for book keeping and, later, skin mesh
  bcPartVec_.push_back(part);

  const int nDim = meta_data().spatial_dimension();

  // register fields
  MasterElement* meFC = MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsIp = meFC->num_integration_points();

  GenericFieldType* exposedAreaVec_ =
    &(meta_data().declare_field<GenericFieldType>(
      static_cast<stk::topology::rank_t>(meta_data().side_rank()),
      "exposed_area_vector"));
  stk::mesh::put_field_on_mesh(
    *exposedAreaVec_, *part, nDim * numScsIp, nullptr);

  const AlgorithmType algType = BOUNDARY;
  geometryAlgDriver_->register_face_algorithm<GeometryBoundaryAlg>(
    algType, part, "geometry");
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::register_open_bc(stk::mesh::Part* part, const stk::topology& theTopo)
{
  if (matrixFree_) {
    return;
  }

  //====================================================
  // Register face (boundary condition) data
  //====================================================

  // push back the part for book keeping and, later, skin mesh
  bcPartVec_.push_back(part);

  const int nDim = meta_data().spatial_dimension();

  // register fields
  MasterElement* meFC = MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsIp = meFC->num_integration_points();

  GenericFieldType* exposedAreaVec_ =
    &(meta_data().declare_field<GenericFieldType>(
      static_cast<stk::topology::rank_t>(meta_data().side_rank()),
      "exposed_area_vector"));
  stk::mesh::put_field_on_mesh(
    *exposedAreaVec_, *part, nDim * numScsIp, nullptr);

  const AlgorithmType algType = BOUNDARY;
  geometryAlgDriver_->register_face_algorithm<GeometryBoundaryAlg>(
    algType, part, "geometry");
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
Realm::register_symmetry_bc(stk::mesh::Part* part, const stk::topology& theTopo)
{
  if (matrixFree_) {
    return;
  }
  //====================================================
  // Register face (boundary condition) data
  //====================================================

  // push back the part for book keeping and, later, skin mesh
  bcPartVec_.push_back(part);

  const int nDim = meta_data().spatial_dimension();

  // register fields
  MasterElement* meFC = MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsIp = meFC->num_integration_points();

  GenericFieldType* exposedAreaVec_ =
    &(meta_data().declare_field<GenericFieldType>(
      static_cast<stk::topology::rank_t>(meta_data().side_rank()),
      "exposed_area_vector"));
  stk::mesh::put_field_on_mesh(
    *exposedAreaVec_, *part, nDim * numScsIp, nullptr);

  const AlgorithmType algType = BOUNDARY;
  geometryAlgDriver_->register_face_algorithm<GeometryBoundaryAlg>(
    algType, part, "geometry");
}

//--------------------------------------------------------------------------
//-------- register_periodic_bc --------------------------------------------
//--------------------------------------------------------------------------
void
Realm::register_periodic_bc(
  stk::mesh::Part* masterMeshPart,
  stk::mesh::Part* slaveMeshPart,
  const double& searchTolerance,
  const std::string& searchMethodName)
{
  allPeriodicInteractingParts_.push_back(masterMeshPart);
  allPeriodicInteractingParts_.push_back(slaveMeshPart);

  // push back the part for book keeping and, later, skin mesh
  bcPartVec_.push_back(masterMeshPart);
  bcPartVec_.push_back(slaveMeshPart);

  if (NULL == periodicManager_) {
    periodicManager_ = new PeriodicManager(*this);
    hasPeriodic_ = true;
  }

  // add the parts to the manager
  periodicManager_->add_periodic_pair(
    masterMeshPart, slaveMeshPart, searchTolerance, searchMethodName);
}

//--------------------------------------------------------------------------
//-------- setup_non_conformal_bc ------------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_non_conformal_bc(
  stk::mesh::PartVector currentPartVec,
  stk::mesh::PartVector opposingPartVec,
  const NonConformalBoundaryConditionData& nonConformalBCData)
{
  hasNonConformal_ = true;

  // create manager
  if (NULL == nonConformalManager_) {
    nonConformalManager_ = new NonConformalManager(
      *this, solutionOptions_->ncAlgDetailedOutput_,
      solutionOptions_->ncAlgCoincidentNodesErrorCheck_);
  }

  // create nonconformal info for this surface, extract user data
  NonConformalUserData userData = nonConformalBCData.userData_;

  NonConformalInfo* nonConformalInfo = new NonConformalInfo(
    *this, currentPartVec, opposingPartVec,
    userData.expandBoxPercentage_ / 100.0, userData.searchMethodName_,
    userData.clipIsoParametricCoords_, userData.searchTolerance_,
    userData.dynamicSearchTolAlg_, nonConformalBCData.targetName_);

  nonConformalManager_->nonConformalInfoVec_.push_back(nonConformalInfo);

  for (auto part : currentPartVec)
    allNonConformalInteractingParts_.push_back(part);
  for (auto part : opposingPartVec)
    allNonConformalInteractingParts_.push_back(part);
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
Realm::register_non_conformal_bc(
  stk::mesh::Part* part, const stk::topology& theTopo)
{
  ThrowRequire(!matrixFree_);

  // push back the part for book keeping and, later, skin mesh
  bcPartVec_.push_back(part);

  const int nDim = meta_data().spatial_dimension();
  // register fields
  MasterElement* meFC = MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsIp = meFC->num_integration_points();

  // exposed area vector
  GenericFieldType* exposedAreaVec_ =
    &(meta_data().declare_field<GenericFieldType>(
      static_cast<stk::topology::rank_t>(meta_data().side_rank()),
      "exposed_area_vector"));
  stk::mesh::put_field_on_mesh(
    *exposedAreaVec_, *part, nDim * numScsIp, nullptr);

  const AlgorithmType algType = BOUNDARY;
  geometryAlgDriver_->register_face_algorithm<GeometryBoundaryAlg>(
    algType, part, "geometry");
}

void
Realm::register_overset_bc()
{
  for (auto* superPart : oversetBCPartVec_)
    for (auto* part : superPart->subsets()) {
      const auto topo = part->topology();
      const int nDim = meta_data().spatial_dimension();
      // register fields
      MasterElement* meFC = MasterElementRepo::get_surface_master_element(topo);
      const int numScsIp = meFC->num_integration_points();

      // exposed area vector
      GenericFieldType* exposedAreaVec_ =
        &(meta_data().declare_field<GenericFieldType>(
          static_cast<stk::topology::rank_t>(meta_data().side_rank()),
          "exposed_area_vector"));
      stk::mesh::put_field_on_mesh(
        *exposedAreaVec_, *part, nDim * numScsIp, nullptr);

      const AlgorithmType algType = BOUNDARY;
      geometryAlgDriver_->register_face_algorithm<GeometryBoundaryAlg>(
        algType, part, "geometry");
    }
}

//--------------------------------------------------------------------------
//-------- setup_overset_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_overset_bc(const OversetBoundaryConditionData& oversetBCData)
{
  // setting flag for linear system setup (may have been set via earlier
  // "query")
  hasOverset_ = true;

  // create manager while providing overset data
  if (NULL == oversetManager_) {
    switch (oversetBCData.oversetConnectivityType_) {
    case OversetBoundaryConditionData::TPL_TIOGA:
#ifdef NALU_USES_TIOGA
      oversetManager_ = new OversetManagerTIOGA(*this, oversetBCData.userData_);
      NaluEnv::self().naluOutputP0() << "Realm::setup_overset_bc:: Selecting "
                                        "TIOGA TPL for overset connectivity"
                                     << std::endl;
      break;
#else
      // should not get here... we should have thrown error in input file
      // processing stage
      throw std::runtime_error(
        "TIOGA TPL support not enabled during compilation phase");
#endif

    default:
      throw std::runtime_error("Invalid setting for overset connectivity");
    }
  }

  oversetManager_->setup();
  register_overset_bc();
}

//--------------------------------------------------------------------------
//-------- periodic_field_update -------------------------------------------
//--------------------------------------------------------------------------
void
Realm::periodic_field_update(
  stk::mesh::FieldBase* theField,
  const unsigned& sizeOfField,
  const bool& bypassFieldCheck) const
{
  const bool addSlaves = true;
  const bool setSlaves = true;
  periodicManager_->apply_constraints(
    theField, sizeOfField, bypassFieldCheck, addSlaves, setSlaves);
}

void
Realm::periodic_field_max(
  stk::mesh::FieldBase* theField, const unsigned& sizeOfField) const
{
  periodicManager_->apply_max_field(theField, sizeOfField);
}

//--------------------------------------------------------------------------
//-------- periodic_delta_solution_update
//-------------------------------------------
//--------------------------------------------------------------------------
void
Realm::periodic_delta_solution_update(
  stk::mesh::FieldBase* theField,
  const unsigned& sizeOfField,
  const bool& doCommunication) const
{
  const bool bypassFieldCheck = true;
  const bool addSlaves = false;
  const bool setSlaves = true;
  periodicManager_->ngp_apply_constraints(
    theField, sizeOfField, bypassFieldCheck, addSlaves, setSlaves,
    doCommunication);
}

//--------------------------------------------------------------------------
//-------- periodic_max_field_update ---------------------------------------
//--------------------------------------------------------------------------
void
Realm::periodic_max_field_update(
  stk::mesh::FieldBase* theField, const unsigned& sizeOfField) const
{
  periodicManager_->apply_max_field(theField, sizeOfField);
}

//--------------------------------------------------------------------------
//-------- get_slave_part_vector -------------------------------------------
//--------------------------------------------------------------------------
const stk::mesh::PartVector&
Realm::get_slave_part_vector()
{
  if (hasPeriodic_)
    return periodicManager_->get_slave_part_vector();
  else
    return emptyPartVector_;
}

void
Realm::overset_field_update(
  stk::mesh::FieldBase* field,
  const unsigned nRows,
  const unsigned nCols,
  const bool doFinalSyncToDevice)
{
  if (!hasOverset_ || isExternalOverset_)
    return;

  const double timeA = NaluEnv::self().nalu_time();
  oversetManager_->overset_update_field(
    field, nRows, nCols, doFinalSyncToDevice);
  const double timeB = NaluEnv::self().nalu_time();
  oversetManager_->timerFieldUpdate_ += (timeB - timeA);
}

//--------------------------------------------------------------------------
//-------- provide_output --------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::provide_output()
{
  stk::diag::TimeBlock mesh_output_timeblock(Simulation::outputTimer());
  const double start_time = NaluEnv::self().nalu_time();
  const double currentTime = get_current_time();
  const int timeStepCount = get_time_step_count();
  sideWriters_->write_sides(timeStepCount, currentTime);

  if (outputInfo_->hasOutputBlock_) {

    if (outputInfo_->outputFreq_ == 0)
      return;

    // process output via io
    const int modStep = timeStepCount - outputInfo_->outputStart_;

    // check for elapsed WALL time threshold
    bool forcedOutput = false;
    if (outputInfo_->userWallTimeResults_.first) {
      const double elapsedWallTime = stk::wall_time() - wallTimeStart_;
      // find the max over all core
      double g_elapsedWallTime = 0.0;
      stk::all_reduce_max(
        NaluEnv::self().parallel_comm(), &elapsedWallTime, &g_elapsedWallTime,
        1);
      // convert to hours
      g_elapsedWallTime /= 3600.0;
      // only force output the first time the timer is exceeded
      if (g_elapsedWallTime > outputInfo_->userWallTimeResults_.second) {
        forcedOutput = true;
        outputInfo_->userWallTimeResults_.first = false;
        NaluEnv::self().naluOutputP0()
          << "Realm::provide_output()::Forced Result output will be processed "
             "at current time: "
          << currentTime << std::endl;
        NaluEnv::self().naluOutputP0()
          << " Elapsed (max) WALL time: " << g_elapsedWallTime << " (hours)"
          << std::endl;
        // provide timer information
        dump_simulation_time();
      }
    }

    const bool isOutput = (timeStepCount >= outputInfo_->outputStart_ &&
                           modStep % outputInfo_->outputFreq_ == 0) ||
                          forcedOutput;

    if (isOutput) {
      NaluEnv::self().naluOutputP0()
        << "Realm shall provide output files at : currentTime/timeStepCount: "
        << currentTime << "/" << timeStepCount << " (" << name_ << ")"
        << std::endl;

      // not set up for globals
      if (!doPromotion_) {
        // Sync fields to host on NGP builds before output
        for (auto* fld : meta_data().get_fields()) {
          fld->sync_to_host();
        }

        ioBroker_->process_output_request(resultsFileIndex_, currentTime);
      } else {
        for (auto& stringFieldPair : promotionIO_->get_output_fields()) {
          auto& field = *stringFieldPair.second;
          if (field.type_is<double>()) {
            stk::mesh::get_updated_ngp_field<double>(field).sync_to_host();
          } else if (field.type_is<int>()) {
            stk::mesh::get_updated_ngp_field<int>(field).sync_to_host();
          }
        }
        promotionIO_->write_database_data(currentTime);
      }
      equationSystems_.provide_output();
    }

    const double stop_time = NaluEnv::self().nalu_time();

    // increment time for output
    timerOutputFields_ += (stop_time - start_time);
  }
}

//--------------------------------------------------------------------------
//-------- provide_restart_output ------------------------------------------
//--------------------------------------------------------------------------
void
Realm::provide_restart_output()
{
  stk::diag::TimeBlock mesh_output_timeblock(Simulation::outputTimer());

  if (outputInfo_->hasRestartBlock_) {

    if (outputInfo_->restartFreq_ == 0)
      return;

    const double start_time = NaluEnv::self().nalu_time();

    // process restart via io
    const double currentTime = get_current_time();
    const int timeStepCount = get_time_step_count();
    const int modStep = timeStepCount - outputInfo_->restartStart_;

    // check for elapsed WALL time threshold
    bool forcedOutput = false;
    if (outputInfo_->userWallTimeRestart_.first) {
      const double elapsedWallTime = stk::wall_time() - wallTimeStart_;
      // find the max over all core
      double g_elapsedWallTime = 0.0;
      stk::all_reduce_max(
        NaluEnv::self().parallel_comm(), &elapsedWallTime, &g_elapsedWallTime,
        1);
      // convert to hours
      g_elapsedWallTime /= 3600.0;
      // only force output the first time the timer is exceeded
      if (g_elapsedWallTime > outputInfo_->userWallTimeRestart_.second) {
        forcedOutput = true;
        outputInfo_->userWallTimeRestart_.first = false;
        NaluEnv::self().naluOutputP0()
          << "Realm::provide_restart_output()::Forced Restart output will be "
             "processed at current time: "
          << currentTime << std::endl;
        NaluEnv::self().naluOutputP0()
          << " Elapsed (max) WALL time: " << g_elapsedWallTime << " (hours)"
          << std::endl;
      }
    }

    const bool isRestartOutputStep =
      (timeStepCount >= outputInfo_->restartStart_ &&
       modStep % outputInfo_->restartFreq_ == 0) ||
      forcedOutput;

    if (isRestartOutputStep) {
      NaluEnv::self().naluOutputP0()
        << "Realm shall provide restart files at: currentTime/timeStepCount: "
        << currentTime << "/" << timeStepCount << " (" << name_ << ")"
        << std::endl;
      // handle fields
      ioBroker_->begin_output_step(restartFileIndex_, currentTime);
      ioBroker_->write_defined_output_fields(restartFileIndex_);

      // push global variables for time step
      const double timeStepNm1 = timeIntegrator_->get_time_step();
      globalParameters_->set_value("timeStepNm1", timeStepNm1);
      globalParameters_->set_value("timeStepCount", timeStepCount);

      if (NULL != turbulenceAveragingPostProcessing_) {
        globalParameters_->set_value(
          "currentTimeFilter",
          turbulenceAveragingPostProcessing_->currentTimeFilter_);
      }

      stk::util::ParameterMapType::const_iterator i =
        globalParameters_->begin();
      stk::util::ParameterMapType::const_iterator iend =
        globalParameters_->end();
      for (; i != iend; ++i) {
        std::string parameterName = (*i).first;
        stk::util::Parameter parameter = (*i).second;
        if (parameter.toRestartFile) {
          ioBroker_->write_global(restartFileIndex_, parameterName, parameter);
        }
      }

      ioBroker_->end_output_step(restartFileIndex_);
    }

    const double stop_time = NaluEnv::self().nalu_time();

    // increment time for output
    timerOutputFields_ += (stop_time - start_time);
  }
}

//--------------------------------------------------------------------------
//-------- swap_states -----------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::swap_states()
{
  bulkData_->update_field_data_states();

#if defined(KOKKOS_ENABLE_GPU)
  if (get_time_step_count() < 2)
    return;

  const auto& fieldMgr = ngp_field_manager();
  for (const auto fld : meta_data().get_fields()) {
    const unsigned numStates = fld->number_of_states();
    const auto fieldID = fld->mesh_meta_data_ordinal();
    const auto fieldNp1ID =
      fld->field_state(stk::mesh::StateNP1)->mesh_meta_data_ordinal();

    if ((numStates < 2) || (fieldID != fieldNp1ID))
      continue;

    for (unsigned i = (numStates - 1); i > 0; --i) {
      auto& toField = fieldMgr.get_field<double>(
        fld->field_state(static_cast<stk::mesh::FieldState>(i))
          ->mesh_meta_data_ordinal());
      auto& fromField = fieldMgr.get_field<double>(
        fld->field_state(static_cast<stk::mesh::FieldState>(i - 1))
          ->mesh_meta_data_ordinal());

      toField.swap(fromField);
    }
  }
#endif
}

//--------------------------------------------------------------------------
//-------- predict_state ---------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::predict_state()
{
  equationSystems_.predict_state();
}

//--------------------------------------------------------------------------
//-------- populate_initial_condition --------------------------------------
//--------------------------------------------------------------------------
void
Realm::populate_initial_condition()
{
  for (size_t k = 0; k < initCondAlg_.size(); ++k) {
    initCondAlg_[k]->execute();
  }
}

//--------------------------------------------------------------------------
//-------- boundary_data_to_state_data -------------------------------------
//--------------------------------------------------------------------------
void
Realm::boundary_data_to_state_data()
{
  equationSystems_.boundary_data_to_state_data();
}

//--------------------------------------------------------------------------
//-------- populate_restart ------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::populate_restart(double& timeStepNm1, int& timeStepCount)
{
  double foundRestartTime = get_current_time();
  if (restarted_simulation()) {
    // allow restart to skip missed required fields
    const double restartTime = outputInfo_->restartTime_;
    std::vector<stk::io::MeshField> missingFields;
    foundRestartTime =
      ioBroker_->read_defined_input_fields(restartTime, &missingFields);

    {
      for (const auto& fname : outputInfo_->restartFieldNameSet_) {
        auto* field = stk::mesh::get_field_by_name(fname, meta_data());
        if (field == nullptr)
          continue;

        const unsigned numStates = field->number_of_states();
        for (unsigned i = 0; i < numStates; ++i) {
          auto* fld = field->field_state(static_cast<stk::mesh::FieldState>(i));
          fld->modify_on_host();
          ngp_field_manager().get_field<double>(fld->mesh_meta_data_ordinal());
          fld->sync_to_device();
        }
      }
    }

    if (missingFields.size() > 0) {
      for (size_t k = 0; k < missingFields.size(); ++k) {
        NaluEnv::self().naluOutputP0()
          << "WARNING: Restart value for Field "
          << missingFields[k].field()->name()
          << " is missing; may default to IC specification" << std::endl;
      }
      if (!supportInconsistentRestart_) {
        NaluEnv::self().naluOutputP0()
          << "The user may desire to set the "
             "support_inconsistent_multi_state_restart Realm line command"
          << std::endl;
        NaluEnv::self().naluOutputP0()
          << "This is applicable for a BDF2 restart run from a previously run "
             "Backward Euler simulation"
          << std::endl;
      }
    }
    NaluEnv::self().naluOutputP0()
      << "Realm::populate_restart() candidate restart time: "
      << foundRestartTime << " for Realm: " << name() << std::endl;

    // extract time parameters; okay if they are missing; no need to let the
    // user know
    const bool abortIfNotFound = false;
    ioBroker_->get_global("timeStepNm1", timeStepNm1, abortIfNotFound);
    ioBroker_->get_global("timeStepCount", timeStepCount, abortIfNotFound);
    if (NULL != turbulenceAveragingPostProcessing_) {
      ioBroker_->get_global(
        "currentTimeFilter",
        turbulenceAveragingPostProcessing_->currentTimeFilter_,
        abortIfNotFound);
    }
    if (meshMotionAlg_) {
      // Redo all the mesh and motionAlg setup after reading files from the
      // restart reset the current_coordinate and mesh_velocity fields after
      // reading them
      init_current_coordinates();
      // reset the current time for the meshMotionAlgs
      meshMotionAlg_->restart_reinit(foundRestartTime);
      compute_geometry();
      meshMotionAlg_->post_compute_geometry();
    }
  }
  return foundRestartTime;
}

//--------------------------------------------------------------------------
//-------- populate_variables_from_input -----------------------------------
//--------------------------------------------------------------------------
double
Realm::populate_variables_from_input(const double currentTime)
{
  // no reading fields from mesh if this is a restart
  double foundTime = currentTime;
  if (
    !restarted_simulation() &&
    solutionOptions_->inputVarFromFileMap_.size() > 0) {
    std::vector<stk::io::MeshField> missingFields;
    foundTime = ioBroker_->read_defined_input_fields(
      solutionOptions_->inputVariablesRestorationTime_, &missingFields);
    if (missingFields.size() > 0) {
      for (size_t k = 0; k < missingFields.size(); ++k) {
        NaluEnv::self().naluOutputP0()
          << "WARNING: Realm::populate_variables_from_input for field "
          << missingFields[k].field()->name()
          << " is missing; will default to IC specification" << std::endl;
      }
    }
    NaluEnv::self().naluOutputP0()
      << "Realm::populate_variables_form_input() candidate input time: "
      << foundTime << " for Realm: " << name() << std::endl;
  }
  return foundTime;
}

//--------------------------------------------------------------------------
//-------- populate_derived_quantities -------------------------------------
//--------------------------------------------------------------------------
void
Realm::populate_derived_quantities()
{
  equationSystems_.populate_derived_quantities();
}

//--------------------------------------------------------------------------
//-------- initial_work -----------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::initial_work()
{
  // include initial condition in averaging postprocessor
  if (turbulenceAveragingPostProcessing_ != nullptr) {
    turbulenceAveragingPostProcessing_->execute();
  }

  if (bdyLayerStats_ != nullptr) {
    bdyLayerStats_->execute();
  }

  equationSystems_.initial_work();
}

//--------------------------------------------------------------------------
//-------- set_global_id ---------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::set_global_id()
{
  const stk::mesh::Selector s_universal = meta_data().universal_part();
  stk::mesh::BucketVector const& buckets =
    bulkData_->get_buckets(stk::topology::NODE_RANK, s_universal);

  for (stk::mesh::BucketVector::const_iterator ib = buckets.begin();
       ib != buckets.end(); ++ib) {
    const stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();
    stk::mesh::EntityId* naluGlobalIds =
      stk::mesh::field_data(*naluGlobalId_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      naluGlobalIds[k] = bulkData_->identifier(b[k]);
    }
  }
}

void
Realm::set_hypre_global_id()
{
#ifdef NALU_USES_HYPRE
  /* Create a mapping of Nalu Global ID (nodes) to Hypre Global ID.
   *
   * Background: Hypre requires a contiguous mapping of row IDs for its IJMatrix
   * and IJVector data structure, i.e., the startID(iproc+1) = endID(iproc) + 1.
   * Therefore, this method first determines the total number of rows in each
   * paritition and then determines the starting and ending IDs for the Hypre
   * matrix and finally assigns the hypre ID for all the nodes on this partition
   * in the hypreGlobalId_ field.
   */

  // Fill with an invalid value for future error checking
  stk::mesh::field_fill(
    std::numeric_limits<HypreIntType>::max(), *hypreGlobalId_);

  const stk::mesh::Selector s_local =
    meta_data().locally_owned_part() & !get_inactive_selector();
  const auto& bkts = bulkData_->get_buckets(stk::topology::NODE_RANK, s_local);

  size_t num_nodes = 0;
  int nprocs = bulkData_->parallel_size();
  int iproc = bulkData_->parallel_rank();
  std::vector<int> nodesPerProc(nprocs);
  // std::vector<stk::mesh::EntityId> hypreOffsets(nprocs+1);
  hypreOffsets_.resize(nprocs + 1);

  // 1. Determine the number of nodes per partition and determine appropriate
  // offsets on each MPI rank.
  for (auto b : bkts)
    num_nodes += b->size();

  MPI_Allgather(
    &num_nodes, 1, MPI_INT, nodesPerProc.data(), 1, MPI_INT,
    bulkData_->parallel());

  hypreOffsets_[0] = 0;
  for (int i = 1; i <= nprocs; i++)
    hypreOffsets_[i] = hypreOffsets_[i - 1] + nodesPerProc[i - 1];

  // These are set up for NDOF=1, the actual lower/upper extents will be
  // finalized in HypreLinearSystem class based on the equation being solved.
  hypreILower_ = hypreOffsets_[iproc];
  hypreIUpper_ = hypreOffsets_[iproc + 1];
  hypreNumNodes_ = hypreOffsets_[nprocs];

  // 2. Sort the local STK IDs so that we retain a 1-1 mapping as much as
  // possible
  size_t ii = 0;
  std::vector<stk::mesh::EntityId> localIDs(num_nodes);
  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];
      auto nid = bulkData_->identifier(node);
      localIDs[ii++] = nid;
    }
  }
  std::sort(localIDs.begin(), localIDs.end());

  // 3. Store Hypre global IDs for all the nodes so that this can be used to
  // lookup and populate Hypre data structures.
  HypreIntType nidx = static_cast<HypreIntType>(hypreILower_);
  for (auto nid : localIDs) {
    auto node = bulkData_->get_entity(stk::topology::NODE_RANK, nid);
    HypreIntType* hids = stk::mesh::field_data(*hypreGlobalId_, node);
    *hids = nidx++;
  }

  auto& bulk = bulk_data();
  std::vector<const stk::mesh::FieldBase*> fVec{hypreGlobalId_};

  stk::mesh::copy_owned_to_shared(bulk, fVec);
  stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);

  if (
    oversetManager_ != nullptr && oversetManager_->oversetGhosting_ != nullptr)
    stk::mesh::communicate_field_data(*oversetManager_->oversetGhosting_, fVec);

  if (
    nonConformalManager_ != nullptr &&
    nonConformalManager_->nonConformalGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *nonConformalManager_->nonConformalGhosting_, fVec);

  if (
    periodicManager_ != nullptr &&
    periodicManager_->periodicGhosting_ != nullptr) {
    periodicManager_->parallel_communicate_field(hypreGlobalId_);
    periodicManager_->periodic_parallel_communicate_field(hypreGlobalId_);
  }
#endif
}

//--------------------------------------------------------------------------
//-------- populate_boundary_data ------------------------------------------
//--------------------------------------------------------------------------
void
Realm::populate_boundary_data()
{
  // realm first
  for (size_t k = 0; k < bcDataAlg_.size(); ++k) {
    bcDataAlg_[k]->execute();
  }
  equationSystems_.populate_boundary_data();
}

//--------------------------------------------------------------------------
//-------- output_banner ---------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::output_banner()
{
  if (hasFluids_)
    NaluEnv::self().naluOutputP0()
      << " Max Courant: " << maxCourant_ << " Max Reynolds: " << maxReynolds_
      << " (" << name_ << ")" << std::endl;
}

//--------------------------------------------------------------------------
//-------- check_job -------------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::check_job(bool get_node_count)
{
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0()
    << "Realm memory Review:       " << name_ << std::endl;
  NaluEnv::self().naluOutputP0() << "===========================" << std::endl;

  // set number of nodes, check job run size
  if (get_node_count) {
    size_t localNodeCount =
      ioBroker_->get_input_io_region()->get_property("node_count").get_int();
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &localNodeCount, &nodeCount_, 1);
    NaluEnv::self().naluOutputP0()
      << "Node count from meta data = " << nodeCount_ << std::endl;

    if (doPromotion_) {
      if (meta_data().is_commit()) {
        std::vector<size_t> counts;
        stk::mesh::comm_mesh_counts(*bulkData_, counts);
        nodeCount_ = counts[0];
        NaluEnv::self().naluOutputP0()
          << "Node count after promotion = " << nodeCount_ << std::endl;
      } else {
        nodeCount_ = std::pow(promotionOrder_, spatialDimension_) * nodeCount_;
        NaluEnv::self().naluOutputP0()
          << "(Roughly) Estimated node count after promotion = " << nodeCount_
          << std::endl;
      }
    }
  }

  /// estimate memory based on N*bandwidth, N = nodeCount*nDOF,
  ///   bandwidth = NCon(=27 for Hex mesh)*nDOF - we are very conservative here
  unsigned BWFactor = 27;
  if (doPromotion_) {
    // Ignore boundary terms and assume a structured mesh
    unsigned cornerBWFactor =
      std::pow((2 * promotionOrder_ + 1), spatialDimension_);
    unsigned edgeBWFactor =
      std::pow((2 * promotionOrder_ + 1), spatialDimension_ - 1) *
      (promotionOrder_ + 1);
    unsigned faceBWFactor = (2 * promotionOrder_ + 1) * (promotionOrder_ + 1) *
                            (promotionOrder_ + 1); // only 3D
    unsigned interiorBWFactor =
      std::pow(promotionOrder_ + 1, spatialDimension_);

    unsigned numCornerNodes = (spatialDimension_ == 3) ? 8 : 4;
    unsigned numEdgeNodes = (spatialDimension_ == 3)
                              ? 12 * (promotionOrder_ - 1)
                              : 4 * (promotionOrder_ - 1);
    unsigned numFaceNodes =
      (spatialDimension_ == 3) ? 6 * std::pow(promotionOrder_ - 1, 2) : 0;
    unsigned numInteriorNodes =
      std::pow(promotionOrder_ - 1, spatialDimension_);
    unsigned numNodes = std::pow(promotionOrder_ + 1, spatialDimension_);

    BWFactor =
      (cornerBWFactor * numCornerNodes + edgeBWFactor * numEdgeNodes +
       faceBWFactor * numFaceNodes + interiorBWFactor * numInteriorNodes) /
      numNodes;
  }
  const unsigned MatrixStorageFactor =
    3; // for CRS storage, need one A_IJ, and one I and one J, approx
  SizeType memoryEstimate = 0;
  double procGBScale =
    double(NaluEnv::self().parallel_size()) * (1024. * 1024. * 1024.);
  for (unsigned ieq = 0; ieq < equationSystems_.size(); ++ieq) {
    if (!equationSystems_[ieq]->linsys_)
      continue;
    SizeType numDof = equationSystems_[ieq]->linsys_->numDof();
    SizeType N = nodeCount_ * numDof;
    SizeType bandwidth = BWFactor * numDof;
    memoryEstimate += MatrixStorageFactor * N * bandwidth * sizeof(double);
  }
  NaluEnv::self().naluOutputP0()
    << "Total memory estimate for Matrix solve (per core)= "
    << double(memoryEstimate) / procGBScale << " GB." << std::endl;

  SizeType memoryEstimateFields = 0;
  if (meta_data().is_commit()) {
    std::vector<size_t> counts;
    stk::mesh::comm_mesh_counts(*bulkData_, counts);
    ThrowRequire(counts.size() >= 4);
    size_t nodeCount = counts[stk::topology::NODE_RANK];
    size_t edgeCount = counts[stk::topology::EDGE_RANK];
    size_t faceCount = counts[stk::topology::FACE_RANK];
    size_t elemCount = counts[stk::topology::ELEM_RANK];

    const stk::mesh::FieldVector& fields = meta_data().get_fields();
    unsigned nfields = fields.size();
    for (unsigned ifld = 0; ifld < nfields; ++ifld) {
      stk::mesh::FieldBase* field = fields[ifld];
      unsigned fszNode = field->max_size(stk::topology::NODE_RANK);
      unsigned fszEdge = field->max_size(stk::topology::EDGE_RANK);
      unsigned fszFace = field->max_size(stk::topology::FACE_RANK);
      unsigned fszElem = field->max_size(stk::topology::ELEM_RANK);

      memoryEstimateFields += (nodeCount * fszNode + edgeCount * fszEdge +
                               faceCount * fszFace + elemCount * fszElem) *
                              sizeof(double);
    }
    NaluEnv::self().naluOutputP0()
      << "Total memory estimate for Fields (per core)= "
      << double(memoryEstimateFields) / procGBScale << " GB." << std::endl;
    memoryEstimate += memoryEstimateFields;
  }

  NaluEnv::self().naluOutputP0()
    << "Total memory estimate (per core) = "
    << double(memoryEstimate) / procGBScale << " GB." << std::endl;

  if (meta_data().is_commit() && estimateMemoryOnly_) {
    throw std::runtime_error(
      "Job requested memory estimate only, shutting down");
  }

  // here's where we can check for estimated memory > given available memory
  if (
    availableMemoryPerCoreGB_ != 0 &&
    double(memoryEstimate) / procGBScale > availableMemoryPerCoreGB_) {
    NaluEnv::self().naluOutputP0()
      << "ERROR: property available_memory_per_core_GB is set (= "
      << availableMemoryPerCoreGB_
      << ") and estimated memory (= " << double(memoryEstimate) / procGBScale
      << ") is greater,\n job too large to run, \naborting..." << std::endl;
    throw std::runtime_error("Job shutting down");
  }
}

//--------------------------------------------------------------------------
//-------- dump_simulation_time --------------------------------------------
//--------------------------------------------------------------------------
void
Realm::dump_simulation_time()
{
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0()
    << "-------------------------------- " << std::endl;
  NaluEnv::self().naluOutputP0()
    << "Begin Timer Overview for Realm: " << name_ << std::endl;
  NaluEnv::self().naluOutputP0()
    << "-------------------------------- " << std::endl;

  // equation system time
  equationSystems_.dump_eq_time();

  const int nprocs = NaluEnv::self().parallel_size();

  // common
  const unsigned ntimers = 6;
  double total_time[ntimers] = {timerCreateMesh_,    timerOutputFields_,
                                timerInitializeEqs_, timerPropertyEval_,
                                timerPopulateMesh_,  timerPopulateFieldData_};
  double g_min_time[ntimers] = {}, g_max_time[ntimers] = {},
         g_total_time[ntimers] = {};

  // get min, max and sum over processes
  stk::all_reduce_min(
    NaluEnv::self().parallel_comm(), &total_time[0], &g_min_time[0], ntimers);
  stk::all_reduce_max(
    NaluEnv::self().parallel_comm(), &total_time[0], &g_max_time[0], ntimers);
  stk::all_reduce_sum(
    NaluEnv::self().parallel_comm(), &total_time[0], &g_total_time[0], ntimers);

  NaluEnv::self().naluOutputP0() << "Timing for IO: " << std::endl;
  NaluEnv::self().naluOutputP0()
    << "   io create mesh --  "
    << " \tavg: " << g_total_time[0] / double(nprocs)
    << " \tmin: " << g_min_time[0] << " \tmax: " << g_max_time[0] << std::endl;
  NaluEnv::self().naluOutputP0()
    << " io output fields --  "
    << " \tavg: " << g_total_time[1] / double(nprocs)
    << " \tmin: " << g_min_time[1] << " \tmax: " << g_max_time[1] << std::endl;
  NaluEnv::self().naluOutputP0()
    << " io populate mesh --  "
    << " \tavg: " << g_total_time[4] / double(nprocs)
    << " \tmin: " << g_min_time[4] << " \tmax: " << g_max_time[4] << std::endl;
  NaluEnv::self().naluOutputP0()
    << " io populate fd   --  "
    << " \tavg: " << g_total_time[5] / double(nprocs)
    << " \tmin: " << g_min_time[5] << " \tmax: " << g_max_time[5] << std::endl;
  NaluEnv::self().naluOutputP0()
    << "Timing for property evaluation:         " << std::endl;
  NaluEnv::self().naluOutputP0()
    << "            props --  "
    << " \tavg: " << g_total_time[3] / double(nprocs)
    << " \tmin: " << g_min_time[3] << " \tmax: " << g_max_time[3] << std::endl;

  // now edge creation; if applicable
  if (realmUsesEdges_) {
    double g_total_edge = 0.0, g_min_edge = 0.0, g_max_edge = 0.0;
    stk::all_reduce_min(
      NaluEnv::self().parallel_comm(), &timerCreateEdges_, &g_min_edge, 1);
    stk::all_reduce_max(
      NaluEnv::self().parallel_comm(), &timerCreateEdges_, &g_max_edge, 1);
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &timerCreateEdges_, &g_total_edge, 1);

    NaluEnv::self().naluOutputP0() << "Timing for Edge: " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "    edge creation --  "
      << " \tavg: " << g_total_edge / double(nprocs) << " \tmin: " << g_min_edge
      << " \tmax: " << g_max_edge << std::endl;
  }

  // periodic
  if (hasPeriodic_) {
    double periodicSearchTime = periodicManager_->get_search_time();
    double g_minPeriodicSearchTime = 0.0, g_maxPeriodicSearchTime = 0.0,
           g_periodicSearchTime = 0.0;
    stk::all_reduce_min(
      NaluEnv::self().parallel_comm(), &periodicSearchTime,
      &g_minPeriodicSearchTime, 1);
    stk::all_reduce_max(
      NaluEnv::self().parallel_comm(), &periodicSearchTime,
      &g_maxPeriodicSearchTime, 1);
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &periodicSearchTime,
      &g_periodicSearchTime, 1);

    NaluEnv::self().naluOutputP0() << "Timing for Periodic: " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "           search --  "
      << " \tavg: " << g_periodicSearchTime / double(nprocs)
      << " \tmin: " << g_minPeriodicSearchTime
      << " \tmax: " << g_maxPeriodicSearchTime << std::endl;
  }

  // nonconformal or overset
  if (hasNonConformal_) {
    double g_totalNonconformal = 0.0, g_minNonconformal = 0.0,
           g_maxNonconformal = 0.0;
    stk::all_reduce_min(
      NaluEnv::self().parallel_comm(), &timerNonconformal_, &g_minNonconformal,
      1);
    stk::all_reduce_max(
      NaluEnv::self().parallel_comm(), &timerNonconformal_, &g_maxNonconformal,
      1);
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &timerNonconformal_,
      &g_totalNonconformal, 1);

    NaluEnv::self().naluOutputP0() << "Timing for Nonconformal: " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "  nonconformal bc --  "
      << " \tavg: " << g_totalNonconformal / double(nprocs)
      << " \tmin: " << g_minNonconformal << " \tmax: " << g_maxNonconformal
      << std::endl;
  }

  if (hasOverset_) {
    double connTime[2] = {
      oversetManager_->timerConnectivity_, oversetManager_->timerFieldUpdate_};
    double totTime[2], minTime[2], maxTime[2];
    stk::all_reduce_sum(NaluEnv::self().parallel_comm(), connTime, totTime, 2);
    stk::all_reduce_min(NaluEnv::self().parallel_comm(), connTime, minTime, 2);
    stk::all_reduce_max(NaluEnv::self().parallel_comm(), connTime, maxTime, 2);
    NaluEnv::self().naluOutputP0()
      << "Timing for Overset:" << std::endl
      << "     connectivity --  \tavg: " << totTime[0] / double(nprocs)
      << " \tmin: " << minTime[0] << " \tmax: " << maxTime[0] << std::endl
      << "     field update --  \tavg: " << totTime[1] / double(nprocs)
      << " \tmin: " << minTime[1] << " \tmax: " << maxTime[1] << std::endl;
  }

  // transfer
  if (
    hasMultiPhysicsTransfer_ || hasInitializationTransfer_ || hasIoTransfer_ ||
    hasExternalDataTransfer_) {
    double totalXfer[2] = {timerTransferSearch_, timerTransferExecute_};
    double g_totalXfer[2] = {}, g_minXfer[2] = {}, g_maxXfer[2] = {};
    stk::all_reduce_min(
      NaluEnv::self().parallel_comm(), &totalXfer[0], &g_minXfer[0], 2);
    stk::all_reduce_max(
      NaluEnv::self().parallel_comm(), &totalXfer[0], &g_maxXfer[0], 2);
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &totalXfer[0], &g_totalXfer[0], 2);

    NaluEnv::self().naluOutputP0()
      << "Timing for Tranfer (fromRealm):    " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "           search --  "
      << " \tavg: " << g_totalXfer[0] / double(nprocs)
      << " \tmin: " << g_minXfer[0] << " \tmax: " << g_maxXfer[0] << std::endl;
    NaluEnv::self().naluOutputP0()
      << "          execute --  "
      << " \tavg: " << g_totalXfer[1] / double(nprocs)
      << " \tmin: " << g_minXfer[1] << " \tmax: " << g_maxXfer[1] << std::endl;
  }

  // skin mesh
  if (checkForMissingBcs_ || hasOverset_) {
    double g_totalSkin = 0.0, g_minSkin = 0.0, g_maxSkin = 0.0;
    stk::all_reduce_min(
      NaluEnv::self().parallel_comm(), &timerSkinMesh_, &g_minSkin, 1);
    stk::all_reduce_max(
      NaluEnv::self().parallel_comm(), &timerSkinMesh_, &g_maxSkin, 1);
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &timerSkinMesh_, &g_totalSkin, 1);

    NaluEnv::self().naluOutputP0() << "Timing for skin_mesh :    " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "        skin_mesh --  "
      << " \tavg: " << g_totalSkin / double(nprocs) << " \tmin: " << g_minSkin
      << " \tmax: " << g_maxSkin << std::endl;
  }

  // promotion
  if (doPromotion_) {
    double g_totalPromote = 0.0, g_minPromote = 0.0, g_maxPromote = 0.0;
    stk::all_reduce_min(
      NaluEnv::self().parallel_comm(), &timerPromoteMesh_, &g_minPromote, 1);
    stk::all_reduce_max(
      NaluEnv::self().parallel_comm(), &timerPromoteMesh_, &g_maxPromote, 1);
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &timerPromoteMesh_, &g_totalPromote, 1);

    NaluEnv::self().naluOutputP0()
      << "Timing for promote_mesh :    " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "        promote_mesh --  "
      << " \tavg: " << g_totalPromote / double(nprocs)
      << " \tmin: " << g_minPromote << " \tmax: " << g_maxPromote << std::endl;
  }

  if (timerActuator_ > 0) {
    double g_totalActuator = 0.0, g_minActuator = 0.0, g_maxActuator = 0.0;
    stk::all_reduce_min(
      NaluEnv::self().parallel_comm(), &timerActuator_, &g_minActuator, 1);
    stk::all_reduce_max(
      NaluEnv::self().parallel_comm(), &timerActuator_, &g_maxActuator, 1);
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &timerActuator_, &g_totalActuator, 1);

    NaluEnv::self().naluOutputP0() << "Timing for actuator :    " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "        actuator::execute --  "
      << " \tavg: " << g_totalActuator / double(nprocs)
      << " \tmin: " << g_minActuator << " \tmax: " << g_maxActuator
      << std::endl;
  }

  // consolidated sort
  if (solutionOptions_->useConsolidatedSolverAlg_) {
    double g_totalSort = 0.0, g_minSort = 0.0, g_maxSort = 0.0;
    stk::all_reduce_min(
      NaluEnv::self().parallel_comm(), &timerSortExposedFace_, &g_minSort, 1);
    stk::all_reduce_max(
      NaluEnv::self().parallel_comm(), &timerSortExposedFace_, &g_maxSort, 1);
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), &timerSortExposedFace_, &g_totalSort, 1);

    NaluEnv::self().naluOutputP0() << "Timing for sort_mesh: " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "       sort_mesh  -- "
      << " \tavg: " << g_totalSort / double(nprocs) << " \tmin: " << g_minSort
      << " \tmax: " << g_maxSort << std::endl;
  }

  NaluEnv::self().naluOutputP0() << std::endl;
}

//--------------------------------------------------------------------------
//-------- provide_mean_norm -----------------------------------------------
//--------------------------------------------------------------------------
double
Realm::provide_mean_norm()
{
  return equationSystems_.provide_mean_system_norm();
}

//--------------------------------------------------------------------------
//-------- get_hybrid_factor -----------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_hybrid_factor(const std::string dofName)
{
  double factor = solutionOptions_->hybridDefault_;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->hybridMap_.find(dofName);
  if (iter != solutionOptions_->hybridMap_.end()) {
    factor = (*iter).second;
  }
  return factor;
}

//--------------------------------------------------------------------------
//-------- get_alpha_factor ------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_alpha_factor(const std::string dofName)
{
  double factor = solutionOptions_->alphaDefault_;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->alphaMap_.find(dofName);
  if (iter != solutionOptions_->alphaMap_.end()) {
    factor = (*iter).second;
  }
  return factor;
}

//--------------------------------------------------------------------------
//-------- get_alpha_upw_factor --------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_alpha_upw_factor(const std::string dofName)
{
  double factor = solutionOptions_->alphaUpwDefault_;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->alphaUpwMap_.find(dofName);
  if (iter != solutionOptions_->alphaUpwMap_.end()) {
    factor = (*iter).second;
  }
  return factor;
}

//--------------------------------------------------------------------------
//-------- get_upw_factor ------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_upw_factor(const std::string dofName)
{
  double factor = solutionOptions_->upwDefault_;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->upwMap_.find(dofName);
  if (iter != solutionOptions_->upwMap_.end()) {
    factor = (*iter).second;
  }
  return factor;
}

//--------------------------------------------------------------------------
//-------- primitive_uses_limiter ------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::primitive_uses_limiter(const std::string dofName)
{
  bool usesIt = false;
  std::map<std::string, bool>::const_iterator iter =
    solutionOptions_->limiterMap_.find(dofName);
  if (iter != solutionOptions_->limiterMap_.end()) {
    usesIt = (*iter).second;
  }
  return usesIt;
}

//--------------------------------------------------------------------------
//-------- get_lam_schmidt -------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_lam_schmidt(const std::string dofName)
{
  double factor = solutionOptions_->lamScDefault_;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->lamScMap_.find(dofName);
  if (iter != solutionOptions_->lamScMap_.end()) {
    factor = (*iter).second;
  }
  return factor;
}

//--------------------------------------------------------------------------
//-------- get_lam_prandtl -------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_lam_prandtl(const std::string dofName, bool& prProvided)
{
  double factor = 1.0;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->lamPrMap_.find(dofName);
  if (iter != solutionOptions_->lamPrMap_.end()) {
    factor = (*iter).second;
    prProvided = true;
  } else {
    prProvided = false;
  }
  return factor;
}

//--------------------------------------------------------------------------
//-------- get_turb_schmidt ------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_turb_schmidt(const std::string dofName)
{
  double factor = solutionOptions_->turbScDefault_;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->turbScMap_.find(dofName);
  if (iter != solutionOptions_->turbScMap_.end()) {
    factor = (*iter).second;
  }
  return factor;
}

//--------------------------------------------------------------------------
//-------- get_turb_prandtl ------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_turb_prandtl(const std::string dofName)
{
  double factor = solutionOptions_->turbPrDefault_;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->turbPrMap_.find(dofName);
  if (iter != solutionOptions_->turbPrMap_.end()) {
    factor = (*iter).second;
  }
  return factor;
}

//--------------------------------------------------------------------------
//-------- get_noc_usage ---------------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_noc_usage(const std::string dofName)
{
  bool factor = solutionOptions_->get_noc_usage(dofName);
  return factor;
}

//--------------------------------------------------------------------------
//-------- get_shifted_grad_op ---------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_shifted_grad_op(const std::string dofName)
{
  bool factor = solutionOptions_->shiftedGradOpDefault_;
  std::map<std::string, bool>::const_iterator iter =
    solutionOptions_->shiftedGradOpMap_.find(dofName);
  if (iter != solutionOptions_->shiftedGradOpMap_.end()) {
    factor = (*iter).second;
  }
  return factor;
}

//--------------------------------------------------------------------------
//-------- get_skew_symmetric ----------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_skew_symmetric(const std::string dofName)
{
  return solutionOptions_->get_skew_symmetric(dofName);
}

//--------------------------------------------------------------------------
//-------- get_tanh_functional_form ----------------------------------------
//--------------------------------------------------------------------------
std::string
Realm::get_tanh_functional_form(const std::string dofName)
{
  std::string tanhForm = solutionOptions_->tanhFormDefault_;
  std::map<std::string, std::string>::const_iterator iter =
    solutionOptions_->tanhFormMap_.find(dofName);
  if (iter != solutionOptions_->tanhFormMap_.end()) {
    tanhForm = (*iter).second;
  }
  return tanhForm;
}

//--------------------------------------------------------------------------
//-------- get_tanh_trans --------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_tanh_trans(const std::string dofName)
{
  double tanhTrans = solutionOptions_->tanhTransDefault_;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->tanhTransMap_.find(dofName);
  if (iter != solutionOptions_->tanhTransMap_.end()) {
    tanhTrans = (*iter).second;
  }
  return tanhTrans;
}

//--------------------------------------------------------------------------
//-------- get_tanh_width --------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_tanh_width(const std::string dofName)
{
  double tanhWidth = solutionOptions_->tanhWidthDefault_;
  std::map<std::string, double>::const_iterator iter =
    solutionOptions_->tanhWidthMap_.find(dofName);
  if (iter != solutionOptions_->tanhWidthMap_.end()) {
    tanhWidth = (*iter).second;
  }
  return tanhWidth;
}

//--------------------------------------------------------------------------
//-------- get_consistent_mass_matrix_png ----------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_consistent_mass_matrix_png(const std::string dofName)
{
  bool cmmPng = solutionOptions_->consistentMMPngDefault_;
  std::map<std::string, bool>::const_iterator iter =
    solutionOptions_->consistentMassMatrixPngMap_.find(dofName);
  if (iter != solutionOptions_->consistentMassMatrixPngMap_.end()) {
    cmmPng = (*iter).second;
  }
  return cmmPng;
}

//--------------------------------------------------------------------------
//-------- get_divU --------------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_divU()
{
  return solutionOptions_->includeDivU_;
}

//--------------------------------------------------------------------------
//-------- get_mdot_interp -------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_mdot_interp()
{
  return solutionOptions_->mdotInterpRhoUTogether_ ? 1.0 : 0.0;
}

//--------------------------------------------------------------------------
//-------- get_cvfem_shifted_mdot ------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_cvfem_shifted_mdot()
{
  return solutionOptions_->cvfemShiftMdot_;
}

//--------------------------------------------------------------------------
//-------- get_cvfem_reduced_sens_poisson
//---------------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_cvfem_reduced_sens_poisson()
{
  return solutionOptions_->cvfemReducedSensPoisson_;
}

//--------------------------------------------------------------------------
//-------- has_nc_gauss_labatto_quadrature ---------------------------------
//--------------------------------------------------------------------------
bool
Realm::has_nc_gauss_labatto_quadrature()
{
  return solutionOptions_->ncAlgGaussLabatto_;
}

//--------------------------------------------------------------------------
//-------- get_nc_alg_upwind_advection -------------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_nc_alg_upwind_advection()
{
  return solutionOptions_->ncAlgUpwindAdvection_;
}

//--------------------------------------------------------------------------
//-------- get_nc_alg_include_pstab ----------------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_nc_alg_include_pstab()
{
  return solutionOptions_->ncAlgIncludePstab_;
}

//--------------------------------------------------------------------------
//-------- get_nc_alg_current_normal ---------------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_nc_alg_current_normal()
{
  return solutionOptions_->ncAlgCurrentNormal_;
}

//--------------------------------------------------------------------------
//-------- get_material_prop_eval ------------------------------------------
//--------------------------------------------------------------------------
PropertyEvaluator*
Realm::get_material_prop_eval(const PropertyIdentifier thePropID)
{
  PropertyEvaluator* thePropEval = NULL;
  std::map<PropertyIdentifier, PropertyEvaluator*>::const_iterator iter =
    materialPropertys_.propertyEvalMap_.find(thePropID);
  if (iter != materialPropertys_.propertyEvalMap_.end()) {
    thePropEval = (*iter).second;
  }
  return thePropEval;
}

//--------------------------------------------------------------------------
//-------- is_turbulent ----------------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::is_turbulent()
{
  return solutionOptions_->isTurbulent_;
}

//--------------------------------------------------------------------------
//-------- is_turbulent ----------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::is_turbulent(bool isIt)
{
  isTurbulent_ = isIt;
  solutionOptions_->isTurbulent_ = isIt;
}

//--------------------------------------------------------------------------
//-------- needs_enthalpy --------------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::needs_enthalpy()
{
  return needsEnthalpy_;
}

//--------------------------------------------------------------------------
//-------- needs_enthalpy --------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::needs_enthalpy(bool needsEnthalpy)
{
  needsEnthalpy_ = needsEnthalpy;
}

//--------------------------------------------------------------------------
//-------- number_of_states ------------------------------------------------
//--------------------------------------------------------------------------
int
Realm::number_of_states()
{
  const int numStates = (timeIntegrator_->secondOrderTimeAccurate_) ? 3 : 2;
  return numStates;
}

//--------------------------------------------------------------------------
//-------- name ------------------------------------------------------------
//--------------------------------------------------------------------------
std::string
Realm::name()
{
  return name_;
}

//--------------------------------------------------------------------------
//-------- augment_transfer_vector -----------------------------------------
//--------------------------------------------------------------------------
void
Realm::augment_transfer_vector(
  Transfer* transfer, const std::string transferObjective, Realm* toRealm)
{
  if (transferObjective == "multi_physics") {
    multiPhysicsTransferVec_.push_back(transfer);
    hasMultiPhysicsTransfer_ = true;
  } else if (transferObjective == "initialization") {
    initializationTransferVec_.push_back(transfer);
    hasInitializationTransfer_ = true;
  } else if (transferObjective == "input_output") {
    toRealm->ioTransferVec_.push_back(transfer);
    toRealm->hasIoTransfer_ = true;
  } else if (transferObjective == "external_data") {
    toRealm->externalDataTransferVec_.push_back(transfer);
    toRealm->hasExternalDataTransfer_ = true;
  } else {
    throw std::runtime_error(
      "Real::augment_transfer_vector: Error, none supported transfer "
      "objective: " +
      transferObjective);
  }
}

//--------------------------------------------------------------------------
//-------- process_multi_physics_transfer ----------------------------------
//--------------------------------------------------------------------------
void
Realm::process_multi_physics_transfer()
{
  if (!hasMultiPhysicsTransfer_)
    return;

  double timeXfer = -NaluEnv::self().nalu_time();
  std::vector<Transfer*>::iterator ii;
  for (ii = multiPhysicsTransferVec_.begin();
       ii != multiPhysicsTransferVec_.end(); ++ii)
    (*ii)->execute();
  timeXfer += NaluEnv::self().nalu_time();
  timerTransferExecute_ += timeXfer;
}

//--------------------------------------------------------------------------
//-------- process_initialization_transfer ---------------------------------
//--------------------------------------------------------------------------
void
Realm::process_initialization_transfer()
{
  if (!hasInitializationTransfer_)
    return;

  double timeXfer = -NaluEnv::self().nalu_time();
  std::vector<Transfer*>::iterator ii;
  for (ii = initializationTransferVec_.begin();
       ii != initializationTransferVec_.end(); ++ii) {
    (*ii)->execute();
  }
  timeXfer += NaluEnv::self().nalu_time();
  timerTransferExecute_ += timeXfer;
}

//--------------------------------------------------------------------------
//-------- process_io_transfer ------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::process_io_transfer()
{
  if (!hasIoTransfer_)
    return;

  double timeXfer = -NaluEnv::self().nalu_time();
  // only do at an IO step
  const int timeStepCount = get_time_step_count();
  const bool isOutput = (timeStepCount % outputInfo_->outputFreq_) == 0;
  if (isOutput) {
    std::vector<Transfer*>::iterator ii;
    for (ii = ioTransferVec_.begin(); ii != ioTransferVec_.end(); ++ii)
      (*ii)->execute();
  }
  timeXfer += NaluEnv::self().nalu_time();
  timerTransferExecute_ += timeXfer;
}

//--------------------------------------------------------------------------
//-------- process_external_data_transfer ----------------------------------
//--------------------------------------------------------------------------
void
Realm::process_external_data_transfer()
{
  if (!hasExternalDataTransfer_)
    return;

  double timeXfer = -NaluEnv::self().nalu_time();
  std::vector<Transfer*>::iterator ii;
  for (ii = externalDataTransferVec_.begin();
       ii != externalDataTransferVec_.end(); ++ii)
    (*ii)->execute();

  equationSystems_.post_external_data_transfer_work();
  timeXfer += NaluEnv::self().nalu_time();
  timerTransferExecute_ += timeXfer;
}

void
Realm::output_lidar()
{
  NaluEnv::self().naluOutputP0()
    << "LidarLineOfSite::output begin" << std::endl;

  const auto& velocity_field =
    *meta_data().get_field(stk::topology::NODE_RANK, "velocity");

  const auto sel = (stk::mesh::selectField(velocity_field) &
                    meta_data().locally_owned_part()) -
                   get_inactive_selector();
  lidarLOS_->output(
    bulk_data(), sel, get_coordinates_name(), timeIntegrator_->get_time_step(),
    timeIntegrator_->get_current_time());
  NaluEnv::self().naluOutputP0() << "LidarLineOfSite::output end" << std::endl;
}

//--------------------------------------------------------------------------
//-------- post_converged_work ---------------------------------------------
//--------------------------------------------------------------------------
void
Realm::post_converged_work()
{
  equationSystems_.post_converged_work();

  // FIXME: Consider a unified collection of post processing work
  if (NULL != solutionNormPostProcessing_)
    solutionNormPostProcessing_->execute();

  if (NULL != turbulenceAveragingPostProcessing_)
    turbulenceAveragingPostProcessing_->execute();

  if (NULL != dataProbePostProcessing_) {
    dataProbePostProcessing_->execute();
  }

  if (nullptr != bdyLayerStats_)
    bdyLayerStats_->execute();

  if (lidarLOS_) {
    output_lidar();
  }
}

//--------------------------------------------------------------------------
//-------- setup_element_promotion() ---------------------------------------
//--------------------------------------------------------------------------
void
Realm::setup_element_promotion()
{
  // Create a description of the element and deal with the part naming styles

  // Struct containing information about the element (e.g. number of nodes,
  // nodes per face, etc.)
  HexNElementDescription desc(promotionOrder_);

  // Every mesh part is promoted for now
  basePartVector_ = meta_data().get_mesh_parts();

  // Create new parts if not restarted
  // otherwise, super element parts are read from the restart file
  // However, the super face / edge parts are not and must be re-created
  for (const auto& targetName : materialPropertys_.targetNames_) {
    auto* basePart = meta_data().get_part(targetName);

    if (basePart->topology().rank() == stk::topology::ELEM_RANK) {
      const auto superName = super_element_part_name(targetName);

      // declare the part then set the topology.  Change to declaring the part
      // with topology when STK fixes declare_part_with_topology to work with
      // super elements
      stk::mesh::Part* superPart;
      if (!restarted_simulation()) {
        if (meta_data().get_part(superName) != nullptr) {
          throw std::runtime_error(
            "A part with name " + superName +
            " already exists in the mesh.  "
            "This can happen if a restart mesh was used but a restart_time was "
            "not specified");
        }

        superPart = &meta_data().declare_part_with_topology(
          superName, stk::create_superelement_topology(
                       static_cast<unsigned>(desc.nodesPerElement)));
        stk::io::put_io_part_attribute(*superPart);
      } else {
        superPart = meta_data().get_part(superName);
        if (superPart == nullptr) {
          throw std::runtime_error(
            "A restart was requested with promotion, "
            "but the promoted mesh parts are not in the restart file.");
        }
      }
      superPartVector_.push_back(superPart);
      superTargetNames_.push_back(superName);
    }
  }

  // always create side-ranked super parts
  for (auto* targetPart : basePartVector_) {
    if (!targetPart->subsets().empty()) {
      auto sideRank = meta_data().side_rank();
      auto* superSuperset = &meta_data().declare_part(
        super_element_part_name(targetPart->name()), sideRank);
      for (const auto* subset : targetPart->subsets()) {
        if (subset->topology().rank() == sideRank) {
          unsigned nodesPerSide = desc.nodesPerSide;
          auto sideTopo = (meta_data().spatial_dimension() == 2)
                            ? stk::create_superedge_topology(nodesPerSide)
                            : stk::create_superface_topology(nodesPerSide);

          // parts are named like "surface_se_super_super_1"
          auto partName = super_subset_part_name(subset->name());
          stk::mesh::Part* superFacePart =
            &meta_data().declare_part_with_topology(partName, sideTopo);
          superPartVector_.push_back(superFacePart);
          meta_data().declare_part_subset(*superSuperset, *superFacePart);
        }
      }
    }
  }
  meta_data().declare_part("edge_part", stk::topology::EDGE_RANK);
}

//--------------------------------------------------------------------------
//-------- promote_element -------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::promote_mesh()
{
  NaluEnv::self().naluOutputP0()
    << "Realm::promote_elements() Begin " << std::endl;
  auto timeA = stk::wall_time();

  auto& coords = *meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  if (!restarted_simulation()) {
    const auto gllNodes =
      gauss_lobatto_legendre_rule(promotionOrder_ + 1).first;
    promotion::create_tensor_product_hex_elements(
      gllNodes, *bulkData_, coords, basePartVector_);
  } else {
    promotion::create_promoted_boundary_elements(
      promotionOrder_, *bulkData_, basePartVector_);
  }

  auto timeB = stk::wall_time();
  timerPromoteMesh_ = timeB - timeA;
  NaluEnv::self().naluOutputP0()
    << "Realm::promote_elements() End " << std::endl;
}

//--------------------------------------------------------------------------
//-------- create_promoted_output_mesh -------------------------------------
//--------------------------------------------------------------------------
void
Realm::create_promoted_output_mesh()
{
  NaluEnv::self().naluOutputP0()
    << "Realm::create_promoted_output_mesh() Begin " << std::endl;

  if (outputInfo_->hasOutputBlock_) {
    if (outputInfo_->outputFreq_ == 0) {
      return;
    }

    auto* coords = meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates");
    promotionIO_ = std::make_unique<PromotedElementIO>(
      promotionOrder_, meta_data(), *bulkData_, meta_data().get_mesh_parts(),
      outputInfo_->outputDBName_, *coords);

    std::vector<stk::mesh::FieldBase*> outputFields;
    for (const auto& varName : outputInfo_->outputFieldNameSet_) {
      outputFields.push_back(
        stk::mesh::get_field_by_name(varName, meta_data()));
    }
    promotionIO_->add_fields(outputFields);
  }
  NaluEnv::self().naluOutputP0()
    << "Realm::create_promoted_output_mesh() End " << std::endl;
}

//--------------------------------------------------------------------------
//-------- part_name(std::string) ----------------------------------------------
//--------------------------------------------------------------------------
std::string
Realm::physics_part_name(std::string name) const
{
  if (doPromotion_) {
    return super_element_part_name(name);
  }
  return name;
}

std::vector<std::string>
Realm::physics_part_names(std::vector<std::string> names) const
{
  names = handle_all_element_part_alias(names);
  if (doPromotion_) {
    std::transform(
      names.begin(), names.end(), names.begin(),
      [&](const std::string& name) { return super_element_part_name(name); });
  }
  return names;
}

//--------------------------------------------------------------------------
//-------- get_current_time() ----------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_current_time()
{
  return timeIntegrator_->get_current_time();
}

//--------------------------------------------------------------------------
//-------- get_time_step() ----------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_time_step()
{
  return timeIntegrator_->get_time_step();
}

double
Realm::get_time_step_from_file()
{
  return timeIntegrator_->get_time_step_from_file();
}

bool
Realm::get_is_fixed_time_step()
{
  return timeIntegrator_->get_is_fixed_time_step();
}

bool
Realm::get_is_terminate_based_on_time()
{
  return timeIntegrator_->get_is_terminate_based_on_time();
}

double
Realm::get_total_sim_time()
{
  return timeIntegrator_->get_total_sim_time();
}

int
Realm::get_max_time_step_count()
{
  return timeIntegrator_->get_max_time_step_count();
}

//--------------------------------------------------------------------------
//-------- get_gamma1() ----------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_gamma1()
{
  return timeIntegrator_->get_gamma1();
}

//--------------------------------------------------------------------------
//-------- get_gamma2() ----------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_gamma2()
{
  return timeIntegrator_->get_gamma2();
}

//--------------------------------------------------------------------------
//-------- get_gamma3() ----------------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_gamma3()
{
  return timeIntegrator_->get_gamma3();
}

//--------------------------------------------------------------------------
//-------- get_time_step_count() ----------------------------------------------
//--------------------------------------------------------------------------
int
Realm::get_time_step_count() const
{
  return timeIntegrator_->get_time_step_count();
}

//--------------------------------------------------------------------------
//-------- restarted_simulation() ------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::restarted_simulation()
{
  return outputInfo_->activateRestart_;
}

//--------------------------------------------------------------------------
//-------- support_inconsistent_restart() ----------------------------------
//--------------------------------------------------------------------------
bool
Realm::support_inconsistent_restart()
{
  return supportInconsistentRestart_;
}

//--------------------------------------------------------------------------
//-------- get_stefan_boltzmann() ------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_stefan_boltzmann()
{
  return solutionOptions_->stefanBoltzmann_;
}

//--------------------------------------------------------------------------
//-------- get_turb_model_constant() ---------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_turb_model_constant(const TurbulenceModelConstant turbModelEnum)
{
  std::map<TurbulenceModelConstant, double>::iterator it =
    solutionOptions_->turbModelConstantMap_.find(turbModelEnum);
  if (it != solutionOptions_->turbModelConstantMap_.end()) {
    return it->second;
  } else {
    throw std::runtime_error("unknown (not found) turbulence model constant");
  }
}

//--------------------------------------------------------------------------
//-------- get_turbulence_model() ------------------------------------------
//--------------------------------------------------------------------------
TurbulenceModel
Realm::get_turbulence_model() const
{
  return solutionOptions_->turbulenceModel_;
}

//--------------------------------------------------------------------------
//-------- get_buckets() ---------------------------------------------------
//--------------------------------------------------------------------------
stk::mesh::BucketVector const&
Realm::get_buckets(
  stk::mesh::EntityRank rank, const stk::mesh::Selector& selector) const
{
  return bulkData_->get_buckets(rank, selector);
}

//--------------------------------------------------------------------------
//-------- bulk_data() -----------------------------------------------------
//--------------------------------------------------------------------------
stk::mesh::BulkData&
Realm::bulk_data()
{
  return *bulkData_;
}

const stk::mesh::BulkData&
Realm::bulk_data() const
{
  return *bulkData_;
}

//--------------------------------------------------------------------------
//-------- meta_data() -----------------------------------------------------
//--------------------------------------------------------------------------
stk::mesh::MetaData&
Realm::meta_data()
{
  return bulkData_->mesh_meta_data();
}

const stk::mesh::MetaData&
Realm::meta_data() const
{
  return bulkData_->mesh_meta_data();
}

//--------------------------------------------------------------------------
//-------- get_activate_aura()
//-----------------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::get_activate_aura()
{
  return activateAura_;
}

/** Return a selector containing inactive parts
 *
 *  The selector returned from this method will contain entities from
 *  parts that are do not participate in the PDE solution process, but
 *  are created/used for pre and post-processing purposes. Examples include:
 *  data probes, inactive sub-blocks from overset simulations after hole
 *  cut, etc.
 *
 *  \return stk::mesh::Selector Inactive entities
 */
stk::mesh::Selector
Realm::get_inactive_selector()
{
  // Return early if matrix free is active, nothing to do
  if (matrixFree_) {
    return stk::mesh::Selector{};
  }

  // provide inactive Overset part that excludes background surface
  //
  // Treat this selector differently because certain entities from interior
  // blocks could have been inactivated by the overset algorithm.
  stk::mesh::Selector nothing;
  stk::mesh::Selector inactiveOverSetSelector =
    (hasOverset_) ? oversetManager_->get_inactive_selector() : nothing;

  stk::mesh::Selector inactiveDPSel =
    (dataProbePostProcessing_ != nullptr)
      ? dataProbePostProcessing_->get_inactive_selector()
      : nothing;

  stk::mesh::Selector otherInactiveSelector =
    (meta_data().universal_part() &
     !(stk::mesh::selectUnion(interiorPartVec_)) &
     !(stk::mesh::selectUnion(bcPartVec_)));

  if (interiorPartVec_.empty() && bcPartVec_.empty()) {
    otherInactiveSelector = nothing;
  }

  return inactiveOverSetSelector | otherInactiveSelector | inactiveDPSel;
}

//--------------------------------------------------------------------------
//-------- push_equation_to_systems() --------------------------------------
//--------------------------------------------------------------------------
void
Realm::push_equation_to_systems(EquationSystem* eqSystem)
{
  equationSystems_.equationSystemVector_.push_back(eqSystem);
}

//--------------------------------------------------------------------------
//-------- get_physics_target_names() --------------------------------------
//--------------------------------------------------------------------------
const std::vector<std::string>&
Realm::get_physics_target_names()
{
  // in the future, possibly check for more advanced names;
  // for now, material props holds this'
  if (doPromotion_) {
    return superTargetNames_;
  }
  return materialPropertys_.targetNames_;
}

//--------------------------------------------------------------------------
//-------- get_tanh_blending() ---------------------------------------------
//--------------------------------------------------------------------------
double
Realm::get_tanh_blending(const std::string dofName)
{
  // assumes interval starts at a = 0 and ends at b = 1
  double omegaBlend = 1.0;
  if (get_tanh_functional_form(dofName) == "tanh") {
    const double c1 = get_tanh_trans(dofName);
    const double c2 = get_tanh_width(dofName);
    TanhFunction<double> tanhFunction(c1, c2);
    const double currentTime = get_current_time();
    omegaBlend = tanhFunction.execute(currentTime);
  }
  return omegaBlend;
}

//--------------------------------------------------------------------------
//-------- rebalance_mesh() ------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::rebalance_mesh()
{
#ifndef HAVE_ZOLTAN2_PARMETIS
  if (rebalanceMethod_ == "parmetis")
    throw std::runtime_error(
      "Zoltan2 is not built with parmetis enabled, "
      "try a geometric balance method instead (rcb or rib)");
#endif
  stk::balance::GraphCreationSettings rebalanceSettings;
  rebalanceSettings.setDecompMethod(rebalanceMethod_);
  stk::balance::balanceStkMesh(rebalanceSettings, *bulkData_);
}

//--------------------------------------------------------------------------
//-------- balance_nodes() -------------------------------------------------
//--------------------------------------------------------------------------
void
Realm::balance_nodes()
{
  stk::balance::GraphCreationSettings nodeBalanceSettings;
  nodeBalanceSettings.setUseNodeBalancer(true);
  nodeBalanceSettings.setNodeBalancerTargetLoadBalance(
    balanceNodeOptions_.target);
  nodeBalanceSettings.setNodeBalancerMaxIterations(
    balanceNodeOptions_.numIters);
  stk::balance::balanceStkMeshNodes(nodeBalanceSettings, *bulkData_);
}

std::vector<std::string>
Realm::handle_all_element_part_alias(
  const std::vector<std::string>& names) const
{
  if (names.size() == 1u && names.front() == allElementPartAlias) {
    std::vector<std::string> new_names;
    for (const auto* part : meta_data().get_mesh_parts()) {
      ThrowRequire(part);
      if (part->topology().rank() == stk::topology::ELEMENT_RANK) {
        new_names.push_back(part->name());
      }
    }
    return new_names;
  }

  if (
    std::find(names.begin(), names.end(), allElementPartAlias) != names.end()) {
    NaluEnv::self().naluOutputP0()
      << "Part alias " << allElementPartAlias << " present with other parts; "
      << allElementPartAlias << " must be a valid mesh part" << std::endl;
  }
  return names;
}

//--------------------------------------------------------------------------
//-------- polynomial_order() ----------------------------------------------
//--------------------------------------------------------------------------
int
Realm::polynomial_order() const
{
  return promotionOrder_;
}

//--------------------------------------------------------------------------
//-------- matrix_free() ---------------------------------------------------
//--------------------------------------------------------------------------
bool
Realm::matrix_free() const
{
  return matrixFree_;
}

Teuchos::ParameterList
Realm::solver_parameters(std::string name) const
{
  return root()->linearSolvers_->get_solver_configuration(
    equationSystems_.get_solver_block_name(name));
}

} // namespace nalu
} // namespace sierra
