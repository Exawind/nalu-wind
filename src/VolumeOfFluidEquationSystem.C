// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <property_evaluator/MaterialPropertyData.h>
#include <VolumeOfFluidEquationSystem.h>
#include <ProjectedNodalGradientEquationSystem.h>
#include <NaluParsing.h>
#include <Enums.h>
#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSystem.h>
#include <Realms.h>
#include <Realm.h>
#include <Simulation.h>
#include <CopyFieldAlgorithm.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>
#include <AssembleNGPNodeSolverAlgorithm.h>
#include <stdexcept>
#include <AuxFunctionAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <DirichletBC.h>

// edge kernels
#include "edge_kernels/VOFAdvectionEdgeAlg.h"
#include "user_functions/ZalesakDiskMassFlowRateKernel.h"
#include "user_functions/ZalesakSphereMassFlowRateKernel.h"

// node kernels
#include "node_kernels/NodeKernelUtils.h"
#include "node_kernels/VOFMassBDFNodeKernel.h"
#include "node_kernels/VOFGclNodeKernel.h"

// ngp
#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_algorithms/NodalGradBndryElemAlg.h"
#include "stk_topology/topology.hpp"
#include "user_functions/ZalesakDiskVOFAuxFunction.h"
#include "user_functions/ZalesakSphereVOFAuxFunction.h"
#include "user_functions/DropletVOFAuxFunction.h"
#include "user_functions/SloshingTankVOFAuxFunction.h"
#include "ngp_utils/NgpFieldBLAS.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "stk_io/IossBridge.hpp"

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// VolumeOfFluidEquationSystem - manages VOF pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
VolumeOfFluidEquationSystem::VolumeOfFluidEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "VolumeOfFluidEQS", "volume_of_fluid"),
    managePNG_(realm_.get_consistent_mass_matrix_png("volume_of_fluid")),
    volumeOfFluid_(NULL),
    dvolumeOfFluiddx_(NULL),
    vofTmp_(NULL),
    nodalGradAlgDriver_(realm_, "volume_of_fluid", "dvolume_of_fluiddx"),
    projectedNodalGradEqs_(NULL),
    isInit_(false)
{
  dofName_ = "volume_of_fluid";

  // extract solver name and solver object
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("volume_of_fluid");
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, realm_.name(), EQ_VOLUME_OF_FLUID);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // Require div(u) = 0 instead of div(density*u) = 0
  realm_.solutionOptions_->solveIncompressibleContinuity_ = true;
  realm_.solutionOptions_->realm_has_vof_ = true;

  // determine nodal gradient form
  set_nodal_gradient("volume_of_fluid");
  NaluEnv::self().naluOutputP0()
    << "Edge projected nodal gradient for volume_of_fluid: "
    << edgeNodalGradient_ << std::endl;

  // push back EQ to manager
  realm_.equationSystems_.equationSystemVector_.push_back(this);

  // create projected nodal gradient equation system
  if (managePNG_) {
    manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
VolumeOfFluidEquationSystem::~VolumeOfFluidEquationSystem() {}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_nodal_fields(
  const stk::mesh::PartVector& part_vec)
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);

  // register dof; set it as a restart variable
  const int numStates = realm_.number_of_states();

  auto density_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "density", numStates));
  stk::mesh::put_field_on_mesh(*density_, selector, nullptr);
  realm_.augment_restart_variable_list("density");

  // push to property list
  realm_.augment_property_map(DENSITY_ID, density_);

  volumeOfFluid_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "volume_of_fluid", numStates));
  stk::mesh::put_field_on_mesh(*volumeOfFluid_, selector, nullptr);
  realm_.augment_restart_variable_list("volume_of_fluid");

  auto velocity_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "velocity", numStates));
  stk::mesh::put_field_on_mesh(*velocity_, selector, nullptr);
  realm_.augment_restart_variable_list("velocity");

  dvolumeOfFluiddx_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "dvolume_of_fluiddx"));
  stk::mesh::put_field_on_mesh(*dvolumeOfFluiddx_, selector, nDim, nullptr);
  stk::io::set_field_output_type(
    *dvolumeOfFluiddx_, stk::io::FieldOutputType::VECTOR_3D);

  // delta solution for linear solver; share delta since this is a split system
  vofTmp_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "vofTmp"));
  stk::mesh::put_field_on_mesh(*vofTmp_, selector, nullptr);

  levelSet_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "level_set"));
  stk::mesh::put_field_on_mesh(*levelSet_, selector, nullptr);

  hFunction_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "h_function"));
  stk::mesh::put_field_on_mesh(*hFunction_, selector, nullptr);

  if (
    numStates > 2 &&
    (!realm_.restarted_simulation() || realm_.support_inconsistent_restart())) {

    ScalarFieldType& vofN = volumeOfFluid_->field_of_state(stk::mesh::StateN);
    ScalarFieldType& vofNp1 =
      volumeOfFluid_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
      realm_, part_vec, &vofNp1, &vofN, 0, 1, stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_element_fields -------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_element_fields(
  const stk::mesh::PartVector& /*part_vec*/, const stk::topology& /* theTopo */)
{
  // nothing as of yet
}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_edge_fields(
  const stk::mesh::PartVector& part_vec)
{
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  auto massFlowRate_ = &(meta_data.declare_field<double>(
    stk::topology::EDGE_RANK, "mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*massFlowRate_, selector, nullptr);
  auto massVofBalancedFlowRate_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::EDGE_RANK, "mass_vof_balanced_flow_rate"));
  stk::mesh::put_field_on_mesh(*massVofBalancedFlowRate_, selector, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_interior_algorithm(stk::mesh::Part* part)
{

  // non-solver, dpdx
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType& vofNp1 = volumeOfFluid_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dvofdxNone =
    dvolumeOfFluiddx_->field_of_state(stk::mesh::StateNone);

  if (!managePNG_) {
    nodalGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
      algType, part, "volume_of_fluid_nodal_grad", &vofNp1, &dvofdxNone);
  }

  if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {

    std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = NULL;
      if (realm_.realmUsesEdges_) {
        const bool useAvgMdot = (realm_.solutionOptions_->turbulenceModel_ ==
                                 TurbulenceModel::SST_AMS)
                                  ? true
                                  : false;
        theAlg = new VOFAdvectionEdgeAlg(
          realm_, part, this, volumeOfFluid_, dvolumeOfFluiddx_, useAvgMdot);

      } else {
        throw std::runtime_error(
          "VOFEQS: Attempt to use non-NGP element solver algorithm");
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;

    } else {
      itsi->second->partVec_.push_back(part);
    }

    NaluEnv::self().naluOutputP0() << "register vof interior: " << std::endl;
    std::vector<std::string> checkAlgNames = {
      "volume_of_fluid_time_derivative",
      "lumped_volume_of_fluid_time_derivative"};
    bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
    if (elementMassAlg) {
      throw std::runtime_error("consistent mass integration of volume of fluid "
                               "time-derivative unavailable");
    }
    auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
    process_ngp_node_kernels(
      solverAlgMap, realm_, part, this,
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg) {
        nodeAlg.add_kernel<VOFMassBDFNodeKernel>(
          realm_.bulk_data(), volumeOfFluid_);
      },
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg, std::string& srcName) {
        if (srcName == "gcl") {
          nodeAlg.add_kernel<VOFGclNodeKernel>(
            realm_.bulk_data(), volumeOfFluid_);
          NaluEnv::self().naluOutputP0() << " - " << srcName << std::endl;
        } else
          throw std::runtime_error("VOFEqSys: Invalid source term: " + srcName);
      });

  } else {
    throw std::runtime_error("VOFEQS: Element terms not supported");
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_inflow_bc(
  stk::mesh::Part* part,
  const stk::topology& /* partTopo */,
  const InflowBoundaryConditionData& inflowBCData)
{
  // algorithm type
  const AlgorithmType algType = INFLOW;
  ScalarFieldType& vofNp1 = volumeOfFluid_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dvofdxNone =
    dvolumeOfFluiddx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; vof_bc
  ScalarFieldType* theBcField =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "vof_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified vof and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  std::string vofName = "volume_of_fluid";
  UserDataType theDataType = get_bc_data_type(userData, vofName);

  AuxFunction* theAuxFunc = NULL;

  if (CONSTANT_UD == theDataType) {
    VolumeOfFluid volumeOfFluid = userData.volumeOfFluid_;
    std::vector<double> userSpec(1);
    userSpec[0] = volumeOfFluid.volumeOfFluid_;
    theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  } else if (FUNCTION_UD == theDataType) {
    throw std::runtime_error("VolumeOfFluidEquationSystem::register_inflow_bc: "
                             "limited functions supported");
  } else {
    throw std::runtime_error("VolumeOfFluidEquationSystem::register_inflow_bc: "
                             "only constant functions supported");
  }

  // bc data alg
  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);

  // how to populate the field?
  if (userData.externalData_) {
    // xfer will handle population; only need to populate the initial value
    realm_.initCondAlg_.push_back(auxAlg);
  } else {
    // put it on bcData
    bcDataAlg_.push_back(auxAlg);
  }

  // copy vof_bc to vof_transition np1...
  CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
    realm_, part, theBcField, &vofNp1, 0, 1, stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dvofdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "vof_nodal_grad", &vofNp1, &dvofdxNone, edgeNodalGradient_);

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg =
      new DirichletBC(realm_, this, part, &vofNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology& /* partTopo */,
  const OpenBoundaryConditionData&)
{
  const AlgorithmType algType = OPEN;

  ScalarFieldType& vofNp1 = volumeOfFluid_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dvofdxNone =
    dvolumeOfFluiddx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dvofdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "vof_nodal_grad", &vofNp1, &dvofdxNone, edgeNodalGradient_);
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const WallBoundaryConditionData& wallBCData)
{
  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  ScalarFieldType& vofNp1 = volumeOfFluid_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dvofdxNone =
    dvolumeOfFluiddx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dvofdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "vof_nodal_grad", &vofNp1, &dvofdxNone, edgeNodalGradient_);
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_symmetry_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const SymmetryBoundaryConditionData& /* symmetryBCData */)
{
  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  ScalarFieldType& vofNp1 = volumeOfFluid_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dvofdxNone =
    dvolumeOfFluiddx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dvofdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "vof_nodal_grad", &vofNp1, &dvofdxNone, edgeNodalGradient_);
}

//--------------------------------------------------------------------------
//-------- register_abltop_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_abltop_bc(
  stk::mesh::Part* /* part */,
  const stk::topology& /* partTopo */,
  const ABLTopBoundaryConditionData& /* abltopBCData */)
{
  // Nothing to do
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_non_conformal_bc(
  stk::mesh::Part* /* part */, const stk::topology& /* theTopo */)
{
  // Nothing to do
}
//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(volumeOfFluid_);
  equationSystems_.register_overset_field_update(volumeOfFluid_, 1, 1);
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::reinitialize_linear_system()
{
  // If this is decoupled overset simulation and the user has requested that the
  // linear system be reused, then do nothing
  if (decoupledOverset_ && linsys_->config().reuseLinSysIfPossible())
    return;

  // delete linsys
  delete linsys_;

  // create new solver
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("volume_of_fluid");
  LinearSolver* solver = realm_.root()->linearSolvers_->reinitialize_solver(
    solverName, realm_.name(), EQ_VOLUME_OF_FLUID);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::register_initial_condition_fcn(
  stk::mesh::Part* part,
  const std::map<std::string, std::string>& theNames,
  const std::map<std::string, std::vector<double>>& theParams)
{
  // iterate map and check for name
  const std::string dofName = "volume_of_fluid";
  std::map<std::string, std::string>::const_iterator iterName =
    theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;
    AuxFunction* theAuxFunc = NULL;

    if (fcnName == "zalesak_disk") {
      theAuxFunc = new ZalesakDiskVOFAuxFunction();
      // Initialize mass flow rate until momentum connection implemented
      {
        const bool useAvgMdot = (realm_.solutionOptions_->turbulenceModel_ ==
                                 TurbulenceModel::SST_AMS)
                                  ? true
                                  : false;
        VectorFieldType* velocity_ = realm_.meta_data().get_field<double>(
          stk::topology::NODE_RANK, "velocity");
        ScalarFieldType* density_ = realm_.meta_data().get_field<double>(
          stk::topology::NODE_RANK, "density");
        std::vector<double> userSpec(3);
        userSpec[0] = 1.0;
        userSpec[1] = 0.0;
        userSpec[2] = 0.0;
        AuxFunction* constantAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
        AuxFunctionAlgorithm* constantAuxAlg = new AuxFunctionAlgorithm(
          realm_, part, density_, constantAuxFunc, stk::topology::NODE_RANK);
        realm_.initCondAlg_.push_back(constantAuxAlg);
        AuxFunction* constantVelFunc = new ConstantAuxFunction(0, 3, userSpec);
        AuxFunctionAlgorithm* constantVelAlg = new AuxFunctionAlgorithm(
          realm_, part, velocity_, constantVelFunc, stk::topology::NODE_RANK);
        realm_.initCondAlg_.push_back(constantVelAlg);
        auto VOFSetMassFlowRate =
          new ZalesakDiskMassFlowRateEdgeAlg(realm_, part, this, useAvgMdot);
        realm_.initCondAlg_.push_back(VOFSetMassFlowRate);
      }
    } else if (fcnName == "zalesak_sphere") {
      theAuxFunc = new ZalesakSphereVOFAuxFunction();
      // Initialize mass flow rate until momentum connection implemented
      {
        const bool useAvgMdot = (realm_.solutionOptions_->turbulenceModel_ ==
                                 TurbulenceModel::SST_AMS)
                                  ? true
                                  : false;
        ScalarFieldType* density_ = realm_.meta_data().get_field<double>(
          stk::topology::NODE_RANK, "density");
        std::vector<double> userSpec(1);
        userSpec[0] = 1.0;
        AuxFunction* constantAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
        AuxFunctionAlgorithm* constantAuxAlg = new AuxFunctionAlgorithm(
          realm_, part, density_, constantAuxFunc, stk::topology::NODE_RANK);
        realm_.initCondAlg_.push_back(constantAuxAlg);
        auto VOFSetMassFlowRate =
          new ZalesakSphereMassFlowRateEdgeAlg(realm_, part, this, useAvgMdot);
        realm_.initCondAlg_.push_back(VOFSetMassFlowRate);
      }
    } else if (fcnName == "droplet") {
      theAuxFunc = new DropletVOFAuxFunction();
    } else if (fcnName == "sloshing_tank") {
      std::map<std::string, std::vector<double>>::const_iterator iterParams =
        theParams.find(dofName);
      std::vector<double> fcnParams = (iterParams != theParams.end())
                                        ? (*iterParams).second
                                        : std::vector<double>();
      theAuxFunc = new SloshingTankVOFAuxFunction(fcnParams);
    } else {
      throw std::runtime_error("VolumeOfFluidEquationSystem::register_initial_"
                               "condition_fcn: limited functions supported");
    }
    // create the algorithm
    AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
      realm_, part, volumeOfFluid_, theAuxFunc, stk::topology::NODE_RANK);

    // push to ic
    realm_.initCondAlg_.push_back(auxAlg);
  }
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& /* eqSystems */)
{
  throw std::runtime_error("VolumeOfFluidEquationSystem::manage_projected_"
                           "nodal_gradient: Not supported");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::compute_projected_nodal_gradient()
{

  using Traits = nalu_ngp::NGPMeshTraits<>;

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  stk::mesh::Selector sel =
    (meta_data.locally_owned_part() | meta_data.globally_shared_part()) &
    stk::mesh::selectField(*volumeOfFluid_);

  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  auto ngpVof =
    fieldMgr.get_field<double>(volumeOfFluid_->mesh_meta_data_ordinal());

  ngpVof.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "vof_update_and_clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
      if (ngpVof.get(mi, 0) < 0.0) {
        ngpVof.get(mi, 0) = 0.0;
      }
      if (ngpVof.get(mi, 0) > 1.0) {
        ngpVof.get(mi, 0) = 1.0;
      }
    });
  ngpVof.modify_on_device();

  if (!managePNG_) {
    const double timeA = -NaluEnv::self().nalu_time();
    nodalGradAlgDriver_.execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  } else {
    projectedNodalGradEqs_->solve_and_update_external();
  }
}

//--------------------------------------------------------------------------
//-------- calculate_smooth_vof --------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::calculate_smooth_vof()
{

  const int ndim = realm_.meta_data().spatial_dimension();

  using Traits = nalu_ngp::NGPMeshTraits<>;
  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;

  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  const double desired_interface_width =
    realm_.solutionOptions_->interface_width_;

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  stk::mesh::Selector sel =
    (meta_data.locally_owned_part() | meta_data.globally_shared_part()) &
    stk::mesh::selectField(*volumeOfFluid_);

  auto ngpVof =
    fieldMgr.get_field<double>(volumeOfFluid_->mesh_meta_data_ordinal());

  auto ngpDVof =
    fieldMgr.get_field<double>(dvolumeOfFluiddx_->mesh_meta_data_ordinal());

  auto ngpLset =
    fieldMgr.get_field<double>(levelSet_->mesh_meta_data_ordinal());

  auto ngpHfun =
    fieldMgr.get_field<double>(hFunction_->mesh_meta_data_ordinal());

  auto ngpVol = fieldMgr.get_field<double>(
    get_field_ordinal(meta_data, "dual_nodal_volume", stk::mesh::StateNP1));

  nalu_ngp::run_entity_algorithm(
    "set_level_set", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
      const double gamma = desired_interface_width > 0.0
                             ? 0.5 * desired_interface_width
                             : 0.75 * stk::math::cbrt(ngpVol.get(mi, 0));
      const double eps =
        desired_interface_width > 0.0 ? desired_interface_width : 2.0 * gamma;

      const double pi_const = 4.0 * stk::math::atan(1.0);

      ngpLset.get(mi, 0) = 2.0 * (ngpVof.get(mi, 0) - 0.5) * gamma;

      if (ngpLset.get(mi, 0) < -eps)
        ngpHfun.get(mi, 0) = 0.0;
      else if (ngpLset.get(mi, 0) > eps)
        ngpHfun.get(mi, 0) = 1.0;
      else
        ngpHfun.get(mi, 0) =
          0.5 * (1.0 + ngpLset.get(mi, 0) / eps +
                 1.0 / pi_const *
                   stk::math::sin(pi_const * ngpLset.get(mi, 0) / eps));

      ngpHfun.get(mi, 0) = 2.0 * (ngpHfun.get(mi, 0) - 0.5);

      // Re-use vof gradient routines to calculate gradient of Level set
      const double stored_value = ngpVof.get(mi, 0);
      ngpVof.get(mi, 0) = ngpLset.get(mi, 0);
      ngpLset.get(mi, 0) = stored_value;
    });

  for (int _iteration = 0; _iteration < 25; ++_iteration) {

    if (!managePNG_) {
      nodalGradAlgDriver_.execute();
    } else {
      projectedNodalGradEqs_->solve_and_update_external();
    }

    nalu_ngp::run_entity_algorithm(
      "smooth_level_set", ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
        const double dx = stk::math::cbrt(ngpVol.get(mi, 0));
        const double tau = 0.1 * dx;

        double interface_magnitude = 0.0;
        for (int dim = 0; dim < 3; ++dim)
          interface_magnitude += ngpDVof.get(mi, dim) * ngpDVof.get(mi, dim);

        interface_magnitude = stk::math::sqrt(interface_magnitude);

        ngpVof.get(mi, 0) +=
          tau * ngpHfun.get(mi, 0) * (1.0 - interface_magnitude);
      });
  }

  nalu_ngp::run_entity_algorithm(
    "reset_vof", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
      const double stored_lset_value = ngpVof.get(mi, 0);
      const double gamma = desired_interface_width > 0.0
                             ? 0.5 * desired_interface_width
                             : 0.75 * stk::math::cbrt(ngpVol.get(mi, 0));
      const double eps = desired_interface_width > 0.0
                           ? desired_interface_width
                           : 1.5 * stk::math::cbrt(ngpVol.get(mi, 0));
      const double pi_const = 4.0 * stk::math::atan(1.0);

      ngpVof.get(mi, 0) = ngpLset.get(mi, 0);

      if (stored_lset_value < -eps)
        ngpLset.get(mi, 0) = 0.0;
      else if (stored_lset_value > eps)
        ngpLset.get(mi, 0) = 1.0;
      else
        ngpLset.get(mi, 0) =
          0.5 *
          (1.0 + stored_lset_value / eps +
           1.0 / pi_const * stk::math::sin(pi_const * stored_lset_value / eps));
      // ngpLset.get(mi, 0) = 0.5 * stored_lset_value / eps + 0.5;
    });

  ngpLset.modify_on_device();

  // Fill in gradient
  if (!managePNG_) {
    nodalGradAlgDriver_.execute();
  } else {
    projectedNodalGradEqs_->solve_and_update_external();
  }
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::solve_and_update()
{

  // compute dvof/dx
  if (!isInit_) {
    compute_projected_nodal_gradient();
    isInit_ = true;
  }

  for (int k = 0; k < maxIterations_; ++k) {

    NaluEnv::self().naluOutputP0()
      << " " << k + 1 << "/" << maxIterations_ << std::setw(15) << std::right
      << userSuppliedName_ << std::endl;
    calculate_smooth_vof();
    assemble_and_solve(vofTmp_);
    solution_update(1.0, *vofTmp_, 1.0, *volumeOfFluid_);

    compute_projected_nodal_gradient();
  }
}

//--------------------------------------------------------------------------
//-------- predict_state ---------------------------------------------------
//--------------------------------------------------------------------------
void
VolumeOfFluidEquationSystem::predict_state()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  auto& vofN = fieldMgr.get_field<double>(
    volumeOfFluid_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& vofNp1 = fieldMgr.get_field<double>(
    volumeOfFluid_->field_of_state(stk::mesh::StateNP1)
      .mesh_meta_data_ordinal());

  vofN.sync_to_device();

  const auto& meta = realm_.meta_data();
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part() |
     meta.aura_part()) &
    stk::mesh::selectField(*volumeOfFluid_);
  nalu_ngp::field_copy(ngpMesh, sel, vofNp1, vofN, 1);
  vofNp1.modify_on_device();
}

} // namespace nalu
} // namespace sierra
