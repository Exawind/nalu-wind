// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <LowMachEquationSystem.h>
#include <wind_energy/ABLForcingAlgorithm.h>
#include <AlgorithmDriver.h>
#include <aero/AeroContainer.h>
#include <AssembleContinuityNonConformalSolverAlgorithm.h>
#include <AssembleMomentumEdgeWallFunctionSolverAlgorithm.h>
#ifdef NALU_USES_FFTW
#include <AssembleMomentumEdgeABLTopBC.h>
#endif
#include <AssembleMomentumNonConformalSolverAlgorithm.h>
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <AssembleNodalGradUNonConformalAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
#include <AuxFunctionAlgorithm.h>
#include <ComputeMdotNonConformalAlgorithm.h>
#include <ComputeWallFrictionVelocityAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <ContinuityLowSpeedCompressibleNodeSuppAlg.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
#include <EffectiveDiffFluxCoeffAlgorithm.h>
#include <Enums.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <FieldFunctions.h>
#include <LinearSolver.h>
#include <LinearSolvers.h>
#include <LinearSystem.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <MomentumBuoyancySrcNodeSuppAlg.h>
#include <MomentumBoussinesqRASrcNodeSuppAlg.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <NonConformalManager.h>
#include <NonConformalInfo.h>
#include <PeriodicManager.h>
#include <ProjectedNodalGradientEquationSystem.h>
#include <PostProcessingData.h>
#include <Realm.h>
#include <Realms.h>
#include <SurfaceForceAndMomentAlgorithmDriver.h>
#include <SurfaceForceAndMomentAlgorithm.h>
#include <SurfaceForceAndMomentWallFunctionAlgorithm.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>
#include <TurbViscSmagorinskyAlgorithm.h>
#include <TurbViscWaleAlgorithm.h>
#include <wind_energy/ABLForcingAlgorithm.h>
#include <FixPressureAtNodeAlgorithm.h>
#include <FixPressureAtNodeInfo.h>

#ifdef NALU_USES_HYPRE
#include <HypreLinearSystem.h>
#endif

// template for kernels
#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

// bc kernels
#include <kernel/ContinuityInflowElemKernel.h>
#include <kernel/MomentumWallFunctionElemKernel.h>

// edge kernels
#include <edge_kernels/ContinuityEdgeSolverAlg.h>
#include <edge_kernels/ContinuityOpenEdgeKernel.h>
#include <edge_kernels/MomentumEdgeSolverAlg.h>
#include <edge_kernels/MomentumOpenEdgeKernel.h>
#include <edge_kernels/MomentumABLWallShearStressEdgeKernel.h>
#include <edge_kernels/MomentumSymmetryEdgeKernel.h>
#include <edge_kernels/MomentumEdgePecletAlg.h>
#include <edge_kernels/StreletsUpwindEdgeAlg.h>
#include <edge_kernels/AMSMomentumEdgePecletAlg.h>

// node kernels
#include "node_kernels/NodeKernelUtils.h"
#include "node_kernels/MomentumABLForceNodeKernel.h"
#include "node_kernels/MomentumActuatorNodeKernel.h"
#include "node_kernels/MomentumBodyForceNodeKernel.h"
#include "node_kernels/MomentumBodyForceBoxNodeKernel.h"
#include "node_kernels/MomentumBoussinesqNodeKernel.h"
#include "node_kernels/MomentumCoriolisNodeKernel.h"
#include "node_kernels/MomentumMassBDFNodeKernel.h"
#include "node_kernels/MomentumGclSrcNodeKernel.h"
#include "node_kernels/ContinuityGclNodeKernel.h"
#include "node_kernels/ContinuityMassBDFNodeKernel.h"

// ngp
#include "ngp_algorithms/ABLWallFrictionVelAlg.h"
#include "ngp_algorithms/ABLWallFluxesAlg.h"
#include "ngp_algorithms/CourantReAlg.h"
#include "ngp_algorithms/GeometryAlgDriver.h"
#include "ngp_algorithms/MdotEdgeAlg.h"
#include "ngp_algorithms/MdotAlgDriver.h"
#include "ngp_algorithms/MdotDensityAccumAlg.h"
#include "ngp_algorithms/MdotInflowAlg.h"
#include "ngp_algorithms/MdotOpenEdgeAlg.h"
#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_algorithms/NodalGradElemAlg.h"
#include "ngp_algorithms/NodalGradBndryElemAlg.h"
#include "ngp_algorithms/NodalGradPOpenBoundaryAlg.h"
#include "ngp_algorithms/EffDiffFluxCoeffAlg.h"
#include "ngp_algorithms/TurbViscKsgsAlg.h"
#include "ngp_algorithms/TurbViscSSTAlg.h"
#include "ngp_algorithms/TurbViscSSTLRAlg.h"
#include "ngp_algorithms/TurbViscKEAlg.h"
#include "ngp_algorithms/TurbViscKOAlg.h"
#include "ngp_algorithms/WallFuncGeometryAlg.h"
#include "ngp_algorithms/DynamicPressureOpenAlg.h"
#include "ngp_algorithms/MomentumABLWallFuncMaskUtil.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldBLAS.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "stk_mesh/base/NgpFieldParallel.hpp"
#include "ngp_utils/NgpTypes.h"

// UT Austin Hybrid AMS kernels
#include <edge_kernels/AssembleAMSEdgeKernelAlg.h>
#include <node_kernels/MomentumSSTAMSForcingNodeKernel.h>

// user function
#include <user_functions/ConvectingTaylorVortexVelocityAuxFunction.h>
#include <user_functions/ConvectingTaylorVortexPressureAuxFunction.h>
#include <user_functions/TornadoAuxFunction.h>

#include <user_functions/WindEnergyTaylorVortexAuxFunction.h>
#include <user_functions/WindEnergyTaylorVortexPressureAuxFunction.h>

#include <user_functions/SteadyTaylorVortexMomentumSrcNodeSuppAlg.h>
#include <user_functions/SteadyTaylorVortexVelocityAuxFunction.h>
#include <user_functions/SteadyTaylorVortexPressureAuxFunction.h>

#include <user_functions/VariableDensityVelocityAuxFunction.h>
#include <user_functions/VariableDensityPressureAuxFunction.h>
#include <user_functions/VariableDensityContinuitySrcNodeSuppAlg.h>
#include <user_functions/VariableDensityMomentumSrcNodeSuppAlg.h>

#include <user_functions/VariableDensityNonIsoContinuitySrcNodeSuppAlg.h>
#include <user_functions/VariableDensityNonIsoMomentumSrcNodeSuppAlg.h>
#include <user_functions/BoussinesqNonIsoMomentumSrcNodeSuppAlg.h>

#include <user_functions/TaylorGreenPressureAuxFunction.h>
#include <user_functions/TaylorGreenVelocityAuxFunction.h>

#include <user_functions/BoussinesqNonIsoVelocityAuxFunction.h>

#include <user_functions/SinProfileChannelFlowVelocityAuxFunction.h>

#include <user_functions/BoundaryLayerPerturbationAuxFunction.h>

#include <user_functions/WindEnergyPowerLawAuxFunction.h>

#include <user_functions/KovasznayVelocityAuxFunction.h>
#include <user_functions/KovasznayPressureAuxFunction.h>

#include <overset/UpdateOversetFringeAlgorithmDriver.h>
#include <overset/AssembleOversetPressureAlgorithm.h>

#include <user_functions/OneTwoTenVelocityAuxFunction.h>

#include <user_functions/PerturbedShearLayerAuxFunctions.h>
#include <user_functions/GaussJetVelocityAuxFunction.h>

// deprecated

// stk_util
#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_util/util/SortAndUnique.hpp>

// stk_mesh/base/fem
#include "stk_mesh/base/Types.hpp"
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/SkinMesh.hpp>
#include <stk_mesh/base/Comm.hpp>
#include "stk_mesh/base/NgpMesh.hpp"

// stk_topo
#include <stk_topology/topology.hpp>

#include <utils/StkHelpers.h>

// basic c++
#include <vector>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// LowMachEquationSystem - manage the low Mach equation system (uvw_p)
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
LowMachEquationSystem::LowMachEquationSystem(
  EquationSystems& eqSystems, const bool elementContinuityEqs)
  : EquationSystem(eqSystems, "LowMachEOSWrap", "low_mach_type"),
    elementContinuityEqs_(elementContinuityEqs),
    density_(NULL),
    viscosity_(NULL),
    dualNodalVolume_(NULL),
    edgeAreaVec_(NULL),
    surfaceForceAndMomentAlgDriver_(NULL),
    xyBCType_(2, 0),
    isInit_(true)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create momentum and pressure
  momentumEqSys_ = new MomentumEquationSystem(eqSystems);
  continuityEqSys_ =
    new ContinuityEquationSystem(eqSystems, elementContinuityEqs_);

  momentumEqSys_->dofName_ = "velocity";
  continuityEqSys_->dofName_ = "pressure";

  // inform realm
  realm_.hasFluids_ = true;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
LowMachEquationSystem::~LowMachEquationSystem()
{
  if (NULL != surfaceForceAndMomentAlgDriver_)
    delete surfaceForceAndMomentAlgDriver_;
}

void
LowMachEquationSystem::load(const YAML::Node& node)
{
  EquationSystem::load(node);

  if (realm_.query_for_overset()) {
    bool momDecoupled = decoupledOverset_;
    bool presDecoupled = decoupledOverset_;
    int momNumIters = numOversetIters_;
    int presNumIters = numOversetIters_;

    get_if_present_no_default(node, "momentum_decoupled_overset", momDecoupled);
    get_if_present_no_default(
      node, "continuity_decoupled_overset", presDecoupled);
    get_if_present_no_default(
      node, "momentum_num_overset_correctors", momNumIters);
    get_if_present_no_default(
      node, "continuity_num_overset_correctors", presNumIters);

    momentumEqSys_->decoupledOverset_ = momDecoupled;
    momentumEqSys_->numOversetIters_ = momNumIters;
    continuityEqSys_->decoupledOverset_ = presDecoupled;
    continuityEqSys_->numOversetIters_ = presNumIters;

    // LowMach is considered decoupled only if both momentum and continuity are
    // decoupled.
    decoupledOverset_ = momDecoupled && presDecoupled;
  }
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::initialize()
{
  // let equation systems that are owned some information
  momentumEqSys_->convergenceTolerance_ = convergenceTolerance_;
  continuityEqSys_->convergenceTolerance_ = convergenceTolerance_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_nodal_fields(stk::mesh::Part* part)
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // add properties; denisty needs to be a restart field
  const int numStates = realm_.number_of_states();
  density_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density", numStates));
  stk::mesh::put_field_on_mesh(*density_, *part, nullptr);
  realm_.augment_restart_variable_list("density");

  viscosity_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*viscosity_, *part, nullptr);

  // push to property list
  realm_.augment_property_map(DENSITY_ID, density_);
  realm_.augment_property_map(VISCOSITY_ID, viscosity_);

  // dual nodal volume (should push up...)
  const int numVolStates =
    realm_.does_mesh_move() ? realm_.number_of_states() : 1;
  dualNodalVolume_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume", numVolStates));
  stk::mesh::put_field_on_mesh(*dualNodalVolume_, *part, nullptr);
  if (numVolStates > 1)
    realm_.augment_restart_variable_list("dual_nodal_volume");

  // make sure all states are properly populated (restart can handle this)
  if (
    numStates > 2 &&
    (!realm_.restarted_simulation() || realm_.support_inconsistent_restart())) {
    ScalarFieldType& densityN = density_->field_of_state(stk::mesh::StateN);
    ScalarFieldType& densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm* theCopyAlgDens = new CopyFieldAlgorithm(
      realm_, part, &densityNp1, &densityN, 0, 1, stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlgDens);

    if (numVolStates <= 2)
      return;

    ScalarFieldType& dualNdVolN =
      dualNodalVolume_->field_of_state(stk::mesh::StateN);
    ScalarFieldType& dualNdVolNp1 =
      dualNodalVolume_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm* theCopyAlgDlNdVol = new CopyFieldAlgorithm(
      realm_, part, &dualNdVolNp1, &dualNdVolN, 0, 1, stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlgDlNdVol);
  }
}

//--------------------------------------------------------------------------
//-------- register_element_fields -------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_element_fields(
  stk::mesh::Part* part, const stk::topology& theTopo)
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register mdot for element-based scheme...
  if (elementContinuityEqs_) {
    // extract master element and get scs points
    MasterElement* meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
    const int numScsIp = meSCS->num_integration_points();
    GenericFieldType* massFlowRate =
      &(meta_data.declare_field<GenericFieldType>(
        stk::topology::ELEMENT_RANK, "mass_flow_rate_scs"));
    stk::mesh::put_field_on_mesh(*massFlowRate, *part, numScsIp, nullptr);
  }
  // register the intersected elemental field
  if (realm_.query_for_overset()) {
    const int sizeOfElemField = 1;
    GenericFieldType* intersectedElement =
      &(meta_data.declare_field<GenericFieldType>(
        stk::topology::ELEMENT_RANK, "intersected_element"));
    stk::mesh::put_field_on_mesh(
      *intersectedElement, *part, sizeOfElemField, nullptr);
  }

  // provide mean element Peclet and Courant fields; always...
  GenericFieldType* elemReynolds = &(meta_data.declare_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "element_reynolds"));
  stk::mesh::put_field_on_mesh(*elemReynolds, *part, 1, nullptr);
  GenericFieldType* elemCourant = &(meta_data.declare_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "element_courant"));
  stk::mesh::put_field_on_mesh(*elemCourant, *part, 1, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_edge_fields(stk::mesh::Part* part)
{

  if (realm_.realmUsesEdges_) {
    stk::mesh::MetaData& meta_data = realm_.meta_data();
    const int nDim = meta_data.spatial_dimension();
    edgeAreaVec_ = &(meta_data.declare_field<VectorFieldType>(
      stk::topology::EDGE_RANK, "edge_area_vector"));
    stk::mesh::put_field_on_mesh(*edgeAreaVec_, *part, nDim, nullptr);
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology& theTopo,
  const OpenBoundaryConditionData& openBCData)
{

  // register boundary data
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const int nDim = metaData.spatial_dimension();

  VectorFieldType* velocityBC = &(metaData.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "open_velocity_bc"));
  stk::mesh::put_field_on_mesh(*velocityBC, *part, nDim, nullptr);

  // extract the value for user specified velocity and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  Velocity ux = userData.u_;
  std::vector<double> userSpecUbc(nDim);
  userSpecUbc[0] = ux.ux_;
  userSpecUbc[1] = ux.uy_;
  if (nDim > 2)
    userSpecUbc[2] = ux.uz_;

  // new it
  ConstantAuxFunction* theAuxFuncUbc =
    new ConstantAuxFunction(0, nDim, userSpecUbc);

  // bc data alg
  AuxFunctionAlgorithm* auxAlgUbc = new AuxFunctionAlgorithm(
    realm_, part, velocityBC, theAuxFuncUbc, stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlgUbc);

  // extract the value for user specified pressure and save off the AuxFunction
  if (!realm_.solutionOptions_->activateOpenMdotCorrection_) {
    ScalarFieldType* pressureBC = &(metaData.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "pressure_bc"));
    stk::mesh::put_field_on_mesh(*pressureBC, *part, nullptr);

    Pressure pSpec = userData.p_;
    std::vector<double> userSpecPbc(1);
    userSpecPbc[0] = pSpec.pressure_;

    // new it
    ConstantAuxFunction* theAuxFuncPbc =
      new ConstantAuxFunction(0, 1, userSpecPbc);

    // bc data alg
    AuxFunctionAlgorithm* auxAlgPbc = new AuxFunctionAlgorithm(
      realm_, part, pressureBC, theAuxFuncPbc, stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlgPbc);
  } else {
    if (userData.pSpec_)
      NaluEnv::self().naluOutputP0()
        << "LowMachEqs::register_open_bc Error: Pressure specified at an open "
           "bc while global correction algorithm has been activated"
        << std::endl;
  }

  // mdot at open bc; register field
  MasterElement* meFC =
    sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsBip = meFC->num_integration_points();
  GenericFieldType* mdotBip = &(metaData.declare_field<GenericFieldType>(
    static_cast<stk::topology::rank_t>(metaData.side_rank()),
    "open_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*mdotBip, *part, numScsBip, nullptr);

  auto& dynPress = metaData.declare_field<GenericFieldType>(
    static_cast<stk::topology::rank_t>(metaData.side_rank()),
    "dynamic_pressure");
  std::vector<double> ic(numScsBip, 0);
  stk::mesh::put_field_on_mesh(dynPress, *part, numScsBip, ic.data());
}

//--------------------------------------------------------------------------
//-------- register_surface_pp_algorithm ----------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_surface_pp_algorithm(
  const PostProcessingData& theData, stk::mesh::PartVector& partVector)
{
  const std::string thePhysics = theData.physics_;

  // register nodal fields in common
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  VectorFieldType* pressureForce = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "pressure_force"));
  stk::mesh::put_field_on_mesh(
    *pressureForce, stk::mesh::selectUnion(partVector),
    meta_data.spatial_dimension(), nullptr);
  VectorFieldType* viscousForce = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "viscous_force"));
  stk::mesh::put_field_on_mesh(
    *viscousForce, stk::mesh::selectUnion(partVector),
    meta_data.spatial_dimension(), nullptr);
  VectorFieldType* tauWallVector = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "tau_wall_vector"));
  stk::mesh::put_field_on_mesh(
    *tauWallVector, stk::mesh::selectUnion(partVector),
    meta_data.spatial_dimension(), nullptr);
  ScalarFieldType* tauWall = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "tau_wall"));
  stk::mesh::put_field_on_mesh(
    *tauWall, stk::mesh::selectUnion(partVector), nullptr);
  ScalarFieldType* yplus = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "yplus"));
  stk::mesh::put_field_on_mesh(
    *yplus, stk::mesh::selectUnion(partVector), nullptr);

  // force output for these variables
  realm_.augment_output_variable_list(pressureForce->name());
  realm_.augment_output_variable_list(viscousForce->name());
  realm_.augment_output_variable_list(tauWall->name());
  realm_.augment_output_variable_list(yplus->name());

  const bool RANSAblBcApproach = momentumEqSys_->RANSAblBcApproach_;

  if (thePhysics == "surface_force_and_moment") {
    if (RANSAblBcApproach) {
      std::cout << "surface_force_and_moment not implemented with RANS_abl_bc."
                << std::endl;
    }

    ScalarFieldType* assembledArea = &(meta_data.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "assembled_area_force_moment"));
    stk::mesh::put_field_on_mesh(
      *assembledArea, stk::mesh::selectUnion(partVector), nullptr);
    if (NULL == surfaceForceAndMomentAlgDriver_)
      surfaceForceAndMomentAlgDriver_ =
        new SurfaceForceAndMomentAlgorithmDriver(realm_);
    SurfaceForceAndMomentAlgorithm* ppAlg = new SurfaceForceAndMomentAlgorithm(
      realm_, partVector, theData.outputFileName_, theData.frequency_,
      theData.parameters_, realm_.realmUsesEdges_);
    surfaceForceAndMomentAlgDriver_->algVec_.push_back(ppAlg);
  } else if (thePhysics == "surface_force_and_moment_wall_function") {
    if (RANSAblBcApproach) {
      std::cout << "surface_force_and_moment_wall_function not implemented "
                   "with RANS_abl_bc."
                << std::endl;
    }

    ScalarFieldType* assembledArea = &(meta_data.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "assembled_area_force_moment_wf"));
    stk::mesh::put_field_on_mesh(
      *assembledArea, stk::mesh::selectUnion(partVector), nullptr);
    if (NULL == surfaceForceAndMomentAlgDriver_)
      surfaceForceAndMomentAlgDriver_ =
        new SurfaceForceAndMomentAlgorithmDriver(realm_);
    SurfaceForceAndMomentWallFunctionAlgorithm* ppAlg =
      new SurfaceForceAndMomentWallFunctionAlgorithm(
        realm_, partVector, theData.outputFileName_, theData.frequency_,
        theData.parameters_, realm_.realmUsesEdges_);
    surfaceForceAndMomentAlgDriver_->algVec_.push_back(ppAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_initial_condition_fcn(
  stk::mesh::Part* part,
  const std::map<std::string, std::string>& theNames,
  const std::map<std::string, std::vector<double>>& theParams)
{
  // extract nDim
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  // iterate map and check for name
  const std::string dofName = "velocity";
  std::map<std::string, std::string>::const_iterator iterName =
    theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;

    // save off the field (np1 state)
    VectorFieldType* velocityNp1 = meta_data.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity");

    // create a few Aux things
    AuxFunction* theAuxFunc = NULL;
    AuxFunctionAlgorithm* auxAlg = NULL;

    if (fcnName == "wind_energy_taylor_vortex") {

      // extract the params
      std::map<std::string, std::vector<double>>::const_iterator iterParams =
        theParams.find(dofName);
      if (iterParams != theParams.end()) {
        std::vector<double> fcnParams = (*iterParams).second;
        // create the function
        theAuxFunc = new WindEnergyTaylorVortexAuxFunction(0, nDim, fcnParams);
      } else {
        throw std::runtime_error(
          "Wind_energy_taylor_vortex missing parameters");
      }
    } else if (fcnName == "boundary_layer_perturbation") {

      // extract the params
      std::map<std::string, std::vector<double>>::const_iterator iterParams =
        theParams.find(dofName);
      if (iterParams != theParams.end()) {
        std::vector<double> fcnParams = (*iterParams).second;
        // create the function
        theAuxFunc =
          new BoundaryLayerPerturbationAuxFunction(0, nDim, fcnParams);
      } else {
        throw std::runtime_error(
          "Boundary_layer_perturbation missing parameters");
      }
    } else if (fcnName == "wind_energy_power_law") {
      auto it = theParams.find(dofName);
      if (it != theParams.end()) {
        auto& fp = it->second;
        theAuxFunc = new WindEnergyPowerLawAuxFunction(0, nDim, fp);
      } else {
        throw std::runtime_error("Wind Energy Power Law aux function missing "
                                 "parameters in initial condition");
      }
    } else if (fcnName == "kovasznay") {
      theAuxFunc = new KovasznayVelocityAuxFunction(0, nDim);
    } else if (fcnName == "SteadyTaylorVortex") {
      theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0, nDim);
    } else if (fcnName == "VariableDensity") {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0, nDim);
    } else if (fcnName == "VariableDensityNonIso") {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0, nDim);
    } else if (fcnName == "OneTwoTenVelocity") {
      theAuxFunc = new OneTwoTenVelocityAuxFunction(0, nDim);
    } else if (fcnName == "convecting_taylor_vortex") {
      theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0, nDim);
    } else if (fcnName == "TaylorGreen") {
      theAuxFunc = new TaylorGreenVelocityAuxFunction(0, nDim);
    } else if (fcnName == "BoussinesqNonIso") {
      theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0, nDim);
    } else if (fcnName == "SinProfileChannelFlow") {
      theAuxFunc = new SinProfileChannelFlowVelocityAuxFunction(0, nDim);
    } else if (fcnName == "PerturbedShearLayer") {
      theAuxFunc = new PerturbedShearLayerVelocityAuxFunction(0, nDim);
    } else {
      throw std::runtime_error(
        "InitialCondFunction::non-supported velocity IC");
    }

    // create the algorithm
    auxAlg = new AuxFunctionAlgorithm(
      realm_, part, velocityNp1, theAuxFunc, stk::topology::NODE_RANK);

    // push to ic
    realm_.initCondAlg_.push_back(auxAlg);
  }
}

void
LowMachEquationSystem::pre_iter_work()
{
  momentumEqSys_->pre_iter_work();
  if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
    momentumEqSys_->AMSAlgDriver_->execute();
  continuityEqSys_->pre_iter_work();
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::solve_and_update()
{
  // wrap timing
  double timeA, timeB;
  if (isInit_) {
    continuityEqSys_->compute_projected_nodal_gradient();
    timeA = NaluEnv::self().nalu_time();
    continuityEqSys_->mdotAlgDriver_->execute();
    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerMisc_ += (timeB - timeA);

    if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS) {
      momentumEqSys_->AMSAlgDriver_->initial_mdot();
    }

    isInit_ = false;
  } else if (
    realm_.has_mesh_deformation() && (realm_.currentNonlinearIteration_ == 1)) {
    // continuity assemble, load_complete and solve
    continuityEqSys_->assemble_and_solve(continuityEqSys_->pTmp_);

    // update pressure
    timeA = NaluEnv::self().nalu_time();
    solution_update(
      1.0, *continuityEqSys_->pTmp_, 1.0, *continuityEqSys_->pressure_);
    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerAssemble_ += (timeB - timeA);

    // compute mdot
    timeA = NaluEnv::self().nalu_time();
    continuityEqSys_->mdotAlgDriver_->execute();
    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerMisc_ += (timeB - timeA);

    // project nodal velocity
    project_nodal_velocity();

    // update pressure
    const std::string dofName = "pressure";
    const double relaxFP =
      realm_.solutionOptions_->get_relaxation_factor(dofName);
    if (std::fabs(1.0 - relaxFP) > 1.0e-3) {
      timeA = NaluEnv::self().nalu_time();
      solution_update(
        (relaxFP - 1.0), *continuityEqSys_->pTmp_, 1.0,
        *continuityEqSys_->pressure_);
      continuityEqSys_->compute_projected_nodal_gradient();
      timeB = NaluEnv::self().nalu_time();
      continuityEqSys_->timerAssemble_ += (timeB - timeA);
    }
  }

  // compute tvisc and effective viscosity
  momentumEqSys_->compute_turbulence_parameters();

  // start the iteration loop
  for (int k = 0; k < maxIterations_; ++k) {

    NaluEnv::self().naluOutputP0()
      << " " << k + 1 << "/" << maxIterations_ << std::setw(15) << std::right
      << userSuppliedName_ << std::endl;

    for (int oi = 0; oi < momentumEqSys_->numOversetIters_; ++oi) {
      momentumEqSys_->dynPressAlgDriver_.execute();
      if (momentumEqSys_->pecletAlg_)
        momentumEqSys_->pecletAlg_->execute();
      momentumEqSys_->assemble_and_solve(momentumEqSys_->uTmp_);

      timeA = NaluEnv::self().nalu_time();
      solution_update(
        1.0, *momentumEqSys_->uTmp_, 1.0,
        momentumEqSys_->velocity_->field_of_state(stk::mesh::StateNP1),
        realm_.meta_data().spatial_dimension());
      timeB = NaluEnv::self().nalu_time();
      momentumEqSys_->timerAssemble_ += (timeB - timeA);

      if (momentumEqSys_->decoupledOverset_ && realm_.hasOverset_)
        realm_.overset_field_update(
          &momentumEqSys_->velocity_->field_of_state(stk::mesh::StateNP1), 1,
          realm_.meta_data().spatial_dimension());
    }

    // compute velocity relative to mesh with new velocity
    realm_.compute_vrtm();

    // activate global correction scheme
    if (realm_.solutionOptions_->activateOpenMdotCorrection_) {
      timeA = NaluEnv::self().nalu_time();
      continuityEqSys_->mdotAlgDriver_->execute();
      timeB = NaluEnv::self().nalu_time();
      continuityEqSys_->timerMisc_ += (timeB - timeA);
    }

    for (int oi = 0; oi < continuityEqSys_->numOversetIters_; ++oi) {
      continuityEqSys_->assemble_and_solve(continuityEqSys_->pTmp_);

      timeA = NaluEnv::self().nalu_time();
      solution_update(
        1.0, *continuityEqSys_->pTmp_, 1.0, *continuityEqSys_->pressure_);
      timeB = NaluEnv::self().nalu_time();
      continuityEqSys_->timerAssemble_ += (timeB - timeA);

      if (continuityEqSys_->decoupledOverset_ && realm_.hasOverset_)
        realm_.overset_field_update(continuityEqSys_->pressure_, 1, 1);
    }

    // compute mdot
    timeA = NaluEnv::self().nalu_time();
    continuityEqSys_->mdotAlgDriver_->execute();
    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerMisc_ += (timeB - timeA);

    // project nodal velocity
    project_nodal_velocity();

    // update pressure
    const std::string dofName = "pressure";
    const double relaxFP =
      realm_.solutionOptions_->get_relaxation_factor(dofName);
    timeA = NaluEnv::self().nalu_time();
    if (std::fabs(1.0 - relaxFP) > 1.0e-3) {
      // Take care of the possibility that we have multiple overset correctors
      // and we need to do a pressure update that is the sum of all the deltaP
      // that were accumulated over the multiple correctors.
      if (continuityEqSys_->decoupledOverset_ && realm_.hasOverset_) {
        solution_update(
          (1.0 - relaxFP),
          continuityEqSys_->pressure_->field_of_state(stk::mesh::StateN),
          relaxFP,
          continuityEqSys_->pressure_->field_of_state(stk::mesh::StateNP1));

        realm_.overset_field_update(
          &continuityEqSys_->pressure_->field_of_state(stk::mesh::StateNP1), 1,
          1);
      } else {
        solution_update(
          (relaxFP - 1.0), *continuityEqSys_->pTmp_, 1.0,
          *continuityEqSys_->pressure_);
      }

      continuityEqSys_->compute_projected_nodal_gradient();
      timeB = NaluEnv::self().nalu_time();
      continuityEqSys_->timerAssemble_ += (timeB - timeA);
    }
    // Pressure isn't actually a state, we do this to support multiple overset
    // correctors when the relaxation factor is not 1.0. So copy the current
    // pressure into `StateN` so that we can perform solution update correction
    // with the correct relaxation factor
    nalu_ngp::field_copy(
      realm_.mesh_info(),
      continuityEqSys_->pressure_->field_of_state(stk::mesh::StateN),
      continuityEqSys_->pressure_->field_of_state(stk::mesh::StateNP1));

    // compute velocity relative to mesh with new velocity
    realm_.compute_vrtm();

    // velocity gradients based on current values;
    // note timing of this algorithm relative to initial_work
    // we use this approach to avoid two evals per
    // solve/update since dudx is required for tke
    // production
    momentumEqSys_->compute_projected_nodal_gradient();
    timeA = NaluEnv::self().nalu_time();
    momentumEqSys_->compute_wall_function_params();
    timeB = NaluEnv::self().nalu_time();
    momentumEqSys_->timerMisc_ += (timeB - timeA);
  }

  // process CFL/Reynolds
  momentumEqSys_->cflReAlgDriver_.execute();
}

//--------------------------------------------------------------------------
//-------- project_nodal_velocity ------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::project_nodal_velocity()
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  auto uTmp =
    fieldMgr.get_field<double>(momentumEqSys_->uTmp_->mesh_meta_data_ordinal());
  auto dpdx = fieldMgr.get_field<double>(
    continuityEqSys_->dpdx_->mesh_meta_data_ordinal());
  auto Udiag = fieldMgr.get_field<double>(
    momentumEqSys_->get_diagonal_field()->mesh_meta_data_ordinal());
  auto velNp1 = fieldMgr.get_field<double>(
    momentumEqSys_->velocity_->field_of_state(stk::mesh::StateNP1)
      .mesh_meta_data_ordinal());
  auto rhoNp1 = fieldMgr.get_field<double>(
    density_->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal());

  //==========================================================
  // save off dpdx to uTmp (do it everywhere)
  //==========================================================
  {
    const stk::mesh::Selector sel =
      stk::mesh::selectField(*continuityEqSys_->dpdx_);
    nalu_ngp::field_copy(ngpMesh, sel, uTmp, dpdx, nDim);
  }

  //==========================================================
  // safe to update pressure gradient
  //==========================================================
  continuityEqSys_->compute_projected_nodal_gradient();

  uTmp.sync_to_device();
  dpdx.sync_to_device();
  Udiag.sync_to_device();
  velNp1.sync_to_device();
  rhoNp1.sync_to_device();

  //==========================================================
  // project u, u^n+1 = u^k+1 - dt/rho*(Gjp^N+1 - uTmp);
  //==========================================================
  {
    using Traits = nalu_ngp::NGPMeshTraits<>;
    using MeshIndex = Traits::MeshIndex;
    const stk::mesh::Selector sel =
      ((!stk::mesh::selectUnion(momentumEqSys_->notProjectedPart_)) &
       stk::mesh::selectField(*continuityEqSys_->dpdx_));
    nalu_ngp::run_entity_algorithm(
      "nodal_velocity_projection", ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const MeshIndex& mi) {
        // Scaling factor
        const double fac = 1.0 / (rhoNp1.get(mi, 0) * Udiag.get(mi, 0));
        // Projection step
        for (int d = 0; d < nDim; ++d) {
          velNp1.get(mi, d) -= fac * (dpdx.get(mi, d) - uTmp.get(mi, d));
        }
      });
    const stk::mesh::Selector selX =
      (stk::mesh::selectUnion(momentumEqSys_->notProjectedDir_[0]));
    nalu_ngp::run_entity_algorithm(
      "nodal_velocity_projection_strongX", ngpMesh, stk::topology::NODE_RANK,
      selX, KOKKOS_LAMBDA(const MeshIndex& mi) {
        // Scaling factor
        const double fac = 1.0 / (rhoNp1.get(mi, 0) * Udiag.get(mi, 0));
        //  undo Projection step
        velNp1.get(mi, 0) += fac * (dpdx.get(mi, 0) - uTmp.get(mi, 0));
      });
    const stk::mesh::Selector selY =
      (stk::mesh::selectUnion(momentumEqSys_->notProjectedDir_[1]));
    nalu_ngp::run_entity_algorithm(
      "nodal_velocity_project_strongY", ngpMesh, stk::topology::NODE_RANK, selY,
      KOKKOS_LAMBDA(const MeshIndex& mi) {
        // Scaling factor
        const double fac = 1.0 / (rhoNp1.get(mi, 0) * Udiag.get(mi, 0));
        //  undo Projection step
        velNp1.get(mi, 1) += fac * (dpdx.get(mi, 1) - uTmp.get(mi, 1));
      });
    if (nDim == 3) {
      const stk::mesh::Selector selZ =
        (stk::mesh::selectUnion(momentumEqSys_->notProjectedDir_[2]));
      nalu_ngp::run_entity_algorithm(
        "nodal_velocity_projection_strongZ", ngpMesh, stk::topology::NODE_RANK,
        selZ, KOKKOS_LAMBDA(const MeshIndex& mi) {
          // Scaling factor
          const double fac = 1.0 / (rhoNp1.get(mi, 0) * Udiag.get(mi, 0));
          //  undo Projection step
          velNp1.get(mi, 2) += fac * (dpdx.get(mi, 2) - uTmp.get(mi, 2));
        });
    }
  }

  velNp1.modify_on_device();
}

void
LowMachEquationSystem::predict_state()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  auto& rhoN = fieldMgr.get_field<double>(
    density_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& rhoNp1 = fieldMgr.get_field<double>(
    density_->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal());
  auto& presN =
    nalu_ngp::get_ngp_field(meshInfo, "pressure", stk::mesh::StateN);
  auto& presNp1 =
    nalu_ngp::get_ngp_field(meshInfo, "pressure", stk::mesh::StateNP1);

  rhoN.sync_to_device();
  presN.sync_to_device();

  const auto& meta = realm_.meta_data();
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part() |
     meta.aura_part()) &
    stk::mesh::selectField(*density_);
  nalu_ngp::field_copy(ngpMesh, sel, rhoNp1, rhoN, 1);
  nalu_ngp::field_copy(ngpMesh, sel, presNp1, presN, 1);
  rhoNp1.modify_on_device();
  presNp1.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- post_converged_work ---------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::post_converged_work()
{
  if (NULL != surfaceForceAndMomentAlgDriver_) {
    surfaceForceAndMomentAlgDriver_->execute();
  }

  // output mass closure
  continuityEqSys_->mdotAlgDriver_->provide_output();

  if (realm_.realmUsesEdges_) {
    // get max peclet factor touching each node
    // (host only operation since this is a post processor)
    determine_max_peclet_number(realm_.bulk_data(), realm_.meta_data());
    determine_max_peclet_factor(realm_.bulk_data(), realm_.meta_data());
  }
}

//--------------------------------------------------------------------------
//-------- post_iter_work --------------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::post_iter_work()
{
  if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
    momentumEqSys_->AMSAlgDriver_->post_iter_work();
}

//==========================================================================
// Class Definition
//==========================================================================
// MomentumEquationSystem - manages uvw pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MomentumEquationSystem::MomentumEquationSystem(EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "MomentumEQS", "momentum"),
    managePNG_(realm_.get_consistent_mass_matrix_png("velocity")),
    velocity_(NULL),
    dudx_(NULL),
    coordinates_(NULL),
    uTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    nodalGradAlgDriver_(realm_, "dudx"),
    wallFuncAlgDriver_(realm_),
    dynPressAlgDriver_(realm_),
    cflReAlgDriver_(realm_),
    projectedNodalGradEqs_(NULL),
    firstPNGResidual_(0.0)
{
  dofName_ = "velocity";

  // extract solver name and solver object
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("velocity");
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, realm_.name(), EQ_MOMENTUM);
  linsys_ =
    LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("velocity");
  NaluEnv::self().naluOutputP0()
    << "Edge projected nodal gradient for velocity: " << edgeNodalGradient_
    << std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create projected nodal gradient equation system
  if (managePNG_) {
    manage_projected_nodal_gradient(eqSystems);
  }

  if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
    AMSAlgDriver_.reset(new AMSAlgDriver(realm_));
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MomentumEquationSystem::~MomentumEquationSystem() {}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::initial_work()
{
  // call base class method (BDF2 state management, etc)
  EquationSystem::initial_work();
  if (ablWallNodeMask_)
    ablWallNodeMask_->execute();

  // proceed with a bunch of initial work; wrap in timer
  {
    const double timeA = NaluEnv::self().nalu_time();
    realm_.compute_vrtm();
    const double timeB = NaluEnv::self().nalu_time();
    timerMisc_ += (timeB - timeA);
  }

  compute_projected_nodal_gradient();

  if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
    AMSAlgDriver_->initial_work();

  {
    const double timeA = NaluEnv::self().nalu_time();
    compute_wall_function_params();
    compute_turbulence_parameters();
    if (pecletAlg_)
      pecletAlg_->execute();
    cflReAlgDriver_.execute();

    const double timeB = NaluEnv::self().nalu_time();
    timerMisc_ += (timeB - timeA);
  }

  if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
    AMSAlgDriver_->initial_production();
}

//--------------------------------------------------------------------------
//-------- pre_timestep_work -----------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::pre_timestep_work()
{
  // call base class method due to override
  EquationSystem::pre_timestep_work();

  if (
    (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS) &&
    (realm_.solutionOptions_->meshMotion_ ||
     realm_.solutionOptions_->externalMeshDeformation_)) {
    AMSAlgDriver_->compute_metric_tensor();
  }
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_nodal_fields(stk::mesh::Part* part)
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  velocity_ = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity", numStates));
  stk::mesh::put_field_on_mesh(*velocity_, *part, nDim, nullptr);
  realm_.augment_restart_variable_list("velocity");

  dudx_ = &(meta_data.declare_field<GenericFieldType>(
    stk::topology::NODE_RANK, "dudx"));
  stk::mesh::put_field_on_mesh(*dudx_, *part, nDim * nDim, nullptr);

  // delta solution for linear solver
  uTmp_ = &(
    meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "uTmp"));
  stk::mesh::put_field_on_mesh(*uTmp_, *part, nDim, nullptr);

  coordinates_ = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates"));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);

  visc_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  if (realm_.is_turbulent()) {
    tvisc_ = &(meta_data.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "turbulent_viscosity"));
    stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);
    evisc_ = &(meta_data.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "effective_viscosity_u"));
    stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);

    if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
      AMSAlgDriver_->register_nodal_fields(part);

    if (
      realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_IDDES) {
      iddesRansIndicator_ = &(meta_data.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "iddes_rans_indicator"));
      stk::mesh::put_field_on_mesh(*iddesRansIndicator_, *part, nullptr);
    }
  }

  if (realm_.realmUsesEdges_) {
    ScalarFieldType* pecletAtNodes =
      &(realm_.meta_data().declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "max_peclet_factor"));
    stk::mesh::put_field_on_mesh(*pecletAtNodes, *part, nullptr);
    ScalarFieldType* pecletNumAtNodes =
      &(realm_.meta_data().declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "max_peclet_number"));
    stk::mesh::put_field_on_mesh(*pecletNumAtNodes, *part, nullptr);
  }

  Udiag_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "momentum_diag"));
  stk::mesh::put_field_on_mesh(*Udiag_, *part, nullptr);
  realm_.augment_restart_variable_list("momentum_diag");

  // make sure all states are properly populated (restart can handle this)
  if (
    numStates > 2 &&
    (!realm_.restarted_simulation() || realm_.support_inconsistent_restart())) {
    VectorFieldType& velocityN = velocity_->field_of_state(stk::mesh::StateN);
    VectorFieldType& velocityNp1 =
      velocity_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
      realm_, part, &velocityNp1, &velocityN, 0, nDim,
      stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }

  // register specialty fields for PNG
  if (managePNG_) {
    // create temp vector field for duidx that will hold the active dudx
    VectorFieldType* duidx = &(meta_data.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "duidx"));
    stk::mesh::put_field_on_mesh(*duidx, *part, nDim, nullptr);
  }

  // Add actuator and other source terms
  // put it here because the parts to register are sorted on the equation system
  // probably should go in Realm::register_nodal_fields at some point
  if (realm_.aeroModels_->is_active()) {
    realm_.aeroModels_->register_nodal_fields(meta_data, part);
  }

  ScalarFieldType& node_mask =
    realm_.meta_data().declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "abl_wall_no_slip_wall_func_node_mask");
  double one = 1;
  stk::mesh::put_field_on_mesh(node_mask, *part, 1, &one);
}

//--------------------------------------------------------------------------
//-------- register_element_fields -----------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_element_fields(
  stk::mesh::Part* /* part */, const stk::topology& /* theTopo */)
{
}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_edge_fields(stk::mesh::Part* part)
{
  ScalarFieldType* pecletFactor =
    &(realm_.meta_data().declare_field<ScalarFieldType>(
      stk::topology::EDGE_RANK, "peclet_factor"));
  stk::mesh::put_field_on_mesh(*pecletFactor, *part, nullptr);
  ScalarFieldType* pecletNumber =
    &(realm_.meta_data().declare_field<ScalarFieldType>(
      stk::topology::EDGE_RANK, "peclet_number"));
  stk::mesh::put_field_on_mesh(*pecletNumber, *part, nullptr);
  if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
    AMSAlgDriver_->register_edge_fields(part);
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_interior_algorithm(stk::mesh::Part* part)
{
  // types of algorithms
  const AlgorithmType algType = INTERIOR;
  const AlgorithmType algMass = SRC;

  // non-solver CFL alg
  cflReAlgDriver_.register_elem_algorithm<CourantReAlg>(
    algType, part, "courant_reynolds", cflReAlgDriver_);

  VectorFieldType& velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType& dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if (!managePNG_) {
    if (edgeNodalGradient_ && realm_.realmUsesEdges_)
      nodalGradAlgDriver_.register_edge_algorithm<VectorNodalGradEdgeAlg>(
        algType, part, "momentum_nodal_grad", &velocityNp1, &dudxNone);
    else
      nodalGradAlgDriver_.register_elem_algorithm<VectorNodalGradElemAlg>(
        algType, part, "momentum_nodal_grad", &velocityNp1, &dudxNone,
        edgeNodalGradient_);
  }

  const auto theTurbModel = realm_.solutionOptions_->turbulenceModel_;

  // solver; interior contribution (advection + diffusion) [possible CMM time]
  if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {
    std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theSolverAlg = NULL;
      if (realm_.realmUsesEdges_) {
        theSolverAlg = new MomentumEdgeSolverAlg(realm_, part, this);
        if (theTurbModel == TurbulenceModel::SST_AMS) {
          SolverAlgorithm* theSolverSrcAlg = NULL;
          theSolverSrcAlg = new AssembleAMSEdgeKernelAlg(realm_, part, this);
          solverAlgDriver_->solverAlgMap_[SRC] = theSolverSrcAlg;
        }
        if (
          realm_.is_turbulent() &&
          realm_.solutionOptions_->useStreletsUpwinding_) {
          pecletAlg_.reset(new StreletsUpwindEdgeAlg(realm_, part));
        } else if (
          realm_.is_turbulent() && theTurbModel == TurbulenceModel::SST_AMS) {
          pecletAlg_.reset(new AMSMomentumEdgePecletAlg(realm_, part, this));
        } else {
          pecletAlg_.reset(new MomentumEdgePecletAlg(realm_, part, this));
        }
      } else {
        throw std::runtime_error(
          "MomentumEQS: Attempting to use non-NGP element algorithm");
      }
      solverAlgDriver_->solverAlgMap_[algType] = theSolverAlg;

      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string>>::iterator isrc =
        realm_.solutionOptions_->elemSrcTermsMap_.find("momentum");
      if (isrc != realm_.solutionOptions_->elemSrcTermsMap_.end()) {
        throw std::runtime_error(
          "MomentumElemSrcTerms::Error can not use element source terms for an "
          "edge-based scheme");
      }
    } else {
      itsi->second->partVec_.push_back(part);

      const bool hasAMS =
        realm_.realmUsesEdges_ && (theTurbModel == TurbulenceModel::SST_AMS);
      if (hasAMS) {
        auto* tamsAlg = solverAlgDriver_->solverAlgMap_.at(SRC);
        tamsAlg->partVec_.push_back(part);
      }

      if (pecletAlg_)
        pecletAlg_->partVec_.push_back(part);
    }
  } else {
    throw std::runtime_error("MomentumEQS: Element terms not supported");
  }

  // Check if the user has requested CMM or LMM algorithms; if so, do not
  // include Nodal Mass algorithms
  std::vector<std::string> checkAlgNames = {
    "momentum_time_derivative", "lumped_momentum_time_derivative"};
  bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
  // solver; time contribution (lumped mass matrix)
  if (!elementMassAlg || nodal_src_is_requested()) {
    // Handle error checking during transition period. Some kernels are handled
    // through the NGP-ready interface while others are handled via legacy
    // interface and only supported on CPUs.
    int ngpSrcSkipped = 0;
    int nonNgpSrcSkipped = 0;
    int numUsrSrc = 0;

    // Process NGP-ready nodal source terms first
    auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
    process_ngp_node_kernels(
      solverAlgMap, realm_, part, this,
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg) {
        if (!elementMassAlg)
          nodeAlg.add_kernel<MomentumMassBDFNodeKernel>(realm_.bulk_data());
        if (
          realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
          nodeAlg.add_kernel<MomentumSSTAMSForcingNodeKernel>(
            realm_.bulk_data(), *realm_.solutionOptions_);
      },
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg, std::string& srcName) {
        bool added = true;
        if (srcName == "buoyancy_boussinesq") {
          nodeAlg.add_kernel<MomentumBoussinesqNodeKernel>(
            realm_.bulk_data(), *realm_.solutionOptions_);
        } else if (srcName == "body_force") {
          const auto it =
            realm_.solutionOptions_->srcTermParamMap_.find("momentum");
          if (it != realm_.solutionOptions_->srcTermParamMap_.end())
            nodeAlg.add_kernel<MomentumBodyForceNodeKernel>(
              realm_.bulk_data(), it->second);
          else
            throw std::runtime_error(
              "MomentumEQS::body_force: No force vector found");
        } else if (srcName == "body_force_box") {
          const auto it =
            realm_.solutionOptions_->srcTermParamMap_.find("momentum");
          if (it != realm_.solutionOptions_->srcTermParamMap_.end()) {
            const auto bx =
              realm_.solutionOptions_->srcTermParamMap_.find("momentum_box");
            if (bx != realm_.solutionOptions_->srcTermParamMap_.end()) {
              nodeAlg.add_kernel<MomentumBodyForceBoxNodeKernel>(
                realm_, it->second, bx->second);
            } else {
              throw std::runtime_error(
                "MomentumEQS::body_force_box: No box vector found");
            }
          } else {
            throw std::runtime_error(
              "MomentumEQS::body_force_box: No force vector found");
          }
        } else if (srcName == "abl_forcing") {
          ThrowRequireMsg(
            ((NULL != realm_.ablForcingAlg_) &&
             (realm_.ablForcingAlg_->momentumForcingOn())),
            "ERROR! ABL Forcing parameters not "
            "initialized for momentum");
          nodeAlg.add_kernel<MomentumABLForceNodeKernel>(
            realm_.bulk_data(), *realm_.solutionOptions_);
        } else if (srcName == "actuator") {
          nodeAlg.add_kernel<MomentumActuatorNodeKernel>(realm_.meta_data());
        } else if ((srcName == "coriolis") || (srcName == "EarthCoriolis")) {
          nodeAlg.add_kernel<MomentumCoriolisNodeKernel>(
            realm_.bulk_data(), *realm_.solutionOptions_);
        } else if (srcName == "gcl") {
          nodeAlg.add_kernel<MomentumGclSrcNodeKernel>(realm_.bulk_data());
        } else {
          // Encountered a source term not yet supported by NGP
          added = false;
          ++ngpSrcSkipped;
        }

        if (added)
          NaluEnv::self().naluOutputP0() << "  - " << srcName << std::endl;
      });

    // Process non-NGP nodal source terms via legacy interface
    std::map<AlgorithmType, SolverAlgorithm*>::iterator itsm =
      solverAlgDriver_->solverAlgMap_.find(algMass);
    if (itsm == solverAlgDriver_->solverAlgMap_.end()) {
      AssembleNodeSolverAlgorithm* theAlg =
        new AssembleNodeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algMass] = theAlg;

      // Add src term supp alg...; limited number supported
      std::map<std::string, std::vector<std::string>>::iterator isrc =
        realm_.solutionOptions_->srcTermsMap_.find("momentum");
      if (isrc != realm_.solutionOptions_->srcTermsMap_.end()) {
        std::vector<std::string> mapNameVec = isrc->second;
        numUsrSrc = mapNameVec.size();
        for (size_t k = 0; k < mapNameVec.size(); ++k) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm* suppAlg = NULL;
          if (sourceName == "buoyancy") {
            suppAlg = new MomentumBuoyancySrcNodeSuppAlg(realm_);
          } else if (sourceName == "buoyancy_boussinesq_ra") {
            suppAlg = new MomentumBoussinesqRASrcNodeSuppAlg(realm_);
          } else if (sourceName == "SteadyTaylorVortex") {
            suppAlg = new SteadyTaylorVortexMomentumSrcNodeSuppAlg(realm_);
          } else if (sourceName == "VariableDensity") {
            suppAlg = new VariableDensityMomentumSrcNodeSuppAlg(realm_);
          } else if (sourceName == "VariableDensityNonIso") {
            suppAlg = new VariableDensityNonIsoMomentumSrcNodeSuppAlg(realm_);
          } else if (sourceName == "BoussinesqNonIso") {
            suppAlg = new BoussinesqNonIsoMomentumSrcNodeSuppAlg(realm_);
          } else {
            ++nonNgpSrcSkipped;
          }
          if (suppAlg != NULL) {
            NaluEnv::self().naluOutputP0()
              << "MomentumNodalSrcTerms::added() " << sourceName << std::endl;
            theAlg->supplementalAlg_.push_back(suppAlg);
          }
        }
      }
    } else {
      itsm->second->partVec_.push_back(part);
    }

    // Ensure that all user source terms were processed by either interface
    if ((ngpSrcSkipped + nonNgpSrcSkipped) != numUsrSrc)
      throw std::runtime_error(
        "Error processing nodal source terms for Momentum");
  }

  // effective viscosity alg
  if (realm_.is_turbulent()) {
    if (!diffFluxCoeffAlg_) {
      diffFluxCoeffAlg_.reset(new EffDiffFluxCoeffAlg(
        realm_, part, visc_, tvisc_, evisc_, 1.0, 1.0, realm_.is_turbulent()));
    } else {
      diffFluxCoeffAlg_->partVec_.push_back(part);
    }

    // deal with tvisc better? - possibly should be on EqSysManager?
    if (!tviscAlg_) {
      switch (realm_.solutionOptions_->turbulenceModel_) {
      case TurbulenceModel::KSGS:
        tviscAlg_.reset(new TurbViscKsgsAlg(realm_, part, tvisc_));
        break;

      case TurbulenceModel::SMAGORINSKY:
        tviscAlg_.reset(new TurbViscSmagorinskyAlgorithm(realm_, part));
        break;

      case TurbulenceModel::WALE:
        tviscAlg_.reset(new TurbViscWaleAlgorithm(realm_, part));
        break;

      case TurbulenceModel::SST:
      case TurbulenceModel::SST_DES:
      case TurbulenceModel::SST_IDDES:

        tviscAlg_.reset(new TurbViscSSTAlg(realm_, part, tvisc_));
        break;

      case TurbulenceModel::SST_AMS:
        tviscAlg_.reset(new TurbViscSSTAlg(realm_, part, tvisc_, true));
        break;

      case TurbulenceModel::KE:
        tviscAlg_.reset(new TurbViscKEAlg(realm_, part, tvisc_));
        break;

      case TurbulenceModel::KO:
        tviscAlg_.reset(new TurbViscKOAlg(realm_, part, tvisc_));
        break;

      case TurbulenceModel::SSTLR:
        tviscAlg_.reset(new TurbViscSSTLRAlg(realm_, part, tvisc_));
        break;

      default:
        throw std::runtime_error("Unsupported turbulence model provided");
      }
    } else {
      tviscAlg_->partVec_.push_back(part);
    }

    if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
      AMSAlgDriver_->register_interior_algorithm(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_inflow_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const InflowBoundaryConditionData& inflowBCData)
{

  // push mesh part
  notProjectedPart_.push_back(part);

  // algorithm type
  const AlgorithmType algType = INFLOW;

  // velocity np1
  VectorFieldType& velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType& dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();

  // register boundary data; velocity_bc
  VectorFieldType* theBcField = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

  // extract the value for user specified velocity and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  std::string velocityName = "velocity";
  UserDataType theDataType = get_bc_data_type(userData, velocityName);

  AuxFunction* theAuxFunc = NULL;
  if (CONSTANT_UD == theDataType) {
    Velocity ux = userData.u_;
    std::vector<double> userSpec(nDim);
    userSpec[0] = ux.ux_;
    userSpec[1] = ux.uy_;
    if (nDim > 2)
      userSpec[2] = ux.uz_;

    // new it
    theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);

  } else if (FUNCTION_UD == theDataType) {
    // extract the name/params
    std::string fcnName = get_bc_function_name(userData, velocityName);
    std::vector<double> theParams =
      get_bc_function_params(userData, velocityName);
    if (theParams.size() == 0)
      NaluEnv::self().naluOutputP0()
        << "function parameter size is zero" << std::endl;
    // switch on the name found...
    if (fcnName == "convecting_taylor_vortex") {
      theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0, nDim);
    } else if (fcnName == "SteadyTaylorVortex") {
      theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0, nDim);
    } else if (fcnName == "VariableDensity") {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0, nDim);
    } else if (fcnName == "VariableDensityNonIso") {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0, nDim);
    } else if (fcnName == "TaylorGreen") {
      theAuxFunc = new TaylorGreenVelocityAuxFunction(0, nDim);
    } else if (fcnName == "BoussinesqNonIso") {
      theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0, nDim);
    } else if (fcnName == "kovasznay") {
      theAuxFunc = new KovasznayVelocityAuxFunction(0, nDim);
    } else if (fcnName == "wind_energy_power_law") {
      theAuxFunc = new WindEnergyPowerLawAuxFunction(0, nDim, theParams);
    } else if (fcnName == "GaussJet") {
      theAuxFunc = new GaussJetVelocityAuxFunction(0, nDim);
    } else {
      throw std::runtime_error("MomentumEquationSystem::register_inflow_bc: "
                               "limited functions supported");
    }
  } else {
    throw std::runtime_error("MomentumEquationSystem::register_inflow_bc: only "
                             "constant and user function supported");
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

  // copy velocity_bc to velocity np1...
  CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
    realm_, part, theBcField, &velocityNp1, 0, nDim, stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<VectorNodalGradBndryElemAlg>(
      algType, part, "momentum_nodal_grad", theBcField, &dudxNone,
      edgeNodalGradient_);
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg =
      new DirichletBC(realm_, this, part, &velocityNp1, theBcField, 0, nDim);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const OpenBoundaryConditionData& openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  // register boundary data; open_velocity_bc
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  VectorFieldType* theBcField = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "open_velocity_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

  // extract the value for user specified velocity and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  Velocity ux = userData.u_;
  std::vector<double> userSpec(nDim);
  userSpec[0] = ux.ux_;
  userSpec[1] = ux.uy_;
  if (nDim > 2)
    userSpec[2] = ux.uz_;

  // new it
  ConstantAuxFunction* theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);

  // bc data alg
  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  VectorFieldType& velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType& dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<VectorNodalGradBndryElemAlg>(
      algType, part, "momentum_nodal_grad", &velocityNp1, &dudxNone,
      edgeNodalGradient_);
  }

  if (realm_.realmUsesEdges_) {
    // solver for continuity open
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    stk::topology elemTopo = get_elem_topo(realm_, *part);

    AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(faceElemSolverAlg, solverAlgWasBuilt) =
      build_or_add_part_to_face_elem_solver_alg(
        algType, *this, *part, elemTopo, solverAlgMap, "open");

    auto& activeKernels = faceElemSolverAlg->activeKernels_;

    if (solverAlgWasBuilt) {
      build_face_elem_topo_kernel_automatic<MomentumOpenEdgeKernel>(
        partTopo, elemTopo, *this, activeKernels, "momentum_open",
        realm_.meta_data(), realm_.solutionOptions_,
        realm_.is_turbulent() ? evisc_ : visc_,
        faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_,
        userData.entrainMethod_);
    }
  } else {
    throw std::runtime_error(
      "MomentumEQS: Attempt to use element open algorithm");
  }

  if (userData.totalP_) {
    dynPressAlgDriver_.register_face_algorithm<DynamicPressureOpenAlg>(
      algType, part, "dyn_press");
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const WallBoundaryConditionData& wallBCData)
{

  // find out if this is a wall function approach
  WallUserData userData = wallBCData.userData_;
  const bool wallFunctionApproach = userData.wallFunctionApproach_;
  const bool ablWallFunctionApproach = userData.ablWallFunctionApproach_;
  const bool anyWallFunctionActivated =
    wallFunctionApproach || ablWallFunctionApproach;
  auto& ablWallFunctionNode = userData.ablWallFunctionNode_;

  // find out if this is RANS SST for modeling the ABL
  RANSAblBcApproach_ = userData.RANSAblBcApproach_;

  // push mesh part
  if (!anyWallFunctionActivated)
    notProjectedPart_.push_back(part);

  // np1 velocity
  VectorFieldType& velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType& dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();

  const std::string bcFieldName =
    anyWallFunctionActivated ? "wall_velocity_bc" : "velocity_bc";

  // register boundary data; velocity_bc
  VectorFieldType* theBcField = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, bcFieldName));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

  // if mesh motion is enabled ...
  if (realm_.solutionOptions_->meshMotion_) {
    NaluEnv::self().naluOutputP0()
      << "MomentumEquationSystem::register_wall_bc(): Mesh motion active! "
         "Velocity definition under wall_user_data will be ignored"
      << std::endl;

    // get the mesh velocity field
    VectorFieldType* meshVelocity = meta_data.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "mesh_velocity");

    // create algorithm to copy mesh velocity to wall velocity
    CopyFieldAlgorithm* wallVelCopyAlg = new CopyFieldAlgorithm(
      realm_, part, meshVelocity, theBcField, 0, nDim,
      stk::topology::NODE_RANK);

    bcDataAlg_.push_back(wallVelCopyAlg);
  }

  // if mesh motion is not enabled...
  else {
    // extract the value for user specified velocity and save off the
    // AuxFunction
    AuxFunction* theAuxFunc = NULL;
    Algorithm* auxAlg = NULL;

    std::string velocityName = "velocity";

    if (bc_data_specified(userData, velocityName)) {

      UserDataType theDataType = get_bc_data_type(userData, velocityName);
      if (CONSTANT_UD == theDataType) {
        // constant data type specification
        Velocity ux = userData.u_;
        std::vector<double> userSpec(nDim);
        userSpec[0] = ux.ux_;
        userSpec[1] = ux.uy_;
        if (nDim > 2)
          userSpec[2] = ux.uz_;
        theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);
      } else if (FUNCTION_UD == theDataType) {
        // extract the name and parameters (double and string)
        std::string fcnName = get_bc_function_name(userData, velocityName);
        // switch on the name found...
        if (fcnName == "tornado") {
          theAuxFunc = new TornadoAuxFunction(0, nDim);
        } else if (fcnName == "wind_energy") {
          NaluEnv::self().naluOutputP0()
            << "MomentumEqSys: WARNING! mesh_motion user function for wall BC "
               "has been deprecated"
            << std::endl;
        } else {
          throw std::runtime_error("MomentumEqSys::register_wall_function: "
                                   "Only tornado user functions supported");
        }
      }
    } else {
      throw std::runtime_error("Invalid Wall Data Specification; must provide "
                               "const or fcn for velocity");
    }

    auxAlg = new AuxFunctionAlgorithm(
      realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);

    // check to see if this is an FSI interface to determine how we handle
    // velocity population
    if (userData.isFsiInterface_) {
      // xfer will handle population; only need to populate the initial value
      realm_.initCondAlg_.push_back(auxAlg);
    } else {
      bcDataAlg_.push_back(auxAlg);
    }
  }

  // Only set velocityNp1 at the wall boundary if we are not using any wall
  // functions
  if (!anyWallFunctionActivated || userData.isNoSlip_) {
    // copy velocity_bc to velocity np1
    CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
      realm_, part, theBcField, &velocityNp1, 0, nDim,
      stk::topology::NODE_RANK);

    bcDataMapAlg_.push_back(theCopyAlg);
  }

  // non-solver; contribution to Gjui; allow for element-based shifted
  if (!managePNG_) {
    const AlgorithmType algTypePNG = anyWallFunctionActivated ? WALL_FCN : WALL;
    nodalGradAlgDriver_.register_face_algorithm<VectorNodalGradBndryElemAlg>(
      algTypePNG, part, "momentum_nodal_grad", theBcField, &dudxNone,
      edgeNodalGradient_);
  }

  if (anyWallFunctionActivated || RANSAblBcApproach_) {
    ScalarFieldType* assembledWallArea =
      &(meta_data.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "assembled_wall_area_wf"));
    stk::mesh::put_field_on_mesh(*assembledWallArea, *part, nullptr);

    ScalarFieldType* assembledWallNormalDistance =
      &(meta_data.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "assembled_wall_normal_distance"));
    stk::mesh::put_field_on_mesh(*assembledWallNormalDistance, *part, nullptr);

    // integration point; size it based on number of boundary integration points
    MasterElement* meFC =
      sierra::nalu::MasterElementRepo::get_surface_master_element(partTopo);
    const int numScsBip = meFC->num_integration_points();

    stk::topology::rank_t sideRank =
      static_cast<stk::topology::rank_t>(meta_data.side_rank());

    GenericFieldType* wallFrictionVelocityBip =
      &(meta_data.declare_field<GenericFieldType>(
        sideRank, "wall_friction_velocity_bip"));
    stk::mesh::put_field_on_mesh(
      *wallFrictionVelocityBip, *part, numScsBip, nullptr);

    GenericFieldType* wallNormalDistanceBip =
      &(meta_data.declare_field<GenericFieldType>(
        sideRank, "wall_normal_distance_bip"));
    stk::mesh::put_field_on_mesh(
      *wallNormalDistanceBip, *part, numScsBip, nullptr);

    // need wall friction velocity for TKE boundary condition
    if (RANSAblBcApproach_) {
      const AlgorithmType wfAlgType = WALL_FCN;

      wallFuncAlgDriver_
        .register_legacy_algorithm<ComputeWallFrictionVelocityAlgorithm>(
          wfAlgType, part, "wall_func", realm_.realmUsesEdges_, wallBCData);
    }

    // Wall models.
    if (anyWallFunctionActivated) {
      GenericFieldType* wallShearStressBip =
        &(meta_data.declare_field<GenericFieldType>(
          sideRank, "wall_shear_stress_bip"));
      stk::mesh::put_field_on_mesh(
        *wallShearStressBip, *part, nDim * numScsBip, nullptr);

      // register the standard time-space-invariant wall heat flux (not used by
      // the ABL wall model).
      NormalHeatFlux heatFlux = userData.q_;
      std::vector<double> userSpec(1);
      userSpec[0] = heatFlux.qn_;
      ConstantAuxFunction* theHeatFluxAuxFunc =
        new ConstantAuxFunction(0, 1, userSpec);
      ScalarFieldType* theHeatFluxBcField =
        &(meta_data.declare_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "heat_flux_bc"));
      stk::mesh::put_field_on_mesh(*theHeatFluxBcField, *part, nullptr);
      bcDataAlg_.push_back(new AuxFunctionAlgorithm(
        realm_, part, theHeatFluxBcField, theHeatFluxAuxFunc,
        stk::topology::NODE_RANK));

      // Atmospheric-boundary-layer-style wall model.
      if (ablWallFunctionApproach) {

        const AlgorithmType wfAlgType = WALL_ABL;

        // register boundary data: wall_heat_flux_bip.  This is the ABL
        // integration-point-based heat flux field.
        GenericFieldType* wallHeatFluxBip =
          &(meta_data.declare_field<GenericFieldType>(
            sideRank, "wall_heat_flux_bip"));
        stk::mesh::put_field_on_mesh(
          *wallHeatFluxBip, *part, numScsBip, nullptr);

        // register the algorithm that computes geometry that the wall model
        // uses.
        realm_.geometryAlgDriver_
          ->register_wall_func_algorithm<WallFuncGeometryAlg>(
            wfAlgType, part, get_elem_topo(realm_, *part),
            "geometry_wall_func");

        // register the algorithm that calculates the momentum and heat flux on
        // the wall.
        wallFuncAlgDriver_.register_face_elem_algorithm<ABLWallFluxesAlg>(
          wfAlgType, part, get_elem_topo(realm_, *part), "abl_wall_func",
          wallFuncAlgDriver_, realm_.realmUsesEdges_, ablWallFunctionNode);

        // Assemble wall stresses via the edge algorithm.
        auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
        stk::topology elemTopo = get_elem_topo(realm_, *part);
        AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
        bool solverAlgWasBuilt = false;

        std::tie(faceElemSolverAlg, solverAlgWasBuilt) =
          build_or_add_part_to_face_elem_solver_alg(
            wfAlgType, *this, *part, elemTopo, solverAlgMap, "wall_func");

        if (userData.isNoSlip_) {
          notProjectedPart_.push_back(part);
          if (!ablWallNodeMask_) {
            ablWallNodeMask_.reset(
              new MomentumABLWallFuncMaskUtil(realm_, part));
          } else {
            ablWallNodeMask_->partVec_.push_back(part);
          }
        }

        auto& activeKernels = faceElemSolverAlg->activeKernels_;
        if (solverAlgWasBuilt) {
          build_face_elem_topo_kernel_automatic<
            MomentumABLWallShearStressEdgeKernel>(
            partTopo, elemTopo, *this, activeKernels, "momentum_abl_wall",
            !userData.isNoSlip_, realm_.meta_data(),
            faceElemSolverAlg->faceDataNeeded_,
            faceElemSolverAlg->elemDataNeeded_);
        }
      }

      // Engineering-style wall model.
      else {

        const AlgorithmType wfAlgType = WALL_FCN;

        wallFuncAlgDriver_
          .register_legacy_algorithm<ComputeWallFrictionVelocityAlgorithm>(
            wfAlgType, part, "wall_func", realm_.realmUsesEdges_, wallBCData);

        // create lhs/rhs algorithm; generalized for edge (nearest node usage)
        // and element
        if (realm_.solutionOptions_->useConsolidatedBcSolverAlg_) {
          // element-based uses consolidated approach fully
          auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

          AssembleElemSolverAlgorithm* solverAlg = nullptr;
          bool solverAlgWasBuilt = false;

          std::tie(solverAlg, solverAlgWasBuilt) =
            build_or_add_part_to_face_bc_solver_alg(
              *this, *part, solverAlgMap, "wall_fcn");

          ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
          auto& activeKernels = solverAlg->activeKernels_;

          if (solverAlgWasBuilt) {
            build_face_topo_kernel_automatic<MomentumWallFunctionElemKernel>(
              partTopo, *this, activeKernels, "momentum_wall_function",
              realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);
            report_built_supp_alg_names();
          }
        } else {
          // deprecated element-based and supported edge-based approach
          std::map<AlgorithmType, SolverAlgorithm*>::iterator it_wf =
            solverAlgDriver_->solverAlgMap_.find(wfAlgType);
          if (it_wf == solverAlgDriver_->solverAlgMap_.end()) {
            SolverAlgorithm* theAlg = NULL;
            if (realm_.realmUsesEdges_) {
              theAlg = new AssembleMomentumEdgeWallFunctionSolverAlgorithm(
                realm_, part, this);
            } else {
              throw std::runtime_error(
                "MomentumEQS: Cannot use non-NGP wall function algorithm");
            }
            solverAlgDriver_->solverAlgMap_[wfAlgType] = theAlg;
          } else {
            it_wf->second->partVec_.push_back(part);
          }
        }
      }
    }
  }

  // Dirichlet wall boundary condition.
  if (!anyWallFunctionActivated || userData.isNoSlip_) {
    const AlgorithmType algType = WALL;

    std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
      DirichletBC* theAlg =
        new DirichletBC(realm_, this, part, &velocityNp1, theBcField, 0, nDim);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    } else {
      itd->second->partVec_.push_back(part);
    }
  }

  // specialty FSI
  if (userData.isFsiInterface_) {
    // FIXME: need p^n+1/2; requires "old" pressure... need a utility to save it
    // and compute it...
    NaluEnv::self().naluOutputP0()
      << "Warning: Second-order FSI requires p^n+1/2; BC is using p^n+1"
      << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_symmetry_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const SymmetryBoundaryConditionData& symmBCData)
{
  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  VectorFieldType& velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType& dudxNone = dudx_->field_of_state(stk::mesh::StateNone);
  using SYMMTYPES = SymmetryUserData::SymmetryTypes;
  const SYMMTYPES symmType = symmBCData.userData_.symmType_;
  unsigned beginPos{0}, endPos{1};
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();
  AlgorithmType pickTheType = algType;
  // non-solver; contribution to Gjui; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<VectorNodalGradBndryElemAlg>(
      algType, part, "momentum_nodal_grad", &velocityNp1, &dudxNone,
      edgeNodalGradient_);
  }
  switch (symmType) {
  case SYMMTYPES::GENERAL_WEAK:
    if (!realm_.realmUsesEdges_) {
      throw std::runtime_error(
        "MomentumEQS: Attempt to use element symm algorithm");
    } else {
      auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

      stk::topology elemTopo = get_elem_topo(realm_, *part);

      AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
      bool solverAlgWasBuilt = false;
      const std::string algName =
        realm_.realmUsesEdges_ ? "symm_edge" : "symm_elem";

      std::tie(faceElemSolverAlg, solverAlgWasBuilt) =
        build_or_add_part_to_face_elem_solver_alg(
          algType, *this, *part, elemTopo, solverAlgMap, algName);

      auto& activeKernels = faceElemSolverAlg->activeKernels_;

      if (solverAlgWasBuilt) {

        const stk::mesh::MetaData& metaData = realm_.meta_data();
        const std::string viscName =
          realm_.is_turbulent() ? "effective_viscosity_u" : "viscosity";

        build_face_elem_topo_kernel_automatic<MomentumSymmetryEdgeKernel>(
          partTopo, elemTopo, *this, activeKernels, "momentum_symmetry_edge",
          metaData, *realm_.solutionOptions_,
          metaData.get_field<VectorFieldType>(
            stk::topology::NODE_RANK, "velocity"),
          metaData.get_field<ScalarFieldType>(
            stk::topology::NODE_RANK, viscName),
          faceElemSolverAlg->faceDataNeeded_,
          faceElemSolverAlg->elemDataNeeded_);
      }
    }
    return;
// Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
    break;
#endif
  case SYMMTYPES::X_DIR_STRONG:
    pickTheType = AlgorithmType::X_SYM_STRONG;
    beginPos = 0;
    break;
  case SYMMTYPES::Y_DIR_STRONG:
    pickTheType = AlgorithmType::Y_SYM_STRONG;
    beginPos = 1;
    break;
  case SYMMTYPES::Z_DIR_STRONG:
    pickTheType = AlgorithmType::Z_SYM_STRONG;
    beginPos = 2;
    break;
  }

#ifdef NALU_USES_HYPRE
  if (dynamic_cast<HypreLinearSystem*>(linsys_) != nullptr) {
    throw std::runtime_error("Hypre is not supported for a momentum solver "
                             "when using strong_symmetry bc's.");
  }
#endif

  endPos = beginPos + 1;
  if (!symmBCData.userData_.useProjections_) {
    notProjectedDir_[beginPos].push_back(part);
  }
  if (linsys_->useSegregatedSolver()) {
    NaluEnv::self().naluOutputP0()
      << "Warning: You are currently using a segregated solver with a strong "
         "symmetry boundary "
      << "condition. This leads to an approximation of the momentum equation "
         "for the tangential "
      << "velocity component(s) at the symmetry surface because it deletes LHS "
         "sensitivities."
      << std::endl
      << "Warning (cont): Testing shows the error to be negligible, but "
      << "if strange behavior is encountered it is recommended that you "
      << "switch to the monolithic solve (segregated_solver: no)." << std::endl;
  }

  // register boundary data; velocity_bc
  const std::string bcFieldName = "strong_sym_velocity";
  VectorFieldType* theBcField = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, bcFieldName));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

  std::vector<double> userSpec(nDim, 0.0);
  AuxFunction* theAuxFunc = NULL;
  Algorithm* auxAlg = NULL;

  theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);
  auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // copy velocity_bc to velocity np1
  CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
    realm_, part, theBcField, &velocityNp1, 0, nDim, stk::topology::NODE_RANK);

  bcDataMapAlg_.push_back(theCopyAlg);
  const AlgorithmType symAlgType = pickTheType;

  std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(symAlgType);

  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg = new DirichletBC(
      realm_, this, part, &velocityNp1, theBcField, beginPos, endPos);
    solverAlgDriver_->solverDirichAlgMap_[symAlgType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_abltop_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_abltop_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const ABLTopBoundaryConditionData& abltopBCData)
{
  auto userData = abltopBCData.userData_;

  if (!userData.ABLTopBC_) {
    SymmetryBoundaryConditionData symData;
    symData.userData_ = abltopBCData.symmetryUserData_;
    register_symmetry_bc(part, partTopo, symData);
    return;
  }

#ifdef NALU_USES_FFTW
  auto& meta_data = realm_.meta_data();
  // algorithm type
  const AlgorithmType algType = TOP_ABL;
  auto user_data = abltopBCData.userData_;

  VectorFieldType& velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType& dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // push mesh part
  notProjectedPart_.push_back(part);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<VectorNodalGradBndryElemAlg>(
      algType, part, "momentum_nodal_grad", &velocityNp1, &dudxNone,
      edgeNodalGradient_);
  }

  if (!realm_.solutionOptions_->useConsolidatedBcSolverAlg_) {
    // solver algs; lhs
    std::string bcFieldName =
      realm_.solutionOptions_->activateOpenMdotCorrection_ ? "velocity_bc"
                                                           : "cont_velocity_bc";
    VectorFieldType* theBcField = &(meta_data.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, bcFieldName));
    stk::mesh::put_field_on_mesh(*theBcField, *part, 3, nullptr);

    auto it = solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverDirichAlgMap_.end()) {
      SolverAlgorithm* theAlg = new AssembleMomentumEdgeABLTopBC(
        realm_, part, this, user_data.grid_dims_, user_data.horiz_bcs_,
        user_data.z_sample_);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    } else {
      it->second->partVec_.push_back(part);
    }
  } else {
    // auto &solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

    // stk::topology elemTopo = get_elem_topo(realm_, *part);

    // AssembleFaceElemSolverAlgorithm *faceElemSolverAlg = nullptr;
    // bool solverAlgWasBuilt = false;

    // std::tie(faceElemSolverAlg, solverAlgWasBuilt) =
    //     build_or_add_part_to_face_elem_solver_alg(
    //         algType, *this, *part, elemTopo, solverAlgMap, "symm");

    // auto &activeKernels = faceElemSolverAlg->activeKernels_;

    // if (solverAlgWasBuilt) {

    //   const stk::mesh::MetaData &metaData = realm_.meta_data();
    //   const std::string viscName =
    //       realm_.is_turbulent() ? "effective_viscosity_u" : "viscosity";

    //   build_face_elem_topo_kernel_automatic<MomentumSymmetryElemKernel>(
    //       partTopo, elemTopo, *this, activeKernels, "momentum_symmetry",
    //       metaData, *realm_.solutionOptions_,
    //       metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK,
    //                                           "velocity"),
    //       metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK,
    //                                           viscName),
    //       faceElemSolverAlg->faceDataNeeded_,
    //       faceElemSolverAlg->elemDataNeeded_);
    // }
    throw std::runtime_error("MomentumEqSys: Consolidated algorithm not "
                             "supported at this time for ABL Top BC.");
  }
#else
  throw std::runtime_error(
    "Cannot initialize ABL top BC because FFTW support is mising.\n Set "
    "ENABLE_FFTW to ON in nalu-wind/CMakeLists.txt, reconfigure and "
    "recompile.");
#endif
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_non_conformal_bc(
  stk::mesh::Part* part, const stk::topology& theTopo)
{
  const AlgorithmType algType = NON_CONFORMAL;

  VectorFieldType& velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType& dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // mdot at nc bc; register field; require topo and num ips
  MasterElement* meFC =
    sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsBip = meFC->num_integration_points();

  stk::topology::rank_t sideRank =
    static_cast<stk::topology::rank_t>(meta_data.side_rank());
  GenericFieldType* mdotBip =
    &(meta_data.declare_field<GenericFieldType>(sideRank, "nc_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*mdotBip, *part, numScsBip, nullptr);

  // non-solver; contribution to Gjui; DG algorithm decides on locations for
  // integration points
  if (!managePNG_) {
    if (edgeNodalGradient_) {
      nodalGradAlgDriver_.register_face_algorithm<VectorNodalGradBndryElemAlg>(
        algType, part, "momentum_nodal_grad", &velocityNp1, &dudxNone,
        edgeNodalGradient_);
    } else {
      nodalGradAlgDriver_
        .register_legacy_algorithm<AssembleNodalGradUNonConformalAlgorithm>(
          algType, part, "momentum_nodal_grad", &velocityNp1, &dudxNone);
    }
  }

  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
    AssembleMomentumNonConformalSolverAlgorithm* theAlg =
      new AssembleMomentumNonConformalSolverAlgorithm(
        realm_, part, this, &velocityNp1,
        realm_.is_turbulent() ? evisc_ : visc_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    itsi->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(velocity_);

  int nDim = realm_.meta_data().spatial_dimension();
  equationSystems_.register_overset_field_update(velocity_, 1, nDim);
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();

  // Set flag to extract diagonal if the user activates it in input file
  extractDiagonal_ = (realm_.solutionOptions_->tscaleType_ == TSCALE_UDIAGINV);

  // We need an estimate of projTimeScale for the computational of mdot in
  // initialization phase
  if (!realm_.restarted_simulation() || !extractDiagonal_) {
    const double dt = realm_.get_time_step();
    const double gamma1 = realm_.get_gamma1();
    stk::mesh::field_fill(gamma1 / dt, *Udiag_);

    Udiag_->modify_on_host();
  }
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::reinitialize_linear_system()
{
  // If this is decoupled overset simulation and the user has requested that the
  // linear system be reused, then do nothing
  if (decoupledOverset_ && linsys_->config().reuseLinSysIfPossible())
    return;

  // delete linsys
  delete linsys_;

  // create new solver
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("velocity");
  LinearSolver* solver = realm_.root()->linearSolvers_->reinitialize_solver(
    solverName, realm_.name(), EQ_MOMENTUM);
  linsys_ =
    LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

  // initialize new solver
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- predict_state ---------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::predict_state()
{
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  auto& velN = fieldMgr.get_field<double>(
    velocity_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& velNp1 = fieldMgr.get_field<double>(
    velocity_->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal());

  velN.sync_to_device();
  velNp1.sync_to_device();

  const auto& meta = realm_.meta_data();
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part() |
     meta.aura_part()) &
    stk::mesh::selectField(*velocity_);
  nalu_ngp::field_copy(ngpMesh, sel, velNp1, velN, meta.spatial_dimension());
  velNp1.modify_on_device();

  if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
    AMSAlgDriver_->predict_state();
}

//--------------------------------------------------------------------------
//-------- compute_wall_function_params ------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::compute_wall_function_params()
{
  wallFuncAlgDriver_.execute();
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if (NULL == projectedNodalGradEqs_) {
    projectedNodalGradEqs_ = new ProjectedNodalGradientEquationSystem(
      eqSystems, EQ_PNG_U, "duidx", "qTmp", "pTmp", "PNGradUEQS");

    // turn off output
    projectedNodalGradEqs_->deactivate_output();
  }
  // fill the map for expected boundary condition names; recycle pTmp (ui copied
  // in as needed)
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "pTmp");
  projectedNodalGradEqs_->set_data_map(
    WALL_BC, "pTmp"); // might want wall_function velocity_bc?
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "pTmp");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "pTmp");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::compute_projected_nodal_gradient()
{
  if (!managePNG_) {
    const double timeA = -NaluEnv::self().nalu_time();
    nodalGradAlgDriver_.execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  } else {
    // this option is more complex... Rather than solving a nDim*nDim system, we
    // copy each velocity component i to the expected dof for the PNG system;
    // pTmp

    // extract fields
    ScalarFieldType* pTmp = realm_.meta_data().get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "pTmp");
    VectorFieldType* duidx = realm_.meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "duidx");

    const int nDim = realm_.meta_data().spatial_dimension();

    // manage norms here
    bool isFirst = realm_.currentNonlinearIteration_ == 1;
    if (isFirst)
      firstPNGResidual_ = 0.0;

    double sumNonlinearResidual = 0.0;
    double sumLinearResidual = 0.0;
    int sumLinearIterations = 0;
    for (int i = 0; i < nDim; ++i) {
      // copy velocity, component i to pTmp
      field_index_copy(
        realm_.meta_data(), realm_.bulk_data(), *velocity_, i, *pTmp, 0,
        realm_.get_activate_aura());

      // copy active tensor, dudx to vector, duidx
      for (int k = 0; k < nDim; ++k) {
        field_index_copy(
          realm_.meta_data(), realm_.bulk_data(), *dudx_, i * nDim + k, *duidx,
          k, realm_.get_activate_aura());
      }

      projectedNodalGradEqs_->solve_and_update_external();

      // extract the solver history info
      const double nonlinearRes =
        projectedNodalGradEqs_->linsys_->nonLinearResidual();
      const double linearRes =
        projectedNodalGradEqs_->linsys_->linearResidual();
      const int linearIter =
        projectedNodalGradEqs_->linsys_->linearSolveIterations();

      // sum system norms for this iteration
      sumNonlinearResidual += nonlinearRes;
      sumLinearResidual += linearRes;
      sumLinearIterations += linearIter;

      // increment first nonlinear residual
      if (isFirst)
        firstPNGResidual_ += nonlinearRes;

      // copy vector, duidx_k to tensor, dudx; this one might hurt as compared
      // to a specialty loop..
      for (int k = 0; k < nDim; ++k) {
        field_index_copy(
          realm_.meta_data(), realm_.bulk_data(), *duidx, k, *dudx_,
          nDim * i + k, realm_.get_activate_aura());
      }
    }

    // output norms
    const double scaledNonLinearResidual =
      sumNonlinearResidual /
      std::max(std::numeric_limits<double>::epsilon(), firstPNGResidual_);
    std::string pngName = projectedNodalGradEqs_->linsys_->name();
    const int nameOffset = pngName.length() + 8;
    NaluEnv::self().naluOutputP0()
      << std::setw(nameOffset) << std::right << pngName
      << std::setw(32 - nameOffset) << std::right
      << sumLinearIterations / (int)nDim << std::setw(18) << std::right
      << sumLinearResidual / (int)nDim << std::setw(15) << std::right
      << sumNonlinearResidual / (int)nDim << std::setw(14) << std::right
      << scaledNonLinearResidual << std::endl;

    // a bit covert, provide linsys with the new norm which is the sum of all
    // norms
    projectedNodalGradEqs_->linsys_->setNonLinearResidual(sumNonlinearResidual);
  }
}

void
MomentumEquationSystem::save_diagonal_term(
  const std::vector<stk::mesh::Entity>& entities,
  const std::vector<int>& /* scratchIds */,
  const std::vector<double>& lhs)
{
  auto& bulk = realm_.bulk_data();
  const int nEntities = entities.size();
  const int nDim = realm_.spatialDimension_;
  const int offset = nEntities * nDim;

  for (int in = 0; in < nEntities; in++) {
    const auto naluID =
      *stk::mesh::field_data(*realm_.naluGlobalId_, entities[in]);
    const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluID);
    int ix = in * nDim * (offset + 1);
    double* diagVal = (double*)stk::mesh::field_data(*Udiag_, mnode);
    diagVal[0] += lhs[ix];
  }
}

void
MomentumEquationSystem::save_diagonal_term(
  unsigned nEntities,
  const stk::mesh::Entity* entities,
  const SharedMemView<const double**>& lhs)
{
  auto& bulk = realm_.bulk_data();
  const int nDim = realm_.spatialDimension_;
  constexpr bool forceAtomic =
    !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;

  for (unsigned in = 0; in < nEntities; in++) {
    const auto naluID =
      *stk::mesh::field_data(*realm_.naluGlobalId_, entities[in]);
    const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluID);
    int ix = in * nDim;
    double* diagVal = (double*)stk::mesh::field_data(*Udiag_, mnode);
    if (forceAtomic)
      Kokkos::atomic_add(diagVal, lhs(ix, ix));
    else
      diagVal[0] += lhs(ix, ix);
  }
}

#ifndef KOKKOS_ENABLE_CUDA
void
MomentumEquationSystem::save_diagonal_term(
  unsigned nEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<const double**, DeviceShmem>& lhs)
{
  auto& bulk = realm_.bulk_data();
  const int nDim = realm_.spatialDimension_;
  constexpr bool forceAtomic =
    !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;

  for (unsigned in = 0; in < nEntities; in++) {
    const auto naluID =
      *stk::mesh::field_data(*realm_.naluGlobalId_, entities[in]);
    const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluID);
    int ix = in * nDim;
    double* diagVal = (double*)stk::mesh::field_data(*Udiag_, mnode);
    if (forceAtomic)
      Kokkos::atomic_add(diagVal, lhs(ix, ix));
    else
      diagVal[0] += lhs(ix, ix);
  }
}
#else
void
MomentumEquationSystem::save_diagonal_term(
  unsigned,
  const stk::mesh::NgpMesh::ConnectedNodes&,
  const SharedMemView<const double**, DeviceShmem>&)
{
}
#endif

void
MomentumEquationSystem::assemble_and_solve(stk::mesh::FieldBase* deltaSolution)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<>::MeshIndex;
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();

  extractDiagonal_ = (realm_.solutionOptions_->tscaleType_ == TSCALE_UDIAGINV);
  auto ngpUdiag = realm_.ngp_field_manager().get_field<double>(
    Udiag_->mesh_meta_data_ordinal());
  // Reset timescale field before momentum solve
  {
    double projTimeScale = 0.0;
    if (realm_.solutionOptions_->tscaleType_ == TSCALE_DEFAULT) {
      const double dt = realm_.get_time_step();
      const double gamma1 = realm_.get_gamma1();
      projTimeScale = gamma1 / dt;
    }

    ngpUdiag.set_all(realm_.ngp_mesh(), projTimeScale);
    ngpUdiag.modify_on_device();
  }

  // Perform actual solve
  EquationSystem::assemble_and_solve(deltaSolution);

  // Post-process the Udiag term
  ScalarFieldType* dualVol = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  ScalarFieldType* density =
    meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");

  if (realm_.solutionOptions_->tscaleType_ == TSCALE_UDIAGINV) {
    const std::string dofName = "velocity";
    const double dt = realm_.get_time_step();
    const double gamma1 = realm_.get_gamma1();
    const double projTimeScale = gamma1 / dt;
    const double alphaU =
      realm_.solutionOptions_->get_relaxation_factor(dofName);

    // Sum up contributions on the nodes shared amongst processors
    const std::vector<NGPDoubleFieldType*> fVecNgp{&ngpUdiag};
    bool doFinalSyncBackToDevice = true;
    stk::mesh::parallel_sum(
      realm_.bulk_data(), fVecNgp, doFinalSyncBackToDevice);

    const auto sel = stk::mesh::selectField(*Udiag_) &
                     meta.locally_owned_part() &
                     !(stk::mesh::selectUnion(realm_.get_slave_part_vector())) &
                     !(realm_.get_inactive_selector());
    const auto& ngpMesh = realm_.ngp_mesh();
    const auto& fieldMgr = realm_.ngp_field_manager();
    const auto& ngpRho =
      fieldMgr.get_field<double>(density->mesh_meta_data_ordinal());
    const auto& ngpdVol =
      fieldMgr.get_field<double>(dualVol->mesh_meta_data_ordinal());

    // Remove momentum relaxation factor from diagonal term
    nalu_ngp::run_entity_algorithm(
      "LowMach::udiag_post_processing", ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const MeshIndex& mi) {
        double udiagTmp =
          ngpUdiag.get(mi, 0) / (ngpRho.get(mi, 0) * ngpdVol.get(mi, 0));
        ngpUdiag.get(mi, 0) =
          (udiagTmp - projTimeScale) * alphaU + projTimeScale;
      });
    ngpUdiag.modify_on_device();
    ngpUdiag.sync_to_host();

    // Communicate to shared and ghosted nodes (all synchronization on host)
    std::vector<const stk::mesh::FieldBase*> fVec{Udiag_};
    stk::mesh::copy_owned_to_shared(bulk, fVec);
    stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);
    if (realm_.hasPeriodic_) {
      const bool bypassFieldCheck = true;
      const bool addMirrorNodes = false;
      const bool setMirrorNodes = true;
      realm_.periodicManager_->apply_constraints(
        Udiag_, 1, bypassFieldCheck, addMirrorNodes, setMirrorNodes);
    }
    if (
      realm_.nonConformalManager_ != nullptr &&
      realm_.nonConformalManager_->nonConformalGhosting_ != nullptr)
      stk::mesh::communicate_field_data(
        *realm_.nonConformalManager_->nonConformalGhosting_, fVec);
    if (realm_.hasOverset_) {
      const bool doFinalSyncToDevice = false;
      realm_.overset_field_update(Udiag_, 1, 1, doFinalSyncToDevice);
    }

    // Push back to device
    ngpUdiag.modify_on_host();
    ngpUdiag.sync_to_device();
  }
}

void
MomentumEquationSystem::compute_turbulence_parameters()
{
  if (realm_.is_turbulent()) {
    tviscAlg_->execute();
    diffFluxCoeffAlg_->execute();
  }
}

//==========================================================================
// Class Definition
//==========================================================================
// ContinuityEquationSystem - manages p pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ContinuityEquationSystem::ContinuityEquationSystem(
  EquationSystems& eqSystems, const bool elementContinuityEqs)
  : EquationSystem(eqSystems, "ContinuityEQS", "continuity"),
    elementContinuityEqs_(elementContinuityEqs),
    managePNG_(realm_.get_consistent_mass_matrix_png("pressure")),
    pressure_(NULL),
    dpdx_(NULL),
    massFlowRate_(NULL),
    coordinates_(NULL),
    pTmp_(NULL),
    nodalGradAlgDriver_(realm_, "dpdx"),
    mdotAlgDriver_(new MdotAlgDriver(realm_, elementContinuityEqs)),
    projectedNodalGradEqs_(NULL)
{
  dofName_ = "pressure";

  // message to user
  if (realm_.realmUsesEdges_ && elementContinuityEqs_)
    NaluEnv::self().naluOutputP0()
      << "Edge scheme active (all scalars); element-based (continuity)!"
      << std::endl;

  // error check
  if (!elementContinuityEqs_ && !realm_.realmUsesEdges_)
    throw std::runtime_error("If using the non-element-based continuity "
                             "system, edges must be active at realm level");

  // extract solver name and solver object
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("pressure");
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, realm_.name(), EQ_CONTINUITY);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("pressure");
  NaluEnv::self().naluOutputP0()
    << "Edge projected nodal gradient for pressure: " << edgeNodalGradient_
    << std::endl;

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
ContinuityEquationSystem::~ContinuityEquationSystem() {}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_nodal_fields(stk::mesh::Part* part)
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // register dof; set it as a restart variable
  const int numStates = 2;
  pressure_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "pressure", numStates));
  stk::mesh::put_field_on_mesh(*pressure_, *part, nullptr);
  realm_.augment_restart_variable_list("pressure");

  dpdx_ = &(
    meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx"));
  stk::mesh::put_field_on_mesh(*dpdx_, *part, nDim, nullptr);

  // delta solution for linear solver; share delta with other split systems
  pTmp_ = &(
    meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pTmp"));
  stk::mesh::put_field_on_mesh(*pTmp_, *part, nullptr);

  coordinates_ = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates"));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_element_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_element_fields(
  stk::mesh::Part* /* part */, const stk::topology& /* theTopo */)
{
  // nothing as of yet
}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_edge_fields(stk::mesh::Part* part)
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  massFlowRate_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::EDGE_RANK, "mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*massFlowRate_, *part, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_interior_algorithm(stk::mesh::Part* part)
{

  // non-solver, dpdx
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType& pressureNp1 = pressure_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  bool lumpedMass = false;
  bool hasMass = false;

  // non-solver; contribution to Gjp; allow for element-based shifted
  if (!managePNG_) {
    if (!elementContinuityEqs_ && edgeNodalGradient_)
      nodalGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
        algType, part, "continuity_nodal_grad", &pressureNp1, &dpdxNone);
    else
      nodalGradAlgDriver_.register_elem_algorithm<ScalarNodalGradElemAlg>(
        algType, part, "continuity_nodal_grad", &pressureNp1, &dpdxNone,
        edgeNodalGradient_);
  }

  if (!elementContinuityEqs_) {

    // pure edge-based scheme

    mdotAlgDriver_->register_edge_algorithm<MdotEdgeAlg>(
      algType, part, "mdot_edge_interior");

    auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
    const auto it = solverAlgMap.find(algType);
    if (it == solverAlgMap.end()) {
      solverAlgMap[algType] = new ContinuityEdgeSolverAlg(realm_, part, this);
    } else {
      it->second->partVec_.push_back(part);
    }
  } else {
    throw std::runtime_error("ContinuityEQS: Element algorithms not supported");
  }

  // time term using lumped mass
  std::map<std::string, std::vector<std::string>>::iterator isrc =
    realm_.solutionOptions_->srcTermsMap_.find("continuity");
  if (isrc != realm_.solutionOptions_->srcTermsMap_.end()) {
    // Handle error checking during transition period. Some kernels are handled
    // through the NGP-ready interface while others are handled via legacy
    // interface and only supported on CPUs.
    int ngpSrcSkipped = 0;
    int nonNgpSrcSkipped = 0;
    int numUsrSrc = 0;
    {
      auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
      process_ngp_node_kernels(
        solverAlgMap, realm_, part, this,
        [&](AssembleNGPNodeSolverAlgorithm&) {
          // Time derivative terms not yet implemented
        },
        [&](AssembleNGPNodeSolverAlgorithm& nodeAlg, std::string& srcName) {
          bool added = true;
          if (srcName == "gcl") {
            nodeAlg.add_kernel<ContinuityGclNodeKernel>(realm_.bulk_data());
          } else if (srcName == "density_time_derivative") {
            nodeAlg.add_kernel<ContinuityMassBDFNodeKernel>(realm_.bulk_data());
            hasMass = true;
            lumpedMass = true;
          } else {
            added = false;
            ++ngpSrcSkipped;
          }

          if (added)
            NaluEnv::self().naluOutputP0() << " - " << srcName << std::endl;
        });
    }
    const AlgorithmType algMass = SRC;
    std::map<AlgorithmType, SolverAlgorithm*>::iterator itsm =
      solverAlgDriver_->solverAlgMap_.find(algMass);
    if (itsm == solverAlgDriver_->solverAlgMap_.end()) {
      // create the solver alg
      AssembleNodeSolverAlgorithm* theAlg =
        new AssembleNodeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algMass] = theAlg;
      std::vector<std::string> mapNameVec = isrc->second;
      numUsrSrc = mapNameVec.size();
      for (size_t k = 0; k < mapNameVec.size(); ++k) {
        std::string sourceName = mapNameVec[k];

        SupplementalAlgorithm* suppAlg = NULL;
        if (sourceName == "low_speed_compressible") {
          suppAlg = new ContinuityLowSpeedCompressibleNodeSuppAlg(realm_);
        } else if (sourceName == "VariableDensity") {
          suppAlg = new VariableDensityContinuitySrcNodeSuppAlg(realm_);
        } else if (sourceName == "VariableDensityNonIso") {
          suppAlg = new VariableDensityNonIsoContinuitySrcNodeSuppAlg(realm_);
        } else {
          ++nonNgpSrcSkipped;
        }
        if (suppAlg != NULL) {
          NaluEnv::self().naluOutputP0()
            << "ContinuityNodalSrcTerms::added " << sourceName << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg);
        }
      }
    } else {
      itsm->second->partVec_.push_back(part);
    }
    // Ensure that all user source terms were processed by either interface
    if ((ngpSrcSkipped + nonNgpSrcSkipped) != numUsrSrc)
      throw std::runtime_error(
        "Error processing nodal source terms for Continuity");
  }

  // Register density accumulation calculations if the user has requested
  // density time derivative terms.
  if (hasMass)
    mdotAlgDriver_->register_elem_algorithm<MdotDensityAccumAlg>(
      algType, part, "mdot_rho_accum", *mdotAlgDriver_, lumpedMass);
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_inflow_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const InflowBoundaryConditionData& inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType& pressureNp1 = pressure_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();

  // register boundary data; cont_velocity_bc
  if (!realm_.solutionOptions_->activateOpenMdotCorrection_) {
    VectorFieldType* theBcField = &(meta_data.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "cont_velocity_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

    // extract the value for user specified velocity and save off the
    // AuxFunction
    InflowUserData userData = inflowBCData.userData_;
    std::string velocityName = "velocity";
    UserDataType theDataType = get_bc_data_type(userData, velocityName);

    AuxFunction* theAuxFunc = NULL;
    if (CONSTANT_UD == theDataType) {
      Velocity ux = userData.u_;
      std::vector<double> userSpec(nDim);
      userSpec[0] = ux.ux_;
      userSpec[1] = ux.uy_;
      if (nDim > 2)
        userSpec[2] = ux.uz_;

      // new it
      theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);
    } else if (FUNCTION_UD == theDataType) {
      // extract the name/params
      std::string fcnName = get_bc_function_name(userData, velocityName);
      std::vector<double> theParams =
        get_bc_function_params(userData, velocityName);
      if (theParams.size() == 0)
        NaluEnv::self().naluOutputP0()
          << "function parameter size is zero" << std::endl;
      // switch on the name found...
      if (fcnName == "convecting_taylor_vortex") {
        theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0, nDim);
      } else if (fcnName == "SteadyTaylorVortex") {
        theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0, nDim);
      } else if (fcnName == "VariableDensity") {
        theAuxFunc = new VariableDensityVelocityAuxFunction(0, nDim);
      } else if (fcnName == "VariableDensityNonIso") {
        theAuxFunc = new VariableDensityVelocityAuxFunction(0, nDim);
      } else if (fcnName == "kovasznay") {
        theAuxFunc = new KovasznayVelocityAuxFunction(0, nDim);
      } else if (fcnName == "TaylorGreen") {
        theAuxFunc = new TaylorGreenVelocityAuxFunction(0, nDim);
      } else if (fcnName == "BoussinesqNonIso") {
        theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0, nDim);
      } else if (fcnName == "wind_energy_power_law") {
        theAuxFunc = new WindEnergyPowerLawAuxFunction(0, nDim, theParams);
      } else if (fcnName == "GaussJet") {
        theAuxFunc = new GaussJetVelocityAuxFunction(0, nDim);
      } else {
        throw std::runtime_error("ContEquationSystem::register_inflow_bc: "
                                 "limited functions supported");
      }
    } else {
      throw std::runtime_error("ContEquationSystem::register_inflow_bc: only "
                               "constant and user function supported");
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
  }

  // non-solver; contribution to Gjp; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "continuity_nodal_grad", &pressureNp1, &dpdxNone,
      edgeNodalGradient_);
  }

  // check to see if we are using shifted as inflow is shared
  const bool useShifted =
    !elementContinuityEqs_ ? true : realm_.get_cvfem_shifted_mdot();

  mdotAlgDriver_->register_face_algorithm<MdotInflowAlg>(
    algType, part, "mdot_inflow", *mdotAlgDriver_, useShifted);

  // solver; lhs
  if (
    realm_.solutionOptions_->useConsolidatedBcSolverAlg_ ||
    realm_.realmUsesEdges_) {

    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(solverAlg, solverAlgWasBuilt) =
      build_or_add_part_to_face_bc_solver_alg(
        *this, *part, solverAlgMap, "inflow");

    ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
    auto& activeKernels = solverAlg->activeKernels_;

    if (solverAlgWasBuilt) {
      build_face_topo_kernel_automatic<ContinuityInflowElemKernel>(
        partTopo, *this, activeKernels, "continuity_inflow", realm_.bulk_data(),
        *realm_.solutionOptions_, useShifted, dataPreReqs);
    }
  } else {
    throw std::runtime_error(
      "MomentumEQS: Attempt to use non-NGP element inflow algorithm");
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const OpenBoundaryConditionData&)
{

  const AlgorithmType algType = OPEN;

  // register boundary data
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  ScalarFieldType* pressureBC = NULL;
  if (!realm_.solutionOptions_->activateOpenMdotCorrection_) {
    pressureBC = &(meta_data.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "pressure_bc"));
    stk::mesh::put_field_on_mesh(*pressureBC, *part, nullptr);
  }

  // non-solver; contribution to Gjp; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_elem_algorithm<NodalGradPOpenBoundary>(
      algType, part, get_elem_topo(realm_, *part), "continuity_nodal_grad",
      edgeNodalGradient_);
  }

  // mdot at open and lhs
  if (!elementContinuityEqs_) {
    // non-solver edge alg; compute open mdot
    mdotAlgDriver_->register_open_mdot_algorithm<MdotOpenEdgeAlg>(
      algType, part, get_elem_topo(realm_, *part), "mdot_open_edge",
      realm_.solutionOptions_->activateOpenMdotCorrection_, *mdotAlgDriver_);

    {
      auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

      stk::topology elemTopo = get_elem_topo(realm_, *part);

      AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
      bool solverAlgWasBuilt = false;
      const std::string algName = "open_edge";

      std::tie(faceElemSolverAlg, solverAlgWasBuilt) =
        build_or_add_part_to_face_elem_solver_alg(
          algType, *this, *part, elemTopo, solverAlgMap, algName);

      auto& activeKernels = faceElemSolverAlg->activeKernels_;

      if (solverAlgWasBuilt)
        build_face_elem_topo_kernel_automatic<ContinuityOpenEdgeKernel>(
          partTopo, elemTopo, *this, activeKernels, "continuity_open_edge",
          realm_.meta_data(), realm_.solutionOptions_,
          faceElemSolverAlg->faceDataNeeded_,
          faceElemSolverAlg->elemDataNeeded_);
    }
  } else {
    throw std::runtime_error(
      "ContinuityEQS: Attempt to use element open algorithm");
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const WallBoundaryConditionData& /* wallBCData */)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  ScalarFieldType& pressureNp1 = pressure_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "continuity_nodal_grad", &pressureNp1, &dpdxNone,
      edgeNodalGradient_);
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_symmetry_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const SymmetryBoundaryConditionData& /* symmetryBCData */)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  ScalarFieldType& pressureNp1 = pressure_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "continuity_nodal_grad", &pressureNp1, &dpdxNone,
      edgeNodalGradient_);
  }
}

//--------------------------------------------------------------------------
//-------- register_abltop_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_abltop_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const ABLTopBoundaryConditionData& abltopBCData)
{
  auto userData = abltopBCData.userData_;

  if (!userData.ABLTopBC_) {
    SymmetryBoundaryConditionData symData;
    register_symmetry_bc(part, partTopo, symData);
    return;
  }

#ifdef NALU_USES_FFTW
  // algorithm type
  const AlgorithmType algType = TOP_ABL;

  if (!realm_.realmUsesEdges_)
    throw std::runtime_error(
      "ABLTopBoundaryCondition::Error you must use the edge-based scheme");

  ScalarFieldType& pressureNp1 = pressure_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "continuity_nodal_grad", &pressureNp1, &dpdxNone,
      edgeNodalGradient_);
  }

  // check to see if we are using shifted as inflow is shared
  const bool useShifted =
    !elementContinuityEqs_ ? true : realm_.get_cvfem_shifted_mdot();

  // non-solver inflow mdot - shared by both elem/edge
  mdotAlgDriver_->register_face_algorithm<MdotInflowAlg>(
    algType, part, "mdot_inflow", *mdotAlgDriver_, useShifted);

  // solver; lhs
  auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

  AssembleElemSolverAlgorithm* solverAlg = nullptr;
  bool solverAlgWasBuilt = false;

  std::tie(solverAlg, solverAlgWasBuilt) =
    build_or_add_part_to_face_bc_solver_alg(
      *this, *part, solverAlgMap, "inflow");

  ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
  auto& activeKernels = solverAlg->activeKernels_;

  if (solverAlgWasBuilt) {
    build_face_topo_kernel_automatic<ContinuityInflowElemKernel>(
      partTopo, *this, activeKernels, "continuity_inflow", realm_.bulk_data(),
      *realm_.solutionOptions_, useShifted, dataPreReqs);
  }
#else
  throw std::runtime_error(
    "Cannot initialize ABL top BC because FFTW support is mising.\n Set "
    "ENABLE_FFTW to ON in nalu-wind/CMakeLists.txt, reconfigure and "
    "recompile.");
#endif
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_non_conformal_bc(
  stk::mesh::Part* part, const stk::topology& theTopo)
{
  const AlgorithmType algType = NON_CONFORMAL;

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // mdot at nc bc; register field; require topo and num ips
  MasterElement* meFC =
    sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsBip = meFC->num_integration_points();

  stk::topology::rank_t sideRank =
    static_cast<stk::topology::rank_t>(meta_data.side_rank());
  GenericFieldType* mdotBip =
    &(meta_data.declare_field<GenericFieldType>(sideRank, "nc_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*mdotBip, *part, numScsBip, nullptr);

  // non-solver; contribution to Gjp; DG algorithm decides on locations for
  // integration points
  if (!managePNG_) {
    if (edgeNodalGradient_) {
      nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
        algType, part, "continuity_nodal_grad", pressure_, dpdx_,
        edgeNodalGradient_);
    } else {
      // proceed with DG
      nodalGradAlgDriver_
        .register_legacy_algorithm<AssembleNodalGradNonConformalAlgorithm>(
          algType, part, "continuity_nodal_grad", pressure_, dpdx_);
    }
  }

  // non-solver alg; compute nc mdot (same for edge and element-based)
  mdotAlgDriver_->register_legacy_algorithm<ComputeMdotNonConformalAlgorithm>(
    algType, part, "mdot_non_conformal", pressure_, dpdx_);

  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
    AssembleContinuityNonConformalSolverAlgorithm* theAlg =
      new AssembleContinuityNonConformalSolverAlgorithm(
        realm_, part, this, pressure_, dpdx_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    itsi->second->partVec_.push_back(part);
  }
}
//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_overset_bc()
{
  if (decoupledOverset_)
    EquationSystem::create_constraint_algorithm(pressure_);
  else
    create_constraint_algorithm(pressure_);

  equationSystems_.register_overset_field_update(pressure_, 1, 1);
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::initialize()
{
  if (realm_.solutionOptions_->needPressureReference_) {
    const AlgorithmType algType = REF_PRESSURE;
    // Process parts if necessary
    realm_.solutionOptions_->fixPressureInfo_->create_part_vector(
      realm_.meta_data());
    stk::mesh::PartVector& pvec =
      realm_.solutionOptions_->fixPressureInfo_->partVec_;

    // The user could have provided just a Node ID instead of a part vector
    stk::mesh::Part* firstPart = pvec.size() > 0 ? pvec.at(0) : nullptr;

    auto it = solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverDirichAlgMap_.end()) {
      FixPressureAtNodeAlgorithm* theAlg =
        new FixPressureAtNodeAlgorithm(realm_, firstPart, this);
      // populate the remaining parts if necessary
      for (size_t i = 1; i < pvec.size(); i++)
        theAlg->partVec_.push_back(pvec[i]);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    } else {
      throw std::runtime_error(
        "ContinuityEquationSystem::initialize: logic error. Multiple "
        "initializations of FixPressureAtNodeAlgorithm.");
    }
  }

  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::reinitialize_linear_system()
{
  // If this is decoupled overset simulation and the user has requested that the
  // linear system be reused, then do nothing
  if (decoupledOverset_ && linsys_->config().reuseLinSysIfPossible())
    return;

  // delete linsys
  delete linsys_;

  // create new solver
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("pressure");
  LinearSolver* solver = realm_.root()->linearSolvers_->reinitialize_solver(
    solverName, realm_.name(), EQ_CONTINUITY);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_initial_condition_fcn(
  stk::mesh::Part* part,
  const std::map<std::string, std::string>& theNames,
  const std::map<std::string, std::vector<double>>& theParams)
{
  // iterate map and check for name
  const std::string dofName = "pressure";
  std::map<std::string, std::string>::const_iterator iterName =
    theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;
    AuxFunction* theAuxFunc = NULL;
    if (fcnName == "convecting_taylor_vortex") {
      // create the function
      theAuxFunc = new ConvectingTaylorVortexPressureAuxFunction();
    } else if (fcnName == "wind_energy_taylor_vortex") {
      // extract the params
      auto iterParams = theParams.find(dofName);
      std::vector<double> fcnParams = (iterParams != theParams.end())
                                        ? (*iterParams).second
                                        : std::vector<double>();
      theAuxFunc = new WindEnergyTaylorVortexPressureAuxFunction(fcnParams);
    } else if (fcnName == "SteadyTaylorVortex") {
      // create the function
      theAuxFunc = new SteadyTaylorVortexPressureAuxFunction();
    } else if (fcnName == "VariableDensity") {
      // create the function
      theAuxFunc = new VariableDensityPressureAuxFunction();
    } else if (fcnName == "VariableDensityNonIso") {
      // create the function
      theAuxFunc = new VariableDensityPressureAuxFunction();
    } else if (fcnName == "TaylorGreen") {
      // create the function
      theAuxFunc = new TaylorGreenPressureAuxFunction();
    } else if (fcnName == "kovasznay") {
      theAuxFunc = new KovasznayPressureAuxFunction();
    } else {
      throw std::runtime_error("ContinuityEquationSystem::register_initial_"
                               "condition_fcn: limited functions supported");
    }

    // create the algorithm
    AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
      realm_, part, pressure_, theAuxFunc, stk::topology::NODE_RANK);

    // push to ic
    realm_.initCondAlg_.push_back(auxAlg);
  }
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if (NULL == projectedNodalGradEqs_) {
    projectedNodalGradEqs_ = new ProjectedNodalGradientEquationSystem(
      eqSystems, EQ_PNG_P, "dpdx", "qTmp", "pressure", "PNGradPEQS");
  }
  // fill the map for expected boundary condition names...
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "pressure");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "pressure");
  projectedNodalGradEqs_->set_data_map(
    OPEN_BC, realm_.solutionOptions_->activateOpenMdotCorrection_
               ? "pressure"
               : "pressure_bc");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "pressure");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::compute_projected_nodal_gradient()
{
  if (!managePNG_) {
    const double timeA = -NaluEnv::self().nalu_time();
    nodalGradAlgDriver_.execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  } else {
    projectedNodalGradEqs_->solve_and_update_external();
  }
}

void
ContinuityEquationSystem::create_constraint_algorithm(
  stk::mesh::FieldBase* theField)
{
  const AlgorithmType algType = OVERSET;

  auto it = solverAlgDriver_->solverConstraintAlgMap_.find(algType);
  if (it == solverAlgDriver_->solverConstraintAlgMap_.end()) {
    AssembleOversetPressureAlgorithm* theAlg =
      new AssembleOversetPressureAlgorithm(realm_, nullptr, this, theField);
    solverAlgDriver_->solverConstraintAlgMap_[algType] = theAlg;
  } else {
    throw std::runtime_error("ContinuityEquationSystem::register_overset_bc: "
                             "Multiple invocations of overset is not allowed");
  }
}

} // namespace nalu
} // namespace sierra
