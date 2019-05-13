/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <LowMachEquationSystem.h>
#include <wind_energy/ABLForcingAlgorithm.h>
#include <AlgorithmDriver.h>
#include <AssembleCourantReynoldsElemAlgorithm.h>
#include <AssembleContinuityEdgeSolverAlgorithm.h>
#include <AssembleContinuityElemSolverAlgorithm.h>
#include <AssembleContinuityInflowSolverAlgorithm.h>
#include <AssembleContinuityEdgeOpenSolverAlgorithm.h>
#include <AssembleContinuityElemOpenSolverAlgorithm.h>
#include <AssembleContinuityNonConformalSolverAlgorithm.h>
#include <AssembleMomentumEdgeSolverAlgorithm.h>
#include <AssembleMomentumElemSolverAlgorithm.h>
#include <AssembleMomentumEdgeOpenSolverAlgorithm.h>
#include <AssembleMomentumElemOpenSolverAlgorithm.h>
#include <AssembleMomentumEdgeSymmetrySolverAlgorithm.h>
#include <AssembleMomentumElemSymmetrySolverAlgorithm.h>
#include <AssembleMomentumEdgeWallFunctionSolverAlgorithm.h>
#include <AssembleMomentumElemWallFunctionSolverAlgorithm.h>
#include <AssembleMomentumElemABLWallFunctionSolverAlgorithm.h>
#include <AssembleMomentumEdgeABLWallFunctionSolverAlgorithm.h>
#ifdef NALU_USES_FFTW
#include <AssembleMomentumEdgeABLTopBC.h>
#endif
#include <AssembleMomentumNonConformalSolverAlgorithm.h>
#include <AssembleNodalGradAlgorithmDriver.h>
#include <AssembleNodalGradPAlgorithmDriver.h>
#include <AssembleNodalGradEdgeAlgorithm.h>
#include <AssembleNodalGradElemAlgorithm.h>
#include <AssembleNodalGradBoundaryAlgorithm.h>
#include <AssembleNodalGradPOpenBoundaryAlgorithm.h>
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <AssembleNodalGradUAlgorithmDriver.h>
#include <AssembleNodalGradUEdgeAlgorithm.h>
#include <AssembleNodalGradUElemAlgorithm.h>
#include <AssembleNodalGradUBoundaryAlgorithm.h>
#include <AssembleNodalGradUNonConformalAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
#include <AuxFunctionAlgorithm.h>
#include <ComputeMdotAlgorithmDriver.h>
#include <ComputeMdotInflowAlgorithm.h>
#include <ComputeMdotEdgeAlgorithm.h>
#include <ComputeMdotElemAlgorithm.h>
#include <ComputeMdotEdgeOpenAlgorithm.h>
#include <ComputeMdotElemOpenAlgorithm.h>
#include <ComputeMdotElemOpenPenaltyAlgorithm.h>
#include <ComputeMdotNonConformalAlgorithm.h>
#include <ComputeWallFrictionVelocityAlgorithm.h>
#include <ComputeABLWallFrictionVelocityAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <ContinuityGclNodeSuppAlg.h>
#include <ContinuityLowSpeedCompressibleNodeSuppAlg.h>
#include <ContinuityMassBackwardEulerNodeSuppAlg.h>
#include <ContinuityMassBDF2NodeSuppAlg.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
#include <EffectiveDiffFluxCoeffAlgorithm.h>
#include <Enums.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <ErrorIndicatorAlgorithmDriver.h>
#include <FieldFunctions.h>
#include <LinearSolver.h>
#include <LinearSolvers.h>
#include <LinearSystem.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <MomentumActuatorSrcNodeSuppAlg.h>
#include <MomentumBuoyancySrcNodeSuppAlg.h>
#include <MomentumBoussinesqSrcNodeSuppAlg.h>
#include <MomentumBoussinesqRASrcNodeSuppAlg.h>
#include <MomentumBodyForceSrcNodeSuppAlg.h>
#include <MomentumABLForceSrcNodeSuppAlg.h>
#include <MomentumCoriolisSrcNodeSuppAlg.h>
#include <MomentumGclSrcNodeSuppAlg.h>
#include <MomentumMassBackwardEulerNodeSuppAlg.h>
#include <MomentumMassBDF2NodeSuppAlg.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <NonConformalManager.h>
#include <NonConformalInfo.h>
#include <ProjectedNodalGradientEquationSystem.h>
#include <PostProcessingData.h>
#include <PstabErrorIndicatorEdgeAlgorithm.h>
#include <PstabErrorIndicatorElemAlgorithm.h>
#include <LimiterErrorIndicatorElemAlgorithm.h>
#include <SimpleErrorIndicatorElemAlgorithm.h>
#include <Realm.h>
#include <Realms.h>
#include <SurfaceForceAndMomentAlgorithmDriver.h>
#include <SurfaceForceAndMomentAlgorithm.h>
#include <SurfaceForceAndMomentWallFunctionAlgorithm.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>
#include <TurbViscKsgsAlgorithm.h>
#include <TurbViscSmagorinskyAlgorithm.h>
#include <TurbViscSSTAlgorithm.h>
#include <TurbViscWaleAlgorithm.h>
#include <wind_energy/ABLForcingAlgorithm.h>
#include <FixPressureAtNodeAlgorithm.h>
#include <FixPressureAtNodeInfo.h>

// template for kernels
#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

// kernels
#include <kernel/ContinuityAdvElemKernel.h>
#include <kernel/ContinuityMassElemKernel.h>
#include <kernel/MomentumAdvDiffElemKernel.h>
#include <kernel/MomentumActuatorSrcElemKernel.h>
#include <kernel/MomentumBuoyancyBoussinesqSrcElemKernel.h>
#include <kernel/MomentumBuoyancySrcElemKernel.h>
#include <kernel/MomentumCoriolisSrcElemKernel.h>
#include <kernel/MomentumMassElemKernel.h>
#include <kernel/MomentumUpwAdvDiffElemKernel.h>

#include <kernel/MomentumMassHOElemKernel.h>
#include <kernel/MomentumAdvDiffHOElemKernel.h>
#include <kernel/MomentumBuoyancySrcHOElemKernel.h>
#include <kernel/PressurePoissonHOElemKernel.h>
#include <kernel/ContinuityMassHOElemKernel.h>

//mms kernels
#include <user_functions/TGMMSHOElemKernel.h>


// bc kernels
#include <kernel/ContinuityInflowElemKernel.h>
#include <kernel/ContinuityOpenElemKernel.h>
#include <kernel/MomentumOpenAdvDiffElemKernel.h>
#include <kernel/MomentumSymmetryElemKernel.h>
#include <kernel/MomentumWallFunctionElemKernel.h>

// nso
#include <nso/MomentumNSOElemKernel.h>
#include <nso/MomentumNSOKeElemKernel.h>
#include <nso/MomentumNSOSijElemKernel.h>
#include <nso/MomentumNSOGradElemSuppAlg.h>

// hybrid turbulence
#include <kernel/MomentumHybridTurbElemKernel.h>

// user function
#include <user_functions/ConvectingTaylorVortexVelocityAuxFunction.h>
#include <user_functions/ConvectingTaylorVortexPressureAuxFunction.h>
#include <user_functions/TornadoAuxFunction.h>

#include <user_functions/WindEnergyTaylorVortexAuxFunction.h>
#include <user_functions/WindEnergyTaylorVortexPressureAuxFunction.h>

#include <user_functions/SteadyTaylorVortexMomentumSrcElemSuppAlg.h>
#include <user_functions/SteadyTaylorVortexContinuitySrcElemSuppAlg.h>
#include <user_functions/SteadyTaylorVortexMomentumSrcNodeSuppAlg.h>
#include <user_functions/SteadyTaylorVortexVelocityAuxFunction.h>
#include <user_functions/SteadyTaylorVortexPressureAuxFunction.h>

#include <user_functions/VariableDensityVelocityAuxFunction.h>
#include <user_functions/VariableDensityPressureAuxFunction.h>
#include <user_functions/VariableDensityContinuitySrcElemSuppAlg.h>
#include <user_functions/VariableDensityContinuitySrcNodeSuppAlg.h>
#include <user_functions/VariableDensityMomentumSrcElemSuppAlg.h>
#include <user_functions/VariableDensityMomentumSrcNodeSuppAlg.h>

#include <user_functions/VariableDensityMomentumMMSHOElemKernel.h>
#include <user_functions/VariableDensityContinuityMMSHOElemKernel.h>

#include <user_functions/VariableDensityNonIsoContinuitySrcNodeSuppAlg.h>
#include <user_functions/VariableDensityNonIsoMomentumSrcNodeSuppAlg.h>
#include <user_functions/BoussinesqNonIsoMomentumSrcNodeSuppAlg.h>

#include <user_functions/TaylorGreenPressureAuxFunction.h>
#include <user_functions/TaylorGreenVelocityAuxFunction.h>

#include <user_functions/BoussinesqNonIsoVelocityAuxFunction.h>

#include <user_functions/SinProfileChannelFlowVelocityAuxFunction.h>

#include <user_functions/BoundaryLayerPerturbationAuxFunction.h>

#include <user_functions/KovasznayVelocityAuxFunction.h>
#include <user_functions/KovasznayPressureAuxFunction.h>

#include <overset/UpdateOversetFringeAlgorithmDriver.h>
#include <overset/AssembleOversetPressureAlgorithm.h>

#include <user_functions/OneTwoTenVelocityAuxFunction.h>

#include <user_functions/PerturbedShearLayerAuxFunctions.h>

// deprecated
#include <ContinuityMassElemSuppAlgDep.h>
#include <MomentumMassElemSuppAlgDep.h>
#include <MomentumBuoyancySrcElemSuppAlgDep.h>
#include <nso/MomentumNSOKeElemSuppAlgDep.h>
#include <nso/MomentumNSOElemSuppAlgDep.h>

// stk_util
#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_util/util/SortAndUnique.hpp>

// stk_mesh/base/fem
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

// stk_topo
#include <stk_topology/topology.hpp>

#include <utils/StkHelpers.h>

// basic c++
#include <vector>


namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// LowMachEquationSystem - manage the low Mach equation system (uvw_p)
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
LowMachEquationSystem::LowMachEquationSystem(
  EquationSystems& eqSystems,
  const bool elementContinuityEqs)
  : EquationSystem(eqSystems, "LowMachEOSWrap","low_mach_type"),
    elementContinuityEqs_(elementContinuityEqs),
    density_(NULL),
    viscosity_(NULL),
    dualNodalVolume_(NULL),
    edgeAreaVec_(NULL),
    surfaceForceAndMomentAlgDriver_(NULL),
    xyBCType_(2,0),
    isInit_(true)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create momentum and pressure
  momentumEqSys_= new MomentumEquationSystem(eqSystems);
  continuityEqSys_ = new ContinuityEquationSystem(eqSystems, elementContinuityEqs_);

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
  if ( NULL != surfaceForceAndMomentAlgDriver_ )
    delete surfaceForceAndMomentAlgDriver_;
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
LowMachEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // add properties; denisty needs to be a restart field
  const int numStates = realm_.number_of_states();
  density_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "density", numStates));
  stk::mesh::put_field_on_mesh(*density_, *part, nullptr);
  realm_.augment_restart_variable_list("density");

  viscosity_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*viscosity_, *part, nullptr);

  // push to property list
  realm_.augment_property_map(DENSITY_ID, density_);
  realm_.augment_property_map(VISCOSITY_ID, viscosity_);

  // dual nodal volume (should push up...)
  const int numVolStates = realm_.does_mesh_move() ? realm_.number_of_states() : 1;
  dualNodalVolume_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume", numVolStates));
  stk::mesh::put_field_on_mesh(*dualNodalVolume_, *part, nullptr);
  if (numVolStates > 1) realm_.augment_restart_variable_list("dual_nodal_volume");

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &densityN = density_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlgDens
      = new CopyFieldAlgorithm(realm_, part,
                               &densityNp1, &densityN,
                               0, 1,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlgDens);

    if ( numVolStates <= 2 ) return;

    ScalarFieldType &dualNdVolN = dualNodalVolume_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &dualNdVolNp1 = dualNodalVolume_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlgDlNdVol
      = new CopyFieldAlgorithm(realm_, part,
                               &dualNdVolNp1, &dualNdVolN,
                               0, 1,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlgDlNdVol);
  }
}

//--------------------------------------------------------------------------
//-------- register_element_fields -------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_element_fields(
  stk::mesh::Part *part,
  const stk::topology &theTopo)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register mdot for element-based scheme...
  if ( elementContinuityEqs_ ) {
    // extract master element and get scs points
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
    const int numScsIp = meSCS->num_integration_points();
    GenericFieldType *massFlowRate = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "mass_flow_rate_scs"));
    stk::mesh::put_field_on_mesh(*massFlowRate, *part, numScsIp , nullptr);
  }

  // deal with fluids error indicator; elemental field of size unity
  if ( realm_.solutionOptions_->activateAdaptivity_) {
    const int numIp = 1;
    GenericFieldType *pstabEI= &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "error_indicator"));
    stk::mesh::put_field_on_mesh(*pstabEI, *part, numIp, nullptr);
  }

  // register the intersected elemental field
  if ( realm_.query_for_overset() ) {
    const int sizeOfElemField = 1;
    GenericFieldType *intersectedElement
      = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "intersected_element"));
    stk::mesh::put_field_on_mesh(*intersectedElement, *part, sizeOfElemField, nullptr);
  }

  // provide mean element Peclet and Courant fields; always...
  GenericFieldType *elemReynolds
    = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "element_reynolds"));
  stk::mesh::put_field_on_mesh(*elemReynolds, *part, 1, nullptr);
  GenericFieldType *elemCourant
    = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "element_courant"));
  stk::mesh::put_field_on_mesh(*elemCourant, *part, 1, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_edge_fields(
  stk::mesh::Part *part)
{

  if ( realm_.realmUsesEdges_ ) {
    stk::mesh::MetaData &meta_data = realm_.meta_data();
    const int nDim = meta_data.spatial_dimension();
    edgeAreaVec_ = &(meta_data.declare_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector"));
    stk::mesh::put_field_on_mesh(*edgeAreaVec_, *part, nDim, nullptr);
  }

}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{
  // types of algorithms
  const AlgorithmType algType = INTERIOR;
  if ( realm_.solutionOptions_->activateAdaptivity_) {

    // non-solver alg
    std::map<AlgorithmType, Algorithm *>::iterator it
      = realm_.errorIndicatorAlgDriver_->algMap_.find(algType);
    if ( it == realm_.errorIndicatorAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( realm_.solutionOptions_->errorIndicatorType_ & EIT_PSTAB ) {
        if ( realm_.realmUsesEdges_)
          theAlg = new PstabErrorIndicatorEdgeAlgorithm(realm_, part, continuityEqSys_->pressure_, continuityEqSys_->dpdx_);
        else
          theAlg = new PstabErrorIndicatorElemAlgorithm(realm_, part, continuityEqSys_->pressure_, continuityEqSys_->dpdx_);
      }
      else if ( realm_.solutionOptions_->errorIndicatorType_ & EIT_LIMITER ) {
        theAlg = new LimiterErrorIndicatorElemAlgorithm(realm_, part);
      }
      else if ( realm_.solutionOptions_->errorIndicatorType_ & EIT_SIMPLE_BASE ) {
        theAlg = new SimpleErrorIndicatorElemAlgorithm(realm_, part);
      }
      realm_.errorIndicatorAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &theTopo,
  const OpenBoundaryConditionData &openBCData)
{

  // register boundary data
  stk::mesh::MetaData &metaData = realm_.meta_data();

  const int nDim = metaData.spatial_dimension();

  VectorFieldType *velocityBC = &(metaData.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "open_velocity_bc"));
  stk::mesh::put_field_on_mesh(*velocityBC, *part, nDim, nullptr);

  // extract the value for user specified velocity and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  Velocity ux = userData.u_;
  std::vector<double> userSpecUbc(nDim);
  userSpecUbc[0] = ux.ux_;
  userSpecUbc[1] = ux.uy_;
  if ( nDim > 2)
    userSpecUbc[2] = ux.uz_;

  // new it
  ConstantAuxFunction *theAuxFuncUbc = new ConstantAuxFunction(0, nDim, userSpecUbc);

  // bc data alg
  AuxFunctionAlgorithm *auxAlgUbc
    = new AuxFunctionAlgorithm(realm_, part,
                               velocityBC, theAuxFuncUbc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlgUbc);

  // extract the value for user specified pressure and save off the AuxFunction
  if ( !realm_.solutionOptions_->activateOpenMdotCorrection_ ) {
    ScalarFieldType *pressureBC
      = &(metaData.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure_bc"));
    stk::mesh::put_field_on_mesh(*pressureBC, *part , nullptr);
    
    Pressure pSpec = userData.p_;
    std::vector<double> userSpecPbc(1);
    userSpecPbc[0] = pSpec.pressure_;
    
    // new it
    ConstantAuxFunction *theAuxFuncPbc = new ConstantAuxFunction(0, 1, userSpecPbc);
    
    // bc data alg
    AuxFunctionAlgorithm *auxAlgPbc
      = new AuxFunctionAlgorithm(realm_, part,
                                 pressureBC, theAuxFuncPbc,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlgPbc);
  }
  else {
    if ( userData.pSpec_ ) 
      NaluEnv::self().naluOutputP0() << "LowMachEqs::register_open_bc Error: Pressure specified at an open bc while global correction algorithm has been activated" << std::endl;    
  }

  // mdot at open bc; register field
  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsBip = meFC->num_integration_points();
  GenericFieldType *mdotBip 
    = &(metaData.declare_field<GenericFieldType>(static_cast<stk::topology::rank_t>(metaData.side_rank()), 
                                                 "open_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*mdotBip, *part, numScsBip, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_surface_pp_algorithm ----------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::register_surface_pp_algorithm(
  const PostProcessingData &theData,
  stk::mesh::PartVector &partVector)
{
  const std::string thePhysics = theData.physics_;

  // register nodal fields in common
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  VectorFieldType *pressureForce =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "pressure_force"));
  stk::mesh::put_field_on_mesh(*pressureForce, stk::mesh::selectUnion(partVector), meta_data.spatial_dimension(), nullptr);
  VectorFieldType *viscousForce =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "viscous_force"));
  stk::mesh::put_field_on_mesh(*viscousForce, stk::mesh::selectUnion(partVector), meta_data.spatial_dimension(), nullptr);
  ScalarFieldType *tauWall =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "tau_wall"));
  stk::mesh::put_field_on_mesh(*tauWall, stk::mesh::selectUnion(partVector), nullptr);
  ScalarFieldType *yplus =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "yplus"));
  stk::mesh::put_field_on_mesh(*yplus, stk::mesh::selectUnion(partVector), nullptr);
 
  // force output for these variables
  realm_.augment_output_variable_list(pressureForce->name());
  realm_.augment_output_variable_list(viscousForce->name());
  realm_.augment_output_variable_list(tauWall->name());
  realm_.augment_output_variable_list(yplus->name());


  if ( thePhysics == "surface_force_and_moment" ) {
    ScalarFieldType *assembledArea =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_area_force_moment"));
    stk::mesh::put_field_on_mesh(*assembledArea, stk::mesh::selectUnion(partVector), nullptr);
    if ( NULL == surfaceForceAndMomentAlgDriver_ )
      surfaceForceAndMomentAlgDriver_ = new SurfaceForceAndMomentAlgorithmDriver(realm_);
    SurfaceForceAndMomentAlgorithm *ppAlg
      = new SurfaceForceAndMomentAlgorithm(
          realm_, partVector, theData.outputFileName_, theData.frequency_,
          theData.parameters_, realm_.realmUsesEdges_);
    surfaceForceAndMomentAlgDriver_->algVec_.push_back(ppAlg);
  }
  else if ( thePhysics == "surface_force_and_moment_wall_function" ) {
    ScalarFieldType *assembledArea =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_area_force_moment_wf"));
    stk::mesh::put_field_on_mesh(*assembledArea, stk::mesh::selectUnion(partVector), nullptr);
    if ( NULL == surfaceForceAndMomentAlgDriver_ )
      surfaceForceAndMomentAlgDriver_ = new SurfaceForceAndMomentAlgorithmDriver(realm_);
    SurfaceForceAndMomentWallFunctionAlgorithm *ppAlg
      = new SurfaceForceAndMomentWallFunctionAlgorithm(
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
  stk::mesh::Part *part,
  const std::map<std::string, std::string> &theNames,
  const std::map<std::string, std::vector<double> > &theParams)
{
  // extract nDim
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  // iterate map and check for name
  const std::string dofName = "velocity";
  std::map<std::string, std::string>::const_iterator iterName
    = theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;
    
    // save off the field (np1 state)
    VectorFieldType *velocityNp1 = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
    
    // create a few Aux things
    AuxFunction *theAuxFunc = NULL;
    AuxFunctionAlgorithm *auxAlg = NULL;

    if ( fcnName == "wind_energy_taylor_vortex") {
      
      // extract the params
      std::map<std::string, std::vector<double> >::const_iterator iterParams
        = theParams.find(dofName);
      if (iterParams != theParams.end()) {
        std::vector<double> fcnParams = (*iterParams).second;	
        // create the function
        theAuxFunc = new WindEnergyTaylorVortexAuxFunction(0,nDim,fcnParams);
      }
      else {
        throw std::runtime_error("Wind_energy_taylor_vortex missing parameters");
      }
    }
    else if ( fcnName == "boundary_layer_perturbation") {
      
      // extract the params
      std::map<std::string, std::vector<double> >::const_iterator iterParams
        = theParams.find(dofName);
      if (iterParams != theParams.end()) {
        std::vector<double> fcnParams = (*iterParams).second;	
        // create the function
        theAuxFunc = new BoundaryLayerPerturbationAuxFunction(0,nDim,fcnParams);
      }
      else {
        throw std::runtime_error("Boundary_layer_perturbation missing parameters");
      }
    }
    else if (fcnName == "kovasznay") {
      theAuxFunc = new KovasznayVelocityAuxFunction(0, nDim);
    }
    else if ( fcnName == "SteadyTaylorVortex" ) {
      theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensity" ) {      
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensityNonIso" ) {      
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "OneTwoTenVelocity" ) {      
      theAuxFunc = new OneTwoTenVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "convecting_taylor_vortex" ) {
      theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0,nDim); 
    }
    else if ( fcnName == "TaylorGreen"  ) {
      theAuxFunc = new TaylorGreenVelocityAuxFunction(0,nDim); 
    }
    else if ( fcnName == "BoussinesqNonIso" ) {
      theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "SinProfileChannelFlow" ) {
      theAuxFunc = new SinProfileChannelFlowVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "PerturbedShearLayer" ) {
      theAuxFunc = new PerturbedShearLayerVelocityAuxFunction(0,nDim);
    }
    else {
      throw std::runtime_error("InitialCondFunction::non-supported velocity IC"); 
    }

    // create the algorithm
    auxAlg = new AuxFunctionAlgorithm(realm_, part,
                                      velocityNp1, theAuxFunc,
                                      stk::topology::NODE_RANK);
    
    // push to ic
    realm_.initCondAlg_.push_back(auxAlg);
  }
}

void
LowMachEquationSystem::pre_iter_work()
{
  momentumEqSys_->pre_iter_work();
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
  if ( isInit_ ) {
    continuityEqSys_->compute_projected_nodal_gradient();
    timeA = NaluEnv::self().nalu_time();
    continuityEqSys_->computeMdotAlgDriver_->execute();

    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerMisc_ += (timeB-timeA);
    isInit_ = false;
  }

  // compute tvisc
  momentumEqSys_->tviscAlgDriver_->execute();

  // compute effective viscosity
  momentumEqSys_->diffFluxCoeffAlgDriver_->execute();

  // start the iteration loop
  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    // momentum assemble, load_complete and solve
    momentumEqSys_->assemble_and_solve(momentumEqSys_->uTmp_);

    // update all of velocity
    timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(),
      realm_.bulk_data(),
      1.0, *momentumEqSys_->uTmp_,
      1.0, momentumEqSys_->velocity_->field_of_state(stk::mesh::StateNP1),
      realm_.get_activate_aura());
    timeB = NaluEnv::self().nalu_time();
    momentumEqSys_->timerAssemble_ += (timeB-timeA);

    // compute velocity relative to mesh with new velocity
    realm_.compute_vrtm();

    // activate global correction scheme
    if ( realm_.solutionOptions_->activateOpenMdotCorrection_ ) {
      timeA = NaluEnv::self().nalu_time();
      continuityEqSys_->computeMdotAlgDriver_->execute();
      timeB = NaluEnv::self().nalu_time();
      continuityEqSys_->timerMisc_ += (timeB-timeA);
    }

    // continuity assemble, load_complete and solve
    continuityEqSys_->assemble_and_solve(continuityEqSys_->pTmp_);

    // update pressure
    timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(),
      realm_.bulk_data(),
      1.0, *continuityEqSys_->pTmp_,
      1.0, *continuityEqSys_->pressure_,
      realm_.get_activate_aura());
    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerAssemble_ += (timeB-timeA);

    // compute mdot
    timeA = NaluEnv::self().nalu_time();
    continuityEqSys_->computeMdotAlgDriver_->execute();
    timeB = NaluEnv::self().nalu_time();
    continuityEqSys_->timerMisc_ += (timeB-timeA);

    // project nodal velocity
    project_nodal_velocity();

    // update pressure
    const std::string dofName="pressure";
    const double relaxFP = realm_.solutionOptions_->get_relaxation_factor(dofName);
    if (std::fabs(1.0 - relaxFP) > 1.0e-3) {
      timeA = NaluEnv::self().nalu_time();
      field_axpby(
        realm_.meta_data(),
        realm_.bulk_data(),
        (relaxFP - 1.0), *continuityEqSys_->pTmp_,
        1.0, *continuityEqSys_->pressure_,
        realm_.get_activate_aura());
      continuityEqSys_->compute_projected_nodal_gradient();
      timeB = NaluEnv::self().nalu_time();
      continuityEqSys_->timerAssemble_ += (timeB-timeA);
    }

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
    momentumEqSys_->timerMisc_ += (timeB-timeA);

  }

  // process CFL/Reynolds
  momentumEqSys_->cflReyAlgDriver_->execute();
 }

//--------------------------------------------------------------------------
//-------- post_adapt_work -------------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::post_adapt_work()
{

  // at the very least, we need to populate ip values at edge/element
  if ( realm_.process_adaptivity() ) {
    
    NaluEnv::self().naluOutputP0() << "--LowMachEquationSystem::post_adapt_work()" << std::endl;

    // compute new nodal pressure gradient
    continuityEqSys_->compute_projected_nodal_gradient();
    
    // continuity assemble, load_complete and solve
    const bool solveCont = false;
    if ( solveCont ) {

      // compute new nodal pressure gradient
      continuityEqSys_->compute_projected_nodal_gradient();
      
      continuityEqSys_->assemble_and_solve(continuityEqSys_->pTmp_);
      
      // update pressure
      field_axpby(
          realm_.meta_data(),
          realm_.bulk_data(),
          1.0, *continuityEqSys_->pTmp_,
          1.0, *continuityEqSys_->pressure_,
          realm_.get_activate_aura());
    }
    
    // compute mdot
    continuityEqSys_->computeMdotAlgDriver_->execute();
    
    // project nodal velocity/gradU
    const bool processU = false;
    if ( processU ) {
      project_nodal_velocity();
      momentumEqSys_->assembleNodalGradAlgDriver_->execute();
    }
    
    // compute wall function parameters (bip values)
    momentumEqSys_->compute_wall_function_params();
    
  }

}

//--------------------------------------------------------------------------
//-------- project_nodal_velocity ------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::project_nodal_velocity()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // field that we need
  VectorFieldType *velocity = momentumEqSys_->velocity_;
  VectorFieldType &velocityNp1 = velocity->field_of_state(stk::mesh::StateNP1);
  VectorFieldType *uTmp = momentumEqSys_->uTmp_;
  VectorFieldType *dpdx = continuityEqSys_->dpdx_;
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType *Udiag = momentumEqSys_->Udiag_;

  //==========================================================
  // save off dpdx to uTmp (do it everywhere)
  //==========================================================
 
  // selector (everywhere dpdx lives) and node_buckets 
  stk::mesh::Selector s_nodes = stk::mesh::selectField(*dpdx);
  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_nodes );
  
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * ut = stk::mesh::field_data(*uTmp, b);
    double * dp = stk::mesh::field_data(*dpdx, b);
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const int offSet = k*nDim;
      for ( int j = 0; j < nDim; ++j ) {
        ut[offSet+j] = dp[offSet+j];
      }
    }
  }

  //==========================================================
  // safe to update pressure gradient
  //==========================================================
  continuityEqSys_->compute_projected_nodal_gradient();

  //==========================================================
  // project u, u^n+1 = u^k+1 - dt/rho*(Gjp^N+1 - uTmp);
  //==========================================================
  
  // selector and node_buckets (only projected nodes)
  stk::mesh::Selector s_projected_nodes
    = (!stk::mesh::selectUnion(momentumEqSys_->notProjectedPart_)) &
    stk::mesh::selectField(*dpdx);
  stk::mesh::BucketVector const& p_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_projected_nodes );
  
  // process loop
  for ( stk::mesh::BucketVector::const_iterator ib = p_node_buckets.begin() ;
        ib != p_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * uNp1 = stk::mesh::field_data(velocityNp1, b);
    double * ut = stk::mesh::field_data(*uTmp, b);
    double * dp = stk::mesh::field_data(*dpdx, b);
    double * rho = stk::mesh::field_data(densityNp1, b);
    double * udiagN = stk::mesh::field_data(*Udiag, b);
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // Get scaling factor
      const double fac = 1.0/(rho[k] * udiagN[k]);
      
      // projection step
      const size_t offSet = k*nDim;
      for ( int j = 0; j < nDim; ++j ) {
        const double gdpx = dp[offSet+j] - ut[offSet+j];
        uNp1[offSet+j] -= fac*gdpx;
      }
    }
  }
}

void
LowMachEquationSystem::predict_state()
{
  // Does Nothing
}

//--------------------------------------------------------------------------
//-------- post_converged_work ---------------------------------------------
//--------------------------------------------------------------------------
void
LowMachEquationSystem::post_converged_work()
{
  if (NULL != surfaceForceAndMomentAlgDriver_){
    surfaceForceAndMomentAlgDriver_->execute();
  }
  
  // output mass closure
  continuityEqSys_->computeMdotAlgDriver_->provide_output();
}

//==========================================================================
// Class Definition
//==========================================================================
// MomentumEquationSystem - manages uvw pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MomentumEquationSystem::MomentumEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "MomentumEQS","momentum"),
    managePNG_(realm_.get_consistent_mass_matrix_png("velocity")),
    velocity_(NULL),
    dudx_(NULL),
    coordinates_(NULL),
    uTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradUAlgorithmDriver(realm_, "dudx")),
    diffFluxCoeffAlgDriver_(new AlgorithmDriver(realm_)),
    tviscAlgDriver_(new AlgorithmDriver(realm_)),
    cflReyAlgDriver_(new AlgorithmDriver(realm_)),
    wallFunctionParamsAlgDriver_(NULL),
    projectedNodalGradEqs_(NULL),
    firstPNGResidual_(0.0)
{
  dofName_ = "velocity";

  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("velocity");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_MOMENTUM);
  linsys_ = LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("velocity");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for velocity: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create projected nodal gradient equation system
  if ( managePNG_ ) {
     manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MomentumEquationSystem::~MomentumEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
  delete diffFluxCoeffAlgDriver_;
  delete tviscAlgDriver_;
  delete cflReyAlgDriver_;

  if ( NULL != wallFunctionParamsAlgDriver_)
    delete wallFunctionParamsAlgDriver_;
}



//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::initial_work()
{
  // call base class method (BDF2 state management, etc)
  EquationSystem::initial_work();

  // proceed with a bunch of initial work; wrap in timer
  {
    const double timeA = NaluEnv::self().nalu_time();
    realm_.compute_vrtm();
    const double timeB = NaluEnv::self().nalu_time();
    timerMisc_ += (timeB-timeA);
  }

  compute_projected_nodal_gradient();

  {
    const double timeA = NaluEnv::self().nalu_time();
    compute_wall_function_params();
    tviscAlgDriver_->execute();
    diffFluxCoeffAlgDriver_->execute();
    cflReyAlgDriver_->execute();

    const double timeB = NaluEnv::self().nalu_time();
    timerMisc_ += (timeB-timeA);
  }
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  velocity_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity", numStates));
  stk::mesh::put_field_on_mesh(*velocity_, *part, nDim, nullptr);
  realm_.augment_restart_variable_list("velocity");

  dudx_ =  &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx"));
  stk::mesh::put_field_on_mesh(*dudx_, *part, nDim*nDim, nullptr);

  // delta solution for linear solver
  uTmp_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "uTmp"));
  stk::mesh::put_field_on_mesh(*uTmp_, *part, nDim, nullptr);

  coordinates_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates"));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);

  visc_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  if ( realm_.is_turbulent() ) {
    tvisc_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity"));
    stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);
    evisc_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "effective_viscosity_u"));
    stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);
  }

  Udiag_ = &(meta_data.declare_field<ScalarFieldType>(
               stk::topology::NODE_RANK, "momentum_diag"));
  stk::mesh::put_field_on_mesh(*Udiag_, *part, nullptr);
  realm_.augment_restart_variable_list("momentum_diag");

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    VectorFieldType &velocityN = velocity_->field_of_state(stk::mesh::StateN);
    VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
    
    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &velocityNp1, &velocityN,
                               0, nDim,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }

  // register specialty fields for PNG
  if (managePNG_ ) {
    // create temp vector field for duidx that will hold the active dudx
    VectorFieldType *duidx =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "duidx"));
    stk::mesh::put_field_on_mesh(*duidx, *part, nDim, nullptr);
  }

  // speciality source
  if ( NULL != realm_.actuator_ ) {
    VectorFieldType *actuatorSource 
      =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "actuator_source"));
    VectorFieldType *actuatorSourceLHS
      =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "actuator_source_lhs"));
    ScalarFieldType *g
      =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "g"));
    stk::mesh::put_field_on_mesh(*actuatorSource, *part, nDim, nullptr);
    stk::mesh::put_field_on_mesh(*actuatorSourceLHS, *part, nDim, nullptr);
    stk::mesh::put_field_on_mesh(*g, *part, nullptr);
  }

}

//--------------------------------------------------------------------------
//-------- register_element_fields -----------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_element_fields(
  stk::mesh::Part * /* part */,
  const stk::topology & /* theTopo */)
{
  // nothing as of yet
}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_edge_fields(
  stk::mesh::Part * /* part */)
{
  // nothing as of yet
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;
  const AlgorithmType algMass = MASS;

  // non-solver CFL alg
  std::map<AlgorithmType, Algorithm *>::iterator it
    = cflReyAlgDriver_->algMap_.find(algType);
  if ( it == cflReyAlgDriver_->algMap_.end() ) {
    AssembleCourantReynoldsElemAlgorithm*theAlg
      = new AssembleCourantReynoldsElemAlgorithm(realm_, part);
    cflReyAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator itgu
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( itgu == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( edgeNodalGradient_ && realm_.realmUsesEdges_ ) {
        theAlg = new AssembleNodalGradUEdgeAlgorithm(realm_, part, &velocityNp1, &dudxNone);
      }
      else {
        theAlg = new AssembleNodalGradUElemAlgorithm(realm_, part, &velocityNp1, &dudxNone, edgeNodalGradient_);
      }
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      itgu->second->partVec_.push_back(part);
    }
  }

  // solver; interior contribution (advection + diffusion) [possible CMM time]
  if ( !realm_.solutionOptions_->useConsolidatedSolverAlg_ ) {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theSolverAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theSolverAlg = new AssembleMomentumEdgeSolverAlgorithm(realm_, part, this);
      }
      else {
        theSolverAlg = new AssembleMomentumElemSolverAlgorithm(realm_, part, this);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theSolverAlg;

      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string> >::iterator isrc
        = realm_.solutionOptions_->elemSrcTermsMap_.find("momentum");
      if ( isrc != realm_.solutionOptions_->elemSrcTermsMap_.end() ) {

        if ( realm_.realmUsesEdges_ )
          throw std::runtime_error("MomentumElemSrcTerms::Error can not use element source terms for an edge-based scheme");

        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if (sourceName == "momentum_time_derivative" ) {
            suppAlg = new MomentumMassElemSuppAlgDep(realm_, false);
          }
          else if (sourceName == "lumped_momentum_time_derivative" ) {
            suppAlg = new MomentumMassElemSuppAlgDep(realm_, true);
          }
          else if (sourceName == "SteadyTaylorVortex" ) {
            suppAlg = new SteadyTaylorVortexMomentumSrcElemSuppAlg(realm_);
          }
          else if (sourceName == "VariableDensity" ) {
            suppAlg = new VariableDensityMomentumSrcElemSuppAlg(realm_);
          }
          else if (sourceName == "NSO_2ND" ) {
            suppAlg = new MomentumNSOElemSuppAlgDep(realm_, velocity_, dudx_, realm_.is_turbulent() ? evisc_ : visc_, 0.0, 0.0);
          }
          else if (sourceName == "NSO_2ND_ALT" ) {
            suppAlg = new MomentumNSOElemSuppAlgDep(realm_, velocity_, dudx_, realm_.is_turbulent() ? evisc_ : visc_, 0.0, 1.0);
          }
          else if (sourceName == "NSO_4TH" ) {
            suppAlg = new MomentumNSOElemSuppAlgDep(realm_, velocity_, dudx_, realm_.is_turbulent() ? evisc_ : visc_, 1.0, 0.0);
          }
          else if (sourceName == "NSO_4TH_ALT" ) {
            suppAlg = new MomentumNSOElemSuppAlgDep(realm_, velocity_, dudx_, realm_.is_turbulent() ? evisc_ : visc_, 1.0, 1.0);
          }
          else if (sourceName == "NSO_2ND_KE" ) {
            suppAlg = new MomentumNSOKeElemSuppAlgDep(realm_, velocity_, dudx_, 0.0);
          }
          else if (sourceName == "NSO_4TH_KE" ) {
            suppAlg = new MomentumNSOKeElemSuppAlgDep(realm_, velocity_, dudx_, 1.0);
          }
          else if (sourceName == "NSO_2ND_GRAD" ) {
            suppAlg = new MomentumNSOGradElemSuppAlg(realm_, velocity_, dudx_, 0.0);
          }
          else if (sourceName == "NSO_4TH_GRAD" ) {
            suppAlg = new MomentumNSOGradElemSuppAlg(realm_, velocity_, dudx_, 1.0);
          }
          else if (sourceName == "buoyancy" ) {
            suppAlg = new MomentumBuoyancySrcElemSuppAlgDep(realm_);
          }
          else {
            throw std::runtime_error("MomentumElemSrcTerms::Error Source term is not supported: " + sourceName);
          }
          NaluEnv::self().naluOutputP0() << "MomentumElemSrcTerms::added() " << sourceName << std::endl;
          theSolverAlg->supplementalAlg_.push_back(suppAlg);
        }
      }
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
  }
  else {
    // Homogeneous implementation
    if ( realm_.realmUsesEdges_ )
      throw std::runtime_error("MomentumElemSrcTerms::Error can not use element source terms for an edge-based scheme");

    KernelBuilder kb(*this, *part, solverAlgDriver_->solverAlgorithmMap_, realm_.using_tensor_product_kernels());
    auto& dataPreReqs = kb.data_prereqs();
    auto& dataPreReqsHO = kb.data_prereqs_HO();

    kb.build_topo_kernel_if_requested<MomentumMassElemKernel>
      ("momentum_time_derivative",
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);

    kb.build_topo_kernel_if_requested<MomentumMassElemKernel>
      ("lumped_momentum_time_derivative",
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);

    kb.build_topo_kernel_if_requested<MomentumAdvDiffElemKernel>
      ("advection_diffusion",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_,
       realm_.is_turbulent()? evisc_ : visc_,
       dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumUpwAdvDiffElemKernel>
      ("upw_advection_diffusion",
       realm_.bulk_data(), *realm_.solutionOptions_, this, velocity_,
       realm_.is_turbulent()? evisc_ : visc_, dudx_,
       dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumActuatorSrcElemKernel>
        ("actuator",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);

    kb.build_topo_kernel_if_requested<MomentumActuatorSrcElemKernel>
      ("lumped_actuator",
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);

    kb.build_topo_kernel_if_requested<MomentumBuoyancySrcElemKernel>
      ("buoyancy",
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumBuoyancyBoussinesqSrcElemKernel>
      ("buoyancy_boussinesq",
       realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumNSOElemKernel>
      ("NSO_2ND",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_,
       realm_.is_turbulent()? evisc_ : visc_,
       0.0, 0.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumNSOElemKernel>
      ("NSO_2ND_ALT",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_,
       realm_.is_turbulent()? evisc_ : visc_,
       0.0, 1.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumNSOKeElemKernel>
      ("NSO_2ND_KE",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_, 0.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumNSOSijElemKernel>
      ("NSO_2ND_SIJ",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumNSOElemKernel>
      ("NSO_4TH",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_,
       realm_.is_turbulent()? evisc_ : visc_,
       1.0, 0.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumNSOElemKernel>
      ("NSO_4TH_ALT",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_,
       realm_.is_turbulent()? evisc_ : visc_,
       1.0, 1.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumNSOKeElemKernel>
      ("NSO_4TH_KE",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dudx_, 1.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<MomentumCoriolisSrcElemKernel>
      ("EarthCoriolis",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dataPreReqs, false);

    kb.build_topo_kernel_if_requested<MomentumCoriolisSrcElemKernel>
      ("lumped_EarthCoriolis",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_, dataPreReqs, true);

    kb.build_sgl_kernel_if_requested<MomentumAdvDiffHOElemKernel>
      ("experimental_ho_advection_diffusion",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_,
       realm_.is_turbulent()? evisc_ : visc_,
       dataPreReqsHO, false);

    kb.build_sgl_kernel_if_requested<MomentumAdvDiffHOElemKernel>
      ("experimental_ho_advection_diffusion_reduced_sens",
       realm_.bulk_data(), *realm_.solutionOptions_, velocity_,
       realm_.is_turbulent()? evisc_ : visc_,
       dataPreReqsHO, true);

    kb.build_sgl_kernel_if_requested<MomentumMassHOElemKernel>
      ("experimental_ho_momentum_time_derivative",
        realm_.bulk_data(), *realm_.solutionOptions_,  dataPreReqsHO);

    kb.build_sgl_kernel_if_requested<TGMMSHOElemKernel>
      ("experimental_ho_tgmms",
        realm_.bulk_data(), *realm_.solutionOptions_,  dataPreReqsHO);

    kb.build_sgl_kernel_if_requested<VariableDensityMomentumMMSHOElemKernel>
      ("experimental_ho_vdmms",
        realm_.bulk_data(), *realm_.solutionOptions_,  dataPreReqsHO);

    kb.build_sgl_kernel_if_requested<MomentumBuoyancySrcHOElemKernel>
      ("experimental_ho_buoyancy",
        realm_.bulk_data(), *realm_.solutionOptions_,  dataPreReqsHO);

    kb.report();
 
  }

  // Check if the user has requested CMM or LMM algorithms; if so, do not
  // include Nodal Mass algorithms
  std::vector<std::string> checkAlgNames = {"momentum_time_derivative",
                                            "lumped_momentum_time_derivative",
                                            "experimental_ho_momentum_time_derivative"};
  bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
  // solver; time contribution (lumped mass matrix)
  if ( !elementMassAlg || nodal_src_is_requested() ) {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsm =
      solverAlgDriver_->solverAlgMap_.find(algMass);
    if ( itsm == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleNodeSolverAlgorithm *theAlg
        = new AssembleNodeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algMass] = theAlg;
    
      // now create the supplemental alg for mass term (only when CMM is not in use)
      if ( !elementMassAlg ) {
        if ( realm_.number_of_states() == 2 ) {
          MomentumMassBackwardEulerNodeSuppAlg *theMass
            = new MomentumMassBackwardEulerNodeSuppAlg(realm_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
        else {
          MomentumMassBDF2NodeSuppAlg *theMass
            = new MomentumMassBDF2NodeSuppAlg(realm_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
      }

      // Add src term supp alg...; limited number supported
      std::map<std::string, std::vector<std::string> >::iterator isrc
        = realm_.solutionOptions_->srcTermsMap_.find("momentum");
      if ( isrc != realm_.solutionOptions_->srcTermsMap_.end() ) {
        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if (sourceName == "buoyancy" ) {
            suppAlg = new MomentumBuoyancySrcNodeSuppAlg(realm_);
          }
          else if ( sourceName == "buoyancy_boussinesq") {
            suppAlg = new MomentumBoussinesqSrcNodeSuppAlg(realm_);
          }
          else if ( sourceName == "buoyancy_boussinesq_ra") {
            suppAlg = new MomentumBoussinesqRASrcNodeSuppAlg(realm_);
          }
          else if ( sourceName == "body_force") {
            // extract params
            std::map<std::string, std::vector<double> >::iterator iparams
              = realm_.solutionOptions_->srcTermParamMap_.find("momentum");
            if ( iparams != realm_.solutionOptions_->srcTermParamMap_.end()) {
              std::vector<double> theParams = iparams->second;
              suppAlg = new MomentumBodyForceSrcNodeSuppAlg(realm_, theParams);
            }
            else {
              throw std::runtime_error("SrcTermsError::body_force: No params found");
            }
          }
          else if ( sourceName == "abl_forcing" ) {
            ThrowAssertMsg(
                           ((NULL != realm_.ablForcingAlg_) &&
                            (realm_.ablForcingAlg_->momentumForcingOn())),
                           "ERROR! ABL Forcing parameters must be initialized to use Momentum source.");
            suppAlg = new MomentumABLForceSrcNodeSuppAlg(realm_, realm_.ablForcingAlg_);
          }
          else if ( sourceName == "gcl") {
            suppAlg = new MomentumGclSrcNodeSuppAlg(realm_);
          }
          else if (sourceName == "SteadyTaylorVortex" ) {
            suppAlg = new SteadyTaylorVortexMomentumSrcNodeSuppAlg(realm_);
          }
          else if (sourceName == "VariableDensity" ) {
            suppAlg = new VariableDensityMomentumSrcNodeSuppAlg(realm_);
          }
          else if (sourceName == "VariableDensityNonIso" ) {
            suppAlg = new VariableDensityNonIsoMomentumSrcNodeSuppAlg(realm_);
          }
          else if (sourceName == "BoussinesqNonIso" ) {
            suppAlg = new BoussinesqNonIsoMomentumSrcNodeSuppAlg(realm_);
          }
          else if ( sourceName == "actuator") {
            suppAlg = new MomentumActuatorSrcNodeSuppAlg(realm_);
          }
          else if ( sourceName == "EarthCoriolis") {
            suppAlg = new MomentumCoriolisSrcNodeSuppAlg(realm_);
          }
          else {
            throw std::runtime_error("MomentumNodalSrcTerms::Error Source term is not supported: " + sourceName);
          }
          NaluEnv::self().naluOutputP0() << "MomentumNodalSrcTerms::added() " << sourceName << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg);
        }
      }
    }
    else {
      itsm->second->partVec_.push_back(part);
    }
  }

  // effective viscosity alg
  if ( realm_.is_turbulent() ) {
    std::map<AlgorithmType, Algorithm *>::iterator itev =
      diffFluxCoeffAlgDriver_->algMap_.find(algType);
    if ( itev == diffFluxCoeffAlgDriver_->algMap_.end() ) {
      EffectiveDiffFluxCoeffAlgorithm *theAlg
        = new EffectiveDiffFluxCoeffAlgorithm(realm_, part, visc_, tvisc_, evisc_, 1.0, 1.0);
      diffFluxCoeffAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      itev->second->partVec_.push_back(part);
    }

    // deal with tvisc better? - possibly should be on EqSysManager?
    std::map<AlgorithmType, Algorithm *>::iterator it_tv =
      tviscAlgDriver_->algMap_.find(algType);
    if ( it_tv == tviscAlgDriver_->algMap_.end() ) {
      Algorithm * theAlg = NULL;
      switch (realm_.solutionOptions_->turbulenceModel_ ) {
        case KSGS:
          theAlg = new TurbViscKsgsAlgorithm(realm_, part);
          break;
        case SMAGORINSKY:
          theAlg = new TurbViscSmagorinskyAlgorithm(realm_, part);
          break;
        case WALE:
          theAlg = new TurbViscWaleAlgorithm(realm_, part);
          break;
        case SST: case SST_DES:
          theAlg = new TurbViscSSTAlgorithm(realm_, part);
          break;
        default:
          throw std::runtime_error("non-supported turb model");
      }
      tviscAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it_tv->second->partVec_.push_back(part);
    }
  }

}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{

  // push mesh part
  notProjectedPart_.push_back(part);

  // algorithm type
  const AlgorithmType algType = INFLOW;

  // velocity np1
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();

  // register boundary data; velocity_bc
  VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);
  
  // extract the value for user specified velocity and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  std::string velocityName = "velocity";
  UserDataType theDataType = get_bc_data_type(userData, velocityName);

  AuxFunction *theAuxFunc = NULL;
  if ( CONSTANT_UD == theDataType ) {
    Velocity ux = userData.u_;
    std::vector<double> userSpec(nDim);
    userSpec[0] = ux.ux_;
    userSpec[1] = ux.uy_;
    if ( nDim > 2)
      userSpec[2] = ux.uz_;
    
    // new it
    theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);
    
  }
  else if ( FUNCTION_UD == theDataType ) {
    // extract the name/params
    std::string fcnName = get_bc_function_name(userData, velocityName);
    std::vector<double> theParams = get_bc_function_params(userData, velocityName);
    if ( theParams.size() == 0 )
      NaluEnv::self().naluOutputP0() << "function parameter size is zero" << std::endl;
    // switch on the name found...
    if ( fcnName == "convecting_taylor_vortex" ) {
      theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "SteadyTaylorVortex" ) {
      theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensity" ) {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "VariableDensityNonIso" ) {
      theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
    }
    else if (fcnName == "TaylorGreen" ) {
      theAuxFunc = new TaylorGreenVelocityAuxFunction(0,nDim);
    }
    else if ( fcnName == "BoussinesqNonIso") {
      theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0, nDim);
    }
    else if ( fcnName == "kovasznay") {
      theAuxFunc = new KovasznayVelocityAuxFunction(0,nDim);
    }
    else {
      throw std::runtime_error("MomentumEquationSystem::register_inflow_bc: limited functions supported");
    }
  }
  else {
    throw std::runtime_error("MomentumEquationSystem::register_inflow_bc: only constant and user function supported");
  }
  
  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
			       theBcField, theAuxFunc,
			       stk::topology::NODE_RANK);
  
  // how to populate the field?
  if ( userData.externalData_ ) {
    // xfer will handle population; only need to populate the initial value
    realm_.initCondAlg_.push_back(auxAlg);
  }
  else {
    // put it on bcData
    bcDataAlg_.push_back(auxAlg);
  }

  // copy velocity_bc to velocity np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, &velocityNp1,
                             0, nDim,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);
  
  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg
        = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, theBcField, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg
      = new DirichletBC(realm_, this, part, &velocityNp1, theBcField, 0, nDim);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  }
  else {
    itd->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  // register boundary data; open_velocity_bc
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "open_velocity_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

  // extract the value for user specified velocity and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  Velocity ux = userData.u_;
  std::vector<double> userSpec(nDim);
  userSpec[0] = ux.ux_;
  userSpec[1] = ux.uy_;
  if ( nDim > 2)
    userSpec[2] = ux.uz_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg
        = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, &velocityNp1, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  if ( realm_.solutionOptions_->useConsolidatedBcSolverAlg_ ) {      
    
    // solver for continuity open
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    
    stk::topology elemTopo = get_elem_topo(realm_, *part);
    
    AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
    bool solverAlgWasBuilt = false;
    
    std::tie(faceElemSolverAlg, solverAlgWasBuilt) 
      = build_or_add_part_to_face_elem_solver_alg(algType, *this, *part, elemTopo, solverAlgMap, "open");
    
    auto& activeKernels = faceElemSolverAlg->activeKernels_;
    
    if (solverAlgWasBuilt) {
      
      build_face_elem_topo_kernel_automatic<MomentumOpenAdvDiffElemKernel>
        (partTopo, elemTopo, *this, activeKernels, "momentum_open",
         realm_.meta_data(), *realm_.solutionOptions_, this,
         velocity_, dudx_, realm_.is_turbulent() ? evisc_ : visc_,
         faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_);
      
    }
  }
  else {
    // solver algs; lhs
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theAlg = new AssembleMomentumEdgeOpenSolverAlgorithm(realm_, part, this);
      }
      else {
        theAlg = new AssembleMomentumElemOpenSolverAlgorithm(realm_, part, this);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const WallBoundaryConditionData &wallBCData)
{

  // find out if this is a wall function approach
  WallUserData userData = wallBCData.userData_;
  const bool wallFunctionApproach = userData.wallFunctionApproach_;
  const bool ablWallFunctionApproach = userData.ablWallFunctionApproach_;
  const bool anyWallFunctionActivated = wallFunctionApproach || ablWallFunctionApproach;

  // push mesh part
  if ( !anyWallFunctionActivated )
    notProjectedPart_.push_back(part);

  // np1 velocity
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();

  const std::string bcFieldName = anyWallFunctionActivated ? "wall_velocity_bc" : "velocity_bc";

  // register boundary data; velocity_bc
  VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, bcFieldName));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

  if(realm_.solutionOptions_->meshMotion_) {
    NaluEnv::self().naluOutputP0() << "MomentumEquationSystem::register_wall_bc(): Mesh motion active! Velocity definition under wall_user_data will be ignored" << std::endl;

    // get the mesh velocity field
    VectorFieldType* meshVelocity = meta_data.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "mesh_velocity");

    // create algorithm to copy mesh velocity to wall velocity
    CopyFieldAlgorithm *wallVelCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               meshVelocity, theBcField,
                               0, nDim,
                               stk::topology::NODE_RANK);

    bcDataAlg_.push_back(wallVelCopyAlg);
  }
  else {
    // extract the value for user specified velocity and save off the AuxFunction
    AuxFunction *theAuxFunc = NULL;
    Algorithm* auxAlg = NULL;

    std::string velocityName = "velocity";

    if ( bc_data_specified(userData, velocityName) ) {

      UserDataType theDataType = get_bc_data_type(userData, velocityName);
      if ( CONSTANT_UD == theDataType ) {
        // constant data type specification
        Velocity ux = userData.u_;
        std::vector<double> userSpec(nDim);
        userSpec[0] = ux.ux_;
        userSpec[1] = ux.uy_;
        if ( nDim > 2)
          userSpec[2] = ux.uz_;
        theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);
      }
      else if ( FUNCTION_UD == theDataType ) {
        // extract the name and parameters (double and string)
        std::string fcnName = get_bc_function_name(userData, velocityName);
        // switch on the name found...
        if ( fcnName == "tornado" ) {
          theAuxFunc = new TornadoAuxFunction(0,nDim);
        }
        else if (fcnName == "wind_energy") {
          NaluEnv::self().naluOutputP0()
            << "MomentumEqSys: WARNING! mesh_motion user function for wall BC "
               "has been deprecated" << std::endl;
        } else {
          throw std::runtime_error("MomentumEqSys::register_wall_function: "
                                   "Only tornado user functions supported");
        }
      }
    }
    else {
      throw std::runtime_error("Invalid Wall Data Specification; must provide "
                               "const or fcn for velocity");
    }

    auxAlg = new AuxFunctionAlgorithm(realm_, part,
      theBcField, theAuxFunc,
      stk::topology::NODE_RANK);

    // check to see if this is an FSI interface to determine how we handle velocity population
    if ( userData.isFsiInterface_ ) {
      // xfer will handle population; only need to populate the initial value
      realm_.initCondAlg_.push_back(auxAlg);
    }
    else {
      bcDataAlg_.push_back(auxAlg);
    }
  }
  
  // copy velocity_bc to velocity np1
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
			     theBcField, &velocityNp1,
			     0, nDim,
			     stk::topology::NODE_RANK);

  // wall function activity will only set dof velocity np1 wall value as an IC
  if ( anyWallFunctionActivated )
    realm_.initCondAlg_.push_back(theCopyAlg);
  else
    bcDataMapAlg_.push_back(theCopyAlg);
    
  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    const AlgorithmType algTypePNG = anyWallFunctionActivated ? WALL_FCN : WALL;
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algTypePNG);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg
        = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, theBcField, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algTypePNG] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // Dirichlet or wall function bc
  if ( anyWallFunctionActivated ) {

    // register fields; nodal
    ScalarFieldType *assembledWallArea =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_wf"));
    stk::mesh::put_field_on_mesh(*assembledWallArea, *part, nullptr);

    ScalarFieldType *assembledWallNormalDistance=  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_normal_distance"));
    stk::mesh::put_field_on_mesh(*assembledWallNormalDistance, *part, nullptr);

    // integration point; size it based on number of boundary integration points
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(partTopo);
    const int numScsBip = meFC->num_integration_points();

    stk::topology::rank_t sideRank = static_cast<stk::topology::rank_t>(meta_data.side_rank());
    GenericFieldType *wallFrictionVelocityBip 
      =  &(meta_data.declare_field<GenericFieldType>(sideRank, "wall_friction_velocity_bip"));
    stk::mesh::put_field_on_mesh(*wallFrictionVelocityBip, *part, numScsBip, nullptr);

    GenericFieldType *wallNormalDistanceBip 
      =  &(meta_data.declare_field<GenericFieldType>(sideRank, "wall_normal_distance_bip"));
    stk::mesh::put_field_on_mesh(*wallNormalDistanceBip, *part, numScsBip, nullptr);

    // create wallFunctionParamsAlgDriver
    if ( NULL == wallFunctionParamsAlgDriver_)
      wallFunctionParamsAlgDriver_ = new AlgorithmDriver(realm_);

    if (ablWallFunctionApproach) {

      // register boundary data: heat_flux_bc
      ScalarFieldType *theHeatFluxBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "heat_flux_bc"));
      stk::mesh::put_field_on_mesh(*theHeatFluxBcField, *part, nullptr);

      NormalHeatFlux heatFlux = userData.q_;
      std::vector<double> userSpec(1);
      userSpec[0] = heatFlux.qn_;

      // new it
      ConstantAuxFunction *theHeatFluxAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

      // bc data alg
      bcDataAlg_.push_back( new AuxFunctionAlgorithm(realm_, part,
				   theHeatFluxBcField, theHeatFluxAuxFunc,
				   stk::topology::NODE_RANK));

      const AlgorithmType wfAlgType = WALL_ABL;

      // create algorithm for utau, yp and assembled nodal wall area (_WallFunction)
      //Gravity gravity = userData.gravity_;
      //const double grav = gravity.gravity_;
      std::vector<double> gravity;
      gravity.resize(nDim);
      gravity = realm_.solutionOptions_->gravity_;
      const double grav = std::abs(gravity[userData.gravityComponent_ - 1]);
      RoughnessHeight rough = userData.z0_;
      const double z0 = rough.z0_;
      ReferenceTemperature Tref = userData.referenceTemperature_;
      const double referenceTemperature = Tref.referenceTemperature_;
      std::map<AlgorithmType, Algorithm *>::iterator it_utau =
        wallFunctionParamsAlgDriver_->algMap_.find(wfAlgType);
      if ( it_utau == wallFunctionParamsAlgDriver_->algMap_.end() ) {
        ComputeABLWallFrictionVelocityAlgorithm *theUtauAlg =
          new ComputeABLWallFrictionVelocityAlgorithm(realm_, part, realm_.realmUsesEdges_, grav, z0, referenceTemperature);
        wallFunctionParamsAlgDriver_->algMap_[wfAlgType] = theUtauAlg;
      }
      else {
        it_utau->second->partVec_.push_back(part);
      }

      // create lhs/rhs algorithm; generalized for edge (nearest node usage) and element
      std::map<AlgorithmType, SolverAlgorithm *>::iterator it_wf =
        solverAlgDriver_->solverAlgMap_.find(wfAlgType);
      if ( it_wf == solverAlgDriver_->solverAlgMap_.end() ) {
        SolverAlgorithm *theAlg = NULL;
        if ( realm_.realmUsesEdges_ ) {
          theAlg = new AssembleMomentumEdgeABLWallFunctionSolverAlgorithm(realm_, part, this, 
                                                                          grav, z0, referenceTemperature);
        }
        else {
          theAlg = new AssembleMomentumElemABLWallFunctionSolverAlgorithm(realm_, part, this, realm_.realmUsesEdges_, 
                                                                          grav, z0, referenceTemperature);     
        }
        solverAlgDriver_->solverAlgMap_[wfAlgType] = theAlg;
      }
      else {
        it_wf->second->partVec_.push_back(part);
      }
    }

    else {

      const AlgorithmType wfAlgType = WALL_FCN;

      // create algorithm for utau, yp and assembled nodal wall area (_WallFunction)
      std::map<AlgorithmType, Algorithm *>::iterator it_utau =
        wallFunctionParamsAlgDriver_->algMap_.find(wfAlgType);
      if ( it_utau == wallFunctionParamsAlgDriver_->algMap_.end() ) {
        ComputeWallFrictionVelocityAlgorithm *theUtauAlg =
          new ComputeWallFrictionVelocityAlgorithm(realm_, part, realm_.realmUsesEdges_);
        wallFunctionParamsAlgDriver_->algMap_[wfAlgType] = theUtauAlg;
      }
      else {
        it_utau->second->partVec_.push_back(part);
      }

      // create lhs/rhs algorithm; generalized for edge (nearest node usage) and element
      if ( realm_.solutionOptions_->useConsolidatedBcSolverAlg_ ) {        
        // element-based uses consolidated approach fully
        auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
        
        AssembleElemSolverAlgorithm* solverAlg = nullptr;
        bool solverAlgWasBuilt = false;
        
        std::tie(solverAlg, solverAlgWasBuilt) = build_or_add_part_to_face_bc_solver_alg(*this, *part, solverAlgMap, "wall_fcn");
        
        ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
        auto& activeKernels = solverAlg->activeKernels_;
        
        if (solverAlgWasBuilt) {
          build_face_topo_kernel_automatic<MomentumWallFunctionElemKernel>
            (partTopo, *this, activeKernels, "momentum_wall_function",
             realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);
          report_built_supp_alg_names();   
        }
      }
      else {
        // deprecated element-based and supported edge-based approach
        std::map<AlgorithmType, SolverAlgorithm *>::iterator it_wf =
          solverAlgDriver_->solverAlgMap_.find(wfAlgType);
        if ( it_wf == solverAlgDriver_->solverAlgMap_.end() ) {
          SolverAlgorithm *theAlg = NULL;
          if ( realm_.realmUsesEdges_ ) {
            theAlg = new AssembleMomentumEdgeWallFunctionSolverAlgorithm(realm_, part, this);
          }
          else {
            theAlg = new AssembleMomentumElemWallFunctionSolverAlgorithm(realm_, part, this, realm_.realmUsesEdges_);
          }
          solverAlgDriver_->solverAlgMap_[wfAlgType] = theAlg;
        }
        else {
          it_wf->second->partVec_.push_back(part);
        }
      }
    }
  }
  else {
    const AlgorithmType algType = WALL;
    
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
      DirichletBC *theAlg
        = new DirichletBC(realm_, this, part, &velocityNp1, theBcField, 0, nDim);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    }
    else {
      itd->second->partVec_.push_back(part);
    }
  }
  
  // specialty FSI
  if ( userData.isFsiInterface_ ) {
    // FIXME: need p^n+1/2; requires "old" pressure... need a utility to save it and compute it...
    NaluEnv::self().naluOutputP0() << "Warning: Second-order FSI requires p^n+1/2; BC is using p^n+1" << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const SymmetryBoundaryConditionData &  /* symmetryBCData */)
{
  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg
        = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, &velocityNp1, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  if (!realm_.solutionOptions_->useConsolidatedBcSolverAlg_) {
    // solver algs; lhs
	    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
		    = solverAlgDriver_->solverAlgMap_.find(algType);
	    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
		    SolverAlgorithm *theAlg = NULL;
		    if ( realm_.realmUsesEdges_ ) {
			    theAlg = new AssembleMomentumEdgeSymmetrySolverAlgorithm(realm_, part,this);
		    }
		    else {
			    theAlg = new AssembleMomentumElemSymmetrySolverAlgorithm(realm_, part, this);
		    }
		    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
	    }
	    else {
		    itsi->second->partVec_.push_back(part);
	    }
  }
  else {
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

    stk::topology elemTopo = get_elem_topo(realm_, *part);

    AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(faceElemSolverAlg, solverAlgWasBuilt) 
      = build_or_add_part_to_face_elem_solver_alg(algType, *this, *part, elemTopo, solverAlgMap, "symm");

    auto& activeKernels = faceElemSolverAlg->activeKernels_;

    if (solverAlgWasBuilt) {

      const stk::mesh::MetaData& metaData = realm_.meta_data();
      const std::string viscName = realm_.is_turbulent()
        ? "effective_viscosity_u" : "viscosity";
      
      build_face_elem_topo_kernel_automatic<MomentumSymmetryElemKernel>
        (partTopo, elemTopo, *this, activeKernels, "momentum_symmetry",
         metaData, *realm_.solutionOptions_,
         metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity"),
         metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName),
         faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_
         );
      
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_abltop_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::register_abltop_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const ABLTopBoundaryConditionData & abltopBCData)
{
  auto userData = abltopBCData.userData_;

  if (!userData.ABLTopBC_) {
    SymmetryBoundaryConditionData symData(abltopBCData.boundaryConditions_);
    register_symmetry_bc(part, partTopo, symData);
    return;
  }

#ifdef NALU_USES_FFTW
  auto& meta_data = realm_.meta_data();
  // algorithm type
  const AlgorithmType algType = TOP_ABL;
  auto user_data = abltopBCData.userData_;

  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  // push mesh part
  notProjectedPart_.push_back(part);

  // non-solver; contribution to Gjui; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg
        = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, &velocityNp1, &dudxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  if (!realm_.solutionOptions_->useConsolidatedBcSolverAlg_) {
    // solver algs; lhs
    std::string bcFieldName =
        realm_.solutionOptions_->activateOpenMdotCorrection_
            ? "velocity_bc"
            : "cont_velocity_bc";
    VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, bcFieldName));
    stk::mesh::put_field_on_mesh(*theBcField, *part, 3, nullptr);

    auto it = solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverDirichAlgMap_.end()) {
      SolverAlgorithm *theAlg = new AssembleMomentumEdgeABLTopBC(
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
  stk::mesh::Part *part,
  const stk::topology &theTopo)
{
  const AlgorithmType algType = NON_CONFORMAL;

  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNone = dudx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // mdot at nc bc; register field; require topo and num ips
  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsBip = meFC->num_integration_points();

  stk::topology::rank_t sideRank = static_cast<stk::topology::rank_t>(meta_data.side_rank());
  GenericFieldType *mdotBip =
    &(meta_data.declare_field<GenericFieldType>(sideRank, "nc_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*mdotBip, *part, numScsBip , nullptr);

  // non-solver; contribution to Gjui; DG algorithm decides on locations for integration points
  if ( !managePNG_ ) {
    if ( edgeNodalGradient_ ) {
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        Algorithm *theAlg
          = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, &velocityNp1, &dudxNone, edgeNodalGradient_);
        assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
    else { 
      // proceed with DG
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        AssembleNodalGradUNonConformalAlgorithm *theAlg 
          = new AssembleNodalGradUNonConformalAlgorithm(realm_, part, &velocityNp1, &dudxNone);
        assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
  }
  
  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    AssembleMomentumNonConformalSolverAlgorithm *theAlg
      = new AssembleMomentumNonConformalSolverAlgorithm(realm_, part, this, &velocityNp1, 
                                                        realm_.is_turbulent() ? evisc_ : visc_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
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
  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(velocity_,1,nDim)));
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
    stk::mesh::field_fill(gamma1/dt, *Udiag_);
  }
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_MOMENTUM;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("velocity");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_MOMENTUM);
  linsys_ = LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

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
  // copy state n to state np1
  VectorFieldType &uN = velocity_->field_of_state(stk::mesh::StateN);
  VectorFieldType &uNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), uN, uNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- compute_wall_function_params ------------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::compute_wall_function_params()
{
  if (NULL != wallFunctionParamsAlgDriver_){
    wallFunctionParamsAlgDriver_->execute();
  }
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if ( NULL == projectedNodalGradEqs_ ) {
    projectedNodalGradEqs_
      = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG_U, "duidx", "qTmp", "pTmp", "PNGradUEQS");

    // turn off output
    projectedNodalGradEqs_->deactivate_output();
  }
  // fill the map for expected boundary condition names; recycle pTmp (ui copied in as needed)
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "pTmp");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "pTmp"); // might want wall_function velocity_bc?
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "pTmp");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "pTmp");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
MomentumEquationSystem::compute_projected_nodal_gradient()
{
  if ( !managePNG_ ) {
    const double timeA = -NaluEnv::self().nalu_time();
    assembleNodalGradAlgDriver_->execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  }
  else {
    // this option is more complex... Rather than solving a nDim*nDim system, we
    // copy each velocity component i to the expected dof for the PNG system; pTmp

    // extract fields
    ScalarFieldType *pTmp = realm_.meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pTmp");
    VectorFieldType *duidx = realm_.meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "duidx");

    const int nDim = realm_.meta_data().spatial_dimension();

    // manage norms here
    bool isFirst = realm_.currentNonlinearIteration_ == 1;
    if ( isFirst )
      firstPNGResidual_ = 0.0;

    double sumNonlinearResidual = 0.0;
    double sumLinearResidual = 0.0;
    int sumLinearIterations = 0;
    for ( int i = 0; i < nDim; ++i ) {
      // copy velocity, component i to pTmp
      field_index_copy(realm_.meta_data(), realm_.bulk_data(), *velocity_, i, *pTmp, 0,
        realm_.get_activate_aura());

      // copy active tensor, dudx to vector, duidx
      for ( int k = 0; k < nDim; ++k ) {
        field_index_copy(realm_.meta_data(), realm_.bulk_data(), *dudx_, i*nDim+k, *duidx, k,
          realm_.get_activate_aura());
      }

      projectedNodalGradEqs_->solve_and_update_external();

      // extract the solver history info
      const double nonlinearRes = projectedNodalGradEqs_->linsys_->nonLinearResidual();
      const double linearRes = projectedNodalGradEqs_->linsys_->linearResidual();
      const int linearIter = projectedNodalGradEqs_->linsys_->linearSolveIterations();

      // sum system norms for this iteration
      sumNonlinearResidual += nonlinearRes;
      sumLinearResidual += linearRes;
      sumLinearIterations += linearIter;

      // increment first nonlinear residual
      if ( isFirst )
        firstPNGResidual_ += nonlinearRes;

      // copy vector, duidx_k to tensor, dudx; this one might hurt as compared to a specialty loop..
      for ( int k = 0; k < nDim; ++k ) {
        field_index_copy(realm_.meta_data(), realm_.bulk_data(), *duidx, k, *dudx_, nDim*i+k,
          realm_.get_activate_aura());
      }
    }

    // output norms
    const double scaledNonLinearResidual = sumNonlinearResidual/std::max(std::numeric_limits<double>::epsilon(), firstPNGResidual_);
    std::string pngName = projectedNodalGradEqs_->linsys_->name();
    const int nameOffset = pngName.length()+8;
    NaluEnv::self().naluOutputP0()
        << std::setw(nameOffset) << std::right << pngName
        << std::setw(32-nameOffset)  << std::right << sumLinearIterations/(int)nDim
        << std::setw(18) << std::right << sumLinearResidual/(int)nDim
        << std::setw(15) << std::right << sumNonlinearResidual/(int)nDim
        << std::setw(14) << std::right << scaledNonLinearResidual << std::endl;

    // a bit covert, provide linsys with the new norm which is the sum of all norms
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

  for (int in=0; in < nEntities; in++) {
    const auto naluID = *stk::mesh::field_data(*realm_.naluGlobalId_, entities[in]);
    const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluID);
    int ix = in * nDim * (offset + 1);
    double* diagVal = (double*) stk::mesh::field_data(*Udiag_, mnode);
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
  constexpr bool forceAtomic = !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;

  for (unsigned in=0; in < nEntities; in++) {
    const auto naluID = *stk::mesh::field_data(*realm_.naluGlobalId_, entities[in]);
    const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluID);
    int ix = in * nDim;
    double* diagVal = (double*) stk::mesh::field_data(*Udiag_, mnode);
    if (forceAtomic)
      Kokkos::atomic_add(diagVal, lhs(ix, ix));
    else
      diagVal[0] += lhs(ix, ix);
  }
}

void
MomentumEquationSystem::save_diagonal_term(
  unsigned nEntities,
  const ngp::Mesh::ConnectedNodes& entities,
  const SharedMemView<const double**>& lhs)
{
#ifndef KOKKOS_ENABLE_CUDA
  auto& bulk = realm_.bulk_data();
  const int nDim = realm_.spatialDimension_;
  constexpr bool forceAtomic = !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;

  for (unsigned in=0; in < nEntities; in++) {
    const auto naluID = *stk::mesh::field_data(*realm_.naluGlobalId_, entities[in]);
    const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluID);
    int ix = in * nDim;
    double* diagVal = (double*) stk::mesh::field_data(*Udiag_, mnode);
    if (forceAtomic)
      Kokkos::atomic_add(diagVal, lhs(ix, ix));
    else
      diagVal[0] += lhs(ix, ix);
  }
#endif
}

void
MomentumEquationSystem::assemble_and_solve(
  stk::mesh::FieldBase* deltaSolution)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();

  extractDiagonal_ = (realm_.solutionOptions_->tscaleType_ == TSCALE_UDIAGINV);
  // Reset timescale field before momentum solve
  {
    double projTimeScale = 0.0;
    if (realm_.solutionOptions_->tscaleType_ == TSCALE_DEFAULT) {
      const double dt = realm_.get_time_step();
      const double gamma1 = realm_.get_gamma1();
      projTimeScale = gamma1 / dt;
    }

    const auto sel = meta.universal_part() & stk::mesh::selectField(*Udiag_);
    const auto& bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);
    for (auto b: bkts) {
      double* field = (double*) stk::mesh::field_data(*Udiag_, *b);

      for (size_t in=0; in < b->size(); in++)
        field[in] = projTimeScale;
    }
  }

  // Perform actual solve
  EquationSystem::assemble_and_solve(deltaSolution);

  // Post-process the Udiag term
  ScalarFieldType* dualVol = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  ScalarFieldType* density = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  std::vector<const stk::mesh::FieldBase*> fVec{Udiag_};

  if (realm_.solutionOptions_->tscaleType_ == TSCALE_UDIAGINV) {
    const std::string dofName = "velocity";
    const double dt = realm_.get_time_step();
    const double gamma1 = realm_.get_gamma1();
    const double projTimeScale = gamma1 / dt;
    const double alphaU = realm_.solutionOptions_->get_relaxation_factor(dofName);

    stk::mesh::parallel_sum(bulk, fVec);
    const auto sel = stk::mesh::selectField(*Udiag_)
      & meta.locally_owned_part()
      & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
      & !(realm_.get_inactive_selector());
    const auto& bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);

    for (auto b: bkts) {
      double* field = (double*) stk::mesh::field_data(*Udiag_, *b);
      double* rho = (double*) stk::mesh::field_data(*density, *b);
      double* dVol = (double*) stk::mesh::field_data(*dualVol, *b);

      for (size_t in=0; in < b->size(); in++) {
        field[in] /= (rho[in] * dVol[in]);
        field[in] = (field[in] - projTimeScale) * alphaU + projTimeScale;
      }
    }
  }

  // Communicate to shared and ghosted nodes
  stk::mesh::copy_owned_to_shared(bulk, fVec);
  stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);
  if (realm_.hasPeriodic_)
    realm_.periodic_delta_solution_update(Udiag_, 1);
  if (realm_.nonConformalManager_ != nullptr &&
      realm_.nonConformalManager_->nonConformalGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, fVec);
  if (realm_.hasOverset_)
    realm_.overset_orphan_node_field_update(Udiag_, 1, 1);
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
  EquationSystems& eqSystems,
  const bool elementContinuityEqs)
  : EquationSystem(eqSystems, "ContinuityEQS","continuity"),
    elementContinuityEqs_(elementContinuityEqs),
    managePNG_(realm_.get_consistent_mass_matrix_png("pressure")),
    pressure_(NULL),
    dpdx_(NULL),
    massFlowRate_(NULL),
    coordinates_(NULL),
    pTmp_(NULL),
    assembleNodalGradPAlgDriver_(new AssembleNodalGradPAlgorithmDriver(realm_)),
    computeMdotAlgDriver_(new ComputeMdotAlgorithmDriver(realm_)),
    projectedNodalGradEqs_(NULL)
{
  dofName_ = "pressure";

  // message to user
  if ( realm_.realmUsesEdges_ && elementContinuityEqs_)
    NaluEnv::self().naluOutputP0() << "Edge scheme active (all scalars); element-based (continuity)!" << std::endl;

  // error check
  if ( !elementContinuityEqs_ && !realm_.realmUsesEdges_ )
    throw std::runtime_error("If using the non-element-based continuity system, edges must be active at realm level");

  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("pressure");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_CONTINUITY);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("pressure");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for pressure: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.equationSystems_.equationSystemVector_.push_back(this);
  
  // create projected nodal gradient equation system
  if ( managePNG_ ) {
    manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ContinuityEquationSystem::~ContinuityEquationSystem()
{
  delete assembleNodalGradPAlgDriver_;
  delete computeMdotAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // register dof; set it as a restart variable
  pressure_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure"));
  stk::mesh::put_field_on_mesh(*pressure_, *part, nullptr);
  realm_.augment_restart_variable_list("pressure");

  dpdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx"));
  stk::mesh::put_field_on_mesh(*dpdx_, *part, nDim, nullptr);

  // delta solution for linear solver; share delta with other split systems
  pTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pTmp"));
  stk::mesh::put_field_on_mesh(*pTmp_, *part, nullptr);

  coordinates_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates"));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);

}

//--------------------------------------------------------------------------
//-------- register_element_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_element_fields(
  stk::mesh::Part * /* part */,
  const stk::topology & /* theTopo */)
{
  // nothing as of yet
}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_edge_fields(
  stk::mesh::Part *part)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  massFlowRate_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::EDGE_RANK, "mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*massFlowRate_, *part, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // non-solver, dpdx
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &pressureNone = pressure_->field_of_state(stk::mesh::StateNone);
  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradPAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradPAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( !elementContinuityEqs_ && edgeNodalGradient_ ) {
        theAlg = new AssembleNodalGradEdgeAlgorithm(realm_, part, &pressureNone, &dpdxNone);
      }
      else {
        theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, &pressureNone, &dpdxNone, edgeNodalGradient_);
      }
      assembleNodalGradPAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  if ( !elementContinuityEqs_ ) {

    // pure edge-based scheme

    // mdot
    std::map<AlgorithmType, Algorithm *>::iterator itc =
      computeMdotAlgDriver_->algMap_.find(algType);
    if ( itc == computeMdotAlgDriver_->algMap_.end() ) {
      ComputeMdotEdgeAlgorithm *theAlg
        = new ComputeMdotEdgeAlgorithm(realm_, part);
      computeMdotAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      itc->second->partVec_.push_back(part);
    }

    // solver
    std::map<AlgorithmType, SolverAlgorithm *>::iterator its =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleContinuityEdgeSolverAlgorithm *theAlg
        = new AssembleContinuityEdgeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      its->second->partVec_.push_back(part);
    }
  }
  else {

    // pure element-based scheme

    // mdot
    std::map<AlgorithmType, Algorithm *>::iterator itc =
      computeMdotAlgDriver_->algMap_.find(algType);
    if ( itc == computeMdotAlgDriver_->algMap_.end() ) {
      ComputeMdotElemAlgorithm *theAlg
        = new ComputeMdotElemAlgorithm(realm_, part, realm_.realmUsesEdges_);
      computeMdotAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      itc->second->partVec_.push_back(part);
    }

    // solver
    if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {
      std::map<AlgorithmType, SolverAlgorithm *>::iterator its
        = solverAlgDriver_->solverAlgMap_.find(algType);
      if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
        SolverAlgorithm *theSolverAlg = NULL;
        theSolverAlg = new AssembleContinuityElemSolverAlgorithm(realm_, part, this);
        solverAlgDriver_->solverAlgMap_[algType] = theSolverAlg;

        // look for fully integrated source terms
        std::map<std::string, std::vector<std::string> >::iterator isrc
          = realm_.solutionOptions_->elemSrcTermsMap_.find("continuity");

        if ( isrc != realm_.solutionOptions_->elemSrcTermsMap_.end() ) {

          if ( realm_.realmUsesEdges_ )
            throw std::runtime_error("ContinuityElemSrcTerms::Error can not use element source terms for an edge-based scheme");

          std::vector<std::string> mapNameVec = isrc->second;
          for (size_t k = 0; k < mapNameVec.size(); ++k ) {
            std::string sourceName = mapNameVec[k];
            SupplementalAlgorithm *suppAlg = NULL;
            if (sourceName == "SteadyTaylorVortex" ) {
              suppAlg = new SteadyTaylorVortexContinuitySrcElemSuppAlg(realm_);
            }
            else if ( sourceName == "VariableDensity" ) {
              suppAlg = new VariableDensityContinuitySrcElemSuppAlg(realm_);
            }
            else if (sourceName == "density_time_derivative" ) {
              suppAlg = new ContinuityMassElemSuppAlgDep(realm_, false);
            }
            else if (sourceName == "lumped_density_time_derivative" ) {
              suppAlg = new ContinuityMassElemSuppAlgDep(realm_, true);
            }
            else {
              throw std::runtime_error("ContinuityElemSrcTerms::Error Source term is not supported: " + sourceName);
            }
            NaluEnv::self().naluOutputP0() << "ContinuityElemSrcTerms::added " << sourceName << std::endl;
            theSolverAlg->supplementalAlg_.push_back(suppAlg);
          }
        }
      } else {
        its->second->partVec_.push_back(part);
      }
    } else {
      // Homogeneous kernel implementation
      if ( realm_.realmUsesEdges_ )
        throw std::runtime_error("ContinuityElemSrcTerms::Error can not use element source terms for an edge-based scheme");

      KernelBuilder kb(*this, *part, solverAlgDriver_->solverAlgorithmMap_, realm_.using_tensor_product_kernels());

      kb.build_topo_kernel_if_requested<ContinuityMassElemKernel>
      ("density_time_derivative",
        realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs(), false);

      kb.build_topo_kernel_if_requested<ContinuityMassElemKernel>
      ("lumped_density_time_derivative",
        realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs(), true);

      kb.build_topo_kernel_if_requested<ContinuityAdvElemKernel>
      ("advection",
        realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs());

      kb.build_sgl_kernel_if_requested<ContinuityMassHOElemKernel>
      ("experimental_ho_density_time_derivative",
        realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs_HO());

      kb.build_sgl_kernel_if_requested<ContinuityMassHOElemKernel>
      ("experimental_ho_density_time_derivative",
        realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs_HO());

      kb.build_sgl_kernel_if_requested<PressurePoissonHOElemKernel>
      ("experimental_ho_advection",
        realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs_HO(), false);

      kb.build_sgl_kernel_if_requested<PressurePoissonHOElemKernel>
      ("experimental_ho_advection_reduced_sens",
        realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs_HO(), true);

      kb.build_sgl_kernel_if_requested<VariableDensityContinuityMMSHOElemKernel>
        ("experimental_ho_vdmms",
          realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs_HO());

      kb.report();
    }

  }

  // time term using lumped mass
  std::map<std::string, std::vector<std::string> >::iterator isrc =
    realm_.solutionOptions_->srcTermsMap_.find("continuity");
  if ( isrc != realm_.solutionOptions_->srcTermsMap_.end() ) {
    const AlgorithmType algMass = MASS;
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsm =
      solverAlgDriver_->solverAlgMap_.find(algMass);
    if ( itsm == solverAlgDriver_->solverAlgMap_.end() ) {
      // create the solver alg
      AssembleNodeSolverAlgorithm *theAlg
      = new AssembleNodeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algMass] = theAlg;
      
      std::vector<std::string> mapNameVec = isrc->second;
      
      for (size_t k = 0; k < mapNameVec.size(); ++k ) {
	
        std::string sourceName = mapNameVec[k];

        SupplementalAlgorithm *suppAlg = NULL;
        if ( sourceName == "density_time_derivative" ) {
          // now create the supplemental alg for mass term
          if ( realm_.number_of_states() == 2 ) {
            suppAlg = new ContinuityMassBackwardEulerNodeSuppAlg(realm_);
          }
          else {
            suppAlg = new ContinuityMassBDF2NodeSuppAlg(realm_);
          }
        }
        else if ( sourceName == "low_speed_compressible" ) {
          suppAlg = new ContinuityLowSpeedCompressibleNodeSuppAlg(realm_);
        }
        else if ( sourceName == "gcl" ) {
          suppAlg = new ContinuityGclNodeSuppAlg(realm_);
        }
        else if ( sourceName == "VariableDensity" ) {
          suppAlg = new VariableDensityContinuitySrcNodeSuppAlg(realm_);
        }
        else if ( sourceName == "VariableDensityNonIso" ) {
          suppAlg = new VariableDensityNonIsoContinuitySrcNodeSuppAlg(realm_);
        }
        else {
          throw std::runtime_error("ContinuityNodalSrcTerms::Error Source term is not supported: " + sourceName);
        }
        NaluEnv::self().naluOutputP0() << "ContinuityNodalSrcTerms::added " << sourceName << std::endl;
        theAlg->supplementalAlg_.push_back(suppAlg);
      }
    }
    else {
      itsm->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const InflowBoundaryConditionData &inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType &pressureNone = pressure_->field_of_state(stk::mesh::StateNone);
  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();

  // register boundary data; cont_velocity_bc
  if ( !realm_.solutionOptions_->activateOpenMdotCorrection_ ) {
    VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "cont_velocity_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);
    
    // extract the value for user specified velocity and save off the AuxFunction
    InflowUserData userData = inflowBCData.userData_;
    std::string velocityName = "velocity";
    UserDataType theDataType = get_bc_data_type(userData, velocityName);
    
    AuxFunction *theAuxFunc = NULL;
    if ( CONSTANT_UD == theDataType ) {
      Velocity ux = userData.u_;
      std::vector<double> userSpec(nDim);
      userSpec[0] = ux.ux_;
      userSpec[1] = ux.uy_;
      if ( nDim > 2)
        userSpec[2] = ux.uz_;
      
      // new it
      theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);    
    }
    else if ( FUNCTION_UD == theDataType ) {
      // extract the name/params
      std::string fcnName = get_bc_function_name(userData, velocityName);
      std::vector<double> theParams = get_bc_function_params(userData, velocityName);
      if ( theParams.size() == 0 )
        NaluEnv::self().naluOutputP0() << "function parameter size is zero" << std::endl;
      // switch on the name found...
      if ( fcnName == "convecting_taylor_vortex" ) {
        theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "SteadyTaylorVortex" ) {
        theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "VariableDensity" ) {
        theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "VariableDensityNonIso" ) {
        theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "kovasznay") {
        theAuxFunc = new KovasznayVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "TaylorGreen") {
        theAuxFunc = new TaylorGreenVelocityAuxFunction(0, nDim);
      }
      else if ( fcnName == "BoussinesqNonIso") {
        theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0, nDim);
      }
      else {
        throw std::runtime_error("ContEquationSystem::register_inflow_bc: limited functions supported");
      }
    }
    else {
      throw std::runtime_error("ContEquationSystem::register_inflow_bc: only constant and user function supported");
    }

    
    // bc data alg
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    
    // how to populate the field?
    if ( userData.externalData_ ) {
      // xfer will handle population; only need to populate the initial value
      realm_.initCondAlg_.push_back(auxAlg);
    }
    else {
      // put it on bcData
      bcDataAlg_.push_back(auxAlg);
    }
  }

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradPAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradPAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &pressureNone, &dpdxNone, edgeNodalGradient_);
      assembleNodalGradPAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // check to see if we are using shifted as inflow is shared
  const bool useShifted = !elementContinuityEqs_ ? true : realm_.get_cvfem_shifted_mdot();

  // non-solver inflow mdot - shared by both elem/edge
  std::map<AlgorithmType, Algorithm *>::iterator itmd =
    computeMdotAlgDriver_->algMap_.find(algType);
  if ( itmd == computeMdotAlgDriver_->algMap_.end() ) {
    ComputeMdotInflowAlgorithm *theAlg
      = new ComputeMdotInflowAlgorithm(realm_, part, useShifted);
    computeMdotAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itmd->second->partVec_.push_back(part);
  }
  
  // solver; lhs
  if ( realm_.solutionOptions_->useConsolidatedBcSolverAlg_ ) {      

    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    
    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;
    
    std::tie(solverAlg, solverAlgWasBuilt) 
      = build_or_add_part_to_face_bc_solver_alg(*this, *part, solverAlgMap, "inflow");
    
    ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
    auto& activeKernels = solverAlg->activeKernels_;
    
    if (solverAlgWasBuilt) {
      build_face_topo_kernel_automatic<ContinuityInflowElemKernel>
        (partTopo, *this, activeKernels, "continuity_inflow",
         realm_.bulk_data(), *realm_.solutionOptions_, useShifted, dataPreReqs);
    }

  }
  else {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator its =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleContinuityInflowSolverAlgorithm *theAlg
        = new AssembleContinuityInflowSolverAlgorithm(realm_, part, this, useShifted);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      its->second->partVec_.push_back(part);
    }
  }

}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const OpenBoundaryConditionData & /* openBCData */)
{

  const AlgorithmType algType = OPEN;

  // register boundary data
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  ScalarFieldType *pressureBC = NULL;
  if ( !realm_.solutionOptions_->activateOpenMdotCorrection_ ) {
    pressureBC = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure_bc"));
    stk::mesh::put_field_on_mesh(*pressureBC, *part , nullptr);
  }

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradPAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradPAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradPOpenBoundaryAlgorithm(realm_, part, edgeNodalGradient_);
      assembleNodalGradPAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
}

  // mdot at open and lhs
  if ( !elementContinuityEqs_ ) {
    // non-solver edge alg; compute open mdot
    std::map<AlgorithmType, Algorithm *>::iterator itm =
      computeMdotAlgDriver_->algMap_.find(algType);
    if ( itm == computeMdotAlgDriver_->algMap_.end() ) {
      ComputeMdotEdgeOpenAlgorithm *theAlg
      = new ComputeMdotEdgeOpenAlgorithm(realm_, part);
      computeMdotAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      itm->second->partVec_.push_back(part);
    }

    // solver; lhs
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleContinuityEdgeOpenSolverAlgorithm *theAlg
      = new AssembleContinuityEdgeOpenSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
  }
  else {
    
    if ( realm_.solutionOptions_->useConsolidatedBcSolverAlg_ ) {      
      // non-solver elem alg; compute open mdot (transition to penalty approach)
      std::map<AlgorithmType, Algorithm *>::iterator itm =
        computeMdotAlgDriver_->algMap_.find(algType);
      if ( itm == computeMdotAlgDriver_->algMap_.end() ) {
        ComputeMdotElemOpenPenaltyAlgorithm *theAlg
          = new ComputeMdotElemOpenPenaltyAlgorithm(realm_, part);
        computeMdotAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        itm->second->partVec_.push_back(part);
      }
      
      // solver for continuity open
      auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
      
      stk::topology elemTopo = get_elem_topo(realm_, *part);
      
      AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
      bool solverAlgWasBuilt = false;

      std::tie(faceElemSolverAlg, solverAlgWasBuilt) 
        = build_or_add_part_to_face_elem_solver_alg(algType, *this, *part, elemTopo, solverAlgMap, "open");
      
      auto& activeKernels = faceElemSolverAlg->activeKernels_;
      
      if (solverAlgWasBuilt) {
        
        build_face_elem_topo_kernel_automatic<ContinuityOpenElemKernel>
          (partTopo, elemTopo, *this, activeKernels, "continuity_open",
           realm_.meta_data(), *realm_.solutionOptions_,
           faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_);
        
      }
    }
    else {
      // non-solver elem alg; compute open mdot
      std::map<AlgorithmType, Algorithm *>::iterator itm =
        computeMdotAlgDriver_->algMap_.find(algType);
      if ( itm == computeMdotAlgDriver_->algMap_.end() ) {
        ComputeMdotElemOpenAlgorithm *theAlg
          = new ComputeMdotElemOpenAlgorithm(realm_, part);
        computeMdotAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        itm->second->partVec_.push_back(part);
      }
      
      // solver; lhs
      std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
        solverAlgDriver_->solverAlgMap_.find(algType);
      if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
        AssembleContinuityElemOpenSolverAlgorithm *theAlg
          = new AssembleContinuityElemOpenSolverAlgorithm(realm_, part, this);
        solverAlgDriver_->solverAlgMap_[algType] = theAlg;
      }
      else {
        itsi->second->partVec_.push_back(part);
      }     
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData & /* wallBCData */)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  ScalarFieldType &pressureNone = pressure_->field_of_state(stk::mesh::StateNone);
  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradPAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradPAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &pressureNone, &dpdxNone, edgeNodalGradient_);
      assembleNodalGradPAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData & /* symmetryBCData */)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  ScalarFieldType &pressureNone = pressure_->field_of_state(stk::mesh::StateNone);
  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it = assembleNodalGradPAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradPAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &pressureNone, &dpdxNone, edgeNodalGradient_);
      assembleNodalGradPAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_abltop_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_abltop_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const ABLTopBoundaryConditionData &abltopBCData)
{
  auto userData = abltopBCData.userData_;

  if (!userData.ABLTopBC_) {
    SymmetryBoundaryConditionData symData(abltopBCData.boundaryConditions_);
    register_symmetry_bc(part, partTopo, symData);
    return;
  }

#ifdef NALU_USES_FFTW
  // algorithm type
  const AlgorithmType algType = TOP_ABL;

  if ( !realm_.realmUsesEdges_ )
    throw std::runtime_error("ABLTopBoundaryCondition::Error you must use the edge-based scheme");

  ScalarFieldType &pressureNone = pressure_->field_of_state(stk::mesh::StateNone);
  VectorFieldType &dpdxNone = dpdx_->field_of_state(stk::mesh::StateNone);

/*
  // register boundary data; cont_velocity_bc
  if ( !realm_.solutionOptions_->activateOpenMdotCorrection_ ) {
  std::string bcFieldName = realm_.solutionOptions_->activateOpenMdotCorrection_?"velocity_bc" : "cont_velocity_bc";
    VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, bcFieldName));
    stk::mesh::put_field(*theBcField, *part, nDim);
    
    // extract the value for user specified velocity and save off the AuxFunction
    InflowUserData userData = inflowBCData.userData_;
    std::string velocityName = "velocity";
    UserDataType theDataType = get_bc_data_type(userData, velocityName);
    
    AuxFunction *theAuxFunc = NULL;
    if ( CONSTANT_UD == theDataType ) {
      Velocity ux = userData.u_;
      std::vector<double> userSpec(nDim);
      userSpec[0] = ux.ux_;
      userSpec[1] = ux.uy_;
      if ( nDim > 2)
        userSpec[2] = ux.uz_;
      
      // new it
      theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);    
    }
    else if ( FUNCTION_UD == theDataType ) {
      // extract the name/params
      std::string fcnName = get_bc_function_name(userData, velocityName);
      std::vector<double> theParams = get_bc_function_params(userData, velocityName);
      if ( theParams.size() == 0 )
        NaluEnv::self().naluOutputP0() << "function parameter size is zero" << std::endl;
      // switch on the name found...
      if ( fcnName == "convecting_taylor_vortex" ) {
        theAuxFunc = new ConvectingTaylorVortexVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "SteadyTaylorVortex" ) {
        theAuxFunc = new SteadyTaylorVortexVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "VariableDensity" ) {
        theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "VariableDensityNonIso" ) {
        theAuxFunc = new VariableDensityVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "kovasznay") {
        theAuxFunc = new KovasznayVelocityAuxFunction(0,nDim);
      }
      else if ( fcnName == "TaylorGreen") {
        theAuxFunc = new TaylorGreenVelocityAuxFunction(0, nDim);
      }
      else if ( fcnName == "BoussinesqNonIso") {
        theAuxFunc = new BoussinesqNonIsoVelocityAuxFunction(0, nDim);
      }
      else {
        throw std::runtime_error("ContEquationSystem::register_inflow_bc: limited functions supported");
      }
    }
    else {
      throw std::runtime_error("ContEquationSystem::register_inflow_bc: only constant and user function supported");
    }

    
    // bc data alg
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    
    // how to populate the field?
    if ( userData.externalData_ ) {
      // xfer will handle population; only need to populate the initial value
      realm_.initCondAlg_.push_back(auxAlg);
    }
    else {
      // put it on bcData
      bcDataAlg_.push_back(auxAlg);
    }
  }
*/

  // non-solver; contribution to Gjp; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradPAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradPAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &pressureNone, &dpdxNone, edgeNodalGradient_);
      assembleNodalGradPAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // check to see if we are using shifted as inflow is shared
  const bool useShifted = !elementContinuityEqs_ ? true : realm_.get_cvfem_shifted_mdot();

  // non-solver inflow mdot - shared by both elem/edge
  std::map<AlgorithmType, Algorithm *>::iterator itmd =
    computeMdotAlgDriver_->algMap_.find(algType);
  if ( itmd == computeMdotAlgDriver_->algMap_.end() ) {
    ComputeMdotInflowAlgorithm *theAlg
      = new ComputeMdotInflowAlgorithm(realm_, part, useShifted);
    computeMdotAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itmd->second->partVec_.push_back(part);
  }
  
  // solver; lhs
  if ( realm_.solutionOptions_->useConsolidatedBcSolverAlg_ ) {      

    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    
    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;
    
    std::tie(solverAlg, solverAlgWasBuilt) 
      = build_or_add_part_to_face_bc_solver_alg(*this, *part, solverAlgMap, "inflow");
    
    ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
    auto& activeKernels = solverAlg->activeKernels_;
    
    if (solverAlgWasBuilt) {
      build_face_topo_kernel_automatic<ContinuityInflowElemKernel>
        (partTopo, *this, activeKernels, "continuity_inflow",
         realm_.bulk_data(), *realm_.solutionOptions_, useShifted, dataPreReqs);
    }

  }
  else {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator its =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleContinuityInflowSolverAlgorithm *theAlg
        = new AssembleContinuityInflowSolverAlgorithm(realm_, part, this, useShifted);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      its->second->partVec_.push_back(part);
    }
  }

#else
  throw std::runtime_error("Cannot initialize ABL top BC because FFTW support is mising.\n Set ENABLE_FFTW to ON in nalu-wind/CMakeLists.txt, reconfigure and recompile.");
#endif
}
    

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &theTopo)
{
  const AlgorithmType algType = NON_CONFORMAL;

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // mdot at nc bc; register field; require topo and num ips
  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsBip = meFC->num_integration_points();
  
  stk::topology::rank_t sideRank = static_cast<stk::topology::rank_t>(meta_data.side_rank());
  GenericFieldType *mdotBip =
    &(meta_data.declare_field<GenericFieldType>(sideRank, "nc_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*mdotBip, *part, numScsBip , nullptr);

  // non-solver; contribution to Gjp; DG algorithm decides on locations for integration points
  if ( !managePNG_ ) {
    if ( edgeNodalGradient_ ) {    
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradPAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradPAlgDriver_->algMap_.end() ) {
        Algorithm *theAlg 
          = new AssembleNodalGradBoundaryAlgorithm(realm_, part, pressure_, dpdx_, edgeNodalGradient_);
        assembleNodalGradPAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
    else {
      // proceed with DG
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradPAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradPAlgDriver_->algMap_.end() ) {
        AssembleNodalGradNonConformalAlgorithm *theAlg 
          = new AssembleNodalGradNonConformalAlgorithm(realm_, part, pressure_, dpdx_);
        assembleNodalGradPAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
  }

  // non-solver alg; compute nc mdot (same for edge and element-based)
  std::map<AlgorithmType, Algorithm *>::iterator itm =
    computeMdotAlgDriver_->algMap_.find(algType);
  if ( itm == computeMdotAlgDriver_->algMap_.end() ) {
    ComputeMdotNonConformalAlgorithm *theAlg
      = new ComputeMdotNonConformalAlgorithm(realm_, part, pressure_, dpdx_);
    computeMdotAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itm->second->partVec_.push_back(part);
  }

  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    AssembleContinuityNonConformalSolverAlgorithm *theAlg
      = new AssembleContinuityNonConformalSolverAlgorithm(realm_, part, this, pressure_, dpdx_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    itsi->second->partVec_.push_back(part);
  }
}
//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(pressure_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(pressure_,1,1)));
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
    realm_.solutionOptions_->fixPressureInfo_->create_part_vector(realm_.meta_data());
    stk::mesh::PartVector& pvec = realm_.solutionOptions_->fixPressureInfo_->partVec_;

    // The user could have provided just a Node ID instead of a part vector
    stk::mesh::Part* firstPart = pvec.size() > 0? pvec.at(0) : nullptr;

    auto it = solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverDirichAlgMap_.end()) {
      FixPressureAtNodeAlgorithm* theAlg = new FixPressureAtNodeAlgorithm(
        realm_, firstPart, this);
      // populate the remaining parts if necessary
      for(size_t i=1; i < pvec.size(); i++)
        theAlg->partVec_.push_back( pvec[i]);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    } else {
      throw std::runtime_error("ContinuityEquationSystem::initialize: logic error. Multiple initializations of FixPressureAtNodeAlgorithm.");
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

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_CONTINUITY;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("pressure");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_CONTINUITY);
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
  stk::mesh::Part *part,
  const std::map<std::string, std::string> &theNames,
  const std::map<std::string, std::vector<double> >& theParams)
{
  // iterate map and check for name
  const std::string dofName = "pressure";
  std::map<std::string, std::string>::const_iterator iterName
    = theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;
    AuxFunction *theAuxFunc = NULL;
    if ( fcnName == "convecting_taylor_vortex" ) {
      // create the function
      theAuxFunc = new ConvectingTaylorVortexPressureAuxFunction();      
    }
    else if ( fcnName == "wind_energy_taylor_vortex") {
      // extract the params
      auto iterParams = theParams.find(dofName);
      std::vector<double> fcnParams = (iterParams != theParams.end()) ? (*iterParams).second : std::vector<double>();
      theAuxFunc = new WindEnergyTaylorVortexPressureAuxFunction(fcnParams);
    }
    else if ( fcnName == "SteadyTaylorVortex" ) {
      // create the function
      theAuxFunc = new SteadyTaylorVortexPressureAuxFunction();      
    }
    else if ( fcnName == "VariableDensity" ) {
      // create the function
      theAuxFunc = new VariableDensityPressureAuxFunction();      
    }
    else if ( fcnName == "VariableDensityNonIso" ) {
      // create the function
      theAuxFunc = new VariableDensityPressureAuxFunction();      
    }
    else if ( fcnName == "TaylorGreen") {
      // create the function
      theAuxFunc = new TaylorGreenPressureAuxFunction();      
    }
    else if ( fcnName == "kovasznay" ) {
      theAuxFunc = new KovasznayPressureAuxFunction();
    }
    else {
      throw std::runtime_error("ContinuityEquationSystem::register_initial_condition_fcn: limited functions supported");
    }
    
    // create the algorithm
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
				 pressure_, theAuxFunc,
				 stk::topology::NODE_RANK);
    
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
  if ( NULL == projectedNodalGradEqs_ ) {
    projectedNodalGradEqs_ 
      = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG_P, "dpdx", "qTmp", "pressure", "PNGradPEQS");
  }
  // fill the map for expected boundary condition names...
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "pressure");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "pressure");
  projectedNodalGradEqs_->set_data_map(OPEN_BC, 
   realm_.solutionOptions_->activateOpenMdotCorrection_ ? "pressure" : "pressure_bc");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "pressure");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
ContinuityEquationSystem::compute_projected_nodal_gradient()
{
  if ( !managePNG_ ) {
    const double timeA = -NaluEnv::self().nalu_time();
    assembleNodalGradPAlgDriver_->execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  }
  else {
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
    AssembleOversetPressureAlgorithm* theAlg
      = new AssembleOversetPressureAlgorithm(realm_, nullptr, this, theField);
    solverAlgDriver_->solverConstraintAlgMap_[algType] = theAlg;
  } else {
    throw std::runtime_error("ContinuityEquationSystem::register_overset_bc: "
                             "Multiple invocations of overset is not allowed");
  }
}

} // namespace nalu
} // namespace Sierra
