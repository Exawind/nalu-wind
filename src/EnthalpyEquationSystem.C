/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <EnthalpyEquationSystem.h>
#include <wind_energy/ABLForcingAlgorithm.h>
#include <AlgorithmDriver.h>
#include <AssembleScalarFluxBCSolverAlgorithm.h>
#include <AssembleScalarEdgeOpenSolverAlgorithm.h>
#include <AssembleScalarEdgeSolverAlgorithm.h>
#include <AssembleScalarEigenEdgeSolverAlgorithm.h>
#include <AssembleScalarElemSolverAlgorithm.h>
#include <AssembleScalarElemOpenSolverAlgorithm.h>
#include <AssembleScalarNonConformalSolverAlgorithm.h>
#include <AssembleTemperatureNormalGradientBCSolverAlgorithm.h>
#include <AssembleNodalGradAlgorithmDriver.h>
#include <AssembleNodalGradEdgeAlgorithm.h>
#include <AssembleNodalGradElemAlgorithm.h>
#include <AssembleNodalGradBoundaryAlgorithm.h>
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
#include <AssembleWallHeatTransferAlgorithmDriver.h>
#include <AuxFunctionAlgorithm.h>
#include <ComputeHeatTransferEdgeWallAlgorithm.h>
#include <ComputeHeatTransferElemWallAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
#include <EnthalpyEffectiveDiffFluxCoeffAlgorithm.h>
#include <EnthalpyPmrSrcNodeSuppAlg.h>
#include <EnthalpyLowSpeedCompressibleNodeSuppAlg.h>
#include <EnthalpyPressureWorkNodeSuppAlg.h>
#include <EnthalpyViscousWorkNodeSuppAlg.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <Enums.h>
#include <FieldFunctions.h>
#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSystem.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <ProjectedNodalGradientEquationSystem.h>
#include <Realm.h>
#include <Realms.h>
#include <ScalarGclNodeSuppAlg.h>
#include <ScalarMassBackwardEulerNodeSuppAlg.h>
#include <ScalarMassBDF2NodeSuppAlg.h>
#include <ScalarMassElemSuppAlgDep.h>
#include <EnthalpyABLSrcNodeSuppAlg.h>
#include <Simulation.h>
#include <TimeIntegrator.h>
#include <SolverAlgorithmDriver.h>
#include <SolutionOptions.h>
#include <wind_energy/ABLForcingAlgorithm.h>

// template for kernels
#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

// kernels
#include <AssembleFaceElemSolverAlgorithm.h>
#include <kernel/ScalarMassElemKernel.h>
#include <kernel/ScalarAdvDiffElemKernel.h>
#include <kernel/ScalarUpwAdvDiffElemKernel.h>

#include <kernel/ScalarMassHOElemKernel.h>
#include <kernel/ScalarAdvDiffHOElemKernel.h>

// bc kernels
#include <kernel/ScalarOpenAdvElemKernel.h>

// nso
#include <nso/ScalarNSOElemKernel.h>
#include <nso/ScalarNSOKeElemKernel.h>

// nso to be deprecated
#include <nso/ScalarNSOElemSuppAlgDep.h>
#include <nso/ScalarNSOKeElemSuppAlg.h>

// props
#include <property_evaluator/EnthalpyPropertyEvaluator.h>
#include <MaterialPropertys.h>
#include <property_evaluator/SpecificHeatPropertyEvaluator.h>
#include <property_evaluator/TemperaturePropAlgorithm.h>
#include <property_evaluator/ThermalConductivityFromPrandtlPropAlgorithm.h>

// user functions
#include <user_functions/FlowPastCylinderTempAuxFunction.h>
#include <user_functions/VariableDensityNonIsoTemperatureAuxFunction.h>
#include <user_functions/VariableDensityNonIsoEnthalpySrcNodeSuppAlg.h>
#include <user_functions/VariableDensityEnthalpyMMSHOElemKernel.h>


#include <user_functions/BoussinesqNonIsoTemperatureAuxFunction.h>
#include <user_functions/BoussinesqNonIsoEnthalpySrcNodeSuppAlg.h>

#include <user_functions/CappingInversionTemperatureAuxFunction.h>

// overset
#include <overset/UpdateOversetFringeAlgorithmDriver.h>

// stk_util
#include <stk_util/parallel/Parallel.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// nalu utility
#include <utils/StkHelpers.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// EnthalpyEquationSystem - manages h pde system; with T as dependent var
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
EnthalpyEquationSystem::EnthalpyEquationSystem(
  EquationSystems& eqSystems,
  const double minT,
  const double maxT,
  const bool outputClippingDiag)
  : EquationSystem(eqSystems, "EnthalpyEQS", "enthalpy"),
    minimumT_(minT),
    maximumT_(maxT),
    managePNG_(realm_.get_consistent_mass_matrix_png("enthalpy")),
    outputClippingDiag_(outputClippingDiag),
    enthalpy_(NULL),
    temperature_(NULL),
    dhdx_(NULL),
    hTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    thermalCond_(NULL),
    specHeat_(NULL),
    divQ_(NULL),
    pOld_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(realm_, "enthalpy", "dhdx")),
    diffFluxCoeffAlgDriver_(new AlgorithmDriver(realm_)),
    assembleWallHeatTransferAlgDriver_(NULL),
    pmrCouplingActive_(false),
    lowSpeedCompressActive_(false),
    projectedNodalGradEqs_(NULL),
    isInit_(true)
{
  dofName_ = "enthalpy";

  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("enthalpy");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_ENTHALPY);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("enthalpy");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for enthalpy: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // advertise need for the enthalpy property evaluator
  realm_.needs_enthalpy(true);

  // advertise as non isothermal
  realm_.isothermalFlow_ = false;

  // check for PMR coupling
  std::map<std::string, std::vector<std::string> >::iterator isrc 
    = realm_.solutionOptions_->srcTermsMap_.find("enthalpy");
  if ( isrc != realm_.solutionOptions_->srcTermsMap_.end() ) {
    std::vector<std::string> mapNameVec = isrc->second;
    for (size_t k = 0; k < mapNameVec.size(); ++k ) {
      std::string sourceName = mapNameVec[k];   
      if ( sourceName == "participating_media_radiation" ) {
        pmrCouplingActive_ = true;
      }
      else if ( sourceName == "low_speed_compressible" ) {
        lowSpeedCompressActive_ = true;
      }
    }
  }

  // create projected nodal gradient equation system
  if ( managePNG_ ) {
    manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
EnthalpyEquationSystem::~EnthalpyEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
  delete diffFluxCoeffAlgDriver_;

  if ( NULL != assembleWallHeatTransferAlgDriver_ )
    delete assembleWallHeatTransferAlgDriver_;

  std::vector<TemperaturePropAlgorithm *>::iterator ii;
  for( ii=enthalpyFromTemperatureAlg_.begin(); ii!=enthalpyFromTemperatureAlg_.end(); ++ii )
    delete *ii;

  for( ii=bcEnthalpyFromTemperatureAlg_.begin(); ii!=bcEnthalpyFromTemperatureAlg_.end(); ++ii )
    delete *ii;

  std::vector<Algorithm *>::iterator iib;
  for( iib=bdf2CopyStateAlg_.begin(); iib!=bdf2CopyStateAlg_.end(); ++iib )
    delete *iib;

  for( iib=bcCopyStateAlg_.begin(); iib!=bcCopyStateAlg_.end(); ++iib )
    delete *iib;

}


//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::initial_work()
{
  // compute all enthalpy values given IC
  for ( size_t k = 0; k < enthalpyFromTemperatureAlg_.size(); ++k )
    enthalpyFromTemperatureAlg_[k]->execute();

  // call base class method (will process copyStateAlg)
  EquationSystem::initial_work();

  // manage bdf2; state Np1 to N; not active if restart is requested
  for ( size_t k = 0; k < bdf2CopyStateAlg_.size(); ++k )
    bdf2CopyStateAlg_[k]->execute();

}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  enthalpy_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "enthalpy", numStates));
  stk::mesh::put_field_on_mesh(*enthalpy_, *part, nullptr);
  realm_.augment_restart_variable_list("enthalpy");

  // temperature required in restart
  temperature_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature"));
  stk::mesh::put_field_on_mesh(*temperature_, *part, nullptr);
  realm_.augment_restart_variable_list("temperature");

  dhdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dhdx"));
  stk::mesh::put_field_on_mesh(*dhdx_, *part, nDim, nullptr);

  // props
  specHeat_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_heat"));
  stk::mesh::put_field_on_mesh(*specHeat_, *part, nullptr);
  
  visc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  // push standard props to property list; enthalpy managed along with Cp
  realm_.augment_property_map(SPEC_HEAT_ID, specHeat_);
  realm_.augment_property_map(VISCOSITY_ID, visc_);

  // special thermal conductivity
  thermalCond_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "thermal_conductivity"));
  stk::mesh::put_field_on_mesh(*thermalCond_, *part, nullptr);

  // check to see if Prandtl number was provided
  bool prProvided = false;
  const double providedPr = realm_.get_lam_prandtl("enthalpy", prProvided);
  if ( prProvided ) {
    // compute thermal conductivity using Pr; create and push back the algorithm
    NaluEnv::self().naluOutputP0() << "Laminar Prandtl provided; will compute Thermal conductivity based on this constant value" << std::endl;
    Algorithm *propAlg 
      = new ThermalConductivityFromPrandtlPropAlgorithm(realm_, part, thermalCond_, specHeat_, visc_, providedPr);
    propertyAlg_.push_back(propAlg);
  }
  else {
    // no Pr provided, simply augment property map and expect lambda to be provided in the input file
    realm_.augment_property_map(THERMAL_COND_ID, thermalCond_);
  }

  // delta solution for linear solver; share delta since this is a split system
  hTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pTmp"));
  stk::mesh::put_field_on_mesh(*hTmp_, *part, nullptr);
  
  // turbulent viscosity and effective viscosity
  if ( realm_.is_turbulent() ) {
    tvisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity"));
    stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);
  }

  evisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "effective_viscosity_h"));
  stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);

  // register divergence of radiative heat flux; for now this is an explicit coupling
  if ( pmrCouplingActive_ ) {
    divQ_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_radiative_heat_flux"));
    stk::mesh::put_field_on_mesh(*divQ_, *part, nullptr);
  }

  // need to save off old pressure for pressure time derivative (avoid state for now)
  if ( lowSpeedCompressActive_ ) {
    pOld_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure_old"));
    stk::mesh::put_field_on_mesh(*pOld_, *part, nullptr);
  }

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &enthalpyN = enthalpy_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &enthalpyNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &enthalpyNp1, &enthalpyN,
                               0, 1,
                               stk::topology::NODE_RANK);
    // personally manage enthalpy
    bdf2CopyStateAlg_.push_back(theCopyAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &enthalpyNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dhdxNone = dhdx_->field_of_state(stk::mesh::StateNone);

  // non-solver, dhdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( edgeNodalGradient_ && realm_.realmUsesEdges_ ) {
        theAlg = new AssembleNodalGradEdgeAlgorithm(realm_, part, &enthalpyNp1, &dhdxNone);
      }
      else {
        theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, &enthalpyNp1, &dhdxNone, edgeNodalGradient_);
      }
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // solver; interior contribution (advection + diffusion)
  if ( !realm_.solutionOptions_->useConsolidatedSolverAlg_ ) {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
    = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        if ( !realm_.solutionOptions_->eigenvaluePerturb_ )
          theAlg = new AssembleScalarEdgeSolverAlgorithm(realm_, part, this, enthalpy_, dhdx_, evisc_);
        else
          theAlg = new AssembleScalarEigenEdgeSolverAlgorithm(realm_, part, this, enthalpy_, dhdx_, thermalCond_, specHeat_,
            tvisc_, realm_.get_turb_prandtl(enthalpy_->name()));
      }
      else {
        theAlg = new AssembleScalarElemSolverAlgorithm(realm_, part, this, enthalpy_, dhdx_, evisc_);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;

      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string> >::iterator isrc
      = realm_.solutionOptions_->elemSrcTermsMap_.find("enthalpy");
      if ( isrc != realm_.solutionOptions_->elemSrcTermsMap_.end() ) {

        if ( realm_.realmUsesEdges_ )
          throw std::runtime_error("EnthalpyElemSrcTerms::Error can not use element source terms for an edge-based scheme");

        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if (sourceName == "NSO_2ND" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, enthalpy_, dhdx_, evisc_, 0.0, 0.0);
          }
          else if (sourceName == "NSO_2ND_ALT" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, enthalpy_, dhdx_, evisc_, 0.0, 1.0);
          }
          else if (sourceName == "NSO_4TH" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, enthalpy_, dhdx_, evisc_, 1.0, 0.0);
          }
          else if (sourceName == "NSO_4TH_ALT" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, enthalpy_, dhdx_, evisc_, 1.0, 1.0);
          }
          else if (sourceName == "NSO_2ND_KE" ) {
            const double turbPr = realm_.get_turb_prandtl(enthalpy_->name());
            suppAlg = new ScalarNSOKeElemSuppAlg(realm_, enthalpy_, dhdx_, turbPr, 0.0);
          }
          else if (sourceName == "NSO_4TH_KE" ) {
            const double turbPr = realm_.get_turb_prandtl(enthalpy_->name());
            suppAlg = new ScalarNSOKeElemSuppAlg(realm_, enthalpy_, dhdx_, turbPr, 1.0);
          }
          else if (sourceName == "enthalpy_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, enthalpy_, false);
          }
          else if (sourceName == "lumped_enthalpy_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, enthalpy_, true);
          }
          else {
            throw std::runtime_error("EnthalpyElemSrcTerms::Error Source term is not supported: " + sourceName);
          }
          NaluEnv::self().naluOutputP0() << "EnthalpyElemSrcTerms::added() " << sourceName << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg);
        }
      }
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
  }
  else {
    // Homogeneous kernel implementation
    if ( realm_.realmUsesEdges_ )
      throw std::runtime_error("Enthalpy::Error can not use element source terms for an edge-based scheme");

    KernelBuilder kb(*this, *part, solverAlgDriver_->solverAlgorithmMap_, realm_.using_tensor_product_kernels());
    auto& dataPreReqs = kb.data_prereqs();
    auto& dataPreReqsHO = kb.data_prereqs_HO();

    kb.build_topo_kernel_if_requested<ScalarMassElemKernel>
      ("enthalpy_time_derivative",
        realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, dataPreReqs, false);

    kb.build_topo_kernel_if_requested<ScalarMassElemKernel>
    ("lumped_enthalpy_time_derivative",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, dataPreReqs, true);

    kb.build_topo_kernel_if_requested<ScalarAdvDiffElemKernel>
    ("advection_diffusion",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, evisc_, dataPreReqs);

    kb.build_topo_kernel_if_requested<ScalarUpwAdvDiffElemKernel>
    ("upw_advection_diffusion",
      realm_.bulk_data(), *realm_.solutionOptions_, this, enthalpy_, dhdx_, evisc_, dataPreReqs);

    kb.build_topo_kernel_if_requested<ScalarNSOElemKernel>
    ("NSO_2ND",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, dhdx_, evisc_, 0.0, 0.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<ScalarNSOElemKernel>
    ("NSO_2ND_ALT",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, dhdx_, evisc_, 0.0, 1.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<ScalarNSOElemKernel>
    ("NSO_4TH",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, dhdx_, evisc_, 1.0, 0.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<ScalarNSOElemKernel>
    ("NSO_4TH_ALT",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, dhdx_, evisc_, 1.0, 1.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<ScalarNSOKeElemKernel>
    ("NSO_2ND_KE",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, dhdx_,
      realm_.get_turb_schmidt(enthalpy_->name()), 0.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<ScalarNSOKeElemKernel>
    ("NSO_4TH_KE",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, dhdx_,
      realm_.get_turb_schmidt(enthalpy_->name()), 1.0, dataPreReqs);

    kb.build_sgl_kernel_if_requested<ScalarMassHOElemKernel>
    ("experimental_ho_enthalpy_time_derivative",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, dataPreReqsHO);

    kb.build_sgl_kernel_if_requested<ScalarAdvDiffHOElemKernel>
    ("experimental_ho_advection_diffusion",
      realm_.bulk_data(), *realm_.solutionOptions_, enthalpy_, evisc_, dataPreReqsHO);

    kb.build_sgl_kernel_if_requested<VariableDensityEnthalpyMMSHOElemKernel>
    ("experimental_ho_vdmms",
      realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqsHO);

    kb.report();
  }

  // time term; nodally lumped
  const AlgorithmType algMass = MASS;
  // Check if the user has requested CMM or LMM algorithms; if so, do not
  // include Nodal Mass algorithms
  std::vector<std::string> checkAlgNames = {"enthalpy_time_derivative",
                                            "lumped_enthalpy_time_derivative",
                                            "experimental_ho_enthalpy_time_derivative"};
  bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsm =
    solverAlgDriver_->solverAlgMap_.find(algMass);
  
  if ( itsm == solverAlgDriver_->solverAlgMap_.end() ) {
    // create the solver alg
    AssembleNodeSolverAlgorithm *theAlg
      = new AssembleNodeSolverAlgorithm(realm_, part, this);
    solverAlgDriver_->solverAlgMap_[algMass] = theAlg;

    // now create the supplemental alg for mass term
    if ( !elementMassAlg ) {
      if ( realm_.number_of_states() == 2 ) {
        ScalarMassBackwardEulerNodeSuppAlg *theMass
          = new ScalarMassBackwardEulerNodeSuppAlg(realm_, enthalpy_);
        theAlg->supplementalAlg_.push_back(theMass);
      }
      else {
        ScalarMassBDF2NodeSuppAlg *theMass
          = new ScalarMassBDF2NodeSuppAlg(realm_, enthalpy_);
        theAlg->supplementalAlg_.push_back(theMass);
      }
    }

    // Add src term supp alg...; limited number supported
    std::map<std::string, std::vector<std::string> >::iterator isrc 
      = realm_.solutionOptions_->srcTermsMap_.find("enthalpy");
    if ( isrc != realm_.solutionOptions_->srcTermsMap_.end() ) {      
      std::vector<std::string> mapNameVec = isrc->second;
      for (size_t k = 0; k < mapNameVec.size(); ++k ) {
        std::string sourceName = mapNameVec[k];
        SupplementalAlgorithm *suppAlg = NULL;
        if ( sourceName == "participating_media_radiation" ) {
          suppAlg = new EnthalpyPmrSrcNodeSuppAlg(realm_);
        }
        else if ( sourceName == "low_speed_compressible" ) {
          suppAlg = new EnthalpyLowSpeedCompressibleNodeSuppAlg(realm_);
        }
        else if ( sourceName == "pressure_work" ) {
          suppAlg = new EnthalpyPressureWorkNodeSuppAlg(realm_);
        }
        else if ( sourceName == "viscous_work" ) {
          suppAlg = new EnthalpyViscousWorkNodeSuppAlg(realm_);
        }
        else if ( sourceName == "gcl" ) {
          suppAlg = new ScalarGclNodeSuppAlg(enthalpy_,realm_);
        }
        else if (sourceName == "VariableDensityNonIso" ) {
          suppAlg = new VariableDensityNonIsoEnthalpySrcNodeSuppAlg(realm_);
        }
        else if (sourceName == "BoussinesqNonIso" ) {
          suppAlg = new BoussinesqNonIsoEnthalpySrcNodeSuppAlg(realm_);
        }
        else if (sourceName == "abl_forcing") {
          ThrowAssertMsg(
            ((NULL != realm_.ablForcingAlg_) &&
             (realm_.ablForcingAlg_->temperatureForcingOn())),
            "EnthalpyNodalSrcTerms::ERROR! ABL Forcing parameters must be initialized to use temperature source.");
          suppAlg = new EnthalpyABLSrcNodeSuppAlg(realm_, realm_.ablForcingAlg_);
        }
        else {
          throw std::runtime_error("EnthalpyNodalSrcTerms::Error Source term is not supported: " + sourceName);
        }
        NaluEnv::self().naluOutputP0() << "EnthalpyNodalSrcTerms::added() " << sourceName << std::endl;
        theAlg->supplementalAlg_.push_back(suppAlg);
      }
    }
  }
  else {
    itsm->second->partVec_.push_back(part);
  }

  // effective viscosity alg
  const double turbPr = realm_.get_turb_prandtl(enthalpy_->name());
  std::map<AlgorithmType, Algorithm *>::iterator itev =
    diffFluxCoeffAlgDriver_->algMap_.find(algType);
  if ( itev == diffFluxCoeffAlgDriver_->algMap_.end() ) {
    EnthalpyEffectiveDiffFluxCoeffAlgorithm *theAlg
      = new EnthalpyEffectiveDiffFluxCoeffAlgorithm(realm_, part, thermalCond_, specHeat_, tvisc_, evisc_, turbPr);
    diffFluxCoeffAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itev->second->partVec_.push_back(part);
  }

  // extract material prop evaluation for enthalpy and create alg to compute h
  PropertyEvaluator *thePropEval
    = realm_.get_material_prop_eval(ENTHALPY_ID);

  TemperaturePropAlgorithm *auxAlg
    = new TemperaturePropAlgorithm( realm_, part, &enthalpyNp1, thePropEval);
  enthalpyFromTemperatureAlg_.push_back(auxAlg);

}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType &enthalpyNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dhdxNone = dhdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // extract user data
  InflowUserData userData = inflowBCData.userData_;

  // bc data work (copy, enthalpy evaluation, etc.)
  ScalarFieldType *temperatureBc = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature_bc"));
  stk::mesh::put_field_on_mesh(*temperatureBc, *part, nullptr);
  ScalarFieldType *enthalpyBc = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "enthalpy_bc"));
  stk::mesh::put_field_on_mesh(*enthalpyBc, *part, nullptr);
  temperature_bc_setup(userData, part, temperatureBc, enthalpyBc);

  // non-solver; dhdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &enthalpyNp1, &dhdxNone, edgeNodalGradient_);
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
      = new DirichletBC(realm_, this, part, &enthalpyNp1, enthalpyBc, 0, 1);
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
EnthalpyEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  ScalarFieldType &enthalpyNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dhdxNone = dhdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // extract user data
  OpenUserData userData = openBCData.userData_;

  // check that temperature was specified
  if ( !userData.tempSpec_ )
    throw std::runtime_error("no temperature specified at open");

  // bc data work (copy, enthalpy evaluation, etc.)
  const bool copyBcVal = false;
  const bool isInterface = false;
  ScalarFieldType *temperatureBc = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "open_temperature_bc"));
  stk::mesh::put_field_on_mesh(*temperatureBc, *part, nullptr);
  ScalarFieldType *enthalpyBc = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "open_enthalpy_bc"));
  stk::mesh::put_field_on_mesh(*enthalpyBc, *part, nullptr);
  temperature_bc_setup(userData, part, temperatureBc, enthalpyBc, isInterface, copyBcVal);

  // non-solver; dhdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &enthalpyNp1, &dhdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // solver open; lhs
  if ( realm_.solutionOptions_->useConsolidatedBcSolverAlg_ ) {
    
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    
    stk::topology elemTopo = get_elem_topo(realm_, *part);
    
    AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
    bool solverAlgWasBuilt = false;
    
    std::tie(faceElemSolverAlg, solverAlgWasBuilt) 
      = build_or_add_part_to_face_elem_solver_alg(algType, *this, *part, elemTopo, solverAlgMap, "open");
    
    auto& activeKernels = faceElemSolverAlg->activeKernels_;
    
    if (solverAlgWasBuilt) {
      
      build_face_elem_topo_kernel_automatic<ScalarOpenAdvElemKernel>
        (partTopo, elemTopo, *this, activeKernels, "enthalpy_open",
         realm_.meta_data(), *realm_.solutionOptions_,
         this, enthalpy_, enthalpyBc, dhdx_, evisc_, 
         faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_);
      
    }
  }
  else {    
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theAlg = new AssembleScalarEdgeOpenSolverAlgorithm(realm_, part, this, enthalpy_, enthalpyBc, &dhdxNone, evisc_);
      }
      else {
        theAlg = new AssembleScalarElemOpenSolverAlgorithm(realm_, part, this, enthalpy_, enthalpyBc, &dhdxNone, evisc_);
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
EnthalpyEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  // np1
  ScalarFieldType &enthalpyNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dhdxNone = dhdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // extract user data
  WallUserData userData = wallBCData.userData_;
  std::string temperatureName = "temperature";

  // check to see if this bc is a CHT type
  const bool isInterface = userData.isInterface_;

  // check for wall function; warn user that this is not yet supported
  const bool wallFunctionApproach = userData.wallFunctionApproach_;
  if (wallFunctionApproach)
    NaluEnv::self().naluOutputP0() << "Sorry, wall function not yet supported for energy; will use Dirichlet" << std::endl;

  // check that is was specified (okay if it is not)
  if ( bc_data_specified(userData, temperatureName) ) {

    // bc data work (copy, enthalpy evaluation, etc.)
    ScalarFieldType *temperatureBc = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature_bc"));
    stk::mesh::put_field_on_mesh(*temperatureBc, *part, nullptr);
    ScalarFieldType *enthalpyBc = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "enthalpy_bc"));
    stk::mesh::put_field_on_mesh(*enthalpyBc, *part, nullptr);
    temperature_bc_setup(userData, part, temperatureBc, enthalpyBc, isInterface);

    // Dirichlet bc
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
      DirichletBC *theAlg
        = new DirichletBC(realm_, this, part, &enthalpyNp1, enthalpyBc, 0, 1);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    }
    else {
      itd->second->partVec_.push_back(part);
    }

    // interface bc fields

    // register the fields
    ScalarFieldType *assembledWallArea =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_ht"));
    stk::mesh::put_field_on_mesh(*assembledWallArea, *part, nullptr);
    ScalarFieldType *referenceTemperature =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "reference_temperature"));
    stk::mesh::put_field_on_mesh(*referenceTemperature, *part, nullptr);
    ScalarFieldType *heatTransferCoeff =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "heat_transfer_coefficient"));
    stk::mesh::put_field_on_mesh(*heatTransferCoeff, *part, nullptr);
    ScalarFieldType *normalHeatFlux = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "normal_heat_flux"));
    stk::mesh::put_field_on_mesh(*normalHeatFlux, *part, nullptr);
    ScalarFieldType *robinCouplingParameter = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "robin_coupling_parameter"));
    stk::mesh::put_field_on_mesh(*robinCouplingParameter, *part, nullptr);

    // create the driver
    if ( NULL == assembleWallHeatTransferAlgDriver_ ) {
      assembleWallHeatTransferAlgDriver_ = new AssembleWallHeatTransferAlgorithmDriver(realm_);
    }

    // create the edge or element algorithm for h and Too
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleWallHeatTransferAlgDriver_->algMap_.find(algType);
    if ( it == assembleWallHeatTransferAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theAlg = new ComputeHeatTransferEdgeWallAlgorithm(realm_, part);
      }
      else {
        theAlg = new ComputeHeatTransferElemWallAlgorithm(realm_, part);
      }
      assembleWallHeatTransferAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }

  }
  else if ( userData.heatFluxSpec_ ) {

    ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "heat_flux_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

    NormalHeatFlux heatFlux = userData.q_;
    std::vector<double> userSpec(1);
    userSpec[0] = heatFlux.qn_;

    // new it
    ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

    // bc data alg
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);

    // solver; lhs
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleScalarFluxBCSolverAlgorithm *theAlg
        = new AssembleScalarFluxBCSolverAlgorithm(realm_, part, this,
                                                  theBcField, realm_.realmUsesEdges_);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
  }

  // non-solver; dhdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &enthalpyNp1, &dhdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
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
EnthalpyEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData &symmetryBCData)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  // np1
  ScalarFieldType &enthalpyNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dhdxNone = dhdx_->field_of_state(stk::mesh::StateNone);

  // extract user data
  SymmetryUserData userData = symmetryBCData.userData_;
  std::string temperatureName = "temperature";
  
  // non-solver; dhdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &enthalpyNp1, &dhdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
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
EnthalpyEquationSystem::register_abltop_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const ABLTopBoundaryConditionData &abltopBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  // np1
  ScalarFieldType &enthalpyNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dhdxNone = dhdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // extract user data
  ABLTopUserData userData = abltopBCData.userData_;
  std::string temperatureName = "temperature";
  

  // If specifying the normal temperature gradient.
  if ( userData.normalTemperatureGradientSpec_ ) {  
    ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature_gradient_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

    // Get the specified normal temperature gradient
    NormalTemperatureGradient tempGrad = userData.normalTemperatureGradient_;
    std::vector<double> userSpec(1);
    userSpec[0] = tempGrad.tempGradN_;

    // Use the constant auxiliary function
    ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

    // bc data alg to populate the bc field with constant data.
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);

    // solver; lhs
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleTemperatureNormalGradientBCSolverAlgorithm *theAlg
        = new AssembleTemperatureNormalGradientBCSolverAlgorithm(realm_, part, this,
                                                  theBcField, evisc_, specHeat_, realm_.realmUsesEdges_);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
  }

  // non-solver; dhdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &enthalpyNp1, &dhdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{

  const AlgorithmType algType = NON_CONFORMAL;

  // np1
  ScalarFieldType &hNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dhdxNone = dhdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dhdx; DG algorithm decides on locations for integration points
  if ( !managePNG_ ) {
    if ( edgeNodalGradient_ ) {    
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        Algorithm *theAlg 
          = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &hNp1, &dhdxNone, edgeNodalGradient_);
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
        AssembleNodalGradNonConformalAlgorithm *theAlg 
          = new AssembleNodalGradNonConformalAlgorithm(realm_, part, &hNp1, &dhdxNone);
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
    AssembleScalarNonConformalSolverAlgorithm *theAlg
      = new AssembleScalarNonConformalSolverAlgorithm(realm_, part, this, enthalpy_, evisc_);
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
EnthalpyEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(enthalpy_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(enthalpy_,1,1)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::reinitialize_linear_system()
{
  // delete old solver
  const EquationType theEqID = EQ_ENTHALPY;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // delete linsys
  delete linsys_;

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("enthalpy");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_ENTHALPY);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::register_initial_condition_fcn(
  stk::mesh::Part *part,
  const std::map<std::string, std::string> &theNames,
  const std::map<std::string, std::vector<double> > &/*theParams*/)
{
  // iterate map and check for name
  const std::string dofName = "temperature";
  std::map<std::string, std::string>::const_iterator iterName
    = theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;
    AuxFunction *theAuxFunc = NULL;
    if ( fcnName == "VariableDensityNonIso" ) {
      theAuxFunc = new VariableDensityNonIsoTemperatureAuxFunction();      
    }
    else if ( fcnName == "BoussinesqNonIso" ) {
      theAuxFunc = new BoussinesqNonIsoTemperatureAuxFunction();
    }
    else if ( fcnName == "capping_inversion" ) {
      theAuxFunc = new CappingInversionTemperatureAuxFunction();
    }
    else {
      throw std::runtime_error("EnthalpyEquationSystem::register_initial_condition_fcn: limited user functions supported");
    }
    
    // create the algorithm
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
				 temperature_, theAuxFunc,
				 stk::topology::NODE_RANK);
    
    // push to ic
    realm_.initCondAlg_.push_back(auxAlg);
  }
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::solve_and_update()
{
  // compute bc enthalpy
  for ( size_t k = 0; k < bcEnthalpyFromTemperatureAlg_.size(); ++k )
    bcEnthalpyFromTemperatureAlg_[k]->execute();

  // copy enthalpy_bc to enthalpyNp1
  for ( size_t k = 0; k < bcCopyStateAlg_.size(); ++k )
    bcCopyStateAlg_[k]->execute();

  // compute dh/dx
  if ( isInit_ ) {
    compute_projected_nodal_gradient();
    isInit_ = false;
  }

  // compute effective viscosity
  diffFluxCoeffAlgDriver_->execute();

  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    // enthalpy assemble, load_complete and solve
    assemble_and_solve(hTmp_);

    // update
    double timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(),
      realm_.bulk_data(),
      1.0, *hTmp_,
      1.0, enthalpy_->field_of_state(stk::mesh::StateNP1),
      realm_.get_activate_aura());
    double timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB-timeA);

    // projected nodal gradient
    compute_projected_nodal_gradient();
  }

  // delay extract temperature and h and Too to the end of the iteration over all equations
}

//--------------------------------------------------------------------------
//-------- post_iter_work --------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::post_iter_work_dep()
{

  // compute bc enthalpy based on converged species
  for ( size_t k = 0; k < bcEnthalpyFromTemperatureAlg_.size(); ++k )
    bcEnthalpyFromTemperatureAlg_[k]->execute();

  // copy enthalpy_bc to enthalpyNp1
  for ( size_t k = 0; k < bcCopyStateAlg_.size(); ++k )
    bcCopyStateAlg_[k]->execute();

  // extract temperature now
  extract_temperature();

  // post process h and Too
  if ( NULL != assembleWallHeatTransferAlgDriver_ )
    assembleWallHeatTransferAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- post_adapt_work -------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::post_adapt_work()
{
  if ( realm_.process_adaptivity() ) {
    NaluEnv::self().naluOutputP0() << "--EnthalpyEquationSystem::post_adapt_work()" << std::endl;
    extract_temperature();
  }
}

//--------------------------------------------------------------------------
//-------- extract_temperature ---------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::extract_temperature()
{

  // define some high level quantities
  const int maxIter = 25;
  const double relax = 1.0;
  const double om_relax = 1.0-relax;

  const double tolerance = 1.0e-8;
  double workTemperature[1] = {};

  // quality metrics
  size_t troubleCount[3] = {};

  // extract h evaluator
  PropertyEvaluator *enthEval = NULL;
  std::map<PropertyIdentifier, PropertyEvaluator*>::iterator ith =
    realm_.materialPropertys_.propertyEvalMap_.find(ENTHALPY_ID);
  if ( ith == realm_.materialPropertys_.propertyEvalMap_.end() ) {
    throw std::runtime_error("Enthalpy prop evaluator not found:");
  }
  else {
    enthEval = (*ith).second;
  }

  // extract Cp evaluator
  PropertyEvaluator *cpEval = NULL;
  std::map<PropertyIdentifier, PropertyEvaluator*>::iterator itc =
    realm_.materialPropertys_.propertyEvalMap_.find(SPEC_HEAT_ID);
  if ( itc == realm_.materialPropertys_.propertyEvalMap_.end() ) {
    throw std::runtime_error("Specific heat prop evaluator not found:");
  }
  else {
    cpEval = (*itc).second;
  }

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // np1 state
  ScalarFieldType &enthalpyNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);

  // select all nodes (locally and shared) where enthalpy is defined
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*enthalpy_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // extract the field pointers
    double * temperature = stk::mesh::field_data(*temperature_, b);
    double * enthalpy = stk::mesh::field_data(enthalpyNp1, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // extract the node
      stk::mesh::Entity node = b[k];

      // save the temperature
      const double Tsave = temperature[k];

      // extract the current enthalpy
      const double hNp1 = enthalpy[k];

      // set flag for convergence and possible trouble
      bool convergedT = false;
      bool trouble = false;

      // make an initial guess; current is as good as anything
      double TNp1 = Tsave;
      for ( int j = 0; j < maxIter; ++j ) {

        // populate work temperature array
        workTemperature[0] = TNp1;

        // extract enthalpy and Cp based on guessed temperature
        const double enthalpyWork = enthEval->execute(workTemperature, node);
        const double cpWork = cpEval->execute(workTemperature, node);

        // evaluate diffs
        const double hDiff = hNp1 - enthalpyWork;
        const double tDiff = hDiff/cpWork;
        TNp1 += tDiff;

        // check for convergence
        if ( std::abs(tDiff) < TNp1*tolerance ) {
          convergedT = true;
          break;
        }
      }

      // check for trouble
      if ( !convergedT ) {
        troubleCount[0]++;
        trouble = true;
      }

      // now check for monotonicy issues; too low; too high
      if ( TNp1 < minimumT_ ) {
        TNp1 = minimumT_;
        trouble = true;
        troubleCount[1]++;
      }

      if ( TNp1 > maximumT_ ) {
        TNp1 = maximumT_;
        trouble = true;
        troubleCount[2]++;
      }

      // if trouble, relax temnperature; reset both T and h
      if ( trouble ) {
        const double troubleT = TNp1*relax + om_relax*Tsave;
        workTemperature[0] = troubleT;
        const double enthalpyClip = enthEval->execute(workTemperature, node);
        enthalpy[k] = enthalpyClip;
        temperature[k] = troubleT;
      }
      else {
        temperature[k] = TNp1;
      }
    }
  }

  // parallel assemble not converged
  if ( outputClippingDiag_ ) {
    size_t g_troubleCount[3] = {};
    stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
    stk::all_reduce_sum(comm, &troubleCount[0], &g_troubleCount[0], 3);
    
    if ( g_troubleCount[0] > 0 ) {
      NaluEnv::self().naluOutputP0() << "Temperature extraction failed to converge " << g_troubleCount[0] << " times"
                                     << std::endl;
    }
    
    if ( g_troubleCount[1] > 0 ) {
      NaluEnv::self().naluOutputP0() << "Temperature clipped to min " << g_troubleCount[1] << " times"
                                     << std::endl;
    }
    
    if ( g_troubleCount[2] > 0 ) {
      NaluEnv::self().naluOutputP0() << "Temperature clipped to max " << g_troubleCount[2] << " times"
                                     << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- post_converged_work ---------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::post_converged_work()
{
  if ( lowSpeedCompressActive_ ) {
    stk::mesh::MetaData & meta_data = realm_.meta_data();
    // copy pressure to pOld
    ScalarFieldType *pressure = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
    field_copy(meta_data, realm_.bulk_data(), *pressure, *pOld_, realm_.get_activate_aura());
  } 
}

//--------------------------------------------------------------------------
//-------- predict_state ---------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::predict_state()
{
  // copy state n to state np1
  ScalarFieldType &hN = enthalpy_->field_of_state(stk::mesh::StateN);
  ScalarFieldType &hNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), hN, hNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- temperature_bc_setup --------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::temperature_bc_setup(
  UserData userData,
  stk::mesh::Part *part,
  ScalarFieldType *temperatureBc,
  ScalarFieldType *enthalpyBc,
  const bool isInterface,
  const bool copyBCVal )
{
  ScalarFieldType &enthalpyNp1 = enthalpy_->field_of_state(stk::mesh::StateNP1);

  // extract the type
  std::string temperatureName = "temperature";
  UserDataType theDataType = get_bc_data_type(userData, temperatureName);

  // extract temperature as possibly external; similar to interface
  const bool externalData = userData.externalData_;

  // populate temperature_bc
  AuxFunction *theAuxFunc = NULL;
  if ( CONSTANT_UD == theDataType ) {
    Temperature theTemp = userData.temperature_;
    std::vector<double> userSpec(1);
    userSpec[0] = theTemp.temperature_;
    theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
  }
  else if ( FUNCTION_UD == theDataType ) {
    // extract the name
    std::string fcnName = get_bc_function_name(userData, temperatureName);
    // switch on the name found...
    if ( fcnName == "flow_past_cylinder" ) {
      theAuxFunc = new FlowPastCylinderTempAuxFunction();
    }
    else if ( fcnName == "VariableDensityNonIso" ) {
      theAuxFunc = new VariableDensityNonIsoTemperatureAuxFunction();
    }
    else if ( fcnName == "BoussinesqNonIso" ) {
      theAuxFunc = new BoussinesqNonIsoTemperatureAuxFunction();
    }
    else if ( fcnName == "capping_inversion" ) {
      theAuxFunc = new CappingInversionTemperatureAuxFunction();
    }
    else {
      throw std::runtime_error("EnthalpyEquationSystem::temperature_bc_setup; limited user functions supported");
    }
  } 
  else {
    throw std::runtime_error("EnthalpyEquationSystem::temperature_bc_setup: only function and constants supported (and none specified)");   
  }
  
  AuxFunctionAlgorithm *auxTempAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               temperatureBc, theAuxFunc,
                               stk::topology::NODE_RANK);

  // copy bc value to temperature
  CopyFieldAlgorithm *theTempCopyAlg = NULL;
  if ( copyBCVal ) {
    theTempCopyAlg = new CopyFieldAlgorithm(realm_, part,
					    temperatureBc, temperature_,
					    0, 1,
					    stk::topology::NODE_RANK);
  }

  // if this is an interface bc, then push algorithm to initial condition
  if ( isInterface || externalData ) {
    // xfer will handle population; only need to populate the initial value
    realm_.initCondAlg_.push_back(auxTempAlg);
    if ( copyBCVal )
      realm_.initCondAlg_.push_back(theTempCopyAlg);
  }
  else {
    // will be processed as part of every pre-time step work
    bcDataAlg_.push_back(auxTempAlg);
    if ( copyBCVal )
      bcDataMapAlg_.push_back(theTempCopyAlg);
  }

  // extract material prop evaluation for enthalpy and create alg to compute h_bc
  PropertyEvaluator *thePropEval
    = realm_.get_material_prop_eval(ENTHALPY_ID);

  TemperaturePropAlgorithm *enthAlg
    = new TemperaturePropAlgorithm( realm_, part, enthalpyBc, thePropEval, temperatureBc->name());

  // copy enthalpy_bc to enthalpy np1...
  CopyFieldAlgorithm *theEnthCopyAlg = NULL;
  if ( copyBCVal ) {
    theEnthCopyAlg = new CopyFieldAlgorithm(realm_, part,
					    enthalpyBc, &enthalpyNp1,
					    0, 1,
					    stk::topology::NODE_RANK);
  }

  // enthalpy always manages bc enthalpy population
  bcEnthalpyFromTemperatureAlg_.push_back(enthAlg);
  // only copy enthalpy_bc to enthalpy primitive when required
  if ( copyBCVal ) 
    bcCopyStateAlg_.push_back(theEnthCopyAlg);
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if ( NULL == projectedNodalGradEqs_ ) {
    projectedNodalGradEqs_ 
      = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG_H, "dhdx", "qTmp", "enthalpy", "PNGradHEQS");
  }
  // fill the map for expected boundary condition names; can be more complex...
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "enthalpy");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "enthalpy");
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "enthalpy");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "enthalpy");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
EnthalpyEquationSystem::compute_projected_nodal_gradient()
{
  if ( !managePNG_ ) {
    const double timeA = -NaluEnv::self().nalu_time();
    assembleNodalGradAlgDriver_->execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  }
  else {
    projectedNodalGradEqs_->solve_and_update_external();
  }
}

} // namespace nalu
} // namespace Sierra
