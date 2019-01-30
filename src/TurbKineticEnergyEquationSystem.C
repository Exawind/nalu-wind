/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <TurbKineticEnergyEquationSystem.h>
#include <AlgorithmDriver.h>
#include <AssembleScalarEdgeOpenSolverAlgorithm.h>
#include <AssembleScalarEdgeSolverAlgorithm.h>
#include <AssembleScalarElemSolverAlgorithm.h>
#include <AssembleScalarElemOpenSolverAlgorithm.h>
#include <AssembleScalarNonConformalSolverAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
#include <AssembleNodalGradAlgorithmDriver.h>
#include <AssembleNodalGradEdgeAlgorithm.h>
#include <AssembleNodalGradElemAlgorithm.h>
#include <AssembleNodalGradBoundaryAlgorithm.h>
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <AuxFunctionAlgorithm.h>
#include <ComputeTurbKineticEnergyWallFunctionAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
#include <EffectiveDiffFluxCoeffAlgorithm.h>
#include <EffectiveSSTDiffFluxCoeffAlgorithm.h>
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
#include <Simulation.h>
#include <SolutionOptions.h>
#include <TimeIntegrator.h>
#include <TurbKineticEnergyKsgsNodeSourceSuppAlg.h>
#include <TurbKineticEnergySSTNodeSourceSuppAlg.h>
#include <TurbKineticEnergySSTDESNodeSourceSuppAlg.h>
#include <TurbKineticEnergyKsgsBuoyantElemSuppAlg.h>
#include <TurbKineticEnergyRodiNodeSourceSuppAlg.h>

#include <SolverAlgorithmDriver.h>

// template for kernels
#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

// kernels
#include <AssembleElemSolverAlgorithm.h>
#include <kernel/ScalarMassElemKernel.h>
#include <kernel/ScalarAdvDiffElemKernel.h>
#include <kernel/ScalarUpwAdvDiffElemKernel.h>
#include <kernel/TurbKineticEnergyKsgsSrcElemKernel.h>
#include <kernel/TurbKineticEnergyKsgsDesignOrderSrcElemKernel.h>
#include <kernel/TurbKineticEnergySSTSrcElemKernel.h>
#include <kernel/TurbKineticEnergySSTDESSrcElemKernel.h>

// bc kernels
#include <kernel/ScalarOpenAdvElemKernel.h>

// nso
#include <nso/ScalarNSOElemKernel.h>
#include <nso/ScalarNSOKeElemSuppAlg.h>

// deprecated
#include <ScalarMassElemSuppAlgDep.h>
#include <nso/ScalarNSOKeElemSuppAlg.h>
#include <nso/ScalarNSOElemSuppAlgDep.h>

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
// TurbKineticEnergyEquationSystem - manages tke pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbKineticEnergyEquationSystem::TurbKineticEnergyEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "TurbKineticEnergyEQS","turbulent_ke"),
    managePNG_(realm_.get_consistent_mass_matrix_png("turbulent_ke")),
    tke_(NULL),
    dkdx_(NULL),
    kTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(realm_, "turbulent_ke", "dkdx")),
    diffFluxCoeffAlgDriver_(new AlgorithmDriver(realm_)),
    wallFunctionTurbKineticEnergyAlgDriver_(NULL),
    turbulenceModel_(realm_.solutionOptions_->turbulenceModel_),
    projectedNodalGradEqs_(NULL),
    isInit_(true)
{
  dofName_ = "turbulent_ke";

  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("turbulent_ke");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_TURBULENT_KE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("turbulent_ke");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for turbulent_ke: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // sanity check on turbulence model
  if ( (turbulenceModel_ != SST) && (turbulenceModel_ != KSGS) && (turbulenceModel_ != SST_DES) ) {
    throw std::runtime_error("User has requested TurbKinEnergyEqs, however, turbulence model is not KSGS, SST or SST_DES");
  }

  // create projected nodal gradient equation system
  if ( managePNG_ ) {
    manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
TurbKineticEnergyEquationSystem::~TurbKineticEnergyEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
  delete diffFluxCoeffAlgDriver_;
  if ( NULL!= wallFunctionTurbKineticEnergyAlgDriver_)
    delete wallFunctionTurbKineticEnergyAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  tke_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke", numStates));
  stk::mesh::put_field_on_mesh(*tke_, *part, nullptr);
  realm_.augment_restart_variable_list("turbulent_ke");

  dkdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dkdx"));
  stk::mesh::put_field_on_mesh(*dkdx_, *part, nDim, nullptr);

  // delta solution for linear solver; share delta since this is a split system
  kTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pTmp"));
  stk::mesh::put_field_on_mesh(*kTmp_, *part, nullptr);

  visc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  tvisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity"));
  stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);

  evisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "effective_viscosity_tke"));
  stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &tkeN = tke_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &tkeNp1, &tkeN,
                               0, 1,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  // non-solver, dkdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( edgeNodalGradient_ && realm_.realmUsesEdges_ ) {
        theAlg = new AssembleNodalGradEdgeAlgorithm(realm_, part, &tkeNp1, &dkdxNone);
      }
      else {
        theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, &tkeNp1, &dkdxNone, edgeNodalGradient_);
      }
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // solver; interior contribution (advection + diffusion)
  if ( !realm_.solutionOptions_->useConsolidatedSolverAlg_ ) {
    
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theAlg = new AssembleScalarEdgeSolverAlgorithm(realm_, part, this, tke_, dkdx_, evisc_);
      }
      else {
        theAlg = new AssembleScalarElemSolverAlgorithm(realm_, part, this, tke_, dkdx_, evisc_);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
      
      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string> >::iterator isrc 
        = realm_.solutionOptions_->elemSrcTermsMap_.find("turbulent_ke");
      if ( isrc != realm_.solutionOptions_->elemSrcTermsMap_.end() ) {
        
        if ( realm_.realmUsesEdges_ )
          throw std::runtime_error("TurbKineticEnergyElemSrcTerms::Error can not use element source terms for an edge-based scheme");
        
        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if (sourceName == "ksgs_buoyant" ) {
            if (turbulenceModel_ != KSGS)
              throw std::runtime_error("ElemSrcTermsError::TurbKineticEnergyKsgsBuoyantElemSuppAlg requires Ksgs model");
            suppAlg = new TurbKineticEnergyKsgsBuoyantElemSuppAlg(realm_);
          }
          else if (sourceName == "NSO_2ND_ALT" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, tke_, dkdx_, evisc_, 0.0, 1.0);
          }
          else if (sourceName == "NSO_4TH_ALT" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, tke_, dkdx_, evisc_, 1.0, 1.0);
          }
          else if (sourceName == "NSO_2ND_KE" ) {
            const double turbSc = realm_.get_turb_schmidt(tke_->name());
            suppAlg = new ScalarNSOKeElemSuppAlg(realm_, tke_, dkdx_, turbSc, 0.0);
          }
          else if (sourceName == "NSO_4TH_KE" ) {
            const double turbSc = realm_.get_turb_schmidt(tke_->name());
            suppAlg = new ScalarNSOKeElemSuppAlg(realm_, tke_, dkdx_, turbSc, 1.0);
          }
          else if (sourceName == "turbulent_ke_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, tke_, false);
          }
          else if (sourceName == "lumped_turbulent_ke_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, tke_, true);
          }
          else {
            throw std::runtime_error("TurbKineticEnergyElemSrcTerms::Error Source term is not supported: " + sourceName);
          }     
          NaluEnv::self().naluOutputP0() << "TurbKineticEnergyElemSrcTerms::added() " << sourceName << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg); 
        }
      }
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
    
    // time term; (Pk-Dk); both nodally lumped
    const AlgorithmType algMass = MASS;
    // Check if the user has requested CMM or LMM algorithms; if so, do not
    // include Nodal Mass algorithms
    std::vector<std::string> checkAlgNames = {"turbulent_ke_time_derivative",
                                              "lumped_turbulent_ke_time_derivative"};
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
            = new ScalarMassBackwardEulerNodeSuppAlg(realm_, tke_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
        else {
          ScalarMassBDF2NodeSuppAlg *theMass
            = new ScalarMassBDF2NodeSuppAlg(realm_, tke_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
      }
      
      // now create the src alg for tke source
      SupplementalAlgorithm *theSrc = NULL;
      switch(turbulenceModel_) {
      case KSGS:
        {
          theSrc = new TurbKineticEnergyKsgsNodeSourceSuppAlg(realm_);
        }
        break;
      case SST:
        {
          theSrc = new TurbKineticEnergySSTNodeSourceSuppAlg(realm_);
        }
        break;
      case SST_DES:
        {
          theSrc = new TurbKineticEnergySSTDESNodeSourceSuppAlg(realm_);
        }
        break;
      default:
        throw std::runtime_error("Unsupported turbulence model in TurbKe: only SST, SST_DES and Ksgs supported");
      }
      theAlg->supplementalAlg_.push_back(theSrc);
      
      // Add nodal src term supp alg...; limited number supported
      std::map<std::string, std::vector<std::string> >::iterator isrc 
        = realm_.solutionOptions_->srcTermsMap_.find("turbulent_ke");
      if ( isrc != realm_.solutionOptions_->srcTermsMap_.end() ) {
        std::vector<std::string> mapNameVec = isrc->second;   
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if ( sourceName == "gcl" ) {
            suppAlg = new ScalarGclNodeSuppAlg(tke_,realm_);
          }
          if ( sourceName == "rodi" ) {
            suppAlg = new TurbKineticEnergyRodiNodeSourceSuppAlg(realm_);
          }
          else {
            throw std::runtime_error("TurbKineticEnergyNodalSrcTerms::Error Source term is not supported: " + sourceName);
          }
          NaluEnv::self().naluOutputP0() << "TurbKineticEnergyNodalSrcTerms::added() " << sourceName << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg);
        }
      }
    }
    else {
      itsm->second->partVec_.push_back(part);
    }
  }
  else {
    // Homogeneous kernel implementation
    if ( realm_.realmUsesEdges_ )
      throw std::runtime_error("TurbKineticEnergyEquationSystem::Error can not use element source terms for an edge-based scheme");
    
    stk::topology partTopo = part->topology();
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    
    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;
    
    std::tie(solverAlg, solverAlgWasBuilt) = build_or_add_part_to_solver_alg
      (*this, *part, solverAlgMap);
    
    ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
    auto& activeKernels = solverAlg->activeKernels_;

    if (solverAlgWasBuilt) {
      build_topo_kernel_if_requested<ScalarMassElemKernel>
        (partTopo, *this, activeKernels, "turbulent_ke_time_derivative",
         realm_.bulk_data(), *realm_.solutionOptions_, tke_, dataPreReqs, false);
      
      build_topo_kernel_if_requested<ScalarMassElemKernel>
        (partTopo, *this, activeKernels, "lumped_turbulent_ke_time_derivative",
         realm_.bulk_data(), *realm_.solutionOptions_, tke_, dataPreReqs, true);
      
      build_topo_kernel_if_requested<ScalarAdvDiffElemKernel>
        (partTopo, *this, activeKernels, "advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, tke_, evisc_, dataPreReqs);
      
      build_topo_kernel_if_requested<ScalarUpwAdvDiffElemKernel>
        (partTopo, *this, activeKernels, "upw_advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, this, tke_, dkdx_, evisc_, dataPreReqs);

      build_topo_kernel_if_requested<TurbKineticEnergyKsgsSrcElemKernel>
        (partTopo, *this, activeKernels, "ksgs",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);

      build_topo_kernel_if_requested<TurbKineticEnergyKsgsDesignOrderSrcElemKernel>
        (partTopo, *this, activeKernels, "design_order_ksgs",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);

      build_topo_kernel_if_requested<TurbKineticEnergySSTSrcElemKernel>
        (partTopo, *this, activeKernels, "sst",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);

      build_topo_kernel_if_requested<TurbKineticEnergySSTSrcElemKernel>
        (partTopo, *this, activeKernels, "lumped_sst",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);

      build_topo_kernel_if_requested<TurbKineticEnergySSTDESSrcElemKernel>
        (partTopo, *this, activeKernels, "sst_des",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);

      build_topo_kernel_if_requested<TurbKineticEnergySSTDESSrcElemKernel>
        (partTopo, *this, activeKernels, "lumped_sst_des",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);

      build_topo_kernel_if_requested<ScalarNSOElemKernel>
        (partTopo, *this, activeKernels, "NSO_2ND",
         realm_.bulk_data(), *realm_.solutionOptions_, tke_, dkdx_, evisc_, 0.0, 0.0, dataPreReqs);
      
      build_topo_kernel_if_requested<ScalarNSOElemKernel>
        (partTopo, *this, activeKernels, "NSO_2ND_ALT",
         realm_.bulk_data(), *realm_.solutionOptions_, tke_, dkdx_, evisc_, 0.0, 1.0, dataPreReqs);
      
      build_topo_kernel_if_requested<ScalarNSOElemKernel>
        (partTopo, *this, activeKernels, "NSO_4TH",
         realm_.bulk_data(), *realm_.solutionOptions_, tke_, dkdx_, evisc_, 1.0, 0.0, dataPreReqs);
      
      build_topo_kernel_if_requested<ScalarNSOElemKernel>
        (partTopo, *this, activeKernels, "NSO_4TH_ALT",
         realm_.bulk_data(), *realm_.solutionOptions_, tke_, dkdx_, evisc_, 1.0, 1.0, dataPreReqs);

      report_invalid_supp_alg_names();
      report_built_supp_alg_names();
    }
  }

  // effective viscosity alg
  std::map<AlgorithmType, Algorithm *>::iterator itev =
    diffFluxCoeffAlgDriver_->algMap_.find(algType);
  if ( itev == diffFluxCoeffAlgDriver_->algMap_.end() ) {
    Algorithm *effDiffAlg = NULL;
    switch(turbulenceModel_) {
      case KSGS:
      {
        const double lamSc = realm_.get_lam_schmidt(tke_->name());
        const double turbSc = realm_.get_turb_schmidt(tke_->name());
        effDiffAlg = new EffectiveDiffFluxCoeffAlgorithm(realm_, part, visc_, tvisc_, evisc_, lamSc, turbSc);
      }
      break;
      case SST: case SST_DES:
      {
        const double sigmaKOne = realm_.get_turb_model_constant(TM_sigmaKOne);
        const double sigmaKTwo = realm_.get_turb_model_constant(TM_sigmaKTwo);
        effDiffAlg = new EffectiveSSTDiffFluxCoeffAlgorithm(realm_, part, visc_, tvisc_, evisc_, sigmaKOne, sigmaKTwo);
      }
      break;
      default:
        throw std::runtime_error("Unsupported turbulence model in TurbKe: only SST, SST_DES and Ksgs supported");
    }
    diffFluxCoeffAlgDriver_->algMap_[algType] = effDiffAlg;
  }
  else {
    itev->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; tke_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "tke_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  TurbKinEnergy tke = userData.tke_;
  std::vector<double> userSpec(1);
  userSpec[0] = tke.turbKinEnergy_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

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

  // copy tke_bc to turbulent_ke np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, &tkeNp1,
                             0, 1,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dkdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &tkeNp1, &dkdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd
    = solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg
      = new DirichletBC(realm_, this, part, &tkeNp1, theBcField, 0, 1);
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
TurbKineticEnergyEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; tke_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "open_tke_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  TurbKinEnergy tke = userData.tke_;
  std::vector<double> userSpec(1);
  userSpec[0] = tke.turbKinEnergy_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // non-solver; dkdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &tkeNp1, &dkdxNone, edgeNodalGradient_);
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
        (partTopo, elemTopo, *this, activeKernels, "turbulent_ke_open",
         realm_.meta_data(), *realm_.solutionOptions_,
         this, tke_, theBcField, dkdx_, evisc_, 
         faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_);
      
    }
  }
  else {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theAlg = new AssembleScalarEdgeOpenSolverAlgorithm(realm_, part, this, tke_, theBcField, &dkdxNone, evisc_);
      }
      else {
        theAlg = new AssembleScalarElemOpenSolverAlgorithm(realm_, part, this, tke_, theBcField, &dkdxNone, evisc_);
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
TurbKineticEnergyEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  // np1
  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; tke_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "tke_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  std::string tkeName = "turbulent_ke";
  const bool tkeSpecified = bc_data_specified(userData, tkeName);
  bool wallFunctionApproach = userData.wallFunctionApproach_;
  if ( tkeSpecified && wallFunctionApproach ) {
    NaluEnv::self().naluOutputP0() << "Both wall function and tke specified; will go with dirichlet" << std::endl;
    wallFunctionApproach = false;
  }

  if ( wallFunctionApproach ) {

    // create wallFunctionParamsAlgDriver
    if ( NULL == wallFunctionTurbKineticEnergyAlgDriver_)
      wallFunctionTurbKineticEnergyAlgDriver_ = new AlgorithmDriver(realm_);

    // need to register the assembles wall value for tke; can not share with tke_bc
    ScalarFieldType *theAssembledField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "wall_model_tke_bc"));
    stk::mesh::put_field_on_mesh(*theAssembledField, *part, nullptr);

    // wall function value will prevail at bc intersections
    std::map<AlgorithmType, Algorithm *>::iterator it_tke =
        wallFunctionTurbKineticEnergyAlgDriver_->algMap_.find(algType);
    if ( it_tke == wallFunctionTurbKineticEnergyAlgDriver_->algMap_.end() ) {
      ComputeTurbKineticEnergyWallFunctionAlgorithm *theUtauAlg =
          new ComputeTurbKineticEnergyWallFunctionAlgorithm(realm_, part);
      wallFunctionTurbKineticEnergyAlgDriver_->algMap_[algType] = theUtauAlg;
    }
    else {
      it_tke->second->partVec_.push_back(part);
    }
  }
  else if ( tkeSpecified ) {

    // FIXME: Generalize for constant vs function

    // extract data
    std::vector<double> userSpec(1);
    TurbKinEnergy tke = userData.tke_;
    userSpec[0] = tke.turbKinEnergy_;

    // new it
    ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

    // bc data alg
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);

    // copy tke_bc to tke np1...
    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               theBcField, &tkeNp1,
                               0, 1,
                               stk::topology::NODE_RANK);
    bcDataMapAlg_.push_back(theCopyAlg);

  }
  else {
    throw std::runtime_error("TKE active with wall bc, however, no value of tke or wall function specified");
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg =
        new DirichletBC(realm_, this, part, &tkeNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  }
  else {
    itd->second->partVec_.push_back(part);
  }

  // non-solver; dkdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &tkeNp1, &dkdxNone, edgeNodalGradient_);
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
TurbKineticEnergyEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData & /* symmetryBCData */)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  // np1
  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dkdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &tkeNp1, &dkdxNone, edgeNodalGradient_);
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
TurbKineticEnergyEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{

  const AlgorithmType algType = NON_CONFORMAL;

  // np1
  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dkdx; DG algorithm decides on locations for integration points
  if ( !managePNG_ ) {
    if ( edgeNodalGradient_ ) {    
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        Algorithm *theAlg 
          = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &tkeNp1, &dkdxNone, edgeNodalGradient_);
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
          = new AssembleNodalGradNonConformalAlgorithm(realm_, part, &tkeNp1, &dkdxNone);
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
      = new AssembleScalarNonConformalSolverAlgorithm(realm_, part, this, tke_, evisc_);
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
TurbKineticEnergyEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(tke_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(tke_,1,1)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_TURBULENT_KE;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("turbulent_ke");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_TURBULENT_KE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::solve_and_update()
{

  // sometimes, a higher level equation system manages the solve and update
  if ( turbulenceModel_ != KSGS)
    return;

  // compute dk/dx
  if ( isInit_ ) {
    compute_projected_nodal_gradient();
    isInit_ = false;
  }

  // compute effective viscosity
  compute_effective_diff_flux_coeff();

  // deal with any special wall function approach
  compute_wall_model_parameters();

  // start the iteration loop
  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    // tke assemble, load_complete and solve
    assemble_and_solve(kTmp_);

    // update
    double timeA = NaluEnv::self().nalu_time();
    update_and_clip();
    double timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB-timeA);

    // projected nodal gradient
    compute_projected_nodal_gradient();
  }

}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::initial_work()
{
  // do not let the user specify a negative field
  const double clipValue = 1.0e-16;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*tke_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    double *tke = stk::mesh::field_data(*tke_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const double tkeNp1 = tke[k];
      if ( tkeNp1 < 0.0 ) {
        tke[k] = clipValue;
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- compute_effective_flux_coeff() ----------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::compute_effective_diff_flux_coeff()
{
  const double timeA = NaluEnv::self().nalu_time();
  diffFluxCoeffAlgDriver_->execute();
  timerMisc_ += (NaluEnv::self().nalu_time() - timeA);
}

//--------------------------------------------------------------------------
//-------- compute_wall_model_parameters() ----------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::compute_wall_model_parameters()
{
  if ( NULL != wallFunctionTurbKineticEnergyAlgDriver_)
    wallFunctionTurbKineticEnergyAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- update_and_clip() -----------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::update_and_clip()
{
  const double clipValue = 1.0e-16;
  size_t numClip = 0;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*tke_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    double *tke = stk::mesh::field_data(*tke_, b);
    double *kTmp = stk::mesh::field_data(*kTmp_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const double tkeNp1 = tke[k] + kTmp[k];
      if ( tkeNp1 < 0.0 ) {
        tke[k] = clipValue;
        numClip++;
      }
      else {
        tke[k] = tkeNp1;
      }
    }
  }

  // parallel assemble clipped value
  if (realm_.debug()) {
    size_t g_numClip = 0;
    stk::ParallelMachine comm =  NaluEnv::self().parallel_comm();
    stk::all_reduce_sum(comm, &numClip, &g_numClip, 1);

    if ( g_numClip > 0 ) {
      NaluEnv::self().naluOutputP0() << "tke clipped " << g_numClip << " times " << std::endl;
    }

  }
}

void
TurbKineticEnergyEquationSystem::predict_state()
{
  // copy state n to state np1
  ScalarFieldType &tkeN = tke_->field_of_state(stk::mesh::StateN);
  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), tkeN, tkeNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if ( NULL == projectedNodalGradEqs_ ) {
    projectedNodalGradEqs_ 
      = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG_TKE, "dkdx", "qTmp", "turbulent_ke", "PNGradTkeEQS");
  }
  // fill the map for expected boundary condition names; can be more complex...
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "turbulent_ke");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "turbulent_ke"); // wall function...
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "turbulent_ke");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "turbulent_ke");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient() ---------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::compute_projected_nodal_gradient()
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
