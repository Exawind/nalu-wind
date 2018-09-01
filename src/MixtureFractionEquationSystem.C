/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <MixtureFractionEquationSystem.h>
#include <AlgorithmDriver.h>
#include <AssembleScalarEdgeOpenSolverAlgorithm.h>
#include <AssembleScalarEdgeSolverAlgorithm.h>
#include <AssembleScalarElemSolverAlgorithm.h>
#include <AssembleScalarElemOpenSolverAlgorithm.h>
#include <AssembleScalarNonConformalSolverAlgorithm.h>
#include <AssembleNodalGradAlgorithmDriver.h>
#include <AssembleNodalGradEdgeAlgorithm.h>
#include <AssembleNodalGradElemAlgorithm.h>
#include <AssembleNodalGradBoundaryAlgorithm.h>
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
#include <AuxFunctionAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
#include <EffectiveDiffFluxCoeffAlgorithm.h>
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

#include <kernel/ScalarMassHOElemKernel.h>
#include <kernel/ScalarAdvDiffHOElemKernel.h>

// bc kernels
#include <kernel/ScalarOpenAdvElemKernel.h>

// deprecated
#include <ScalarMassElemSuppAlgDep.h>
#include <nso/ScalarNSOElemSuppAlgDep.h>

// nso
#include <nso/ScalarNSOElemKernel.h>
#include <nso/ScalarNSOKeElemKernel.h>
#include <nso/ScalarNSOKeElemSuppAlg.h>

// user function
#include <user_functions/VariableDensityMixFracSrcElemSuppAlg.h>
#include <user_functions/VariableDensityMixFracSrcNodeSuppAlg.h>
#include <user_functions/VariableDensityMixFracAuxFunction.h>
#include <user_functions/RayleighTaylorMixFracAuxFunction.h>
#include <user_functions/PerturbedShearLayerAuxFunctions.h>

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
// MixtureFractionEquationSystem - manages z pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MixtureFractionEquationSystem::MixtureFractionEquationSystem(
  EquationSystems& eqSystems,
  const bool outputClippingDiag,
  const double deltaZClip)
  : EquationSystem(eqSystems, "MixtureFractionEQS","mixture_fraction"),
    managePNG_(realm_.get_consistent_mass_matrix_png("mixture_fraction")),
    outputClippingDiag_(outputClippingDiag),
    deltaZClip_(deltaZClip),
    mixFrac_(NULL),
    mixFracUF_(NULL),
    dzdx_(NULL),
    zTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    scalarVar_(NULL),
    scalarDiss_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(realm_, "mixture_fraction", "dzdx")),
    diffFluxCoeffAlgDriver_(new AlgorithmDriver(realm_)),
    projectedNodalGradEqs_(NULL),
    isInit_(true)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("mixture_fraction");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_MIXTURE_FRACTION);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("mixture_fraction");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for mixture_fraction: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // advertise as non uniform
  realm_.uniformFlow_ = false;

  // create projected nodal gradient equation system
  if ( managePNG_ ) {
    manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MixtureFractionEquationSystem::~MixtureFractionEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
  delete diffFluxCoeffAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- populate_derived_quantities -------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::populate_derived_quantities()
{
  // placeholder
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  mixFrac_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "mixture_fraction", numStates));
  stk::mesh::put_field_on_mesh(*mixFrac_, *part, nullptr);
  realm_.augment_restart_variable_list("mixture_fraction");

  // for a sanity check, keep around the un-filterd/clipped field
  mixFracUF_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "uf_mixture_fraction", numStates));
  stk::mesh::put_field_on_mesh(*mixFracUF_, *part, nullptr);
 
  dzdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dzdx"));
  stk::mesh::put_field_on_mesh(*dzdx_, *part, nDim, nullptr);

  // delta solution for linear solver; share delta since this is a split system
  zTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pTmp"));
  stk::mesh::put_field_on_mesh(*zTmp_, *part, nullptr);

  visc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  if ( realm_.is_turbulent() ) {
    tvisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity"));
    stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);
  }

  evisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "effective_viscosity_z"));
  stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);

  scalarVar_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar_variance"));
  stk::mesh::put_field_on_mesh(*scalarVar_, *part, nullptr);

  scaledScalarVar_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scaled_scalar_variance"));
  stk::mesh::put_field_on_mesh(*scaledScalarVar_, *part, nullptr);

  scalarDiss_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar_dissipation"));
  stk::mesh::put_field_on_mesh(*scalarDiss_, *part, nullptr);

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &mixFracN = mixFrac_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &mixFracNp1 = mixFrac_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &mixFracNp1, &mixFracN,
                               0, 1,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &mixFracNp1 = mixFrac_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dzdxNone = dzdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to projected nodal gradient; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( edgeNodalGradient_ && realm_.realmUsesEdges_ ) {
        theAlg = new AssembleNodalGradEdgeAlgorithm(realm_, part, &mixFracNp1, &dzdxNone);
      }
      else {
        theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, &mixFracNp1, &dzdxNone, edgeNodalGradient_);
      }
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    } 
  }

  // solver; interior edge contribution (advection + diffusion)
  if ( !realm_.solutionOptions_->useConsolidatedSolverAlg_ ) {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theAlg = new AssembleScalarEdgeSolverAlgorithm(realm_, part, this, mixFrac_, dzdx_, evisc_);
      }
      else {
        theAlg = new AssembleScalarElemSolverAlgorithm(realm_, part, this, mixFrac_, dzdx_, evisc_);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
      
      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string> >::iterator isrc 
        = realm_.solutionOptions_->elemSrcTermsMap_.find("mixture_fraction");
      if ( isrc != realm_.solutionOptions_->elemSrcTermsMap_.end() ) {
        
        if ( realm_.realmUsesEdges_ )
          throw std::runtime_error("MixtureFractionElemSrcTerms::Error can not use element source terms for an edge-based scheme");
        
        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if (sourceName == "VariableDensity" ) {
            suppAlg = new VariableDensityMixFracSrcElemSuppAlg(realm_);
          }
          else if (sourceName == "NSO_2ND" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, mixFrac_, dzdx_, evisc_, 0.0, 0.0);
          }
          else if (sourceName == "NSO_2ND_ALT" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, mixFrac_, dzdx_, evisc_, 0.0, 1.0);
          }
          else if (sourceName == "NSO_4TH" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, mixFrac_, dzdx_, evisc_, 1.0, 0.0);
          }
          else if (sourceName == "NSO_4TH_ALT" ) {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, mixFrac_, dzdx_, evisc_, 1.0, 1.0);
          }
          else if (sourceName == "NSO_2ND_KE" ) {
            const double turbSc = realm_.get_turb_schmidt(mixFrac_->name());
            suppAlg = new ScalarNSOKeElemSuppAlg(realm_, mixFrac_, dzdx_, turbSc, 0.0);
          }
          else if (sourceName == "NSO_4TH_KE" ) {
            const double turbSc = realm_.get_turb_schmidt(mixFrac_->name());
            suppAlg = new ScalarNSOKeElemSuppAlg(realm_, mixFrac_, dzdx_, turbSc, 1.0);
          }
          else if (sourceName == "mixture_fraction_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, mixFrac_, false);
          }
          else if (sourceName == "lumped_mixture_fraction_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, mixFrac_, true);
          }
          else {
            throw std::runtime_error("MixtureFractionElemSrcTerms::Error Source term is not supported: " + sourceName);
          }     
          NaluEnv::self().naluOutputP0() << "MixtureFractionElemSrcTerms::added() " << sourceName << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg); 
        }
      }
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
    
    // time term; nodally lumped
    const AlgorithmType algMass = MASS;
    // Check if the user has requested CMM or LMM algorithms; if so, do not
    // include Nodal Mass algorithms
    std::vector<std::string> checkAlgNames = {"mixture_fraction_time_derivative",
                                              "lumped_mixture_fraction_time_derivative"};
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
            = new ScalarMassBackwardEulerNodeSuppAlg(realm_, mixFrac_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
        else {
          ScalarMassBDF2NodeSuppAlg *theMass
            = new ScalarMassBDF2NodeSuppAlg(realm_, mixFrac_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
      }
      
      // Add src term supp alg...; limited number supported
      std::map<std::string, std::vector<std::string> >::iterator isrc 
        = realm_.solutionOptions_->srcTermsMap_.find("mixture_fraction");
      if ( isrc != realm_.solutionOptions_->srcTermsMap_.end() ) {
        std::vector<std::string> mapNameVec = isrc->second;   
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if ( sourceName == "gcl" ) {
            suppAlg = new ScalarGclNodeSuppAlg(mixFrac_,realm_);
          }
          else if ( sourceName == "VariableDensity" ) {
            suppAlg = new VariableDensityMixFracSrcNodeSuppAlg(realm_);
          }
          else {
            throw std::runtime_error("MixtureFractionNodalSrcTerms::Error Source term is not supported: " + sourceName);
          }
          NaluEnv::self().naluOutputP0() << "MixtureFractionNodalSrcTerms::added() " << sourceName << std::endl;
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
      throw std::runtime_error("MixtureFraction::Error can not use element source terms for an edge-based scheme");
    
    KernelBuilder kb(*this, *part, solverAlgDriver_->solverAlgorithmMap_, realm_.using_tensor_product_kernels());
    auto& dataPreReqs = kb.data_prereqs();
    auto& dataPreReqsHO = kb.data_prereqs_HO();

    kb.build_topo_kernel_if_requested<ScalarMassElemKernel>
        ( "mixture_fraction_time_derivative",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, dataPreReqs, false);
      
    kb.build_topo_kernel_if_requested<ScalarMassElemKernel>
        ("lumped_mixture_fraction_time_derivative",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, dataPreReqs, true);
      
    kb.build_topo_kernel_if_requested<ScalarAdvDiffElemKernel>
        ("advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, evisc_, dataPreReqs);
      
    kb.build_topo_kernel_if_requested<ScalarUpwAdvDiffElemKernel>
        ("upw_advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, this, mixFrac_, dzdx_, evisc_, dataPreReqs);

    kb.build_topo_kernel_if_requested<ScalarNSOElemKernel>
        ("NSO_2ND",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, dzdx_, evisc_, 0.0, 0.0, dataPreReqs);
      
    kb.build_topo_kernel_if_requested<ScalarNSOElemKernel>
        ("NSO_2ND_ALT",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, dzdx_, evisc_, 0.0, 1.0, dataPreReqs);
      
    kb.build_topo_kernel_if_requested<ScalarNSOElemKernel>
        ("NSO_4TH",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, dzdx_, evisc_, 1.0, 0.0, dataPreReqs);
      
    kb.build_topo_kernel_if_requested<ScalarNSOElemKernel>
        ("NSO_4TH_ALT",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, dzdx_, evisc_, 1.0, 1.0, dataPreReqs);

    kb.build_topo_kernel_if_requested<ScalarNSOKeElemKernel>
        ("NSO_2ND_KE",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, dzdx_, realm_.get_turb_schmidt(mixFrac_->name()), 0.0, dataPreReqs);
      
    kb.build_topo_kernel_if_requested<ScalarNSOKeElemKernel>
        ("NSO_4TH_KE",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, dzdx_, realm_.get_turb_schmidt(mixFrac_->name()), 1.0, dataPreReqs);

    kb.build_sgl_kernel_if_requested<ScalarMassHOElemKernel>
        ("experimental_ho_mass",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, dataPreReqsHO);

    kb.build_sgl_kernel_if_requested<ScalarAdvDiffHOElemKernel>
        ("experimental_ho_advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, mixFrac_, evisc_, dataPreReqsHO);

    kb.report();
  }

  // effective viscosity alg
  const double lamSc = realm_.get_lam_schmidt(mixFrac_->name());
  const double turbSc = realm_.get_turb_schmidt(mixFrac_->name());
  std::map<AlgorithmType, Algorithm *>::iterator itev =
    diffFluxCoeffAlgDriver_->algMap_.find(algType);
  if ( itev == diffFluxCoeffAlgDriver_->algMap_.end() ) {
    EffectiveDiffFluxCoeffAlgorithm *theAlg
      = new EffectiveDiffFluxCoeffAlgorithm(realm_, part, visc_, tvisc_, evisc_, lamSc, turbSc);
    diffFluxCoeffAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itev->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType &mixFracNp1 = mixFrac_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dzdxNone = dzdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; mixFrac_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "mixFrac_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified mixFrac and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  std::string mixFracName = "mixture_fraction";
  UserDataType theDataType = get_bc_data_type(userData, mixFracName);

  AuxFunction *theAuxFunc = NULL;
  if ( CONSTANT_UD == theDataType ) {
    MixtureFraction mixFrac = userData.mixFrac_;
    std::vector<double> userSpec(1);
    userSpec[0] = mixFrac.mixFrac_;

    // new it
    theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
  }
  else if ( FUNCTION_UD == theDataType ) {
    std::string fcnName = get_bc_function_name(userData, mixFracName);
    if ( fcnName == "VariableDensity" ) {
      theAuxFunc = new VariableDensityMixFracAuxFunction();
    }
    else {
      throw std::runtime_error("MixFracEquationSystem::register_inflow_bc: Only VariableDensity supported");
    }
  }
  else {
    throw std::runtime_error("MixFracEquationSystem::register_inflow_bc: only constant and user function supported");
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

  // copy mixFrac_bc to mixture_fraction np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, &mixFracNp1,
                             0, 1,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dzdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &mixFracNp1, &dzdxNone, edgeNodalGradient_);
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
      = new DirichletBC(realm_, this, part, &mixFracNp1, theBcField, 0, 1);
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
MixtureFractionEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  ScalarFieldType &mixFracNp1 = mixFrac_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dzdxNone = dzdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; mixFrac_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "open_mixFrac_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified mixFrac and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  MixtureFraction mixFrac = userData.mixFrac_;
  std::vector<double> userSpec(1);
  userSpec[0] = mixFrac.mixFrac_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // non-solver; dzdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &mixFracNp1, &dzdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // now solver contributions; open; lhs
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
        (partTopo, elemTopo, *this, activeKernels, "mixture_fraction_open",
         realm_.meta_data(), *realm_.solutionOptions_,
         this, mixFrac_, theBcField, dzdx_, evisc_, 
         faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_);
      
    }
  }
  else {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theAlg = new AssembleScalarEdgeOpenSolverAlgorithm(realm_, part, this, mixFrac_, theBcField, &dzdxNone, evisc_);
      }
      else {
        theAlg = new AssembleScalarElemOpenSolverAlgorithm(realm_, part, this, mixFrac_, theBcField, &dzdxNone, evisc_);
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
MixtureFractionEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  // np1
  ScalarFieldType &mixFracNp1 = mixFrac_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dzdxNone = dzdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // extract the value for user specified mixFrac and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  std::string mixFracName = "mixture_fraction";
  if ( bc_data_specified(userData, mixFracName) ) {

    // FIXME: Generalize for constant vs function

    // register boundary data; mixFrac_bc
    ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "mixFrac_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

    // extract data
    std::vector<double> userSpec(1);
    MixtureFraction mixFrac = userData.mixFrac_;
    userSpec[0] = mixFrac.mixFrac_;

    // new it
    ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

    // bc data alg
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);

    // copy mixFrac_bc to mixFrac np1...
    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               theBcField, &mixFracNp1,
                               0, 1,
                               stk::topology::NODE_RANK);
    bcDataMapAlg_.push_back(theCopyAlg);

    // Dirichlet bc
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
      DirichletBC *theAlg
        = new DirichletBC(realm_, this, part, &mixFracNp1, theBcField, 0, 1);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    }
    else {
      itd->second->partVec_.push_back(part);
    }
  }

  // non-solver; dzdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &mixFracNp1, &dzdxNone, edgeNodalGradient_);
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
MixtureFractionEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData &/*wallBCData*/)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  // np1
  ScalarFieldType &mixFracNp1 = mixFrac_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dzdxNone = dzdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dzdx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &mixFracNp1, &dzdxNone, edgeNodalGradient_);
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
MixtureFractionEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{
  
  const AlgorithmType algType = NON_CONFORMAL;
  
  // np1
  ScalarFieldType &mixFracNp1 = mixFrac_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dzdxNone = dzdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dzdx; DG algorithm decides on locations for integration points
  if ( !managePNG_ ) {
    if ( edgeNodalGradient_ ) {    
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        Algorithm *theAlg 
          = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &mixFracNp1, &dzdxNone, edgeNodalGradient_);
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
          = new AssembleNodalGradNonConformalAlgorithm(realm_, part, &mixFracNp1, &dzdxNone);
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
      = new AssembleScalarNonConformalSolverAlgorithm(realm_, part, this, mixFrac_, evisc_);
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
MixtureFractionEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(mixFrac_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(mixFrac_,1,1)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_MIXTURE_FRACTION;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("mixture_fraction");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_MIXTURE_FRACTION);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::register_initial_condition_fcn(
  stk::mesh::Part *part,
  const std::map<std::string, std::string> &theNames,
  const std::map<std::string, std::vector<double> > &/*theParams*/)
{
  // iterate map and check for name
  const std::string dofName = "mixture_fraction";
  std::map<std::string, std::string>::const_iterator iterName
    = theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;
    AuxFunction *theAuxFunc = NULL;
    if ( fcnName == "VariableDensity" ) {
      // create the function
      theAuxFunc = new VariableDensityMixFracAuxFunction();      
    }
    else if ( fcnName == "RayleighTaylor" ) {
      // create the function
      theAuxFunc = new RayleighTaylorMixFracAuxFunction();      
    }
    else if ( fcnName == "PerturbedShearLayer" ) {
      // create the function
      theAuxFunc = new PerturbedShearLayerMixFracAuxFunction();
    }
    else {
      throw std::runtime_error("MixtureFractionEquationSystem::register_initial_condition_fcn: VariableDensity only supported");
    }
    
    // create the algorithm
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
				 mixFrac_, theAuxFunc,
				 stk::topology::NODE_RANK);
    
    // push to ic
    realm_.initCondAlg_.push_back(auxAlg);
    
  }
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::solve_and_update()
{

  // compute dz/dx
  if ( isInit_ ) {
    compute_projected_nodal_gradient();
    isInit_ = false;
  }

  // compute effective viscosity
  diffFluxCoeffAlgDriver_->execute();

  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    // mixture fraction assemble, load_complete and solve
    assemble_and_solve(zTmp_);

    // update
    double timeA = NaluEnv::self().nalu_time();
    update_and_clip();
    double timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB-timeA);

    // projected nodal gradient
    compute_projected_nodal_gradient();
  }

  compute_scalar_var_diss();
}

//--------------------------------------------------------------------------
//-------- update_and_clip -------------------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::update_and_clip()
{
  const double deltaZ = deltaZClip_;
  const double lowBound = 0.0-deltaZ;
  const double highBound = 1.0+deltaZ;
  size_t numClip[2] = {0,0};
  double minZ = +1.0e16;
  double maxZ = -1.0e16;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*mixFrac_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    double *mixFrac = stk::mesh::field_data(*mixFrac_, b);
    double *mixFracUF = stk::mesh::field_data(*mixFracUF_, b);
    double *zTmp    = stk::mesh::field_data(*zTmp_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      double mixFracNp1 = mixFrac[k] + zTmp[k];
      // store un-filtered value for numerical methods development purposes
      mixFracUF[k] = mixFracNp1;
      // clip now
      if ( mixFracNp1 < lowBound ) {
        minZ = std::min(mixFracNp1, minZ);
        mixFracNp1 = lowBound;
        numClip[0]++;
      }
      else if ( mixFracNp1 > highBound ) {
        maxZ = std::max(mixFracNp1, maxZ);
        mixFracNp1 = highBound;
        numClip[1]++;
      }
      mixFrac[k] = mixFracNp1;
    }
  }

  // parallel assemble clipped value
  if ( outputClippingDiag_ ) {
    size_t g_numClip[2] = {};
    stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
    stk::all_reduce_sum(comm, numClip, g_numClip, 2);

    if ( g_numClip[0] > 0 ) {
      double g_minZ = 0;
      stk::all_reduce_min(comm, &minZ, &g_minZ, 1);
      NaluEnv::self().naluOutputP0() << "mixFrac clipped (-) " << g_numClip[0] << " times; min: " << g_minZ << std::endl;
    }
    else {
      NaluEnv::self().naluOutputP0() << "mixFrac clipped (-) zero times" << std::endl;
    }

    if ( g_numClip[1] > 0 ) {
      double g_maxZ = 0;
      stk::all_reduce_max(comm, &maxZ, &g_maxZ, 1);
      NaluEnv::self().naluOutputP0() << "mixFrac clipped (+) " << g_numClip[1] << " times; max: " << g_maxZ << std::endl;
    }
    else {
      NaluEnv::self().naluOutputP0() << "mixFrac clipped (+) zero times" << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- compute_scalar_var_diss -----------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::compute_scalar_var_diss()
{
  const double Cv = realm_.solutionOptions_->turbModelConstantMap_.at(TM_zCV);

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  ScalarFieldType *dualNodalVol = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*dzdx_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    double *mixFrac = stk::mesh::field_data(*mixFrac_, b);
    double *scalarVar = stk::mesh::field_data(*scalarVar_, b);
    double *scaledScalarVar = stk::mesh::field_data(*scaledScalarVar_, b);
    double *scalarDiss = stk::mesh::field_data(*scalarDiss_, b);
    const double *dzdx = stk::mesh::field_data(*dzdx_, b);
    const double *rho = stk::mesh::field_data(*density, b);
    const double *evisc = stk::mesh::field_data(*evisc_, b);
    const double *cVol = stk::mesh::field_data(*dualNodalVol, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const double filter = std::pow(cVol[k], 1.0/nDim);
      double sum = 0.0;
      for (int j = 0; j < nDim; ++j ) {
        sum += dzdx[k*nDim+j]*dzdx[k*nDim+j];
      }
      scalarVar[k] = Cv*filter*filter*sum;

      // clip and scale the variance
      // zero scaled scalar variance estimate near free streams
      constexpr double maxVarThreshold = 1.0e-6;
      const double maxScalarVar = std::max(mixFrac[k] * (1 - mixFrac[k]), 0.0);
      const double clippedVar = std::min(std::max(0.0, scalarVar[k]), maxScalarVar);
      scaledScalarVar[k] = (maxScalarVar > maxVarThreshold ) ? clippedVar / maxScalarVar : 0.0;

      scalarDiss[k] = 2.0*evisc[k]/rho[k]*sum;
    }
  }
}

//--------------------------------------------------------------------------
//-------- predict_state ---------------------------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::predict_state()
{
  // copy state n to state np1
  ScalarFieldType &zN = mixFrac_->field_of_state(stk::mesh::StateN);
  ScalarFieldType &zNp1 = mixFrac_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), zN, zNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if ( NULL == projectedNodalGradEqs_ ) {
    projectedNodalGradEqs_ 
      = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG_Z, "dzdx", "qTmp", "mixture_fraction", "PNGradZEQS");
  }
  // fill the map for expected boundary condition names; can be more complex...
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "mixture_fraction");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "mixture_fraction");
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "mixture_fraction");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "mixture_fraction");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient---------------------------------
//--------------------------------------------------------------------------
void
MixtureFractionEquationSystem::compute_projected_nodal_gradient()
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
