/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <SpecificDissipationRateEquationSystem.h>
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
#include <ComputeLowReynoldsSDRWallAlgorithm.h>
#include <ComputeWallModelSDRWallAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
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
#include <Realm.h>
#include <Realms.h>
#include <ScalarGclNodeSuppAlg.h>
#include <ScalarMassBackwardEulerNodeSuppAlg.h>
#include <ScalarMassBDF2NodeSuppAlg.h>
#include <ScalarMassElemSuppAlgDep.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <TimeIntegrator.h>
#include <SpecificDissipationRateSSTNodeSourceSuppAlg.h>
#include <SolverAlgorithmDriver.h>

// template for supp algs
#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

// consolidated
#include <AssembleElemSolverAlgorithm.h>
#include <kernel/ScalarMassElemKernel.h>
#include <kernel/ScalarAdvDiffElemKernel.h>
#include <kernel/ScalarUpwAdvDiffElemKernel.h>
#include <kernel/SpecificDissipationRateSSTSrcElemKernel.h>

// nso
#include <nso/ScalarNSOElemKernel.h>
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

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// SpecificDissipationRateEquationSystem - manages sdr pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SpecificDissipationRateEquationSystem::SpecificDissipationRateEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "SpecDissRateEQS","specific_dissipation_rate"),
    managePNG_(realm_.get_consistent_mass_matrix_png("specific_dissipation_rate")),
    sdr_(NULL),
    dwdx_(NULL),
    wTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    sdrWallBc_(NULL),
    assembledWallSdr_(NULL),
    assembledWallArea_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(realm_, "specific_dissipation_rate", "dwdx")),
    diffFluxCoeffAlgDriver_(new AlgorithmDriver(realm_))
{
  dofName_ = "specific_dissipation_rate";

  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("specific_dissipation_rate");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_SPEC_DISS_RATE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("specific_dissipation_rate");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for specific_dissipation_rate: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create projected nodal gradient equation system
  if ( managePNG_ )
    throw std::runtime_error("SpecificDissipationRateEquationSystem::Error managePNG is not complete");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SpecificDissipationRateEquationSystem::~SpecificDissipationRateEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
  delete diffFluxCoeffAlgDriver_;
  std::vector<Algorithm *>::iterator ii;
  for( ii=wallModelAlg_.begin(); ii!=wallModelAlg_.end(); ++ii )
    delete *ii;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  sdr_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_dissipation_rate", numStates));
  stk::mesh::put_field_on_mesh(*sdr_, *part, nullptr);
  realm_.augment_restart_variable_list("specific_dissipation_rate");

  dwdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dwdx"));
  stk::mesh::put_field_on_mesh(*dwdx_, *part, nDim, nullptr);

  // delta solution for linear solver; share delta since this is a split system
  wTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "wTmp"));
  stk::mesh::put_field_on_mesh(*wTmp_, *part, nullptr);

  visc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  tvisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity"));
  stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);

  evisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "effective_viscosity_sdr"));
  stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &sdrN = sdr_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &sdrNp1, &sdrN,
                               0, 1,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dwdxNone = dwdx_->field_of_state(stk::mesh::StateNone);

  // non-solver, dwdx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg = NULL;
    if ( edgeNodalGradient_ && realm_.realmUsesEdges_ ) {
      theAlg = new AssembleNodalGradEdgeAlgorithm(realm_, part, &sdrNp1, &dwdxNone);
    }
    else {
      theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, &sdrNp1, &dwdxNone, edgeNodalGradient_);
    }
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

  // solver; interior contribution (advection + diffusion)
  if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {

    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = NULL;
      if (realm_.realmUsesEdges_) {
        theAlg = new AssembleScalarEdgeSolverAlgorithm(realm_, part, this, sdr_, dwdx_, evisc_);
      }
      else {
        theAlg = new AssembleScalarElemSolverAlgorithm(realm_, part, this, sdr_, dwdx_, evisc_);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;

      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string> >::iterator isrc
        = realm_.solutionOptions_->elemSrcTermsMap_.find("specific_dissipation_rate");
      if (isrc != realm_.solutionOptions_->elemSrcTermsMap_.end()) {

        if (realm_.realmUsesEdges_)
          throw std::runtime_error("SpecificDissipationElemSrcTerms::Error can not use element source terms for an edge-based scheme");

        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm* suppAlg = NULL;
          if (sourceName == "NSO_2ND_ALT") {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, sdr_, dwdx_, evisc_, 0.0, 1.0);
          }
          else if (sourceName == "NSO_4TH_ALT") {
            suppAlg = new ScalarNSOElemSuppAlgDep(realm_, sdr_, dwdx_, evisc_, 1.0, 1.0);
          }
          else if (sourceName == "NSO_2ND_KE") {
            const double turbSc = realm_.get_turb_schmidt(sdr_->name());
            suppAlg = new ScalarNSOKeElemSuppAlg(realm_, sdr_, dwdx_, turbSc, 0.0);
          }
          else if (sourceName == "NSO_4TH_KE") {
            const double turbSc = realm_.get_turb_schmidt(sdr_->name());
            suppAlg = new ScalarNSOKeElemSuppAlg(realm_, sdr_, dwdx_, turbSc, 1.0);
          }
          else if (sourceName == "specific_dissipation_rate_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, sdr_, false);
          }
          else if (sourceName == "lumped_specific_dissipation_rate_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, sdr_, true);
          }
          else {
            throw std::runtime_error("SpecificDissipationElemSrcTerms::Error Source term is not supported: " + sourceName);
          }
          NaluEnv::self().naluOutputP0() << "SpecificDissipationElemSrcTerms::added() " << sourceName << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg);
        }
      }
    }
    else {
      itsi->second->partVec_.push_back(part);
    }

    // time term; src; both nodally lumped
    const AlgorithmType algMass = MASS;
    // Check if the user has requested CMM or LMM algorithms; if so, do not
    // include Nodal Mass algorithms
    std::vector<std::string> checkAlgNames = {
      "specific_dissipation_rate_time_derivative",
      "lumped_specific_dissipation_rate_time_derivative"};
    bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
    std::map<AlgorithmType, SolverAlgorithm*>::iterator itsm =
      solverAlgDriver_->solverAlgMap_.find(algMass);
    if (itsm == solverAlgDriver_->solverAlgMap_.end()) {
      // create the solver alg
      AssembleNodeSolverAlgorithm *theAlg
        = new AssembleNodeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algMass] = theAlg;

      // now create the supplemental alg for mass term
      if (!elementMassAlg) {
        if (realm_.number_of_states() == 2) {
          ScalarMassBackwardEulerNodeSuppAlg *theMass
            = new ScalarMassBackwardEulerNodeSuppAlg(realm_, sdr_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
        else {
          ScalarMassBDF2NodeSuppAlg *theMass
            = new ScalarMassBDF2NodeSuppAlg(realm_, sdr_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
      }

      // now create the src alg for sdr source
      SpecificDissipationRateSSTNodeSourceSuppAlg *theSrc
        = new SpecificDissipationRateSSTNodeSourceSuppAlg(realm_);
      theAlg->supplementalAlg_.push_back(theSrc);

      // Add src term supp alg...; limited number supported
      std::map<std::string, std::vector<std::string> >::iterator isrc
        = realm_.solutionOptions_->srcTermsMap_.find("specific_dissipation_rate");
      if (isrc != realm_.solutionOptions_->srcTermsMap_.end()) {
        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm* suppAlg = NULL;
          if (sourceName == "gcl") {
            suppAlg = new ScalarGclNodeSuppAlg(sdr_, realm_);
          }
          else {
            throw std::runtime_error("SpecificDissipationRateNodalSrcTerms::Error Source term is not supported: " + sourceName);
          }
          NaluEnv::self().naluOutputP0() << "SpecificDissipationRateNodalSrcTerms::added() " << sourceName << std::endl;
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
    if (realm_.realmUsesEdges_)
      throw std::runtime_error("SpecificDissipationRateEquationSystem::Error can not use element source terms for an edge-based scheme");

    stk::topology partTopo = part->topology();
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(solverAlg, solverAlgWasBuilt) =
      build_or_add_part_to_solver_alg(*this, *part, solverAlgMap);

    ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
    auto& activeKernels = solverAlg->activeKernels_;

    if (solverAlgWasBuilt) {
      build_topo_kernel_if_requested<ScalarMassElemKernel>
        (partTopo, *this, activeKernels, "specific_dissipation_rate_time_derivative",
         realm_.bulk_data(), *realm_.solutionOptions_, sdr_, dataPreReqs, false);

      build_topo_kernel_if_requested<ScalarMassElemKernel>
        (partTopo, *this, activeKernels,  "lumped_specific_dissipation_rate_time_derivative",
         realm_.bulk_data(), *realm_.solutionOptions_, sdr_, dataPreReqs, true);

      build_topo_kernel_if_requested<ScalarAdvDiffElemKernel>
        (partTopo, *this, activeKernels, "advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, sdr_, evisc_, dataPreReqs);

      build_topo_kernel_if_requested<ScalarUpwAdvDiffElemKernel>
        (partTopo, *this, activeKernels, "upw_advection_diffusion",
        realm_.bulk_data(), *realm_.solutionOptions_, this, sdr_, dwdx_, evisc_, dataPreReqs);

      build_topo_kernel_if_requested<SpecificDissipationRateSSTSrcElemKernel>
        (partTopo, *this, activeKernels, "sst",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);

      build_topo_kernel_if_requested<SpecificDissipationRateSSTSrcElemKernel>
        (partTopo, *this, activeKernels, "lumped_sst",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);

      build_topo_kernel_if_requested<ScalarNSOElemKernel>
        (partTopo, *this, activeKernels, "NSO_2ND",
         realm_.bulk_data(), *realm_.solutionOptions_, sdr_, dwdx_, evisc_, 0.0, 0.0, dataPreReqs);

      build_topo_kernel_if_requested<ScalarNSOElemKernel>
        (partTopo, *this, activeKernels, "NSO_2ND_ALT",
         realm_.bulk_data(), *realm_.solutionOptions_, sdr_, dwdx_, evisc_, 0.0, 1.0, dataPreReqs);

      build_topo_kernel_if_requested<ScalarNSOElemKernel>
        (partTopo, *this, activeKernels, "NSO_4TH",
         realm_.bulk_data(), *realm_.solutionOptions_, sdr_, dwdx_, evisc_, 1.0, 0.0, dataPreReqs);

      build_topo_kernel_if_requested<ScalarNSOElemKernel>
        (partTopo, *this, activeKernels, "NSO_4TH_ALT",
         realm_.bulk_data(), *realm_.solutionOptions_, sdr_, dwdx_, evisc_, 1.0, 1.0, dataPreReqs);

      report_invalid_supp_alg_names();
      report_built_supp_alg_names();
    }
  }

  // effective diffusive flux coefficient alg for SST
  std::map<AlgorithmType, Algorithm *>::iterator itev =
    diffFluxCoeffAlgDriver_->algMap_.find(algType);
  if ( itev == diffFluxCoeffAlgDriver_->algMap_.end() ) {
    const double sigmaWOne = realm_.get_turb_model_constant(TM_sigmaWOne);
    const double sigmaWTwo = realm_.get_turb_model_constant(TM_sigmaWTwo);
    EffectiveSSTDiffFluxCoeffAlgorithm *effDiffAlg
      = new EffectiveSSTDiffFluxCoeffAlgorithm(realm_, part, visc_, tvisc_, evisc_, sigmaWOne, sigmaWTwo);
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
SpecificDissipationRateEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dwdxNone = dwdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; sdr_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "sdr_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  SpecDissRate sdr = userData.sdr_;
  std::vector<double> userSpec(1);
  userSpec[0] = sdr.specDissRate_;

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

  // copy sdr_bc to specific_dissipation_rate np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, &sdrNp1,
                             0, 1,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dwdx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg 
      = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &sdrNp1, &dwdxNone, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg
      = new DirichletBC(realm_, this, part, &sdrNp1, theBcField, 0, 1);
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
SpecificDissipationRateEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dwdxNone = dwdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; sdr_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "open_sdr_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  SpecDissRate sdr = userData.sdr_;
  std::vector<double> userSpec(1);
  userSpec[0] = sdr.specDissRate_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // non-solver; dwdx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg 
      = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &sdrNp1, &dwdxNone, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

  // solver open; lhs
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
    = solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    SolverAlgorithm *theAlg = NULL;
    if ( realm_.realmUsesEdges_ ) {
      theAlg = new AssembleScalarEdgeOpenSolverAlgorithm(realm_, part, this, sdr_, theBcField, &dwdxNone, evisc_);
    }
    else {
      theAlg = new AssembleScalarElemOpenSolverAlgorithm(realm_, part, this, sdr_, theBcField, &dwdxNone, evisc_);
    }
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    itsi->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  // np1
  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dwdxNone = dwdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; sdr_bc
  sdrWallBc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "sdr_bc"));
  stk::mesh::put_field_on_mesh(*sdrWallBc_, *part, nullptr);

  // need to register the assembles wall value for sdr; can not share with sdr_bc
  assembledWallSdr_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "wall_model_sdr_bc"));
  stk::mesh::put_field_on_mesh(*assembledWallSdr_, *part, nullptr);

  assembledWallArea_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_sdr"));
  stk::mesh::put_field_on_mesh(*assembledWallArea_, *part, nullptr);

  // are we using wall functions or is this a low Re model?
  WallUserData userData = wallBCData.userData_;
  bool wallFunctionApproach = userData.wallFunctionApproach_;

  // create proper algorithms to fill nodal omega and assembled wall area; utau managed by momentum
  Algorithm *wallAlg = NULL;
  if ( wallFunctionApproach ) {
    wallAlg = new ComputeWallModelSDRWallAlgorithm(realm_, part, realm_.realmUsesEdges_);
  }
  else {
    wallAlg = new ComputeLowReynoldsSDRWallAlgorithm(realm_, part, realm_.realmUsesEdges_);
  }
  wallModelAlg_.push_back(wallAlg);

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg =
        new DirichletBC(realm_, this, part, &sdrNp1, sdrWallBc_, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  }
  else {
    itd->second->partVec_.push_back(part);
  }

  // non-solver; dwdx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg 
      = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &sdrNp1, &dwdxNone, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData & /* symmetryBCData */)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  // np1
  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dwdxNone = dwdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dwdx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg 
      = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &sdrNp1, &dwdxNone, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{

  const AlgorithmType algType = NON_CONFORMAL;

  // np1
  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dwdxNone = dwdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dwdx; DG algorithm decides on locations for integration points
  if ( edgeNodalGradient_ ) {    
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &sdrNp1, &dwdxNone, edgeNodalGradient_);
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
        = new AssembleNodalGradNonConformalAlgorithm(realm_, part, &sdrNp1, &dwdxNone);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    AssembleScalarNonConformalSolverAlgorithm *theAlg
      = new AssembleScalarNonConformalSolverAlgorithm(realm_, part, this, sdr_, evisc_);
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
SpecificDissipationRateEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(sdr_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(sdr_,1,1)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_SPEC_DISS_RATE;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("specific_dissipation_rate");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_SPEC_DISS_RATE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- assemble_nodal_gradient() ---------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::assemble_nodal_gradient()
{
  const double timeA = -NaluEnv::self().nalu_time();
  assembleNodalGradAlgDriver_->execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

//--------------------------------------------------------------------------
//-------- compute_effective_flux_coeff() ----------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::compute_effective_diff_flux_coeff()
{
  const double timeA = -NaluEnv::self().nalu_time();
  diffFluxCoeffAlgDriver_->execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

//--------------------------------------------------------------------------
//-------- compute_wall_model_parameters() ---------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::compute_wall_model_parameters()
{

  // check if we need to process anything
  if ( wallModelAlg_.size() == 0 )
    return;

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // selector; all nodes that have a SST-specific nodal field registered
  stk::mesh::Selector s_all_nodes
     = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
     &stk::mesh::selectField(*assembledWallArea_);
  stk::mesh::BucketVector const& node_buckets
    = realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );

  // zero the fields
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
      ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length  = b.size();
    double * assembledWallSdr = stk::mesh::field_data(*assembledWallSdr_, b);
    double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      assembledWallSdr[k] = 0.0;
      assembledWallArea[k] = 0.0;
    }
  }

  // process the algorithm(s)
  for ( size_t k = 0; k < wallModelAlg_.size(); ++k ) {
    wallModelAlg_[k]->execute();
  }

  stk::mesh::parallel_sum(bulk_data, {assembledWallSdr_, assembledWallArea_});

  // periodic assemble
  if ( realm_.hasPeriodic_) {
    const unsigned fieldSize = 1;
    const bool bypassFieldCheck = false; // fields are not defined at all slave/master node pairs
    realm_.periodic_field_update(assembledWallSdr_, fieldSize, bypassFieldCheck);
    realm_.periodic_field_update(assembledWallArea_, fieldSize, bypassFieldCheck);
  }

  // normalize and set assembled sdr to sdr bc
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
      ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length  = b.size();
    double * sdr = stk::mesh::field_data(*sdr_, b);
    double * sdrWallBc = stk::mesh::field_data(*sdrWallBc_, b);
    double * assembledWallSdr = stk::mesh::field_data(*assembledWallSdr_, b);
    double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const double sdrBnd = assembledWallSdr[k]/assembledWallArea[k];
      sdrWallBc[k] = sdrBnd;
      assembledWallSdr[k] = sdrBnd;
      // make sure that the next matrix assembly uses the proper sdr value
      sdr[k] = sdrBnd;
    }
  }
}

//--------------------------------------------------------------------------
//-------- predict_state() -------------------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::predict_state()
{
  // copy state n to state np1
  ScalarFieldType &sdrN = sdr_->field_of_state(stk::mesh::StateN);
  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), sdrN, sdrNp1, realm_.get_activate_aura());
}

} // namespace nalu
} // namespace Sierra
