// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <SpecificDissipationRateEquationSystem.h>
#include <AlgorithmDriver.h>
#include <AssembleScalarEdgeOpenSolverAlgorithm.h>
#include <AssembleScalarElemSolverAlgorithm.h>
#include <AssembleScalarElemOpenSolverAlgorithm.h>
#include <AssembleScalarNonConformalSolverAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
#include <AssembleNodalGradElemAlgorithm.h>
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <AuxFunctionAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
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
#include <ScalarMassElemSuppAlgDep.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <TimeIntegrator.h>
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
#include <kernel/SpecificDissipationRateSSTDESSrcElemKernel.h>


// edge kernels
#include <edge_kernels/ScalarEdgeSolverAlg.h>
#include <edge_kernels/ScalarOpenEdgeKernel.h>

// node kernels
#include <node_kernels/NodeKernelUtils.h>
#include <node_kernels/ScalarMassBDFNodeKernel.h>
#include <node_kernels/SDRSSTNodeKernel.h>
#include <node_kernels/SDRSSTDESNodeKernel.h>
#include <node_kernels/ScalarGclNodeKernel.h>

// ngp
#include "ngp_utils/NgpFieldBLAS.h"
#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_algorithms/NodalGradElemAlg.h"
#include "ngp_algorithms/NodalGradBndryElemAlg.h"
#include "ngp_algorithms/EffSSTDiffFluxCoeffAlg.h"
#include "ngp_algorithms/SDRWallFuncAlg.h"
#include "ngp_algorithms/SDRLowReWallAlg.h"
#include "ngp_algorithms/SDRWallFuncAlgDriver.h"
#include "utils/StkHelpers.h"

// UT Austin Hybrid TAMS kernel
#include <kernel/SpecificDissipationRateSSTTAMSSrcElemKernel.h>
#include <node_kernels/SDRSSTTAMSNodeKernel.h>

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
    nodalGradAlgDriver_(realm_, "dwdx")
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
SpecificDissipationRateEquationSystem::~SpecificDissipationRateEquationSystem() = default;

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

  if (edgeNodalGradient_ && realm_.realmUsesEdges_)
    nodalGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
      algType, part, "sdr_nodal_grad", &sdrNp1, &dwdxNone);
  else
    nodalGradAlgDriver_.register_legacy_algorithm<AssembleNodalGradElemAlgorithm>(
      algType, part, "sdr_nodal_grad", &sdrNp1, &dwdxNone,
      edgeNodalGradient_);

  // solver; interior contribution (advection + diffusion)
  if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {

    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = NULL;
      if (realm_.realmUsesEdges_) {
        const bool useAvgMdot = (realm_.solutionOptions_->turbulenceModel_ == SST_TAMS) ? true : false;
        theAlg = new ScalarEdgeSolverAlg(realm_, part, this, sdr_, dwdx_, evisc_, useAvgMdot);
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

    // Check if the user has requested CMM or LMM algorithms; if so, do not
    // include Nodal Mass algorithms
    std::vector<std::string> checkAlgNames = {
      "specific_dissipation_rate_time_derivative",
      "lumped_specific_dissipation_rate_time_derivative"};
    bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
    auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
    process_ngp_node_kernels(
      solverAlgMap, realm_, part, this,
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg) {
        if (!elementMassAlg)
          nodeAlg.add_kernel<ScalarMassBDFNodeKernel>(realm_.bulk_data(), sdr_);

        if (SST == realm_.solutionOptions_->turbulenceModel_){
          nodeAlg.add_kernel<SDRSSTNodeKernel>(realm_.meta_data());
        }
        else if (SST_DES == realm_.solutionOptions_->turbulenceModel_){
          nodeAlg.add_kernel<SDRSSTDESNodeKernel>(realm_.meta_data());
        }
        else if (SST_TAMS == realm_.solutionOptions_->turbulenceModel_){
          nodeAlg.add_kernel<SDRSSTTAMSNodeKernel>(realm_.meta_data(), realm_.solutionOptions_->get_coordinates_name());
        }
      },
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg, std::string& srcName) {
        if (srcName == "gcl") {
          nodeAlg.add_kernel<ScalarGclNodeKernel>(realm_.bulk_data(), sdr_);
          NaluEnv::self().naluOutputP0() << " - " << srcName << std::endl;
        }
        else
          throw std::runtime_error("SDREqSys: Invalid source term: " + srcName);
      });
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

      build_topo_kernel_if_requested<ScalarAdvDiffElemKernel>
        (partTopo, *this, activeKernels, "TAMS_advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, sdr_, evisc_, dataPreReqs, true);

      build_topo_kernel_if_requested<ScalarUpwAdvDiffElemKernel>
        (partTopo, *this, activeKernels, "upw_advection_diffusion",
        realm_.bulk_data(), *realm_.solutionOptions_, this, sdr_, dwdx_, evisc_, dataPreReqs);

      build_topo_kernel_if_requested<ScalarUpwAdvDiffElemKernel>
        (partTopo, *this, activeKernels, "TAMS_upw_advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, this, sdr_, dwdx_, evisc_, dataPreReqs, true);

      build_topo_kernel_if_requested<SpecificDissipationRateSSTSrcElemKernel>
        (partTopo, *this, activeKernels, "sst",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);

      build_topo_kernel_if_requested<SpecificDissipationRateSSTDESSrcElemKernel>
        (partTopo, *this, activeKernels, "sst_des",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);

      build_topo_kernel_if_requested<SpecificDissipationRateSSTSrcElemKernel>
        (partTopo, *this, activeKernels, "lumped_sst",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);

      build_topo_kernel_if_requested<SpecificDissipationRateSSTDESSrcElemKernel>
        (partTopo, *this, activeKernels, "lumped_sst_des",
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

      // UT Austin Hybrid TAMS model implementations for SDR source terms
      build_topo_kernel_if_requested<SpecificDissipationRateSSTTAMSSrcElemKernel>
        (partTopo, *this, activeKernels, "sst_tams",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, false);

      build_topo_kernel_if_requested<SpecificDissipationRateSSTTAMSSrcElemKernel>
        (partTopo, *this, activeKernels, "lumped_sst_tams",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs, true);

      report_invalid_supp_alg_names();
      report_built_supp_alg_names();
    }
  }

  // effective diffusive flux coefficient alg for SST
  if (!effDiffFluxAlg_) {
    const double sigmaWOne = realm_.get_turb_model_constant(TM_sigmaWOne);
    const double sigmaWTwo = realm_.get_turb_model_constant(TM_sigmaWTwo);
    effDiffFluxAlg_.reset(new EffSSTDiffFluxCoeffAlg(
      realm_, part, visc_, tvisc_, evisc_, sigmaWOne, sigmaWTwo));
  } else {
    effDiffFluxAlg_->partVec_.push_back(part);
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
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sdr_nodal_grad", &sdrNp1, &dwdxNone, edgeNodalGradient_);

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
  const stk::topology & partTopo,
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
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sdr_nodal_grad", &sdrNp1, &dwdxNone, edgeNodalGradient_);

  if (realm_.realmUsesEdges_) {
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    AssembleElemSolverAlgorithm* elemSolverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(elemSolverAlg, solverAlgWasBuilt)
      = build_or_add_part_to_face_bc_solver_alg(*this, *part, solverAlgMap, "open");

    auto& dataPreReqs = elemSolverAlg->dataNeededByKernels_;
    auto& activeKernels = elemSolverAlg->activeKernels_;

    build_face_topo_kernel_automatic<ScalarOpenEdgeKernel>(
      partTopo, *this, activeKernels, "sdr_open",
      realm_.meta_data(), *realm_.solutionOptions_, sdr_, theBcField, dataPreReqs);
  }
  else {
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
  if (!wallModelAlgDriver_)
    wallModelAlgDriver_.reset(new SDRWallFuncAlgDriver(realm_));
  if (wallFunctionApproach)
    wallModelAlgDriver_->register_face_elem_algorithm<SDRWallFuncAlg>(
      algType, part, get_elem_topo(realm_, *part), "sdr_wall_func");
  else
    wallModelAlgDriver_->register_face_elem_algorithm<SDRLowReWallAlg>(
      algType, part, get_elem_topo(realm_, *part), "sdr_wall_func", realm_.realmUsesEdges_);

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
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sdr_nodal_grad", &sdrNp1, &dwdxNone, edgeNodalGradient_);
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
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sdr_nodal_grad", &sdrNp1, &dwdxNone, edgeNodalGradient_);
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
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
        algType, part, "sdr_nodal_grad", &sdrNp1, &dwdxNone, edgeNodalGradient_);
  }
  else {
    // proceed with DG
    nodalGradAlgDriver_
      .register_legacy_algorithm<AssembleNodalGradNonConformalAlgorithm>(
        algType, part, "sdr_nodal_grad", &sdrNp1, &dwdxNone);
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

  equationSystems_.register_overset_field_update(sdr_, 1, 1);
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  //linsys_->finalizeLinearSystem();
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
  //linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- assemble_nodal_gradient() ---------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::assemble_nodal_gradient()
{
  const double timeA = -NaluEnv::self().nalu_time();
  nodalGradAlgDriver_.execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

//--------------------------------------------------------------------------
//-------- compute_effective_flux_coeff() ----------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::compute_effective_diff_flux_coeff()
{
  const double timeA = -NaluEnv::self().nalu_time();
  effDiffFluxAlg_->execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

//--------------------------------------------------------------------------
//-------- compute_wall_model_parameters() ---------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::compute_wall_model_parameters()
{
  if (wallModelAlgDriver_)
    wallModelAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- predict_state() -------------------------------------------------
//--------------------------------------------------------------------------
void
SpecificDissipationRateEquationSystem::predict_state()
{
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto& sdrN = fieldMgr.get_field<double>(
    sdr_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& sdrNp1 = fieldMgr.get_field<double>(
    sdr_->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal());

  const auto& meta = realm_.meta_data();
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part() | meta.aura_part())
    & stk::mesh::selectField(*sdr_);
  nalu_ngp::field_copy(ngpMesh, sel, sdrNp1, sdrN);
}

} // namespace nalu
} // namespace Sierra
