// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "FieldTypeDef.h"
#include <TurbKineticEnergyEquationSystem.h>
#include <AlgorithmDriver.h>
#include <AssembleScalarNonConformalSolverAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
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
#include <ProjectedNodalGradientEquationSystem.h>
#include <Realm.h>
#include <Realms.h>
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

// UT Austin Hybrid AMS kernel
#include <node_kernels/TKESSTAMSNodeKernel.h>

// edge kernels
#include <edge_kernels/ScalarEdgeSolverAlg.h>
#include <edge_kernels/ScalarOpenEdgeKernel.h>

// node kernels
#include <node_kernels/NodeKernelUtils.h>
#include <node_kernels/ScalarMassBDFNodeKernel.h>
#include <node_kernels/ScalarGclNodeKernel.h>
#include <node_kernels/TKEKsgsNodeKernel.h>
#include <node_kernels/TKESSTDESNodeKernel.h>
#include <node_kernels/TKESSTIDDESNodeKernel.h>
#include <node_kernels/TKESSTNodeKernel.h>
#include <node_kernels/TKESSTLRNodeKernel.h>
#include <node_kernels/TKEKENodeKernel.h>
#include <node_kernels/TKERodiNodeKernel.h>
#include <node_kernels/TKEKONodeKernel.h>

#include <node_kernels/TKESSTBLTM2015NodeKernel.h>
#include <node_kernels/TKESSTIDDESBLTM2015NodeKernel.h>

// ngp
#include <ngp_utils/NgpLoopUtils.h>
#include <ngp_utils/NgpTypes.h>
#include <ngp_utils/NgpFieldBLAS.h>
#include <ngp_utils/NgpFieldManager.h>
#include <ngp_algorithms/NodalGradEdgeAlg.h>
#include <ngp_algorithms/NodalGradElemAlg.h>
#include <ngp_algorithms/NodalGradBndryElemAlg.h>
#include <ngp_algorithms/EffDiffFluxCoeffAlg.h>
#include <ngp_algorithms/EffSSTDiffFluxCoeffAlg.h>
#include <ngp_algorithms/TKEWallFuncAlg.h>

// deprecated

#include <overset/UpdateOversetFringeAlgorithmDriver.h>

// stk_util
#include <stk_util/parallel/Parallel.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpMesh.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// nalu utility
#include <utils/StkHelpers.h>

namespace sierra {
namespace nalu {

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
  : EquationSystem(eqSystems, "TurbKineticEnergyEQS", "turbulent_ke"),
    managePNG_(realm_.get_consistent_mass_matrix_png("turbulent_ke")),
    tke_(NULL),
    dkdx_(NULL),
    kTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    nodalGradAlgDriver_(realm_, "turbulent_ke", "dkdx"),
    turbulenceModel_(realm_.solutionOptions_->turbulenceModel_),
    projectedNodalGradEqs_(NULL),
    isInit_(true)
{
  dofName_ = "turbulent_ke";

  // extract solver name and solver object
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("turbulent_ke");
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, realm_.name(), EQ_TURBULENT_KE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("turbulent_ke");
  NaluEnv::self().naluOutputP0()
    << "Edge projected nodal gradient for turbulent_ke: " << edgeNodalGradient_
    << std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // sanity check on turbulence model
  if (!check_for_valid_turblence_model(turbulenceModel_)) {
    throw std::runtime_error(
      "User has requested TurbKinEnergyEqs, however, turbulence model is not "
      "KSGS, SST, SSTLR, SST_DES, SST_IDDES, SST_AMS, KE, or KO");
  }

  // create projected nodal gradient equation system
  if (managePNG_) {
    manage_projected_nodal_gradient(eqSystems);
  }
}

bool
TurbKineticEnergyEquationSystem::check_for_valid_turblence_model(
  TurbulenceModel turbModel)
{
  switch (turbModel) {
  case TurbulenceModel::SST:
  case TurbulenceModel::SSTLR:
  case TurbulenceModel::KSGS:
  case TurbulenceModel::SST_DES:
  case TurbulenceModel::SST_AMS:
  case TurbulenceModel::SST_IDDES:
  case TurbulenceModel::KE:
  case TurbulenceModel::KO:
    return true;
  default:
    return false;
  }
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_nodal_fields(
  const stk::mesh::PartVector& part_vec)
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  tke_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "turbulent_ke", numStates));
  stk::mesh::put_field_on_mesh(*tke_, selector, nullptr);
  realm_.augment_restart_variable_list("turbulent_ke");

  dkdx_ = &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "dkdx"));
  stk::mesh::put_field_on_mesh(*dkdx_, selector, nDim, nullptr);
  stk::io::set_field_output_type(*dkdx_, stk::io::FieldOutputType::VECTOR_3D);

  // delta solution for linear solver; share delta since this is a split system
  kTmp_ = &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "pTmp"));
  stk::mesh::put_field_on_mesh(*kTmp_, selector, nullptr);

  visc_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, selector, nullptr);

  tvisc_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "turbulent_viscosity"));
  stk::mesh::put_field_on_mesh(*tvisc_, selector, nullptr);

  evisc_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "effective_viscosity_tke"));
  stk::mesh::put_field_on_mesh(*evisc_, selector, nullptr);

  // make sure all states are properly populated (restart can handle this)
  if (
    numStates > 2 &&
    (!realm_.restarted_simulation() || realm_.support_inconsistent_restart())) {
    ScalarFieldType& tkeN = tke_->field_of_state(stk::mesh::StateN);
    ScalarFieldType& tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
      realm_, part_vec, &tkeNp1, &tkeN, 0, 1, stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_interior_algorithm(
  stk::mesh::Part* part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType& tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  // non-solver, dkdx; allow for element-based shifted
  if (!managePNG_) {
    if (edgeNodalGradient_ && realm_.realmUsesEdges_)
      nodalGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
        algType, part, "tke_nodal_grad", &tkeNp1, &dkdxNone);
    else
      nodalGradAlgDriver_.register_elem_algorithm<ScalarNodalGradElemAlg>(
        algType, part, "tke_nodal_grad", &tkeNp1, &dkdxNone,
        edgeNodalGradient_);
  }

  // solver; interior contribution (advection + diffusion)
  if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {

    std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = NULL;
      if (realm_.realmUsesEdges_) {
        const bool useAvgMdot =
          (turbulenceModel_ == TurbulenceModel::SST_AMS) ? true : false;
        theAlg = new ScalarEdgeSolverAlg(
          realm_, part, this, tke_, dkdx_, evisc_, useAvgMdot);
      } else {
        throw std::runtime_error(
          "TKEEQS: Attempt to use non-NGP element solver algorithm");
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;

      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string>>::iterator isrc =
        realm_.solutionOptions_->elemSrcTermsMap_.find("turbulent_ke");
      if (isrc != realm_.solutionOptions_->elemSrcTermsMap_.end()) {
        throw std::runtime_error(
          "TurbKineticEnergyElemSrcTerms::Error can not use element source "
          "terms for an edge-based scheme");
      }
    } else {
      itsi->second->partVec_.push_back(part);
    }

    // Check if the user has requested CMM or LMM algorithms; if so, do not
    // include Nodal Mass algorithms
    std::vector<std::string> checkAlgNames = {
      "turbulent_ke_time_derivative", "lumped_turbulent_ke_time_derivative"};
    bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
    auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
    process_ngp_node_kernels(
      solverAlgMap, realm_, part, this,
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg) {
        if (!elementMassAlg)
          nodeAlg.add_kernel<ScalarMassBDFNodeKernel>(realm_.bulk_data(), tke_);

        switch (turbulenceModel_) {
        case TurbulenceModel::KSGS:
          nodeAlg.add_kernel<TKEKsgsNodeKernel>(realm_.meta_data());
          break;
        case TurbulenceModel::SST:
          if (!realm_.solutionOptions_->gammaEqActive_) {
            nodeAlg.add_kernel<TKESSTNodeKernel>(realm_.meta_data());
          } else {
            nodeAlg.add_kernel<TKESSTBLTM2015NodeKernel>(realm_.meta_data());
          }
          break;
        case TurbulenceModel::SSTLR:
          nodeAlg.add_kernel<TKESSTLRNodeKernel>(realm_.meta_data());
          break;
        case TurbulenceModel::SST_DES:
          nodeAlg.add_kernel<TKESSTDESNodeKernel>(realm_.meta_data());
          break;
        case TurbulenceModel::SST_AMS:
          nodeAlg.add_kernel<TKESSTAMSNodeKernel>(
            realm_.meta_data(),
            realm_.solutionOptions_->get_coordinates_name());
          break;
        case TurbulenceModel::SST_IDDES:
          if (!realm_.solutionOptions_->gammaEqActive_) {
            nodeAlg.add_kernel<TKESSTIDDESNodeKernel>(realm_.meta_data());
          } else {
            nodeAlg.add_kernel<TKESSTIDDESBLTM2015NodeKernel>(
              realm_.meta_data());
          }
          break;
        case TurbulenceModel::KE:
          nodeAlg.add_kernel<TKEKENodeKernel>(realm_.meta_data());
          break;
        case TurbulenceModel::KO:
          nodeAlg.add_kernel<TKEKONodeKernel>(realm_.meta_data());
          break;
        default:
          std::runtime_error("TKEEqSys: Invalid turbulence model");
          break;
        }
      },
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg, std::string& srcName) {
        if (srcName == "rodi") {
          nodeAlg.add_kernel<TKERodiNodeKernel>(
            realm_.meta_data(), *realm_.solutionOptions_);
        } else if (srcName == "gcl") {
          nodeAlg.add_kernel<ScalarGclNodeKernel>(realm_.bulk_data(), tke_);
        } else
          throw std::runtime_error("TKEEqSys: Invalid source term " + srcName);

        NaluEnv::self().naluOutputP0() << " -  " << srcName << std::endl;
      });
  } else {
    throw std::runtime_error("TKEEQS: Element terms not supported");
  }

  // effective viscosity alg
  if (!effDiffFluxCoeffAlg_) {
    switch (turbulenceModel_) {
    case TurbulenceModel::KSGS: {
      const double lamSc = realm_.get_lam_schmidt(tke_->name());
      const double turbSc = realm_.get_turb_schmidt(tke_->name());
      effDiffFluxCoeffAlg_.reset(new EffDiffFluxCoeffAlg(
        realm_, part, visc_, tvisc_, evisc_, lamSc, turbSc,
        realm_.is_turbulent()));
      break;
    }
    case TurbulenceModel::SST:
    case TurbulenceModel::SSTLR:
    case TurbulenceModel::SST_DES:
    case TurbulenceModel::SST_AMS:
    case TurbulenceModel::SST_IDDES: {
      const double sigmaKOne = realm_.get_turb_model_constant(TM_sigmaKOne);
      const double sigmaKTwo = realm_.get_turb_model_constant(TM_sigmaKTwo);
      effDiffFluxCoeffAlg_.reset(new EffSSTDiffFluxCoeffAlg(
        realm_, part, visc_, tvisc_, evisc_, sigmaKOne, sigmaKTwo));
      break;
    }
    case TurbulenceModel::KO: {
      effDiffFluxCoeffAlg_.reset(new EffDiffFluxCoeffAlg(
        realm_, part, visc_, tvisc_, evisc_, 1.0, 2.0, realm_.is_turbulent()));
      break;
    }
    case TurbulenceModel::KE: {
      const double sigmaK = realm_.get_turb_model_constant(TM_sigmaK);
      effDiffFluxCoeffAlg_.reset(new EffDiffFluxCoeffAlg(
        realm_, part, visc_, tvisc_, evisc_, 1.0, sigmaK,
        realm_.is_turbulent()));
      break;
    }

    default:
      throw std::runtime_error("Unsupported turbulence model in TurbKe");
    }
  } else {
    effDiffFluxCoeffAlg_->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_inflow_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const InflowBoundaryConditionData& inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType& tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; tke_bc
  ScalarFieldType* theBcField =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "tke_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  TurbKinEnergy tke = userData.tke_;
  std::vector<double> userSpec(1);
  userSpec[0] = tke.turbKinEnergy_;

  // new it
  ConstantAuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

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

  // copy tke_bc to turbulent_ke np1...
  CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
    realm_, part, theBcField, &tkeNp1, 0, 1, stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dkdx; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "tke_nodal_grad", &tkeNp1, &dkdxNone, edgeNodalGradient_);
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg =
      new DirichletBC(realm_, this, part, &tkeNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const OpenBoundaryConditionData& openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  ScalarFieldType& tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; tke_bc
  ScalarFieldType* theBcField =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "open_tke_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  TurbKinEnergy tke = userData.tke_;
  std::vector<double> userSpec(1);
  userSpec[0] = tke.turbKinEnergy_;

  // new it
  ConstantAuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // non-solver; dkdx; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "tke_nodal_grad", &tkeNp1, &dkdxNone, edgeNodalGradient_);
  }

  if (realm_.realmUsesEdges_) {
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    AssembleElemSolverAlgorithm* elemSolverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(elemSolverAlg, solverAlgWasBuilt) =
      build_or_add_part_to_face_bc_solver_alg(
        *this, *part, solverAlgMap, "open");

    auto& dataPreReqs = elemSolverAlg->dataNeededByKernels_;
    auto& activeKernels = elemSolverAlg->activeKernels_;

    build_face_topo_kernel_automatic<ScalarOpenEdgeKernel>(
      partTopo, *this, activeKernels, "turbulent_ke_open", realm_.meta_data(),
      *realm_.solutionOptions_, tke_, theBcField, dataPreReqs);
  } else {
    throw std::runtime_error(
      "TKEEQS: Attempt to use element open solver algorithm");
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const WallBoundaryConditionData& wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  // np1
  ScalarFieldType& tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; tke_bc
  ScalarFieldType* theBcField =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "tke_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  std::string tkeName = "turbulent_ke";
  const bool tkeSpecified = bc_data_specified(userData, tkeName);
  bool wallFunctionApproach = userData.wallFunctionApproach_;

  // determine if using RANS for ABL
  bool RANSAblBcApproach = userData.RANSAblBcApproach_;

  if (tkeSpecified && wallFunctionApproach) {
    NaluEnv::self().naluOutputP0()
      << "Both wall function and tke specified; will go with dirichlet"
      << std::endl;
    wallFunctionApproach = false;
  }

  if (wallFunctionApproach || RANSAblBcApproach) {
    // need to register the assembles wall value for tke; can not share with
    // tke_bc
    ScalarFieldType* theAssembledField = &(meta_data.declare_field<double>(
      stk::topology::NODE_RANK, "wall_model_tke_bc"));
    stk::mesh::put_field_on_mesh(*theAssembledField, *part, nullptr);

    if (!wallFuncAlgDriver_)
      wallFuncAlgDriver_.reset(new TKEWallFuncAlgDriver(realm_));

    wallFuncAlgDriver_->register_face_algorithm<TKEWallFuncAlg>(
      algType, part, "tke_wall_func");
  } else if (tkeSpecified) {

    // FIXME: Generalize for constant vs function

    // extract data
    std::vector<double> userSpec(1);
    TurbKinEnergy tke = userData.tke_;
    userSpec[0] = tke.turbKinEnergy_;

    // new it
    ConstantAuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

    // bc data alg
    AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
      realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);

    // copy tke_bc to tke np1...
    CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
      realm_, part, theBcField, &tkeNp1, 0, 1, stk::topology::NODE_RANK);
    bcDataMapAlg_.push_back(theCopyAlg);

  } else {
    throw std::runtime_error(
      "TKE active with wall bc, however, no value of "
      "tke or wall function specified");
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg =
      new DirichletBC(realm_, this, part, &tkeNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }

  // non-solver; dkdx; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "tke_nodal_grad", &tkeNp1, &dkdxNone, edgeNodalGradient_);
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_symmetry_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const SymmetryBoundaryConditionData& /* symmetryBCData */)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  // np1
  ScalarFieldType& tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dkdx; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "tke_nodal_grad", &tkeNp1, &dkdxNone, edgeNodalGradient_);
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::register_non_conformal_bc(
  stk::mesh::Part* part, const stk::topology& /*theTopo*/)
{

  const AlgorithmType algType = NON_CONFORMAL;

  // np1
  ScalarFieldType& tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dkdxNone = dkdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dkdx; DG algorithm decides on locations for
  // integration points
  if (!managePNG_) {
    if (edgeNodalGradient_) {
      nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
        algType, part, "tke_nodal_grad", &tkeNp1, &dkdxNone,
        edgeNodalGradient_);
    } else {
      // proceed with DG
      nodalGradAlgDriver_
        .register_legacy_algorithm<AssembleNodalGradNonConformalAlgorithm>(
          algType, part, "tke_nodal_grad", &tkeNp1, &dkdxNone);
    }
  }

  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
    AssembleScalarNonConformalSolverAlgorithm* theAlg =
      new AssembleScalarNonConformalSolverAlgorithm(
        realm_, part, this, tke_, evisc_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
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

  equationSystems_.register_overset_field_update(tke_, 1, 1);
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
  // If this is decoupled overset simulation and the user has requested that the
  // linear system be reused, then do nothing
  if (decoupledOverset_ && linsys_->config().reuseLinSysIfPossible())
    return;

  // delete linsys
  delete linsys_;

  // create new solver
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("turbulent_ke");
  LinearSolver* solver = realm_.root()->linearSolvers_->reinitialize_solver(
    solverName, realm_.name(), EQ_TURBULENT_KE);
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
  if (turbulenceModel_ != TurbulenceModel::KSGS)
    return;

  // compute dk/dx
  if (isInit_) {
    compute_projected_nodal_gradient();
    isInit_ = false;
  }

  // compute effective viscosity
  compute_effective_diff_flux_coeff();

  // deal with any special wall function approach
  compute_wall_model_parameters();

  // start the iteration loop
  for (int k = 0; k < maxIterations_; ++k) {

    NaluEnv::self().naluOutputP0()
      << " " << k + 1 << "/" << maxIterations_ << std::setw(15) << std::right
      << userSuppliedName_ << std::endl;

    for (int oi = 0; oi < numOversetIters_; ++oi) {
      // tke assemble, load_complete and solve
      assemble_and_solve(kTmp_);

      // update
      double timeA = NaluEnv::self().nalu_time();
      update_and_clip();
      double timeB = NaluEnv::self().nalu_time();
      timerAssemble_ += (timeB - timeA);

      if (decoupledOverset_ && realm_.hasOverset_)
        realm_.overset_field_update(tke_, 1, 1);
    }

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
  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using MeshIndex = typename Traits::MeshIndex;

  // do not let the user specify a negative field
  const double clipValue = 1.0e-16;
  const auto& meta = realm_.meta_data();
  const auto& ngpMesh = realm_.ngp_mesh();
  auto ngpTke = realm_.ngp_field_manager().get_field<double>(
    tke_->mesh_meta_data_ordinal());

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*tke_);

  ngpTke.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "clip_tke", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      if (ngpTke.get(mi, 0) < 0.0)
        ngpTke.get(mi, 0) = clipValue;
    });

  ngpTke.modify_on_device();
}

void
TurbKineticEnergyEquationSystem::post_external_data_transfer_work()
{
  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using MeshIndex = typename Traits::MeshIndex;

  // do not let the user specify a negative field
  const double clipValue = 1.0e-16;
  const auto& meta = realm_.meta_data();
  const auto& ngpMesh = realm_.ngp_mesh();
  auto ngpTke = realm_.ngp_field_manager().get_field<double>(
    tke_->mesh_meta_data_ordinal());

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*tke_);

  ngpTke.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "clip_tke", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      if (ngpTke.get(mi, 0) < 0.0)
        ngpTke.get(mi, 0) = clipValue;
    });
  ngpTke.modify_on_device();

  auto* tkeBCField = meta.get_field<double>(stk::topology::NODE_RANK, "tke_bc");
  if (tkeBCField != nullptr) {
    const stk::mesh::Selector bc_sel =
      (meta.locally_owned_part() | meta.globally_shared_part()) &
      stk::mesh::selectField(*tkeBCField);

    auto ngpTkeBC = realm_.ngp_field_manager().get_field<double>(
      tkeBCField->mesh_meta_data_ordinal());
    ngpTkeBC.sync_to_device();
    nalu_ngp::run_entity_algorithm(
      "clip_tke_bc", ngpMesh, stk::topology::NODE_RANK, bc_sel,
      KOKKOS_LAMBDA(const MeshIndex& mi) {
        if (ngpTkeBC.get(mi, 0) < 0.0)
          ngpTkeBC.get(mi, 0) = clipValue;
      });
    ngpTkeBC.modify_on_device();
  }
}

//--------------------------------------------------------------------------
//-------- compute_effective_flux_coeff() ----------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::compute_effective_diff_flux_coeff()
{
  const double timeA = NaluEnv::self().nalu_time();
  effDiffFluxCoeffAlg_->execute();
  timerMisc_ += (NaluEnv::self().nalu_time() - timeA);
}

//--------------------------------------------------------------------------
//-------- compute_wall_model_parameters() ----------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::compute_wall_model_parameters()
{
  if (wallFuncAlgDriver_)
    wallFuncAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- update_and_clip() -----------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::update_and_clip()
{
  using Traits = nalu_ngp::NGPMeshTraits<>;
  const double clipValue = 1.0e-16;
  size_t numClip = 0;

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  stk::mesh::Selector sel =
    (meta_data.locally_owned_part() | meta_data.globally_shared_part()) &
    stk::mesh::selectField(*tke_);

  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto ngpKTmp =
    fieldMgr.get_field<double>(kTmp_->mesh_meta_data_ordinal());
  auto ngpTke = fieldMgr.get_field<double>(tke_->mesh_meta_data_ordinal());

  ngpTke.sync_to_device();

  nalu_ngp::run_entity_par_reduce(
    "tke_update_and_clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& mi, size_t& nClip) {
      const double tmp = ngpTke.get(mi, 0) + ngpKTmp.get(mi, 0);
      if (tmp < 0.0) {
        ngpTke.get(mi, 0) = clipValue;
        nClip++;
      } else {
        ngpTke.get(mi, 0) = tmp;
      }
    },
    numClip);
  ngpTke.modify_on_device();

  // parallel assemble clipped value
  if (NaluEnv::self().debug()) {
    size_t g_numClip = 0;
    stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
    stk::all_reduce_sum(comm, &numClip, &g_numClip, 1);

    if (g_numClip > 0) {
      NaluEnv::self().naluOutputP0()
        << "tke clipped " << g_numClip << " times " << std::endl;
    }
  }
}

void
TurbKineticEnergyEquationSystem::predict_state()
{
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto& tkeN = fieldMgr.get_field<double>(
    tke_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& tkeNp1 = fieldMgr.get_field<double>(
    tke_->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal());

  const auto& meta = realm_.meta_data();
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part() |
     meta.aura_part()) &
    stk::mesh::selectField(*tke_);
  tkeNp1.sync_to_device();
  nalu_ngp::field_copy(ngpMesh, sel, tkeNp1, tkeN);
  tkeNp1.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if (NULL == projectedNodalGradEqs_) {
    projectedNodalGradEqs_ = new ProjectedNodalGradientEquationSystem(
      eqSystems, EQ_PNG_TKE, "dkdx", "qTmp", "turbulent_ke", "PNGradTkeEQS");
  }
  // fill the map for expected boundary condition names; can be more complex...
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "turbulent_ke");
  projectedNodalGradEqs_->set_data_map(
    WALL_BC, "turbulent_ke"); // wall function...
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "turbulent_ke");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "turbulent_ke");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient()
//---------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyEquationSystem::compute_projected_nodal_gradient()
{
  if (!managePNG_) {
    const double timeA = -NaluEnv::self().nalu_time();
    nodalGradAlgDriver_.execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  } else {
    projectedNodalGradEqs_->solve_and_update_external();
  }
}

} // namespace nalu
} // namespace sierra
