// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <SpalartAllmarasEquationSystem.h>
#include <AlgorithmDriver.h>
#include <AssembleScalarNonConformalSolverAlgorithm.h>
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <AssembleNGPNodeSolverAlgorithm.h>
#include <AuxFunctionAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <FieldFunctions.h>
#include <LinearSolver.h>
#include <LinearSolvers.h>
#include <LinearSystem.h>
#include <KynemaUGFEnv.h>
#include <KynemaUGFParsing.h>
#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>

// node kernels
#include <node_kernels/SANuTildaNodeKernel.h>
#include <node_kernels/NodeKernelUtils.h>
#include <node_kernels/ScalarMassBDFNodeKernel.h>
#include <node_kernels/ScalarGclNodeKernel.h>

// template for kernels
#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

// kernels
#include <AssembleElemSolverAlgorithm.h>

// edge kernels
#include <edge_kernels/ScalarEdgeSolverAlg.h>
#include <edge_kernels/ScalarOpenEdgeKernel.h>

// ngp algorithms
#include <ngp_algorithms/EffDiffFluxCoeffSAAlg.h>
#include <ngp_algorithms/NodalGradAlgDriver.h>
#include <ngp_algorithms/NodalGradEdgeAlg.h>
#include <ngp_algorithms/NodalGradElemAlg.h>
#include <ngp_algorithms/NodalGradBndryElemAlg.h>
#include <ngp_algorithms/WallFuncGeometryAlg.h>

// projected nodal gradient
#include <ProjectedNodalGradientEquationSystem.h>

// overset
#include <overset/UpdateOversetFringeAlgorithmDriver.h>

// ngp
#include "FieldTypeDef.h"
#include "ngp_algorithms/GeometryAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "ngp_utils/NgpFieldBLAS.h"

// stk_util
#include <stk_util/parallel/Parallel.hpp>
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// master elements
#include <master_element/MasterElement.h>
#include <master_element/MasterElementRepo.h>

// basic c++
#include <cmath>
#include <vector>
#include <iomanip>

namespace sierra {
namespace kynema_ugf {

//==========================================================================
// Class Definition
//==========================================================================
// SpalartAllmarasEquationSystem - manages SA one-equation turbulence model
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SpalartAllmarasEquationSystem::SpalartAllmarasEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "SpalartAllmarasEQS", "sa_nu_tilda"),
    managePNG_(realm_.get_consistent_mass_matrix_png("sa_nu_tilda")),
    isInit_(true),
    nuTilda_(NULL),
    dnutdx_(NULL),
    nuTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    minDistanceToWall_(NULL),
    nodalGradAlgDriver_(realm_, "sa_nu_tilda", "dnutdx"),
    projectedNodalGradEqs_(NULL)
{
  dofName_ = "sa_nu_tilda";

  // extract solver name and solver object
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("sa_nu_tilda");
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, realm_.name(), EQ_SA_NUTILDA);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("sa_nu_tilda");
  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "Edge projected nodal gradient for sa_nu_tilda: " << edgeNodalGradient_
    << std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create projected nodal gradient equation system
  if (managePNG_) {
    manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SpalartAllmarasEquationSystem::~SpalartAllmarasEquationSystem() {}

void
SpalartAllmarasEquationSystem::load(const YAML::Node& node)
{
  EquationSystem::load(node);
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::reinitialize_linear_system()
{
  // If this is decoupled overset simulation and the user has requested that the
  // linear system be reused, then do nothing
  if (decoupledOverset_ && linsys_->config().reuseLinSysIfPossible())
    return;

  // delete linsys
  delete linsys_;

  // create new solver
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("sa_nu_tilda");
  LinearSolver* solver = realm_.root()->linearSolvers_->reinitialize_solver(
    solverName, realm_.name(), EQ_SA_NUTILDA);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_nodal_fields(
  const stk::mesh::PartVector& part_vec)
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);

  // register the SA variable (nu_tilda); set it as a restart variable
  nuTilda_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "sa_nu_tilda", numStates));
  stk::mesh::put_field_on_mesh(*nuTilda_, selector, nullptr);
  realm_.augment_restart_variable_list("sa_nu_tilda");

  dnutdx_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "dnutdx"));
  stk::mesh::put_field_on_mesh(*dnutdx_, selector, nDim, nullptr);
  stk::io::set_field_output_type(*dnutdx_, stk::io::FieldOutputType::VECTOR_3D);

  // delta solution for linear solver
  nuTmp_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "nuTmp"));
  stk::mesh::put_field_on_mesh(*nuTmp_, selector, nullptr);

  visc_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, selector, nullptr);

  tvisc_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "turbulent_viscosity"));
  stk::mesh::put_field_on_mesh(*tvisc_, selector, nullptr);

  evisc_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "effective_viscosity_sa"));
  stk::mesh::put_field_on_mesh(*evisc_, selector, nullptr);

  // minimum distance to wall
  minDistanceToWall_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*minDistanceToWall_, selector, nullptr);

  // add to restart field
  realm_.augment_restart_variable_list("minimum_distance_to_wall");

  // make sure all states are properly populated (restart can handle this)
  if (
    numStates > 2 &&
    (!realm_.restarted_simulation() || realm_.support_inconsistent_restart())) {
    ScalarFieldType& nuTildaN = nuTilda_->field_of_state(stk::mesh::StateN);
    ScalarFieldType& nuTildaNp1 =
      nuTilda_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
      realm_, part_vec, &nuTildaNp1, &nuTildaN, 0, 1,
      stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_interior_algorithm(
  stk::mesh::Part* part)
{
  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType& nuTildaNp1 = nuTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dnutdxNone = dnutdx_->field_of_state(stk::mesh::StateNone);

  // non-solver, dnutdx; allow for element-based shifted
  if (!managePNG_) {
    if (edgeNodalGradient_ && realm_.realmUsesEdges_)
      nodalGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
        algType, part, "sa_nutilda_nodal_grad", &nuTildaNp1, &dnutdxNone);
    else
      nodalGradAlgDriver_.register_elem_algorithm<ScalarNodalGradElemAlg>(
        algType, part, "sa_nutilda_nodal_grad", &nuTildaNp1, &dnutdxNone,
        edgeNodalGradient_);
  }

  // solver; interior contribution (advection + diffusion)
  if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {

    std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = NULL;
      if (realm_.realmUsesEdges_) {
        theAlg = new ScalarEdgeSolverAlg(
          realm_, part, this, nuTilda_, dnutdx_, evisc_);
      } else {
        throw std::runtime_error(
          "SA_EQS: Attempt to use non-edge solver algorithm. "
          "Only edge-based scheme is supported.");
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;

      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string>>::iterator isrc =
        realm_.solutionOptions_->elemSrcTermsMap_.find("sa_nu_tilda");
      if (isrc != realm_.solutionOptions_->elemSrcTermsMap_.end()) {
        throw std::runtime_error(
          "SpalartAllmarasElemSrcTerms::Error can not use element source "
          "terms for an edge-based scheme");
      }
    } else {
      itsi->second->partVec_.push_back(part);
    }

    // Check if the user has requested CMM or LMM algorithms; if so, do not
    // include Nodal Mass algorithms
    std::vector<std::string> checkAlgNames = {
      "sa_nu_tilda_time_derivative",
      "lumped_sa_nu_tilda_time_derivative"};
    bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
    auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
    process_ngp_node_kernels(
      solverAlgMap, realm_, part, this,
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg) {
        if (!elementMassAlg)
          nodeAlg.add_kernel<ScalarMassBDFNodeKernel>(
            realm_.bulk_data(), nuTilda_);

        nodeAlg.add_kernel<SANuTildaNodeKernel>(realm_.meta_data());
      },
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg, std::string& srcName) {
        if (srcName == "gcl") {
          nodeAlg.add_kernel<ScalarGclNodeKernel>(
            realm_.bulk_data(), nuTilda_);
          KynemaUGFEnv::self().kynema_ugfOutputP0() << " - " << srcName << std::endl;
        } else
          throw std::runtime_error(
            "SA_EQS: Invalid source term: " + srcName);
      });
  } else {
    throw std::runtime_error("SA_EQS: Element terms not supported");
  }

  // effective diffusive flux coefficient alg for SA
  // For SA: effective diffusivity = (mu + rho * nuTilda) / sigma
  // This uses the exact SA diffusion coefficient, not the approximate
  // (mu + mu_t) / sigma form (since mu_t = rho*nuTilda*fv1 != rho*nuTilda)
  if (!effDiffFluxAlg_) {
    const double sigma = realm_.get_turb_model_constant(TM_saSigma);
    effDiffFluxAlg_.reset(new EffDiffFluxCoeffSAAlg(
      realm_, part, visc_, evisc_, sigma));
  } else {
    effDiffFluxAlg_->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_inflow_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const InflowBoundaryConditionData& inflowBCData)
{
  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType& nuTildaNp1 = nuTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dnutdxNone = dnutdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; nuTilda_bc
  ScalarFieldType* theBcField =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "sa_nu_tilda_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specification
  InflowUserData userData = inflowBCData.userData_;
  SANuTilda saNuTilda = userData.saNuTilda_;
  std::vector<double> userSpec(1);
  userSpec[0] = saNuTilda.saNuTilda_;

  // new it
  ConstantAuxFunction* theAuxFunc =
    new ConstantAuxFunction(0, 1, userSpec);

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

  // copy nuTilda_bc to nuTilda np1...
  CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
    realm_, part, theBcField, &nuTildaNp1, 0, 1, stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver: dnutdx; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sa_nutilda_nodal_grad", &nuTildaNp1, &dnutdxNone,
      edgeNodalGradient_);
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg =
      new DirichletBC(realm_, this, part, &nuTildaNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const OpenBoundaryConditionData& openBCData)
{
  // algorithm type
  const AlgorithmType algType = OPEN;

  ScalarFieldType& nuTildaNp1 = nuTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dnutdxNone = dnutdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; nuTilda_bc
  ScalarFieldType* theBcField =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "sa_nu_tilda_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specification
  OpenUserData userData = openBCData.userData_;
  SANuTilda saNuTilda = userData.saNuTilda_;
  std::vector<double> userSpec(1);
  userSpec[0] = saNuTilda.saNuTilda_;

  // new it
  ConstantAuxFunction* theAuxFunc =
    new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // non-solver: dnutdx; allow for element-based shifted
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sa_nutilda_nodal_grad", &nuTildaNp1, &dnutdxNone,
      edgeNodalGradient_);
  }

  // solver; open bc advection (handles outflow and entrainment)
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
      partTopo, *this, activeKernels, "sa_nu_tilda_open", realm_.meta_data(),
      *realm_.solutionOptions_, nuTilda_, theBcField, dataPreReqs);
  } else {
    throw std::runtime_error(
      "SA_EQS: Attempt to use element open solver algorithm");
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const WallBoundaryConditionData& wallBCData)
{
  // algorithm type
  const AlgorithmType algType = WALL;

  // push mesh part
  wallBcPart_.push_back(part);

  ScalarFieldType& nuTildaNp1 = nuTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dnutdxNone = dnutdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; nuTilda_bc = 0 at wall (standard SA)
  ScalarFieldType* theBcField =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "sa_nu_tilda_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // nuTilda = 0 at wall
  std::vector<double> userSpec(1, 0.0);
  ConstantAuxFunction* theAuxFunc =
    new ConstantAuxFunction(0, 1, userSpec);

  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // copy nuTilda_bc to nuTilda np1...
  CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
    realm_, part, theBcField, &nuTildaNp1, 0, 1, stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver: dnutdx
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sa_nutilda_nodal_grad", &nuTildaNp1, &dnutdxNone,
      edgeNodalGradient_);
  }

  // Dirichlet bc (nuTilda = 0 at wall)
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg =
      new DirichletBC(realm_, this, part, &nuTildaNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }

  // Register wall function geometry for distance-to-wall computation
  auto& assembledWallArea = meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "assembled_wall_area_wf");
  stk::mesh::put_field_on_mesh(assembledWallArea, *part, nullptr);
  auto& assembledWallNormDist = meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "assembled_wall_normal_distance");
  stk::mesh::put_field_on_mesh(assembledWallNormDist, *part, nullptr);
  auto& wallNormDistBip =
    meta_data.declare_field<double>(meta_data.side_rank(), "wall_normal_distance_bip");
  auto* meFC = MasterElementRepo::get_surface_master_element_on_host(partTopo);
  const int numScsBip = meFC->num_integration_points();
  stk::mesh::put_field_on_mesh(wallNormDistBip, *part, numScsBip, nullptr);

  WallUserData userData = wallBCData.userData_;
  bool RANSAblBcApproach = userData.RANSAblBcApproach_;
  RoughnessHeight rough = userData.z0_;
  double z0 = rough.z0_;
  realm_.geometryAlgDriver_->register_wall_func_algorithm<WallFuncGeometryAlg>(
    sierra::kynema_ugf::WALL, part, get_elem_topo(realm_, *part),
    "sa_geometry_wall", RANSAblBcApproach, z0);
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_symmetry_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const SymmetryBoundaryConditionData& /* symmetryBCData */)
{
  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  ScalarFieldType& nuTildaNp1 = nuTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dnutdxNone = dnutdx_->field_of_state(stk::mesh::StateNone);

  // non-solver: dnutdx
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sa_nutilda_nodal_grad", &nuTildaNp1, &dnutdxNone,
      edgeNodalGradient_);
  }
}

//--------------------------------------------------------------------------
//-------- register_abltop_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_abltop_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const ABLTopBoundaryConditionData& /* ablTopBCData */)
{
  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType& nuTildaNp1 = nuTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dnutdxNone = dnutdx_->field_of_state(stk::mesh::StateNone);

  // non-solver: dnutdx
  if (!managePNG_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sa_nutilda_nodal_grad", &nuTildaNp1, &dnutdxNone,
      edgeNodalGradient_);
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_non_conformal_bc(
  stk::mesh::Part* part, const stk::topology& /*theTopo*/)
{
  const AlgorithmType algType = NON_CONFORMAL;

  ScalarFieldType& nuTildaNp1 = nuTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dnutdxNone = dnutdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dnutdx; DG algorithm decides on locations for
  // integration points
  if (edgeNodalGradient_) {
    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "sa_nutilda_nodal_grad", &nuTildaNp1, &dnutdxNone,
      edgeNodalGradient_);
  } else {
    // proceed with DG
    nodalGradAlgDriver_
      .register_legacy_algorithm<AssembleNodalGradNonConformalAlgorithm>(
        algType, part, "sa_nutilda_nodal_grad", &nuTildaNp1, &dnutdxNone);
  }

  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
    AssembleScalarNonConformalSolverAlgorithm* theAlg =
      new AssembleScalarNonConformalSolverAlgorithm(
        realm_, part, this, nuTilda_, evisc_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    itsi->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(nuTilda_);
  equationSystems_.register_overset_field_update(nuTilda_, 1, 1);
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::register_initial_condition_fcn(
  stk::mesh::Part* part,
  const std::map<std::string, std::string>& theNames,
  const std::map<std::string, std::vector<double>>& theParams)
{
  // SA does not currently register any specialized IC functions
  // Default IC from input file is used
  EquationSystem::register_initial_condition_fcn(part, theNames, theParams);
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::solve_and_update()
{
  if (isInit_) {
    compute_projected_nodal_gradient();
    clip_min_distance_to_wall();
    isInit_ = false;
  } else if (realm_.has_mesh_motion()) {
    if (realm_.currentNonlinearIteration_ == 1)
      clip_min_distance_to_wall();
  }

  // compute effective viscosity
  compute_effective_diff_flux_coeff();

  // start the iteration loop
  for (int k = 0; k < maxIterations_; ++k) {

    KynemaUGFEnv::self().kynema_ugfOutputP0()
      << " " << k + 1 << "/" << maxIterations_ << std::setw(15) << std::right
      << userSuppliedName_ << std::endl;

    for (int oi = 0; oi < numOversetIters_; ++oi) {
      // SA assemble, load_complete and solve
      assemble_and_solve(nuTmp_);

      // update
      double timeA = KynemaUGFEnv::self().kynema_ugf_time();
      update_and_clip();
      double timeB = KynemaUGFEnv::self().kynema_ugf_time();
      timerAssemble_ += (timeB - timeA);

      if (decoupledOverset_ && realm_.hasOverset_)
        realm_.overset_field_update(nuTilda_, 1, 1);
    }

    // projected nodal gradient
    compute_projected_nodal_gradient();
  }
}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
SpalartAllmarasEquationSystem::initial_work()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& nuTildaNp1 =
    fieldMgr.get_field<double>(nuTilda_->mesh_meta_data_ordinal());
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*nuTilda_);
  clip_nu_tilda(ngpMesh, sel, nuTildaNp1);
}

void
SpalartAllmarasEquationSystem::post_external_data_transfer_work()
{
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& nuTildaNp1 =
    fieldMgr.get_field<double>(nuTilda_->mesh_meta_data_ordinal());

  const stk::mesh::Selector owned_and_shared =
    (meta.locally_owned_part() | meta.globally_shared_part());
  auto interior_sel = owned_and_shared & stk::mesh::selectField(*nuTilda_);
  clip_nu_tilda(ngpMesh, interior_sel, nuTildaNp1);

  auto nuTildaBCField =
    meta.get_field<double>(stk::topology::NODE_RANK, "sa_nu_tilda_bc");
  if (nuTildaBCField != nullptr) {
    auto bc_sel = owned_and_shared & stk::mesh::selectField(*nuTildaBCField);
    auto ngpNuTildaBC =
      fieldMgr.get_field<double>(nuTildaBCField->mesh_meta_data_ordinal());
    clip_nu_tilda(ngpMesh, bc_sel, ngpNuTildaBC);
  }
}

void
SpalartAllmarasEquationSystem::compute_effective_diff_flux_coeff()
{
  if (effDiffFluxAlg_)
    effDiffFluxAlg_->execute();
}

void
SpalartAllmarasEquationSystem::predict_state()
{
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto& nuTildaN = fieldMgr.get_field<double>(
    nuTilda_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& nuTildaNp1 = fieldMgr.get_field<double>(
    nuTilda_->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal());

  const auto& meta = realm_.meta_data();
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part() |
     meta.aura_part()) &
    stk::mesh::selectField(*nuTilda_);
  nuTildaNp1.sync_to_device();
  kynema_ugf_ngp::field_copy(ngpMesh, sel, nuTildaNp1, nuTildaN);
  nuTildaNp1.modify_on_device();
}

void
SpalartAllmarasEquationSystem::compute_wall_model_parameters()
{
  // SA uses Dirichlet BC at wall (nuTilda = 0), no wall model needed
}

void
SpalartAllmarasEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if (NULL == projectedNodalGradEqs_) {
    projectedNodalGradEqs_ = new ProjectedNodalGradientEquationSystem(
      eqSystems, EQ_PNG_SA_NUTILDA, "dnutdx", "nuTmp", "sa_nu_tilda",
      "PNGradSANuTildaEQS");
  }
  // fill the map for expected boundary condition names
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "sa_nu_tilda");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "sa_nu_tilda");
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "sa_nu_tilda");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "sa_nu_tilda");
}

void
SpalartAllmarasEquationSystem::compute_projected_nodal_gradient()
{
  if (!managePNG_) {
    const double timeA = -KynemaUGFEnv::self().kynema_ugf_time();
    nodalGradAlgDriver_.execute();
    timerMisc_ += (KynemaUGFEnv::self().kynema_ugf_time() + timeA);
  } else {
    projectedNodalGradEqs_->solve_and_update_external();
  }
}

void
SpalartAllmarasEquationSystem::update_and_clip()
{
  using MeshIndex = kynema_ugf_ngp::NGPMeshTraits<>::MeshIndex;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& nuTildaNp1 =
    fieldMgr.get_field<double>(nuTilda_->mesh_meta_data_ordinal());
  const auto& nuTmp =
    fieldMgr.get_field<double>(nuTmp_->mesh_meta_data_ordinal());

  auto* turbViscosity =
    meta.get_field<double>(stk::topology::NODE_RANK, "turbulent_viscosity");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*turbViscosity);

  const double nuTildaMinVal = nuTildaMinValue_;

  nuTildaNp1.sync_to_device();

  kynema_ugf_ngp::run_entity_algorithm(
    "SA::update_and_clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double nuTildaNew = nuTildaNp1.get(mi, 0) + nuTmp.get(mi, 0);
      nuTildaNp1.get(mi, 0) = stk::math::max(nuTildaNew, nuTildaMinVal);
    });

  nuTildaNp1.modify_on_device();
}

void
SpalartAllmarasEquationSystem::clip_nu_tilda(
  const stk::mesh::NgpMesh& ngpMesh,
  const stk::mesh::Selector& sel,
  stk::mesh::NgpField<double>& nuTilda)
{
  nuTilda.sync_to_device();

  const double nuTildaMinVal = nuTildaMinValue_;

  kynema_ugf_ngp::run_entity_algorithm(
    "SA::clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const kynema_ugf_ngp::NGPMeshTraits<>::MeshIndex& mi) {
      const double nuTildaNew = nuTilda.get(mi, 0);
      nuTilda.get(mi, 0) = stk::math::max(nuTildaNew, nuTildaMinVal);
    });
  nuTilda.modify_on_device();
}

void
SpalartAllmarasEquationSystem::clip_min_distance_to_wall()
{
  using MeshIndex = kynema_ugf_ngp::NGPMeshTraits<>::MeshIndex;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& meta = meshInfo.meta();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  if (wallBcPart_.empty())
    return;

  auto& ndtw =
    fieldMgr.get_field<double>(minDistanceToWall_->mesh_meta_data_ordinal());
  const auto& wallNormDist =
    kynema_ugf_ngp::get_ngp_field(meshInfo, "assembled_wall_normal_distance");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectUnion(wallBcPart_);

  ndtw.sync_to_device();
  kynema_ugf_ngp::run_entity_algorithm(
    "SA::clip_ndtw", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double minD = ndtw.get(mi, 0);
      ndtw.get(mi, 0) = stk::math::max(minD, wallNormDist.get(mi, 0));
    });
  ndtw.modify_on_device();

  stk::mesh::parallel_max(realm_.bulk_data(), {minDistanceToWall_});
  if (realm_.hasPeriodic_)
    realm_.periodic_field_max(minDistanceToWall_, 1);
}

} // namespace kynema_ugf
} // namespace sierra
