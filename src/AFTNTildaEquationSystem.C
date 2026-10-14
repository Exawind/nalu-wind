// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include <AFTNTildaEquationSystem.h>
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
#include <KynemaUGFEnv.h>
#include <KynemaUGFParsing.h>
#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>

#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

#include <AssembleElemSolverAlgorithm.h>

#include <edge_kernels/ScalarEdgeSolverAlg.h>
#include <edge_kernels/ScalarOpenEdgeKernel.h>

#include <node_kernels/AFTNTildaNodeKernel.h>
#include <node_kernels/NodeKernelUtils.h>
#include <node_kernels/ScalarMassBDFNodeKernel.h>
#include <node_kernels/ScalarGclNodeKernel.h>

#include "ngp_utils/NgpFieldBLAS.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_algorithms/NodalGradElemAlg.h"
#include "ngp_algorithms/NodalGradBndryElemAlg.h"
#include "ngp_algorithms/EffDiffFluxCoeffAlg.h"

#include <overset/UpdateOversetFringeAlgorithmDriver.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>

#include <stk_io/IossBridge.hpp>

#include <iomanip>

namespace sierra {
namespace kynema_ugf {

AFTNTildaEquationSystem::AFTNTildaEquationSystem(EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "AFTNTildaEQS", "aft_n_tilda"),
    managePNG_(realm_.get_consistent_mass_matrix_png("aft_n_tilda")),
    nTilda_(NULL),
    dntdx_(NULL),
    nTmp_(NULL),
    minDistanceToWall_(NULL),
    dwalldistdx_(NULL),
    nDotV_(NULL),
    dnDotVdx_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    nodalGradAlgDriver_(realm_, "aft_n_tilda", "dntdx"),
    wallDistGradAlgDriver_(realm_, "minimum_distance_to_wall", "dwalldistdx"),
    nDotVGradAlgDriver_(realm_, "nDotV", "dnDotVdx")
{
  dofName_ = "aft_n_tilda";

  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("aft_n_tilda");
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, realm_.name(), EQ_AFT_N_TILDA);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  set_nodal_gradient("aft_n_tilda");
  KynemaUGFEnv::self().kynema_ugfOutputP0()
    << "Edge projected nodal gradient for aft_n_tilda: "
    << edgeNodalGradient_ << std::endl;

  realm_.push_equation_to_systems(this);

  if (managePNG_)
    throw std::runtime_error(
      "AFTNTildaEquationSystem::Error managePNG is not complete");
}

AFTNTildaEquationSystem::~AFTNTildaEquationSystem() = default;

void
AFTNTildaEquationSystem::register_nodal_fields(
  const stk::mesh::PartVector& part_vec)
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);

  nTilda_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "aft_n_tilda", numStates));
  stk::mesh::put_field_on_mesh(*nTilda_, selector, nullptr);
  realm_.augment_restart_variable_list("aft_n_tilda");

  dntdx_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "dntdx"));
  stk::mesh::put_field_on_mesh(*dntdx_, selector, nDim, nullptr);
  stk::io::set_field_output_type(*dntdx_, stk::io::FieldOutputType::VECTOR_3D);

  nTmp_ = &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "nTmp"));
  stk::mesh::put_field_on_mesh(*nTmp_, selector, nullptr);

  minDistanceToWall_ = meta_data.get_field<double>(
    stk::topology::NODE_RANK, "minimum_distance_to_wall");

  dwalldistdx_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "dwalldistdx"));
  stk::mesh::put_field_on_mesh(*dwalldistdx_, selector, nDim, nullptr);
  stk::io::set_field_output_type(
    *dwalldistdx_, stk::io::FieldOutputType::VECTOR_3D);

  nDotV_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "nDotV"));
  stk::mesh::put_field_on_mesh(*nDotV_, selector, nullptr);

  dnDotVdx_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "dnDotVdx"));
  stk::mesh::put_field_on_mesh(*dnDotVdx_, selector, nDim, nullptr);
  stk::io::set_field_output_type(
    *dnDotVdx_, stk::io::FieldOutputType::VECTOR_3D);

  visc_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, selector, nullptr);

  tvisc_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "turbulent_viscosity"));
  stk::mesh::put_field_on_mesh(*tvisc_, selector, nullptr);

  evisc_ = &(meta_data.declare_field<double>(
    stk::topology::NODE_RANK, "effective_viscosity_aft_n"));
  stk::mesh::put_field_on_mesh(*evisc_, selector, nullptr);

  if (
    numStates > 2 &&
    (!realm_.restarted_simulation() || realm_.support_inconsistent_restart())) {
    ScalarFieldType& nTildaN = nTilda_->field_of_state(stk::mesh::StateN);
    ScalarFieldType& nTildaNp1 = nTilda_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
      realm_, part_vec, &nTildaNp1, &nTildaN, 0, 1, stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
}

void
AFTNTildaEquationSystem::register_interior_algorithm(stk::mesh::Part* part)
{
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType& nTildaNp1 = nTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dntdxNone = dntdx_->field_of_state(stk::mesh::StateNone);

  if (edgeNodalGradient_ && realm_.realmUsesEdges_) {
    nodalGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
      algType, part, "aft_ntilda_nodal_grad", &nTildaNp1, &dntdxNone);

    wallDistGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
      algType, part, "wall_dist_nodal_grad", minDistanceToWall_, dwalldistdx_);

    nDotVGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
      algType, part, "ndotv_nodal_grad", nDotV_, dnDotVdx_);
  } else {
    nodalGradAlgDriver_.register_elem_algorithm<ScalarNodalGradElemAlg>(
      algType, part, "aft_ntilda_nodal_grad", &nTildaNp1, &dntdxNone,
      edgeNodalGradient_);

    wallDistGradAlgDriver_.register_elem_algorithm<ScalarNodalGradElemAlg>(
      algType, part, "wall_dist_nodal_grad", minDistanceToWall_, dwalldistdx_,
      edgeNodalGradient_);

    nDotVGradAlgDriver_.register_elem_algorithm<ScalarNodalGradElemAlg>(
      algType, part, "ndotv_nodal_grad", nDotV_, dnDotVdx_, edgeNodalGradient_);
  }

  if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {
    auto itsi = solverAlgDriver_->solverAlgMap_.find(algType);
    if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = NULL;
      if (realm_.realmUsesEdges_) {
        theAlg = new ScalarEdgeSolverAlg(
          realm_, part, this, nTilda_, dntdx_, evisc_);
      } else {
        throw std::runtime_error(
          "AFTNTILDA: Attempt to use non-NGP element solver algorithm");
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;

      auto isrc = realm_.solutionOptions_->elemSrcTermsMap_.find("aft_n_tilda");
      if (isrc != realm_.solutionOptions_->elemSrcTermsMap_.end()) {
        throw std::runtime_error(
          "AFTNTildaEquationSystem::Error can not use element source terms "
          "for an edge-based scheme");
      }
    } else {
      itsi->second->partVec_.push_back(part);
    }

    std::vector<std::string> checkAlgNames = {
      "aft_n_tilda_time_derivative", "lumped_aft_n_tilda_time_derivative"};
    bool elementMassAlg = supp_alg_is_requested(checkAlgNames);

    auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
    process_ngp_node_kernels(
      solverAlgMap, realm_, part, this,
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg) {
        if (!elementMassAlg)
          nodeAlg.add_kernel<ScalarMassBDFNodeKernel>(realm_.bulk_data(), nTilda_);
        nodeAlg.add_kernel<AFTNTildaNodeKernel>(realm_.meta_data());
      },
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg, std::string& srcName) {
        if (srcName == "gcl") {
          nodeAlg.add_kernel<ScalarGclNodeKernel>(realm_.bulk_data(), nTilda_);
        } else {
          throw std::runtime_error(
            "AFTNTILDA: Invalid source term: " + srcName);
        }
      });
  } else {
    throw std::runtime_error("AFTNTILDA: Element terms not supported");
  }

  if (!effDiffFluxAlg_) {
    effDiffFluxAlg_.reset(new EffDiffFluxCoeffAlg(
      realm_, part, visc_, tvisc_, evisc_, 1.0, 1.0,
      realm_.is_turbulent()));
  } else {
    effDiffFluxAlg_->partVec_.push_back(part);
  }
}

void
AFTNTildaEquationSystem::register_inflow_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const InflowBoundaryConditionData& inflowBCData)
{
  const AlgorithmType algType = INFLOW;

  ScalarFieldType& nTildaNp1 = nTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dntdxNone = dntdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  ScalarFieldType* theBcField =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, "aft_n_tilda_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  std::vector<double> userSpec(1, 0.0);
  ConstantAuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);

  InflowUserData userData = inflowBCData.userData_;
  if (userData.externalData_)
    realm_.initCondAlg_.push_back(auxAlg);
  else
    bcDataAlg_.push_back(auxAlg);

  CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
    realm_, part, theBcField, &nTildaNp1, 0, 1, stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "aft_ntilda_nodal_grad", &nTildaNp1, &dntdxNone,
    edgeNodalGradient_);

  wallDistGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "wall_dist_nodal_grad", minDistanceToWall_, dwalldistdx_,
    edgeNodalGradient_);

  nDotVGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "ndotv_nodal_grad", nDotV_, dnDotVdx_, edgeNodalGradient_);

  auto itd = solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg =
      new DirichletBC(realm_, this, part, &nTildaNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }
}

void
AFTNTildaEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology& partTopo,
  const OpenBoundaryConditionData& /*openBCData*/)
{
  const AlgorithmType algType = OPEN;

  ScalarFieldType& nTildaNp1 = nTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dntdxNone = dntdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  ScalarFieldType* theBcField = &(
    meta_data.declare_field<double>(stk::topology::NODE_RANK, "open_aft_n_tilda_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  std::vector<double> userSpec(1, 0.0);
  ConstantAuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "aft_ntilda_nodal_grad", &nTildaNp1, &dntdxNone,
    edgeNodalGradient_);

  wallDistGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "wall_dist_nodal_grad", minDistanceToWall_, dwalldistdx_,
    edgeNodalGradient_);

  nDotVGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "ndotv_nodal_grad", nDotV_, dnDotVdx_, edgeNodalGradient_);

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
      partTopo, *this, activeKernels, "AFTN_open", realm_.meta_data(),
      *realm_.solutionOptions_, nTilda_, theBcField, dataPreReqs);
  } else {
    throw std::runtime_error(
      "AFTNTILDA: Attempt to use non-NGP element open algorithm");
  }
}

void
AFTNTildaEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const WallBoundaryConditionData& /*wallBCData*/)
{
  const AlgorithmType algType = WALL;

  ScalarFieldType& nTildaNp1 = nTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dntdxNone = dntdx_->field_of_state(stk::mesh::StateNone);

  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "aft_ntilda_nodal_grad", &nTildaNp1, &dntdxNone,
    edgeNodalGradient_);

  wallDistGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "wall_dist_nodal_grad", minDistanceToWall_, dwalldistdx_,
    edgeNodalGradient_);

  nDotVGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "ndotv_nodal_grad", nDotV_, dnDotVdx_, edgeNodalGradient_);
}

void
AFTNTildaEquationSystem::register_symmetry_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const SymmetryBoundaryConditionData& /*symmetryBCData*/)
{
  const AlgorithmType algType = SYMMETRY;

  ScalarFieldType& nTildaNp1 = nTilda_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dntdxNone = dntdx_->field_of_state(stk::mesh::StateNone);

  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "aft_ntilda_nodal_grad", &nTildaNp1, &dntdxNone,
    edgeNodalGradient_);

  wallDistGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "wall_dist_nodal_grad", minDistanceToWall_, dwalldistdx_,
    edgeNodalGradient_);

  nDotVGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "ndotv_nodal_grad", nDotV_, dnDotVdx_, edgeNodalGradient_);
}

void
AFTNTildaEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(nTilda_);

  equationSystems_.register_overset_field_update(nTilda_, 1, 1);
}

void
AFTNTildaEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

void
AFTNTildaEquationSystem::reinitialize_linear_system()
{
  if (decoupledOverset_ && linsys_->config().reuseLinSysIfPossible())
    return;

  delete linsys_;

  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("aft_n_tilda");
  LinearSolver* solver = realm_.root()->linearSolvers_->reinitialize_solver(
    solverName, realm_.name(), EQ_AFT_N_TILDA);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

void
AFTNTildaEquationSystem::solve_and_update()
{
  assemble_nodal_gradient();
  compute_effective_diff_flux_coeff();

  for (int k = 0; k < maxIterations_; ++k) {
    KynemaUGFEnv::self().kynema_ugfOutputP0()
      << " " << k + 1 << "/" << maxIterations_ << std::setw(15)
      << std::right << userSuppliedName_ << std::endl;

    for (int oi = 0; oi < numOversetIters_; ++oi) {
      assemble_and_solve(nTmp_);
      update_and_clip();

      if (decoupledOverset_ && realm_.hasOverset_)
        realm_.overset_field_update(nTilda_, 1, 1);
    }

    assemble_nodal_gradient();
  }
}

void
AFTNTildaEquationSystem::update_and_clip()
{
  using MeshIndex = kynema_ugf_ngp::NGPMeshTraits<>::MeshIndex;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto& nTildaNp1 = fieldMgr.get_field<double>(nTilda_->mesh_meta_data_ordinal());
  const auto& nTmp = fieldMgr.get_field<double>(nTmp_->mesh_meta_data_ordinal());
  auto* turbViscosity =
    meta.get_field<double>(stk::topology::NODE_RANK, "turbulent_viscosity");

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*turbViscosity);

  nTildaNp1.sync_to_device();

  kynema_ugf_ngp::run_entity_algorithm(
    "AFTN::update_and_clip", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double nTildaNew = nTildaNp1.get(mi, 0) + nTmp.get(mi, 0);
      nTildaNp1.get(mi, 0) = stk::math::max(nTildaNew, 0.0);
    });

  nTildaNp1.modify_on_device();
}

void
AFTNTildaEquationSystem::assemble_nodal_gradient()
{
  using MeshIndex = kynema_ugf_ngp::NGPMeshTraits<>::MeshIndex;

  const auto& meta = realm_.meta_data();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const int ndim = meta.spatial_dimension();

  const double timeA = -KynemaUGFEnv::self().kynema_ugf_time();
  nodalGradAlgDriver_.execute();
  wallDistGradAlgDriver_.execute();

  auto& dwalldistdx =
    fieldMgr.get_field<double>(dwalldistdx_->mesh_meta_data_ordinal());
  auto& nDotV = fieldMgr.get_field<double>(nDotV_->mesh_meta_data_ordinal());
  const std::string velName =
    realm_.does_mesh_move() ? "velocity_rtm" : "velocity";
  auto& vel = fieldMgr.get_field<double>(get_field_ordinal(meta, velName));

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    (stk::mesh::selectField(*dwalldistdx_));

  dwalldistdx.sync_to_device();
  nDotV.sync_to_device();
  vel.sync_to_device();

  kynema_ugf_ngp::run_entity_algorithm(
    "AFTNTilda::compute_nDotV", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      double gradMag = 0.0;
      for (int d = 0; d < ndim; ++d)
        gradMag += dwalldistdx.get(mi, d) * dwalldistdx.get(mi, d);
      gradMag = stk::math::sqrt(gradMag);

      for (int d = 0; d < ndim; ++d)
        dwalldistdx.get(mi, d) /= stk::math::max(gradMag, 1.0e-16);

      double nDotV_tmp = 0.0;
      for (int d = 0; d < ndim; ++d)
        nDotV_tmp +=
          dwalldistdx.get(mi, d) * vel.get(mi, d);
      nDotV.get(mi, 0) = nDotV_tmp;
    });

  dwalldistdx.modify_on_device();
  nDotV.modify_on_device();

  nDotVGradAlgDriver_.execute();
  timerMisc_ += (KynemaUGFEnv::self().kynema_ugf_time() + timeA);
}

void
AFTNTildaEquationSystem::compute_effective_diff_flux_coeff()
{
  const double timeA = -KynemaUGFEnv::self().kynema_ugf_time();
  effDiffFluxAlg_->execute();
  timerMisc_ += (KynemaUGFEnv::self().kynema_ugf_time() + timeA);
}

void
AFTNTildaEquationSystem::predict_state()
{
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto& nTildaN = fieldMgr.get_field<double>(
    nTilda_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& nTildaNp1 = fieldMgr.get_field<double>(
    nTilda_->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal());

  const auto& meta = realm_.meta_data();
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part() |
     meta.aura_part()) &
    stk::mesh::selectField(*nTilda_);
  kynema_ugf_ngp::field_copy(ngpMesh, sel, nTildaNp1, nTildaN);
  nTildaNp1.modify_on_device();
}

} // namespace kynema_ugf
} // namespace sierra
