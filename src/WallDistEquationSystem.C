// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "WallDistEquationSystem.h"

#include "AssembleWallDistNonConformalAlgorithm.h"
#include "AuxFunction.h"
#include "AuxFunctionAlgorithm.h"
#include "ConstantAuxFunction.h"
#include "CopyFieldAlgorithm.h"
#include "DirichletBC.h"
#include "Enums.h"
#include "ElemDataRequests.h"
#include "EquationSystem.h"
#include "EquationSystems.h"
#include "LinearSolver.h"
#include "LinearSolvers.h"
#include "LinearSystem.h"
#include "NaluParsing.h"
#include "NonConformalManager.h"
#include "Realm.h"
#include "Realms.h"
#include "Simulation.h"
#include "SolutionOptions.h"
#include "SolverAlgorithm.h"
#include "SolverAlgorithmDriver.h"

#include "kernel/WallDistElemKernel.h"
#include "kernel/KernelBuilder.h"

// edge kernels
#include "edge_kernels/WallDistEdgeSolverAlg.h"

// node kernels
#include "AssembleNGPNodeSolverAlgorithm.h"
#include "node_kernels/NodeKernelUtils.h"
#include "node_kernels/WallDistNodeKernel.h"

// algorithms
#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_algorithms/NodalGradElemAlg.h"
#include "ngp_algorithms/NodalGradBndryElemAlg.h"
#include "ngp_algorithms/NgpAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"

#include "overset/UpdateOversetFringeAlgorithmDriver.h"
#include "overset/AssembleOversetWallDistAlgorithm.h"

#include "stk_mesh/base/Part.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_topology/topology.hpp"

#include <cmath>

namespace sierra {
namespace nalu {

WallDistEquationSystem::WallDistEquationSystem(EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "WallDistEQS", "ndtw"),
    nodalGradAlgDriver_(realm_, "dwalldistdx"),
    managePNG_(realm_.get_consistent_mass_matrix_png("ndtw"))
{
  if (managePNG_)
    throw std::runtime_error(
      "Consistent mass matrix PNG is not available for WallDistEquationSystem");

  auto solverName = eqSystems.get_solver_block_name("ndtw");
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, realm_.name(), EQ_WALL_DISTANCE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  NaluEnv::self().naluOutputP0()
    << "Edge projected nodal gradient for minimum distance to wall: "
    << edgeNodalGradient_ << std::endl;

  realm_.push_equation_to_systems(this);
}

WallDistEquationSystem::~WallDistEquationSystem() {}

void
WallDistEquationSystem::load(const YAML::Node& node)
{
  EquationSystem::load(node);

  get_if_present(node, "update_frequency", updateFreq_, updateFreq_);
  get_if_present(
    node, "force_init_on_restart", forceInitOnRestart_, forceInitOnRestart_);

  bool exchangeFringeData = true;
  get_if_present(
    node, "exchange_fringe_data", exchangeFringeData, exchangeFringeData);
  resetOversetRows_ = exchangeFringeData;
}

void
WallDistEquationSystem::initial_work()
{
  EquationSystem::initial_work();

  solve_and_update();
}

void
WallDistEquationSystem::register_nodal_fields(
  const stk::mesh::PartVector& part_vec)
{
  auto& meta = realm_.meta_data();
  const int nDim = meta.spatial_dimension();
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);

  wallDistPhi_ = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "wall_distance_phi"));
  stk::mesh::put_field_on_mesh(*wallDistPhi_, selector, nullptr);

  dphidx_ = &(meta.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "dwalldistdx"));
  stk::mesh::put_field_on_mesh(*dphidx_, selector, nDim, nullptr);

  wallDistance_ = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*wallDistance_, selector, nullptr);

  coordinates_ = &(meta.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name()));
  stk::mesh::put_field_on_mesh(*coordinates_, selector, nDim, nullptr);

  const int numVolStates =
    realm_.does_mesh_move() ? realm_.number_of_states() : 1;
  dualNodalVolume_ = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume", numVolStates));
  stk::mesh::put_field_on_mesh(*dualNodalVolume_, selector, nullptr);
}

void
WallDistEquationSystem::register_edge_fields(stk::mesh::Part* part)
{
  auto& meta = realm_.meta_data();

  if (realm_.realmUsesEdges_) {
    const int nDim = meta.spatial_dimension();
    edgeAreaVec_ = &(meta.declare_field<VectorFieldType>(
      stk::topology::EDGE_RANK, "edge_area_vector"));
    stk::mesh::put_field_on_mesh(*edgeAreaVec_, *part, nDim, nullptr);
  }
}

void
WallDistEquationSystem::register_element_fields(
  stk::mesh::Part* part, const stk::topology&)
{
  if (realm_.query_for_overset()) {
    auto& meta = realm_.meta_data();
    GenericFieldType& intersectedElement = meta.declare_field<GenericFieldType>(
      stk::topology::ELEMENT_RANK, "intersected_element");
    stk::mesh::put_field_on_mesh(intersectedElement, *part, 1, nullptr);
  }
}

void
WallDistEquationSystem::register_interior_algorithm(stk::mesh::Part* part)
{
  const AlgorithmType algType = INTERIOR;

  auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
  auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

  // Set up dphi/dx calculation algorithms
  if (edgeNodalGradient_ && realm_.realmUsesEdges_)
    nodalGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
      algType, part, "nodal_grad", &wPhiNp1, &dPhiDxNone);
  else
    nodalGradAlgDriver_.register_elem_algorithm<ScalarNodalGradElemAlg>(
      algType, part, "nodal_grad", &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);

  // Solver algorithms
  if (realm_.realmUsesEdges_) {
    auto it = solverAlgDriver_->solverAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = nullptr;
      theAlg = new WallDistEdgeSolverAlg(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    } else {
      it->second->partVec_.push_back(part);
    }
  } else {
    stk::topology partTopo = part->topology();
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(solverAlg, solverAlgWasBuilt) =
      build_or_add_part_to_solver_alg(*this, *part, solverAlgMap);

    if (solverAlgWasBuilt) {
      ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
      auto& activeKernels = solverAlg->activeKernels_;
      Kernel* compKernel = build_topo_kernel<WallDistElemKernel>(
        partTopo, realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);
      activeKernels.push_back(compKernel);
    }
  }

  if (realm_.realmUsesEdges_) {
    process_ngp_node_kernels(
      solverAlgDriver_->solverAlgMap_, realm_, part, this,
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg) {
        nodeAlg.add_kernel<WallDistNodeKernel>(realm_.bulk_data());
      },
      [&](AssembleNGPNodeSolverAlgorithm&, std::string&) {
        // No user defined kernels available
      });
  }
}

void
WallDistEquationSystem::register_inflow_bc(
  stk::mesh::Part* part,
  const stk::topology&,
  const InflowBoundaryConditionData&)
{
  const AlgorithmType algType = INFLOW;

  auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
  auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

  // Set up dphi/dx calculation algorithms
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "nodal_grad", &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);
}

void
WallDistEquationSystem::register_open_bc(
  stk::mesh::Part* part, const stk::topology&, const OpenBoundaryConditionData&)
{
  const AlgorithmType algType = OPEN;

  auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
  auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

  // Set up dphi/dx calculation algorithms
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "nodal_grad", &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);
}

void
WallDistEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology&,
  const WallBoundaryConditionData& wallBCData)
{
  const AlgorithmType algType = WALL;

  auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
  auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

  // Set up dphi/dx calculation algorithms
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "nodal_grad", &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);

  auto& meta = realm_.meta_data();
  ScalarFieldType& theBCField = meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "wall_distance_phi_bc");
  stk::mesh::put_field_on_mesh(theBCField, *part, nullptr);
  std::vector<double> userSpec(1, 0.0);
  AuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, &theBCField, theAuxFunc, stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // For terrain BC, the wall distance calculations must not compute the
  // distance normal to this wall, but must compute distance from the nearest
  // turbine, so we will disable Dirichlet for the terrain walls.
  WallUserData userData = wallBCData.userData_;
  const bool ablWallFunctionActivated = userData.ablWallFunctionApproach_;

  // Apply Dirichlet BC on non-ABL wall boundaries
  if (!ablWallFunctionActivated) {
    auto it = solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverDirichAlgMap_.end()) {
      DirichletBC* theAlg =
        new DirichletBC(realm_, this, part, &wPhiNp1, &theBCField, 0, 1);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    } else {
      it->second->partVec_.push_back(part);
    }
  }
}

void
WallDistEquationSystem::register_symmetry_bc(
  stk::mesh::Part* part,
  const stk::topology&,
  const SymmetryBoundaryConditionData&)
{
  const AlgorithmType algType = SYMMETRY;

  auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
  auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

  // Set up dphi/dx calculation algorithms
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
    algType, part, "nodal_grad", &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);
}

void
WallDistEquationSystem::register_non_conformal_bc(
  stk::mesh::Part* part, const stk::topology&)
{
  const auto algType = NON_CONFORMAL;

  // Setup dphi/dx calculation algorithms
  {
    auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
    auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

    nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "nodal_grad", &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);
  }

  // LHS contributions at the non-conformal interface
  {
    auto it = solverAlgDriver_->solverAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverAlgMap_.end()) {
      auto* theAlg =
        new AssembleWallDistNonConformalAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    } else {
      it->second->partVec_.push_back(part);
    }
  }
}

void
WallDistEquationSystem::register_overset_bc()
{
  if (resetOversetRows_) {
    if (decoupledOverset_)
      EquationSystem::create_constraint_algorithm(wallDistPhi_);
    else
      create_constraint_algorithm(wallDistPhi_);
  } else {
    for (auto* superPart : realm_.oversetBCPartVec_)
      for (auto* part : superPart->subsets()) {
        const AlgorithmType algType = SYMMETRY;

        auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
        auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

        // Set up dphi/dx calculation algorithms
        nodalGradAlgDriver_
          .register_face_algorithm<ScalarNodalGradBndryElemAlg>(
            algType, part, "nodal_grad", &wPhiNp1, &dPhiDxNone,
            edgeNodalGradient_);
      }
  }

  // No pre-iteration update of field as it is going to be reset in
  // solve_and_update
}

void
WallDistEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();

  // Reset init flag if this is a restarted simulation. The wall distance field
  // is available from the restart file, so we only want to recompute it at
  // user-specified frequency.
  //
  // The user option can override this and force a recompute. This option is
  // useful when "restarting" from a mapped file, e.g., wind-farm mesh where the
  // ABL precursor solution was mapped and is used to initialize the solution
  // using restart section in the input file.
  isInit_ = forceInitOnRestart_ || !realm_.restarted_simulation();
}

void
WallDistEquationSystem::reinitialize_linear_system()
{
  // If this is decoupled overset simulation and the user has requested that the
  // linear system be reused, then do nothing
  if (decoupledOverset_ && linsys_->config().reuseLinSysIfPossible())
    return;

  delete linsys_;
  const EquationType eqID = EQ_WALL_DISTANCE;
  auto solverName = realm_.equationSystems_.get_solver_block_name("ndtw");
  LinearSolver* solver = realm_.root()->linearSolvers_->reinitialize_solver(
    solverName, realm_.name(), eqID);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

void
WallDistEquationSystem::solve_and_update()
{
  // Only execute this equation system if the mesh is changing or upon
  // initialization
  if (
    !isInit_ && !(realm_.has_mesh_motion() &&
                  ((realm_.get_time_step_count() % updateFreq_) == 0) &&
                  (realm_.currentNonlinearIteration_ == 1)))
    return;

  if (isInit_) {
    isInit_ = false;
  } else {
    auto wdistPhi = realm_.ngp_field_manager().get_field<double>(
      wallDistPhi_->mesh_meta_data_ordinal());
    wdistPhi.set_all(realm_.ngp_mesh(), 0.0);
  }

  NaluEnv::self().naluOutputP0()
    << " 1/1" << std::setw(15) << std::right << userSuppliedName_ << std::endl;

  // Since this is purely geometric, we need at least two coupling iterations to
  // inform meshes about the field when using decoupled overset
  const int numOversetIters = (decoupledOverset_ && resetOversetRows_)
                                ? std::max(numOversetIters_, 2)
                                : numOversetIters_;
  for (int k = 0; k < numOversetIters; k++) {
    assemble_and_solve(wallDistPhi_);

    if (decoupledOverset_ && !resetOversetRows_)
      realm_.overset_field_update(wallDistPhi_, 1, 1);
  }

  // projected nodal gradient
  nodalGradAlgDriver_.execute();

  // calculate normal wall distance
  compute_wall_distance();
}

void
WallDistEquationSystem::compute_wall_distance()
{
  using Traits = nalu_ngp::NGPMeshTraits<>;
  using MeshIndex = Traits::MeshIndex;

  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const int nDim = meta.spatial_dimension();

  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto wdistPhi =
    fieldMgr.get_field<double>(wallDistPhi_->mesh_meta_data_ordinal());
  const auto dphidx =
    fieldMgr.get_field<double>(dphidx_->mesh_meta_data_ordinal());
  auto wdist =
    fieldMgr.get_field<double>(wallDistance_->mesh_meta_data_ordinal());
  const stk::mesh::Selector sel = stk::mesh::selectField(*wallDistPhi_);

  wdist.sync_to_device();
  nalu_ngp::run_entity_algorithm(
    "compute_wall_dist", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      double dpdxsq = 0.0;

      for (int d = 0; d < nDim; ++d) {
        double tmp = dphidx.get(mi, d);
        dpdxsq += tmp * tmp;
      }

      wdist.get(mi, 0) = -stk::math::sqrt(dpdxsq) +
                         stk::math::sqrt(dpdxsq + 2.0 * wdistPhi.get(mi, 0));
    });

  // TODO NGP switch to device field comms when STK NGP implements it
  wdist.modify_on_device();
  wdist.sync_to_host();

  // Communicate wall distance to everyone
  std::vector<const stk::mesh::FieldBase*> fVec{wallDistance_};
  stk::mesh::copy_owned_to_shared(bulk, fVec);
  stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);
  if (realm_.hasPeriodic_)
    realm_.periodic_delta_solution_update(wallDistance_, 1);
  if (
    realm_.hasNonConformal_ &&
    (realm_.nonConformalManager_->nonConformalGhosting_ != nullptr))
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, fVec);
  if (realm_.hasOverset_)
    realm_.overset_field_update(wallDistance_, 1, 1);
  wdist.modify_on_host();
  wdist.sync_to_device();
}

void
WallDistEquationSystem::create_constraint_algorithm(
  stk::mesh::FieldBase* theField)
{
  const AlgorithmType algType = OVERSET;

  auto it = solverAlgDriver_->solverConstraintAlgMap_.find(algType);
  if (it == solverAlgDriver_->solverConstraintAlgMap_.end()) {
    AssembleOversetWallDistAlgorithm* theAlg =
      new AssembleOversetWallDistAlgorithm(realm_, nullptr, this, theField);
    solverAlgDriver_->solverConstraintAlgMap_[algType] = theAlg;
  } else {
    throw std::runtime_error("WallDistEquationSystem::register_overset_bc: "
                             "Multiple invocations of overset is not allowed");
  }
}

} // namespace nalu
} // namespace sierra
