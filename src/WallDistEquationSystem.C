/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "WallDistEquationSystem.h"

#include "AssembleNodalGradAlgorithmDriver.h"
#include "AssembleNodalGradEdgeAlgorithm.h"
#include "AssembleNodalGradElemAlgorithm.h"
#include "AssembleNodalGradBoundaryAlgorithm.h"
#include "AssembleNodalGradNonConformalAlgorithm.h"
#include "AssembleNodeSolverAlgorithm.h"
#include "AssembleWallDistEdgeSolverAlgorithm.h"
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
#include "NonConformalManager.h"
#include "Realm.h"
#include "Realms.h"
#include "Simulation.h"
#include "SolutionOptions.h"
#include "SolverAlgorithm.h"
#include "SolverAlgorithmDriver.h"
#include "SupplementalAlgorithm.h"
#include "WallDistSrcNodeSuppAlg.h"

#include "kernel/WallDistElemKernel.h"
#include "kernel/KernelBuilder.h"

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

WallDistEquationSystem::WallDistEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "WallDistEQS", "ndtw"),
    assembleNodalGradAlgDriver_(
      new AssembleNodalGradAlgorithmDriver(realm_, "wall_distance_phi", "dwalldistdx")),
    managePNG_(realm_.get_consistent_mass_matrix_png("ndtw"))
{
  if (managePNG_)
    throw std::runtime_error("Consistent mass matrix PNG is not available for WallDistEquationSystem");

  auto solverName = eqSystems.get_solver_block_name("ndtw");
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, EQ_WALL_DISTANCE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  NaluEnv::self().naluOutputP0()
    << "Edge projected nodal gradient for minimum distance to wall: "
    << edgeNodalGradient_ << std::endl;

  realm_.push_equation_to_systems(this);
}

WallDistEquationSystem::~WallDistEquationSystem()
{}

void
WallDistEquationSystem::load(const YAML::Node& node)
{
  EquationSystem::load(node);

  get_if_present(node, "update_frequency", updateFreq_, updateFreq_);
}

void
WallDistEquationSystem::initial_work(){
  EquationSystem::initial_work();

  solve_and_update();
}

void
WallDistEquationSystem::register_nodal_fields(
  stk::mesh::Part* part)
{
  auto& meta = realm_.meta_data();
  const int nDim = meta.spatial_dimension();

  wallDistPhi_ = &(meta.declare_field<ScalarFieldType>(
                     stk::topology::NODE_RANK, "wall_distance_phi"));
  stk::mesh::put_field_on_mesh(*wallDistPhi_, *part, nullptr);
  realm_.augment_restart_variable_list("wall_distance_phi");

  dphidx_ = &(meta.declare_field<VectorFieldType>(
                stk::topology::NODE_RANK, "dwalldistdx"));
  stk::mesh::put_field_on_mesh(*dphidx_, *part, nDim, nullptr);

  wallDistance_ = &(meta.declare_field<ScalarFieldType>(
                  stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*wallDistance_, *part, nullptr);

  coordinates_ = &(meta.declare_field<VectorFieldType>(
                     stk::topology::NODE_RANK, realm_.get_coordinates_name()));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);
  dualNodalVolume_ = &(meta.declare_field<ScalarFieldType>(
                         stk::topology::NODE_RANK, "dual_nodal_volume"));
  stk::mesh::put_field_on_mesh(*dualNodalVolume_, *part, nullptr);
}

void
WallDistEquationSystem::register_edge_fields(
  stk::mesh::Part* part)
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
  stk::mesh::Part* part,
  const stk::topology&
)
{
  if (realm_.query_for_overset()) {
    auto& meta = realm_.meta_data();
    GenericFieldType& intersectedElement = meta.declare_field<GenericFieldType>(
      stk::topology::ELEMENT_RANK, "intersected_element");
    stk::mesh::put_field_on_mesh(intersectedElement, *part, 1, nullptr);
  }
}

void
WallDistEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{
  const AlgorithmType algType = INTERIOR;
  const AlgorithmType algMass = MASS;

  auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
  auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

  // Set up dphi/dx calculation algorithms
  auto it = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg = nullptr;

    if (edgeNodalGradient_ && realm_.realmUsesEdges_)
      theAlg = new AssembleNodalGradEdgeAlgorithm(
        realm_, part, &wPhiNp1, &dPhiDxNone);
    else
      theAlg = new AssembleNodalGradElemAlgorithm(
        realm_, part, &wPhiNp1, &dPhiDxNone);

    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }

  // Solver algorithms
  if (realm_.realmUsesEdges_) {
    auto it = solverAlgDriver_->solverAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = nullptr;
        theAlg = new AssembleWallDistEdgeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    } else {
      it->second->partVec_.push_back(part);
    }
  } else {
    stk::topology partTopo = part->topology();
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(solverAlg, solverAlgWasBuilt) = build_or_add_part_to_solver_alg
      (*this, *part, solverAlgMap);

    if (solverAlgWasBuilt) {
      ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
      auto& activeKernels = solverAlg->activeKernels_;
      const int dim = realm_.spatialDimension_;

      Kernel* compKernel = build_topo_kernel<WallDistElemKernel>(
        dim, partTopo, realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);
      activeKernels.push_back(compKernel);
    }
  }

  if (realm_.realmUsesEdges_) {
    auto it = solverAlgDriver_->solverAlgMap_.find(algMass);
    if (it == solverAlgDriver_->solverAlgMap_.end()) {
      AssembleNodeSolverAlgorithm* theAlg =
        new AssembleNodeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algMass] = theAlg;

      SupplementalAlgorithm* suppAlg = new WallDistSrcNodeSuppAlg(realm_);
      theAlg->supplementalAlg_.push_back(suppAlg);
    } else {
      it->second->partVec_.push_back(part);
    }
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
  auto it = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg =
      new AssembleNodalGradBoundaryAlgorithm(
        realm_, part, &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);

    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }
}

void
WallDistEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology&,
  const OpenBoundaryConditionData&)
{
  const AlgorithmType algType = OPEN;

  auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
  auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

  // Set up dphi/dx calculation algorithms
  auto it = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg =
      new AssembleNodalGradBoundaryAlgorithm(
        realm_, part, &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);

    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }
}

void
WallDistEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology&,
  const WallBoundaryConditionData&)
{
  const AlgorithmType algType = WALL;

  auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
  auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

  // Set up dphi/dx calculation algorithms
  auto it = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg =
      new AssembleNodalGradBoundaryAlgorithm(
        realm_, part, &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);

    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }

  auto& meta = realm_.meta_data();
  ScalarFieldType& theBCField = meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "wall_distance_phi_bc");
  stk::mesh::put_field_on_mesh(theBCField, *part, nullptr);
  std::vector<double> userSpec(1, 0.0);
  AuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
  AuxFunctionAlgorithm* auxAlg =
    new AuxFunctionAlgorithm(realm_, part, &theBCField, theAuxFunc,
                             stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // Dirichlet BC
  {
    auto it = solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if (it == solverAlgDriver_->solverDirichAlgMap_.end()) {
      DirichletBC* theAlg
        = new DirichletBC(realm_, this, part, &wPhiNp1, &theBCField, 0, 1);
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
  auto it = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg =
      new AssembleNodalGradBoundaryAlgorithm(
        realm_, part, &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);

    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }
}

void
WallDistEquationSystem::register_non_conformal_bc(
  stk::mesh::Part* part,
  const stk::topology&)
{
  const auto algType = NON_CONFORMAL;

  // Setup dphi/dx calculation algorithms
  {
    auto& wPhiNp1 = wallDistPhi_->field_of_state(stk::mesh::StateNone);
    auto& dPhiDxNone = dphidx_->field_of_state(stk::mesh::StateNone);

    auto it = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
      Algorithm* theAlg =
        new AssembleNodalGradBoundaryAlgorithm(
          realm_, part, &wPhiNp1, &dPhiDxNone, edgeNodalGradient_);

      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    } else {
      it->second->partVec_.push_back(part);
    }
  }

  // LHS contributions at the non-conformal interface
  {
    auto it = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( it == solverAlgDriver_->solverAlgMap_.end()) {
      auto* theAlg = new AssembleWallDistNonConformalAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

void
WallDistEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(wallDistPhi_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(wallDistPhi_,1,1)));
}

void
WallDistEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();

  // Reset init flag if this is a restarted simulation. The wall distance field
  // is available from the restart file, so we only want to recompute it at
  // user-specified frequency.
  isInit_ = !realm_.restarted_simulation();
}

void
WallDistEquationSystem::reinitialize_linear_system()
{
  delete linsys_;
  const EquationType eqID = EQ_WALL_DISTANCE;
  auto it = realm_.root()->linearSolvers_->solvers_.find(eqID);
  if (it != realm_.root()->linearSolvers_->solvers_.end())
    delete it->second;

  auto solverName = realm_.equationSystems_.get_solver_block_name("ndtw");
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, eqID);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

void
WallDistEquationSystem::solve_and_update()
{
  // Only execute this equation system if the mesh is changing or upon initialization
  if (!isInit_ &&
      !(realm_.has_mesh_motion() &&
        ((realm_.get_time_step_count() % updateFreq_) == 0) &&
        (realm_.currentNonlinearIteration_ == 1)))
    return;

  if (isInit_) {
    assembleNodalGradAlgDriver_->execute();
    isInit_ = false;
  }

  for (int k=0; k< maxIterations_; k++) {
    NaluEnv::self().naluOutputP0()
      << " " << k+1 << "/" << maxIterations_
      << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    pValue_ = 2 * (k + 1);

    assemble_and_solve(wallDistPhi_);

    // projected nodal gradient
    assembleNodalGradAlgDriver_->execute();
  }

  // calculate normal wall distance
  compute_wall_distance();
}

void
WallDistEquationSystem::compute_wall_distance()
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const int nDim = meta.spatial_dimension();

  stk::mesh::Selector sel = stk::mesh::selectField(*wallDistPhi_);
  const auto& bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);

  for (auto b: bkts) {
    double* phi = stk::mesh::field_data(*wallDistPhi_, *b);
    double* dpdx = stk::mesh::field_data(*dphidx_, *b);
    double* wdist = stk::mesh::field_data(*wallDistance_, *b);

    for (size_t k=0; k < b->size(); k++) {
      const int offset = k*nDim;
      double dpdxsq = 0.0;

      for (int j=0; j<nDim; j++) {
        double tmp = dpdx[offset + j];
        dpdxsq += tmp * tmp;
      }

      wdist[k] = -std::sqrt(dpdxsq) + std::sqrt(dpdxsq + 2.0 * phi[k]);
    }
  }

  // Communicate wall distance to everyone
  std::vector<const stk::mesh::FieldBase*> fVec{wallDistance_};
  stk::mesh::copy_owned_to_shared(bulk, fVec);
  stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);
  if (realm_.hasPeriodic_)
    realm_.periodic_delta_solution_update(wallDistance_, 1);
  if (realm_.hasNonConformal_ &&
      (realm_.nonConformalManager_->nonConformalGhosting_ != nullptr))
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, fVec);
  if (realm_.hasOverset_)
    realm_.overset_orphan_node_field_update(wallDistance_, 1, 1);
}

void
WallDistEquationSystem::create_constraint_algorithm(
  stk::mesh::FieldBase* theField)
{
  const AlgorithmType algType = OVERSET;

  auto it = solverAlgDriver_->solverConstraintAlgMap_.find(algType);
  if (it == solverAlgDriver_->solverConstraintAlgMap_.end()) {
    AssembleOversetWallDistAlgorithm* theAlg
      = new AssembleOversetWallDistAlgorithm(realm_, nullptr, this, theField);
    solverAlgDriver_->solverConstraintAlgMap_[algType] = theAlg;
  } else {
    throw std::runtime_error("WallDistEquationSystem::register_overset_bc: "
                             "Multiple invocations of overset is not allowed");
  }
}

}  // nalu
}  // sierra
