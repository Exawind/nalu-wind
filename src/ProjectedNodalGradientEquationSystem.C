// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <ProjectedNodalGradientEquationSystem.h>

#include <AssemblePNGElemSolverAlgorithm.h>
#include <AssemblePNGBoundarySolverAlgorithm.h>
#include <AssemblePNGNonConformalSolverAlgorithm.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <Enums.h>
#include <FieldFunctions.h>
#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSystem.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>

#include <kernel/KernelBuilder.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/SkinMesh.hpp>
#include <stk_mesh/base/Comm.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>
#include <stk_topology/topology.hpp>

// stk_util
#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

// overset
#include <overset/UpdateOversetFringeAlgorithmDriver.h>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ProjectedNodalGradientEquationSystem - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ProjectedNodalGradientEquationSystem::ProjectedNodalGradientEquationSystem(
  EquationSystems& eqSystems,
  const EquationType eqType,
  const std::string dofName,
  const std::string deltaName,
  const std::string independentDofName,
  const std::string eqSysName,
  const bool managesSolve)
  : EquationSystem(eqSystems, eqSysName),
    eqType_(eqType),
    dofName_(dofName),
    deltaName_(deltaName),
    independentDofName_(independentDofName),
    eqSysName_(eqSysName),
    managesSolve_(managesSolve),
    dqdx_(NULL),
    qTmp_(NULL)
{
  // extract solver name and solver object
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name(dofName);
  LinearSolver* solver = realm_.root()->linearSolvers_->create_solver(
    solverName, realm_.name(), eqType_);
  linsys_ =
    LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

  // push back EQ to manager
  realm_.push_equation_to_systems(this);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ProjectedNodalGradientEquationSystem::~ProjectedNodalGradientEquationSystem()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- set_data_map ----------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::set_data_map(
  BoundaryConditionType BC, std::string name)
{
  dataMap_[BC] = name;
}

//--------------------------------------------------------------------------
//-------- get_name_given_bc -----------------------------------------------
//--------------------------------------------------------------------------
std::string
ProjectedNodalGradientEquationSystem::get_name_given_bc(
  BoundaryConditionType BC)
{
  std::map<BoundaryConditionType, std::string>::iterator it;
  it = dataMap_.find(BC);
  if (it == dataMap_.end())
    throw std::runtime_error(
      "PNGEqSys::missing BC type specification (developer error)!");
  else
    return it->second;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_nodal_fields(
  const stk::mesh::PartVector& part_vec)
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);

  dqdx_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, dofName_));
  stk::mesh::put_field_on_mesh(*dqdx_, selector, nDim, nullptr);
  stk::io::set_field_output_type(*dqdx_, stk::io::FieldOutputType::VECTOR_3D);

  // delta solution for linear solver
  qTmp_ =
    &(meta_data.declare_field<double>(stk::topology::NODE_RANK, deltaName_));
  stk::mesh::put_field_on_mesh(*qTmp_, selector, nDim, nullptr);
  stk::io::set_field_output_type(*qTmp_, stk::io::FieldOutputType::VECTOR_3D);
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_interior_algorithm(
  stk::mesh::Part* part)
{
  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  // solver
  std::map<AlgorithmType, SolverAlgorithm*>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (its == solverAlgDriver_->solverAlgMap_.end()) {
    AssemblePNGElemSolverAlgorithm* theAlg = new AssemblePNGElemSolverAlgorithm(
      realm_, part, this, independentDofName_, dofName_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const WallBoundaryConditionData& /*wallBCData*/)
{

  const AlgorithmType algType = WALL;

  // extract the field name for this bc type
  std::string fieldName = get_name_given_bc(WALL_BC);
  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm*>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (its == solverAlgDriver_->solverAlgMap_.end()) {
    AssemblePNGBoundarySolverAlgorithm* theAlg =
      new AssemblePNGBoundarySolverAlgorithm(realm_, part, this, fieldName);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_inflow_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const InflowBoundaryConditionData& /*inflowBCData*/)
{

  const AlgorithmType algType = INFLOW;

  // extract the field name for this bc type
  std::string fieldName = get_name_given_bc(INFLOW_BC);
  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm*>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (its == solverAlgDriver_->solverAlgMap_.end()) {
    AssemblePNGBoundarySolverAlgorithm* theAlg =
      new AssemblePNGBoundarySolverAlgorithm(realm_, part, this, fieldName);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const OpenBoundaryConditionData& /*openBCData*/)
{
  const AlgorithmType algType = OPEN;

  // extract the field name for this bc type
  std::string fieldName = get_name_given_bc(OPEN_BC);
  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm*>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (its == solverAlgDriver_->solverAlgMap_.end()) {
    AssemblePNGBoundarySolverAlgorithm* theAlg =
      new AssemblePNGBoundarySolverAlgorithm(realm_, part, this, fieldName);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_symmetry_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const SymmetryBoundaryConditionData& /*symmetryBCData*/)
{
  const AlgorithmType algType = SYMMETRY;

  // extract the field name for this bc type
  std::string fieldName = get_name_given_bc(SYMMETRY_BC);
  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm*>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (its == solverAlgDriver_->solverAlgMap_.end()) {
    AssemblePNGBoundarySolverAlgorithm* theAlg =
      new AssemblePNGBoundarySolverAlgorithm(realm_, part, this, fieldName);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_non_conformal_bc(
  stk::mesh::Part* part, const stk::topology& /*theTopo*/)
{
  // FIX THIS
  const AlgorithmType algType = NON_CONFORMAL;

  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm*>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (its == solverAlgDriver_->solverAlgMap_.end()) {
    AssemblePNGNonConformalSolverAlgorithm* theAlg =
      new AssemblePNGNonConformalSolverAlgorithm(
        realm_, part, this, independentDofName_, dofName_,
        realm_.solutionOptions_->ncAlgPngPenalty_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(dqdx_);

  int nDim = realm_.meta_data().spatial_dimension();
  equationSystems_.register_overset_field_update(dqdx_, 1, nDim);
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::reinitialize_linear_system()
{
  // delete linsys; set previously set parameters on linsys
  const bool provideOutput = linsys_->provideOutput_;
  delete linsys_;

  // create new solver; reset parameters
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name(dofName_);
  LinearSolver* solver = realm_.root()->linearSolvers_->reinitialize_solver(
    solverName, realm_.name(), eqType_);
  linsys_ =
    LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);
  linsys_->provideOutput_ = provideOutput;

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::solve_and_update()
{
  if (managesSolve_)
    solve_and_update_external();
}

//--------------------------------------------------------------------------
//-------- solve_and_update_external ---------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::solve_and_update_external()
{
  for (int k = 0; k < maxIterations_; ++k) {

    // projected nodal gradient, load_complete and solve
    assemble_and_solve(qTmp_);

    // update
    double timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(), realm_.bulk_data(), 1.0, *qTmp_, 1.0, *dqdx_,
      realm_.get_activate_aura());
    double timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB - timeA);
  }
}

//--------------------------------------------------------------------------
//-------- deactivate_output -----------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::deactivate_output()
{
  linsys_->provideOutput_ = false;
}

} // namespace nalu
} // namespace sierra
