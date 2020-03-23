#include <GammaEquationSystem.h>
#include <AlgorithmDriver.h>
#include <AssembleScalarEdgeOpenSolverAlgorithm.h>
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

// edge kernels
#include <edge_kernels/ScalarEdgeSolverAlg.h>

// node kernels
#include <node_kernels/NodeKernelUtils.h>
#include <node_kernels/BLTGammaM2015NodeKernel.h>
#include <node_kernels/ScalarGclNodeKernel.h>
#include <node_kernels/ScalarMassBDFNodeKernel.h>

// ngp
#include <ngp_utils/NgpLoopUtils.h>
#include <ngp_utils/NgpTypes.h>
#include "ngp_utils/NgpFieldBLAS.h"
#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_algorithms/NodalGradElemAlg.h"
#include "ngp_algorithms/NodalGradBndryElemAlg.h"
#include "utils/StkHelpers.h"
#include <ngp_algorithms/EffDiffFluxCoeffAlg.h>


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


// GammaEquationSystem manages the gamma equation in the SST transition model

GammaEquationSystem::GammaEquationSystem(
    EquationSystems& eqSystems)
    : EquationSystem(eqSystems, "GammaEQS","gamma_transition"),
      managePNG_(realm_.get_consistent_mass_matrix_png("gamma_transition")),
      gamma_(NULL),
      gammaprod_(NULL),
      gammasink_(NULL),
      gammareth_(NULL),
      dGamdx_(NULL),
      gamTmp_(NULL),
      visc_(NULL),
      tvisc_(NULL),
      evisc_(NULL),
      nodalGradAlgDriver_(realm_, "dGamdx")
{
  dofName_ = "gamma_transition";

  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("gamma_transition");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_GAMMA_TRANS);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("gamma_transition");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for Gamma Transition Model: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

}
GammaEquationSystem::~GammaEquationSystem(){

}
void GammaEquationSystem::register_nodal_fields(stk::mesh::Part *part)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  
  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

    // register dof; set it as a restart variable
  gamma_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "gamma_transition", numStates));
  stk::mesh::put_field_on_mesh(*gamma_, *part, nullptr);
  realm_.augment_restart_variable_list("gamma_transition");

  dGamdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dGamdx"));
  stk::mesh::put_field_on_mesh(*dGamdx_, *part, nDim, nullptr);

  // delta solution for linear solver; share delta since this is a split system
  gamTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "gamTmp"));
  stk::mesh::put_field_on_mesh(*gamTmp_, *part, nullptr);

  visc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  tvisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity"));
  stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);

  evisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "effective_viscosity_gamma"));
  stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);

  gammaprod_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "gamma_production"));
  stk::mesh::put_field_on_mesh(*gammaprod_, *part, nullptr);

  gammasink_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "gamma_sink"));
  stk::mesh::put_field_on_mesh(*gammasink_, *part, nullptr);

  gammareth_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "gamma_reth"));
  stk::mesh::put_field_on_mesh(*gammareth_, *part, nullptr);

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &gammaN = gamma_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &gammaNp1 = gamma_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &gammaNp1, &gammaN,
                               0, 1,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
  
  
}
void GammaEquationSystem::register_interior_algorithm(stk::mesh::Part *part)
{
    // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &gammaNp1 = gamma_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dGamdxNone = dGamdx_->field_of_state(stk::mesh::StateNone);

  if (edgeNodalGradient_ && realm_.realmUsesEdges_)
    nodalGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone);
  else
    nodalGradAlgDriver_.register_legacy_algorithm<AssembleNodalGradElemAlgorithm>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone,
      edgeNodalGradient_);

  // solver; interior contribution (advection + diffusion)
  if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {

    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);

    if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = NULL;
      if (realm_.realmUsesEdges_) {
        theAlg = new ScalarEdgeSolverAlg(realm_, part, this, gamma_, dGamdx_, evisc_, false);
      }
      else {
        throw std::runtime_error("Gamma transition model not implemented for Element-based scheme");
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }

  std::vector<std::string> checkAlgNames = {
    "gamma_time_derivative",
    "lumped_gamma_time_derivative"};
    bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
    auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
    process_ngp_node_kernels(
      solverAlgMap, realm_, part, this,
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg) {
        if (!elementMassAlg)
          nodeAlg.add_kernel<ScalarMassBDFNodeKernel>(realm_.bulk_data(), gamma_);

        nodeAlg.add_kernel<BLTGammaM2015NodeKernel>(realm_.meta_data());
      },
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg, std::string& srcName) {
        if (srcName == "gcl") {
          nodeAlg.add_kernel<ScalarGclNodeKernel>(realm_.bulk_data(), gamma_);
          NaluEnv::self().naluOutputP0() << " - " << srcName << std::endl;
        }
        else
          throw std::runtime_error("Gamma EqSys: Invalid source term: " + srcName);
      });
  }
  else {
    // Homogeneous kernel implementation
    if (realm_.realmUsesEdges_)
      throw std::runtime_error("Gamma EquationSystem:: Not implemented element source terms");
  }

    // effective viscosity alg
  if (!effDiffFluxCoeffAlg_) {
      effDiffFluxCoeffAlg_.reset(new EffDiffFluxCoeffAlg(
          realm_, part, visc_, tvisc_, evisc_, 1.0, 1.0, true));
    }
   else {
    effDiffFluxCoeffAlg_->partVec_.push_back(part);
  }
  
}

void GammaEquationSystem::register_inflow_bc(
    stk::mesh::Part *part,
    const stk::topology &/*theTopo*/,
    const InflowBoundaryConditionData &inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;
  
  ScalarFieldType &gammaNp1 = gamma_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dGamdxNone = dGamdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; gamma_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "gamma_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  GammaInf gamma = userData.gamma_;
  std::vector<double> userSpec(1);
  userSpec[0] = gamma.gamma_;

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

  // copy gamma_bc to gamma np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, &gammaNp1,
                             0, 1,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dGdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
     algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone, edgeNodalGradient_);
  
  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg
      = new DirichletBC(realm_, this, part, &gammaNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  }
  else {
    itd->second->partVec_.push_back(part);
  }

}

#if 0
//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void GammaEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology & partTopo,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  ScalarFieldType &gammaNp1 = gamma_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dGamdxNone = dGamdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; gamma_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "open_gamma_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  Gamma gamma = userData.gamma_;
  std::vector<double> userSpec(1);
  userSpec[0] = gamma.gamma_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // non-solver; dGamdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone, edgeNodalGradient_);

  if (realm_.realmUsesEdges_) {
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    AssembleElemSolverAlgorithm* elemSolverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(elemSolverAlg, solverAlgWasBuilt)
      = build_or_add_part_to_face_bc_solver_alg(*this, *part, solverAlgMap, "open");

    auto& dataPreReqs = elemSolverAlg->dataNeededByKernels_;
    auto& activeKernels = elemSolverAlg->activeKernels_;

    build_face_topo_kernel_automatic<ScalarOpenEdgeKernel>(
      partTopo, *this, activeKernels, "gamma_open",
      realm_.meta_data(), *realm_.solutionOptions_, gamma_, theBcField, dataPreReqs);
  }
  else {
    // solver open; lhs
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theAlg = new AssembleScalarEdgeOpenSolverAlgorithm(realm_, part, this, gamma_, theBcField, &dGamdxNone, evisc_);
      }
      else {
        theAlg = new AssembleScalarElemOpenSolverAlgorithm(realm_, part, this, gamma_, theBcField, &dGamdxNone, evisc_);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
  }

}
#endif

#if 1
void GammaEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology & partTopo,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  // np1
  ScalarFieldType &gammaNp1 = gamma_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dGamdxNone = dGamdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dGamdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone, edgeNodalGradient_);

      NaluEnv::self().naluOutputP0()
      << "*********** register_open_bc **************************" << std::endl;
}

#endif 

void GammaEquationSystem::register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &/*theTopo*/,
    const WallBoundaryConditionData &wallBCData)
{
  const AlgorithmType algType = WALL;
  // np1
  ScalarFieldType &gammaNp1 = gamma_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dGamdxNone = dGamdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dGamdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone, edgeNodalGradient_);

}

void
GammaEquationSystem::register_symmetry_bc(stk::mesh::Part *part,
                                          const stk::topology &/*theTopo*/,
                                          const SymmetryBoundaryConditionData & /* symmetryBCData */)
{
  // algorithm type
  const AlgorithmType algType = SYMMETRY;
  // np1
  ScalarFieldType &gammaNp1 = gamma_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dGamdxNone = dGamdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dGamdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone, edgeNodalGradient_);
}

void GammaEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}
void GammaEquationSystem::reinitialize_linear_system()
{
  delete linsys_;
  const EquationType theEqID = EQ_GAMMA_TRANS;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
      = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }
  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("gamma_transition");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_GAMMA_TRANS);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();

}
void GammaEquationSystem::predict_state()
{
  // copy state n to state np1
  ScalarFieldType &gammaN = gamma_->field_of_state(stk::mesh::StateN);
  ScalarFieldType &gammaNp1 = gamma_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), gammaN, gammaNp1, realm_.get_activate_aura());
}

void GammaEquationSystem::assemble_nodal_gradient()
{
  const double timeA = -NaluEnv::self().nalu_time();
  nodalGradAlgDriver_.execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

void GammaEquationSystem::comp_eff_diff_coeff()
{
  const double timeA = -NaluEnv::self().nalu_time();
  effDiffFluxCoeffAlg_->execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

} //namespace nalu
} //namespace Sierra
