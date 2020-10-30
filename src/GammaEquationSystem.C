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
#include <AssembleWallDistGradElemAlgorithm.h>
#include <AssembleNDotVGradElemAlgorithm.h>
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
#include <WallDistEquationSystem.h>

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
#include <edge_kernels/ScalarOpenEdgeKernel.h>

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

#include "ngp_algorithms/WallDistGradEdgeAlg.h"
#include "ngp_algorithms/WallDistGradElemAlg.h"
#include "ngp_algorithms/WallDistGradBndryElemAlg.h"
#include "ngp_algorithms/NDotVGradEdgeAlg.h"
#include "ngp_algorithms/NDotVGradElemAlg.h"
#include "ngp_algorithms/NDotVGradBndryElemAlg.h"
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
      dWallDistdx_(NULL),
      dNDotVdx_(NULL),
      gamTmp_(NULL),
      minDistanceToWall_(NULL),
      NDotV_(NULL),
      visc_(NULL),
      tvisc_(NULL),
      evisc_(NULL),
      nodalGradAlgDriver_(realm_, "dGamdx"),
      walldistGradAlgDriver_(realm_, "dWallDistdx"),
      ndotvGradAlgDriver_(realm_, "dNDotVdx")
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

  dWallDistdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dWallDistdx"));
  stk::mesh::put_field_on_mesh(*dWallDistdx_, *part, nDim, nullptr);

  dNDotVdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dNDotVdx"));
  stk::mesh::put_field_on_mesh(*dNDotVdx_, *part, nDim, nullptr);

  NDotV_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "NDotV"));
  stk::mesh::put_field_on_mesh(*NDotV_, *part, nullptr);

  minDistanceToWall_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*minDistanceToWall_, *part, nullptr);

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

  ScalarFieldType &minD = minDistanceToWall_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dWallDistdxNone = dWallDistdx_->field_of_state(stk::mesh::StateNone);

  ScalarFieldType &NDotV = NDotV_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dNDotVdxNone = dNDotVdx_->field_of_state(stk::mesh::StateNone);

  if (edgeNodalGradient_ && realm_.realmUsesEdges_) {
    printf("Executing grad option 1\n");
    nodalGradAlgDriver_.register_edge_algorithm<ScalarNodalGradEdgeAlg>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone);

    walldistGradAlgDriver_.register_edge_algorithm<ScalarWallDistGradEdgeAlg>(
      algType, part, "gamma_walldist_grad", &minD, &dWallDistdxNone);

    ndotvGradAlgDriver_.register_edge_algorithm<ScalarNDotVGradEdgeAlg>(
      algType, part, "gamma_ndotv_grad", &NDotV, &dNDotVdxNone);
    }
  else {
    printf("Executing grad option 2\n");
    nodalGradAlgDriver_.register_legacy_algorithm<AssembleNodalGradElemAlgorithm>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone,
      edgeNodalGradient_);

    walldistGradAlgDriver_.register_legacy_algorithm<AssembleWallDistGradElemAlgorithm>(
      algType, part, "gamma_walldist_grad", &minD, &dWallDistdxNone,
      edgeNodalGradient_);

    ndotvGradAlgDriver_.register_legacy_algorithm<AssembleNDotVGradElemAlgorithm>(
      algType, part, "gamma_ndotv_grad", &NDotV, &dNDotVdxNone,
      edgeNodalGradient_);
  }

  //compute_dvdy_parameter();

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

  ScalarFieldType &minD = minDistanceToWall_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dWallDistdxNone = dWallDistdx_->field_of_state(stk::mesh::StateNone);

  ScalarFieldType &NDotV = NDotV_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dNDotVdxNone = dNDotVdx_->field_of_state(stk::mesh::StateNone);

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
  
  walldistGradAlgDriver_.register_face_algorithm<ScalarWallDistGradBndryElemAlg>(
     algType, part, "gamma_walldist_grad", &minD, &dWallDistdxNone, edgeNodalGradient_);
  
  ndotvGradAlgDriver_.register_face_algorithm<ScalarNDotVGradBndryElemAlg>(
     algType, part, "gamma_ndotv_grad", &NDotV, &dNDotVdxNone, edgeNodalGradient_);

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

  ScalarFieldType &minD = minDistanceToWall_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dWallDistdxNone = dWallDistdx_->field_of_state(stk::mesh::StateNone);

  ScalarFieldType &NDotV = NDotV_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dNDotVdxNone = dNDotVdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dGamdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone, edgeNodalGradient_);

  walldistGradAlgDriver_.register_face_algorithm<ScalarWallDistGradBndryElemAlg>(
      algType, part, "gamma_walldist_grad", &minD, &dWallDistdxNone, edgeNodalGradient_);

  ndotvGradAlgDriver_.register_face_algorithm<ScalarNDotVGradBndryElemAlg>(
      algType, part, "gamma_ndotv_grad", &NDotV, &dNDotVdxNone, edgeNodalGradient_);

      NaluEnv::self().naluOutputP0()
      << "*********** register_open_bc **************************" << std::endl;
}

void GammaEquationSystem::register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &/*theTopo*/,
    const WallBoundaryConditionData &wallBCData)
{
  const AlgorithmType algType = WALL;
  // np1
  ScalarFieldType &gammaNp1 = gamma_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dGamdxNone = dGamdx_->field_of_state(stk::mesh::StateNone);

  ScalarFieldType &minD = minDistanceToWall_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dWallDistdxNone = dWallDistdx_->field_of_state(stk::mesh::StateNone);

  ScalarFieldType &NDotV = NDotV_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dNDotVdxNone = dNDotVdx_->field_of_state(stk::mesh::StateNone);
  // non-solver; dGamdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone, edgeNodalGradient_);

  walldistGradAlgDriver_.register_face_algorithm<ScalarWallDistGradBndryElemAlg>(
      algType, part, "gamma_walldist_grad", &minD, &dWallDistdxNone, edgeNodalGradient_);

  ndotvGradAlgDriver_.register_face_algorithm<ScalarNDotVGradBndryElemAlg>(
      algType, part, "gamma_ndotv_grad", &NDotV, &dNDotVdxNone, edgeNodalGradient_);
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

  ScalarFieldType &minD = minDistanceToWall_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dWallDistdxNone = dWallDistdx_->field_of_state(stk::mesh::StateNone);

  ScalarFieldType &NDotV = NDotV_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dNDotVdxNone = dNDotVdx_->field_of_state(stk::mesh::StateNone);
  // non-solver; dGamdx; allow for element-based shifted
  nodalGradAlgDriver_.register_face_algorithm<ScalarNodalGradBndryElemAlg>(
      algType, part, "gamma_nodal_grad", &gammaNp1, &dGamdxNone, edgeNodalGradient_);

  walldistGradAlgDriver_.register_face_algorithm<ScalarWallDistGradBndryElemAlg>(
      algType, part, "gamma_walldist_grad", &minD, &dWallDistdxNone, edgeNodalGradient_);

  ndotvGradAlgDriver_.register_face_algorithm<ScalarNDotVGradBndryElemAlg>(
      algType, part, "gamma_ndotv_grad", &NDotV, &dNDotVdxNone, edgeNodalGradient_);
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


//--------------------------------------------------------------------------
//------------------------- normalize dwalldistdx vector -------------------
//--------------------------------------------------------------------------
  
void
GammaEquationSystem::normalize_dwalldistdx()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  VectorFieldType *dwalldistdx = dWallDistdx_;
  double vmag = 0.0;

  printf("normalize_dwalldistdx\n");

  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*NDotV_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * dwalldist = stk::mesh::field_data(*dwalldistdx, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      vmag = 0.0; 
      for ( int j = 0; j < nDim; ++j ) {
        vmag += dwalldist[k*nDim+j]*dwalldist[k*nDim+j];
      }

      if (stk::math::abs(vmag) >= 1.0e-10) {
        vmag = stk::math::sqrt(vmag);
        for ( int j = 0; j < nDim; ++j ) {
          dwalldist[k*nDim+j] /= vmag;
        }
      }

  }
}
}

void
GammaEquationSystem::compute_norm_dot_vel()
{
  // compute dvdy parameter required for source term of Menter (2015) transition model
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // fields not saved off
  VectorFieldType *coordsNp1 = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  VectorFieldType *velNp1 = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  VectorFieldType *dwalldistdx = dWallDistdx_;

  //select all nodes (locally and shared)
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*NDotV_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    const double * dwalldist = stk::mesh::field_data(*dwalldistdx, b);
    const double *coords = stk::mesh::field_data(*coordsNp1, b);     
    double *vel = stk::mesh::field_data(*velNp1, b);
    double * NDotV = stk::mesh::field_data(*NDotV_, b);

    // compute norm dot velocity
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      NDotV[k] = 0.0;
    // ********************* Temp test of grad operator *************************
    //  vel[k*nDim+0] = 0.0;
    //  vel[k*nDim+1] = 0.0;
    //  vel[k*nDim+2] = 10.0*coords[2];
    // ********************* Temp test of grad operator *************************
      for ( int j = 0; j < nDim; ++j ) {
        NDotV[k] += dwalldist[k*nDim+j] * vel[k*nDim+j];
      }
    }
  }
}

void GammaEquationSystem::assemble_nodal_gradient()
{
  const double timeA = -NaluEnv::self().nalu_time();
  nodalGradAlgDriver_.execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

void GammaEquationSystem::assemble_walldist_gradient()
{
  const double timeA = -NaluEnv::self().nalu_time();
  walldistGradAlgDriver_.execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

void GammaEquationSystem::assemble_ndotv_gradient()
{
  const double timeA = -NaluEnv::self().nalu_time();
  ndotvGradAlgDriver_.execute();
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
