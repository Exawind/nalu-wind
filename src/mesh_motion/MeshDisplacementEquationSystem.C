/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <mesh_motion/MeshDisplacementEquationSystem.h>

#include <mesh_motion/AssembleMeshDisplacementElemSolverAlgorithm.h>
#include <mesh_motion/AssemblePressureForceBCSolverAlgorithm.h>
#include <mesh_motion/MeshDisplacementMassBackwardEulerNodeSuppAlg.h>

#include <user_functions/LinearRampMeshDisplacementAuxFunction.h>
#include <user_functions/SinMeshDisplacementAuxFunction.h>

#include <AlgorithmDriver.h>
#include <AssembleNodalGradUAlgorithmDriver.h>
#include <AssembleNodalGradUElemAlgorithm.h>
#include <AssembleNodalGradUBoundaryAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
#include <AuxFunctionAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
#include <Enums.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <FieldFunctions.h>
#include <LinearSolver.h>
#include <LinearSolvers.h>
#include <LinearSystem.h>
#include <master_element/MasterElement.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>
#include <TimeIntegrator.h>

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
#include <stk_mesh/base/SkinMesh.hpp>
#include <stk_mesh/base/Comm.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

// basic c++
#include <utility>
#include <vector>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// MeshDisplacementEquationSystem - manages uvw pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MeshDisplacementEquationSystem::MeshDisplacementEquationSystem(
  EquationSystems& eqSystems,
  const bool activateMass,
  const bool deformWrtModelCoords)
  : EquationSystem(eqSystems, "MeshDisplacementEQS"),
    activateMass_(activateMass),
    deformWrtModelCoords_(deformWrtModelCoords),
    isInit_(false),
    meshDisplacement_(NULL),
    meshVelocity_(NULL),
    dvdx_(NULL),
    divV_(NULL),
    coordinates_(NULL),
    currentCoordinates_(NULL),
    dualNodalVolume_(NULL),
    density_(NULL),
    lameMu_(NULL),
    lameLambda_(NULL),
    dxTmp_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradUAlgorithmDriver(realm_, "dvdx"))
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("mesh_displacement");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_MESH_DISPLACEMENT);
  linsys_ = LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

  // determine nodal gradient form; use the edgeNodalGradient_ data member since mesh_displacement EQ does not need this
  set_nodal_gradient("mesh_velocity");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for mesh_velocity: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  realm_.solutionOptions_->meshDeformation_ = true;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MeshDisplacementEquationSystem::~MeshDisplacementEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::initial_work()
{
  // call base class method
  EquationSystem::initial_work();

}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  meshDisplacement_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "mesh_displacement", numStates));
  stk::mesh::put_field_on_mesh(*meshDisplacement_, *part, nDim, nullptr);
  realm_.augment_restart_variable_list("mesh_displacement");

  // mesh velocity (used for fluids coupling)
  meshVelocity_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "mesh_velocity"));
  stk::mesh::put_field_on_mesh(*meshVelocity_, *part, nDim, nullptr);

  // projected nodal gradient
  dvdx_ =  &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "dvdx"));
  stk::mesh::put_field_on_mesh(*dvdx_, *part, nDim*nDim, nullptr);

  divV_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_velocity"));
   stk::mesh::put_field_on_mesh(*divV_, *part, nullptr);

   // delta solution for linear solver
  dxTmp_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dxTmp"));
  stk::mesh::put_field_on_mesh(*dxTmp_, *part, nDim, nullptr);

  // geometry
  coordinates_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates"));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);

  currentCoordinates_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "current_coordinates"));
  stk::mesh::put_field_on_mesh(*currentCoordinates_, *part, nDim, nullptr);

  dualNodalVolume_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume"));
  stk::mesh::put_field_on_mesh(*dualNodalVolume_, *part, nullptr);

  // properties
  if ( activateMass_ ) {
    density_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "density"));
    stk::mesh::put_field_on_mesh(*density_, *part, nullptr);
    realm_.augment_property_map(DENSITY_ID, density_);
  }

  lameMu_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "lame_mu"));
  stk::mesh::put_field_on_mesh(*lameMu_, *part, nullptr);

  lameLambda_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "lame_lambda"));
  stk::mesh::put_field_on_mesh(*lameLambda_, *part, nullptr);

  // push to property list
  realm_.augment_property_map(LAME_MU_ID, lameLambda_);
  realm_.augment_property_map(LAME_LAMBDA_ID, lameMu_);

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && !realm_.restarted_simulation() ) {
    VectorFieldType &meshDisplacementN = meshDisplacement_->field_of_state(stk::mesh::StateN);
    VectorFieldType &meshDisplacementNp1 = meshDisplacement_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &meshDisplacementNp1, &meshDisplacementN,
                               0, nDim,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }

}


//--------------------------------------------------------------------------
//-------- register_element_fields -------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::register_element_fields(
  stk::mesh::Part * /* part */,
  const stk::topology & /* theTopo */)
{
  // n/a
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;
  const AlgorithmType algMass = MASS;

  // solver; interior elem contribution
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    AssembleMeshDisplacementElemSolverAlgorithm *theAlg
      = new AssembleMeshDisplacementElemSolverAlgorithm(
              realm_, part, this, deformWrtModelCoords_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    itsi->second->partVec_.push_back(part);
  }

  // solver; time contribution (lumped mass matrix)
  if ( activateMass_ ) {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsm
      = solverAlgDriver_->solverAlgMap_.find(algMass);
    if ( itsm == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleNodeSolverAlgorithm *theAlg
        = new AssembleNodeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algMass] = theAlg;

      // now create the supplemental alg for mass term; generalize for BDF2
      MeshDisplacementMassBackwardEulerNodeSuppAlg *theMass
        = new MeshDisplacementMassBackwardEulerNodeSuppAlg(realm_);
      theAlg->supplementalAlg_.push_back(theMass);
    }
    else {
      itsm->second->partVec_.push_back(part);
    }
  }

  // non-solver; contribution to Gjvi; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator itgv
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( itgv == assembleNodalGradAlgDriver_->algMap_.end() ) {
    AssembleNodalGradUElemAlgorithm *theAlg 
      = new AssembleNodalGradUElemAlgorithm(realm_, part, meshVelocity_, dvdx_, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itgv->second->partVec_.push_back(part);
  }
  
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology & /* theTopo */,
  const WallBoundaryConditionData &wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  // np1 mesh displacement
  VectorFieldType &displacementNp1 = meshDisplacement_->field_of_state(stk::mesh::StateNP1);

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const unsigned nDim = meta_data.spatial_dimension();

  // extract the value for user specified velocity and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  std::string displacementName = "mesh_displacement";
  std::string pressureName = "pressure";

  if ( bc_data_specified(userData, displacementName) ) {

    // register boundary data; mesh_displacement_bc
    VectorFieldType *theBcField = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "mesh_displacement_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, nDim, nullptr);

    AuxFunction *theAuxFunc = NULL;

    UserDataType theDataType = get_bc_data_type(userData, displacementName);
    if ( CONSTANT_UD == theDataType ) {
      // constant data type specification
      Velocity dx = userData.dx_;
      std::vector<double> userSpec(nDim);
      userSpec[0] = dx.ux_;
      userSpec[1] = dx.uy_;
      if ( nDim > 2)
        userSpec[2] = dx.uz_;
      theAuxFunc = new ConstantAuxFunction(0, nDim, userSpec);
    }
    else if ( FUNCTION_UD == theDataType ) {
      // extract the name
      std::string fcnName = get_bc_function_name(userData, displacementName);
      std::vector<double> theParams = get_bc_function_params(userData, displacementName);
      // switch on the name found...
      if ( fcnName == "linear" ) {
        theAuxFunc = new LinearRampMeshDisplacementAuxFunction(0,nDim, theParams);
      }
      else if ( fcnName == "sinusoidal") {
        theAuxFunc = new SinMeshDisplacementAuxFunction(0,nDim, theParams);
      }
      else {
        throw std::runtime_error("Only linear ramp user functions supported");
      }
    }

    // proceed with aux function and dirichlet setup

    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
          theBcField, theAuxFunc,
          stk::topology::NODE_RANK);

    // check to see if this is an FSI interface to determine how we handle mesh_displacement population
    if ( userData.isFsiInterface_ ) {
      // xfer will handle population; only need to populate the initial value
      realm_.initCondAlg_.push_back(auxAlg);
    }
    else {
      bcDataAlg_.push_back(auxAlg);
    }

    // copy mesh_displacement_bc to mesh_displacement np1...
    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
          theBcField, &displacementNp1,
          0, nDim,
          stk::topology::NODE_RANK);
    bcDataMapAlg_.push_back(theCopyAlg);

    // Dirichlet bc
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itd
      = solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
      DirichletBC *theAlg
        = new DirichletBC(realm_, this, part, &displacementNp1, theBcField, 0, nDim);
        solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    }
    else {
      itd->second->partVec_.push_back(part);
    }
  }
  else if (bc_data_specified(userData, pressureName) ) {
    // register the bc pressure field
    ScalarFieldType *bcPressureField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure_bc"));
    stk::mesh::put_field_on_mesh(*bcPressureField, *part, nDim, nullptr);

    // extract the value for user specified pressure and save off the AuxFunction
    Pressure pSpec = userData.pressure_;
    std::vector<double> userSpecPbc(1);
    userSpecPbc[0] = pSpec.pressure_;
    
    ConstantAuxFunction *theAuxFuncPbc = new ConstantAuxFunction(0, 1, userSpecPbc);

    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
          bcPressureField, theAuxFuncPbc,
          stk::topology::NODE_RANK);

    // check to see if this is an FSI interface to determine how we handle pressure population
    if ( userData.isFsiInterface_ ) {
      // xfer will handle population; only need to populate the initial value
      realm_.initCondAlg_.push_back(auxAlg);
    }
    else {
      bcDataAlg_.push_back(auxAlg);
    }

    // now create the RHS algorithm
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      AssemblePressureForceBCSolverAlgorithm *theAlg
        = new AssemblePressureForceBCSolverAlgorithm(realm_, part, this,
                                                     bcPressureField, realm_.realmUsesEdges_);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }

  }
  else {
    NaluEnv::self().naluOutputP0() << "No displacement specified: zero surface traction applied" << std::endl;
  }

  // non-solver; contribution to Gjvi; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator itgv
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( itgv == assembleNodalGradAlgDriver_->algMap_.end() ) {
    AssembleNodalGradUBoundaryAlgorithm *theAlg
      = new AssembleNodalGradUBoundaryAlgorithm(realm_, part, meshVelocity_, dvdx_, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itgv->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(meshVelocity_);

  int nDim = realm_.meta_data().spatial_dimension();
  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(meshVelocity_,1,nDim)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_MESH_DISPLACEMENT;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("mesh_velocity");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_MESH_DISPLACEMENT);
  linsys_ = LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

  // initialize new solver
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- predict_state ---------------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::predict_state()
{
  // copy state n to state np1
  VectorFieldType &dN = meshDisplacement_->field_of_state(stk::mesh::StateN);
  VectorFieldType &dNp1 = meshDisplacement_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), dN, dNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::solve_and_update()
{

  // nothing to do
  if ( isInit_ ) {
    isInit_ = false;
  }

  // start the iteration loop
  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    // tke assemble, load_complete and solve
    assemble_and_solve(dxTmp_);

    // update
    double timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(),
      realm_.bulk_data(),
      1.0, *dxTmp_,
      1.0, meshDisplacement_->field_of_state(stk::mesh::StateNP1), 
      realm_.get_activate_aura());
    double timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB-timeA);

    compute_current_coordinates();
  }
  
  // compute nodal projected gradient and nodal divV
  assembleNodalGradAlgDriver_->execute();
  
  compute_div_mesh_velocity();

}

//--------------------------------------------------------------------------
//-------- compute_current_coordinates -------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::compute_current_coordinates()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  VectorFieldType &displacementN = meshDisplacement_->field_of_state(stk::mesh::StateN);
  VectorFieldType &displacementNp1 = meshDisplacement_->field_of_state(stk::mesh::StateNP1);

  const int nDim = meta_data.spatial_dimension();
  const double dt = realm_.get_time_step();

  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
      &stk::mesh::selectField(*meshDisplacement_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    const double * dxN = stk::mesh::field_data(displacementN, b);
    const double * dxNp1 = stk::mesh::field_data(displacementNp1, b);
    double * meshVelocity = stk::mesh::field_data(*meshVelocity_, b);
    const double * coordinates = stk::mesh::field_data(*coordinates_, b);
    double * currentCoordinates = stk::mesh::field_data(*currentCoordinates_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      size_t offSet = k*nDim;
      for ( int j = 0; j < nDim; ++j ) {
        currentCoordinates[offSet+j] = coordinates[offSet+j] + dxNp1[offSet+j];
        // hack a mesh velocity to be first order backward Euler
        meshVelocity[offSet+j] = (dxNp1[offSet+j] - dxN[offSet+j])/dt;
      }
    }
  }

}

//--------------------------------------------------------------------------
//-------- compute_div_mesh_velocity ---------------------------------------
//--------------------------------------------------------------------------
void
MeshDisplacementEquationSystem::compute_div_mesh_velocity()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
      &stk::mesh::selectField(*divV_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    const double * dvdx = stk::mesh::field_data(*dvdx_, b);
    double * divV = stk::mesh::field_data(*divV_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      size_t offSet = k*nDim*nDim;
      double sum = 0.0;
      for ( int j = 0; j < nDim; ++j )
        sum += dvdx[offSet+nDim*j +j];
      divV[k] = sum;
    }
  }

}

} // namespace nalu
} // namespace Sierra
