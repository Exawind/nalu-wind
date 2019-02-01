/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <MassFractionEquationSystem.h>
#include <AlgorithmDriver.h>
#include <AssembleScalarEdgeOpenSolverAlgorithm.h>
#include <AssembleScalarEdgeSolverAlgorithm.h>
#include <AssembleScalarElemSolverAlgorithm.h>
#include <AssembleScalarElemOpenSolverAlgorithm.h>
#include <AssembleScalarNonConformalSolverAlgorithm.h>
#include <AssembleNodalGradAlgorithmDriver.h>
#include <AssembleNodalGradEdgeAlgorithm.h>
#include <AssembleNodalGradElemAlgorithm.h>
#include <AssembleNodalGradBoundaryAlgorithm.h>
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
#include <AuxFunctionAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <CopyFieldAlgorithm.h>
#include <DirichletBC.h>
#include <EffectiveDiffFluxCoeffAlgorithm.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <Enums.h>
#include <FieldFunctions.h>
#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSystem.h>
#include <master_element/MasterElement.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Realms.h>
#include <ScalarMassBackwardEulerNodeSuppAlg.h>
#include <ScalarMassBDF2NodeSuppAlg.h>
#include <ScalarMassElemSuppAlgDep.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>

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

#include <stk_topology/topology.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// basic c++
#include <cmath>
#include <set>
#include <utility>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// MassFractionEquationSystem - mass fraction Eqs; operator split
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MassFractionEquationSystem::MassFractionEquationSystem(
  EquationSystems& eqSystems,
  const int numMassFraction)
  : EquationSystem(eqSystems, "MassFractionEQS","mass_fraction"),
    managePNG_(realm_.get_consistent_mass_matrix_png("note_follow_momentum_approach")),
    numMassFraction_(numMassFraction),
    massFraction_(NULL),
    currentMassFraction_(NULL),
    dydx_(NULL),
    yTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(realm_, "mass_fraction", "dydx")),
    diffFluxCoeffAlgDriver_(new AlgorithmDriver(realm_)),
    isInit_(true),
    nonLinearResidualSum_(0.0),
    firstNonLinearResidualSum_(0.0)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("mass_fraction");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_MASS_FRACTION);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);
  // turn off standard output
  linsys_->provideOutput_ = false;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // advertise as non-uniform
  realm_.uniformFlow_ = false;

  // create projected nodal gradient equation system
  if ( managePNG_ )
    throw std::runtime_error("MassFractionEquationSystem::Error managePNG is not complete");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MassFractionEquationSystem::~MassFractionEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
  delete diffFluxCoeffAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  massFraction_ =  &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "mass_fraction", numStates));
  stk::mesh::put_field_on_mesh(*massFraction_, *part, numMassFraction_, nullptr);
  realm_.augment_restart_variable_list("mass_fraction");

  currentMassFraction_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "current_mass_fraction", numStates));
  stk::mesh::put_field_on_mesh(*currentMassFraction_, *part, nullptr);

  // delta solution for linear solver
  yTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "yTmp"));
  stk::mesh::put_field_on_mesh(*yTmp_, *part, nullptr);

  dydx_ = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dydx"));
  stk::mesh::put_field_on_mesh(*dydx_, *part, nDim, nullptr);

  visc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  if ( realm_.is_turbulent() ) {
    tvisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity"));
    stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);
  }
  
  evisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "effective_viscosity_y"));
  stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);

}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  // non-solver; contribution to projected nodal gradient; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg = NULL;
    if ( edgeNodalGradient_ && realm_.realmUsesEdges_ ) {
      theAlg = new AssembleNodalGradEdgeAlgorithm(realm_, part, currentMassFraction_, dydx_);
    }
    else {
      theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, currentMassFraction_, dydx_, edgeNodalGradient_);
    }
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

  // solver; interior contribution (advection + diffusion)
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
    = solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    SolverAlgorithm *theAlg = NULL;
    if ( realm_.realmUsesEdges_ ) {
      theAlg = new AssembleScalarEdgeSolverAlgorithm(realm_, part, this, currentMassFraction_, dydx_, evisc_);
    }
    else {
      theAlg = new AssembleScalarElemSolverAlgorithm(realm_, part, this, currentMassFraction_, dydx_, evisc_);
    }
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;

    // look for fully integrated source terms
    std::map<std::string, std::vector<std::string> >::iterator isrc 
      = realm_.solutionOptions_->elemSrcTermsMap_.find("mass_fraction");
    if ( isrc != realm_.solutionOptions_->elemSrcTermsMap_.end() ) {
      
      if ( realm_.realmUsesEdges_ )
        throw std::runtime_error("MassFractionElemSrcTerms::Error can not use element source terms for an edge-based scheme");
    
      std::vector<std::string> mapNameVec = isrc->second;
      for (size_t k = 0; k < mapNameVec.size(); ++k ) {
        std::string sourceName = mapNameVec[k];
        SupplementalAlgorithm *suppAlg = NULL;
        if (sourceName == "mass_fraction_time_derivative" ) {
          suppAlg = new ScalarMassElemSuppAlgDep(realm_, currentMassFraction_, false);
        }
        else if (sourceName == "lumped_mass_fraction_time_derivative" ) {
          suppAlg = new ScalarMassElemSuppAlgDep(realm_, currentMassFraction_, true);
        }
        else {
          throw std::runtime_error("MassFractionElemSrcTerms::Error Source term is not supported: " + sourceName);
        }
        theAlg->supplementalAlg_.push_back(suppAlg);      
      }
    }
  }
  else {
    itsi->second->partVec_.push_back(part);
  }
  
  // time term; nodally lumped
  const AlgorithmType algMass = MASS;
  // Check if the user has requested CMM or LMM algorithms; if so, do not
  // include Nodal Mass algorithms
  std::vector<std::string> checkAlgNames = {"mass_fraction_time_derivative",
                                            "lumped_mass_fraction_time_derivative"};
  bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsm =
    solverAlgDriver_->solverAlgMap_.find(algMass);
  if ( itsm == solverAlgDriver_->solverAlgMap_.end() ) {
    // create the solver alg
    AssembleNodeSolverAlgorithm *theAlg
      = new AssembleNodeSolverAlgorithm(realm_, part, this);
    solverAlgDriver_->solverAlgMap_[algMass] = theAlg;
    
    // now create the supplemental alg for mass term
    if ( !elementMassAlg ) {
      if ( realm_.number_of_states() == 2 ) {
        ScalarMassBackwardEulerNodeSuppAlg *theMass
          = new ScalarMassBackwardEulerNodeSuppAlg(realm_, currentMassFraction_);
        theAlg->supplementalAlg_.push_back(theMass);
      }
      else {
        ScalarMassBDF2NodeSuppAlg *theMass
          = new ScalarMassBDF2NodeSuppAlg(realm_, currentMassFraction_);
        theAlg->supplementalAlg_.push_back(theMass);
      }
    }
    
    // Add src term supp alg...; limited number supported (exactly zero)
    std::map<std::string, std::vector<std::string> >::iterator isrc 
      = realm_.solutionOptions_->srcTermsMap_.find("mass_fraction");
    if ( isrc != realm_.solutionOptions_->srcTermsMap_.end() ) {
      std::vector<std::string> mapNameVec = isrc->second;   
      if ( mapNameVec.size() > 0 )
        throw std::runtime_error("MassFractionNodalSrcTerms::Error No source terms supported");
    }
  }

  // effective viscosity alg
  const double lamSc = realm_.get_lam_schmidt(massFraction_->name());
  const double turbSc = realm_.get_turb_schmidt(massFraction_->name());
  std::map<AlgorithmType, Algorithm *>::iterator itev =
    diffFluxCoeffAlgDriver_->algMap_.find(algType);
  if ( itev == diffFluxCoeffAlgDriver_->algMap_.end() ) {
    EffectiveDiffFluxCoeffAlgorithm *theAlg
      = new EffectiveDiffFluxCoeffAlgorithm(realm_, part, visc_, tvisc_, evisc_, lamSc, turbSc);
    diffFluxCoeffAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itev->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; massFraction_bc for all mass fraction number
  GenericFieldType *theBcField = &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "mass_fraction_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, numMassFraction_, nullptr);

  // register single scalar bc value
  ScalarFieldType *theCurrentBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "current_mass_fraction_bc"));
  stk::mesh::put_field_on_mesh(*theCurrentBcField, *part, nullptr);

  // insert to the set of bcs
  bcMassFractionSet_.insert(std::make_pair(theBcField, theCurrentBcField));

  // extract the value for user specified mass fraction and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  MassFraction massFraction = userData.massFraction_;
  std::vector<double> userSpec(numMassFraction_);
  for ( int k =0; k < numMassFraction_; ++k )
    userSpec[k] = massFraction.massFraction_[k];

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, numMassFraction_, userSpec);

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

  // copy mass fraction_bc to mass fraction np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, massFraction_,
                             0, numMassFraction_,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dydx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg 
      = new AssembleNodalGradBoundaryAlgorithm(realm_, part, currentMassFraction_, dydx_, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg
      = new DirichletBC(realm_, this, part, currentMassFraction_, theCurrentBcField, 0, 1);
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
MassFractionEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; mass fraction_bc for all speecies number
  GenericFieldType *theBcField = &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "mass_fraction_open_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, numMassFraction_, nullptr);

  // register single scalar bc value
  ScalarFieldType *theCurrentBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "current_mass_fraction_open_bc"));
  stk::mesh::put_field_on_mesh(*theCurrentBcField, *part, nullptr);

  // insert to the set of bcs
  bcMassFractionSet_.insert(std::make_pair(theBcField, theCurrentBcField));

  // extract the value for user specified mass fraction and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  MassFraction massFraction = userData.massFraction_;
  std::vector<double> userSpec(numMassFraction_);
  for ( int k =0; k < numMassFraction_; ++k )
    userSpec[k] = massFraction.massFraction_[k];

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // non-solver; dydx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg 
      = new AssembleNodalGradBoundaryAlgorithm(realm_, part, currentMassFraction_, dydx_, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

  // now solver contributions; open; lhs
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
    = solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    SolverAlgorithm *theAlg = NULL;
    if ( realm_.realmUsesEdges_ ) {
      theAlg = new AssembleScalarEdgeOpenSolverAlgorithm(realm_, part, this, currentMassFraction_, theCurrentBcField, dydx_, evisc_);
    }
    else {
      theAlg = new AssembleScalarElemOpenSolverAlgorithm(realm_, part, this, currentMassFraction_, theCurrentBcField, dydx_, evisc_);
    }
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    itsi->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // extract the value for user specified mixFrac and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  std::string massFractionName = "mass_fraction";
  if ( bc_data_specified(userData, massFractionName) ) {

    // FIXME: Generalize for constant vs function

    // register boundary data; mass fraction_bc for all mass fraction number
    GenericFieldType *theBcField = &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "mass_fraction_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, numMassFraction_, nullptr);

    // register single scalar bc value
    ScalarFieldType *theCurrentBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "current_mass_fraction_bc"));
    stk::mesh::put_field_on_mesh(*theCurrentBcField, *part, nullptr);

    // insert to the set of bcs
    bcMassFractionSet_.insert(std::make_pair(theBcField, theCurrentBcField));

    // extract the value for user specified mass fraction and save off the AuxFunction
    MassFraction massFraction = userData.massFraction_;
    std::vector<double> userSpec(numMassFraction_);
    for ( int k =0; k < numMassFraction_; ++k )
      userSpec[k] = massFraction.massFraction_[k];

    // new it
    ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, numMassFraction_, userSpec);

    // bc data alg
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);

    // copy mass fraction_bc to mass fraction np1...
    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               theBcField, massFraction_,
                               0, numMassFraction_,
                               stk::topology::NODE_RANK);
    bcDataMapAlg_.push_back(theCopyAlg);

    // Dirichlet bc
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
    if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
      DirichletBC *theAlg
        = new DirichletBC(realm_, this, part, currentMassFraction_, theBcField, 0, 1);
      solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
    }
    else {
      itd->second->partVec_.push_back(part);
    }
  }

  // non-solver; dydx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg 
      = new AssembleNodalGradBoundaryAlgorithm(realm_, part, currentMassFraction_, dydx_, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  } 
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData & /* symmetryBCData */)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  // non-solver; dwdx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm *>::iterator it
    = assembleNodalGradAlgDriver_->algMap_.find(algType);
  if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
    Algorithm *theAlg
      = new AssembleNodalGradBoundaryAlgorithm(realm_, part, currentMassFraction_, dydx_, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{

  const AlgorithmType algType = NON_CONFORMAL;

  // non-solver; contribution to dwdx; DG algorithm decides on locations for integration points
  if ( !managePNG_ ) {
    if ( edgeNodalGradient_ ) {    
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        Algorithm *theAlg 
          = new AssembleNodalGradBoundaryAlgorithm(realm_, part, currentMassFraction_, dydx_, edgeNodalGradient_);
        assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
    else {
      // proceed with DG
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        AssembleNodalGradNonConformalAlgorithm *theAlg 
          = new AssembleNodalGradNonConformalAlgorithm(realm_, part, currentMassFraction_, dydx_);
        assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
  }

  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    AssembleScalarNonConformalSolverAlgorithm *theAlg
      = new AssembleScalarNonConformalSolverAlgorithm(realm_, part, this, currentMassFraction_, evisc_);
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
MassFractionEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(currentMassFraction_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(currentMassFraction_,1,1)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- predict_state ---------------------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::predict_state()
{
  // copy state n to state np1
  GenericFieldType &yN = massFraction_->field_of_state(stk::mesh::StateN);
  GenericFieldType &yNp1 = massFraction_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), yN, yNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- set_current_mass_fraction ---------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::set_current_mass_fraction(
  const int k)
{
  // copy np1; n and possible nm1
  GenericFieldType &yNp1 = massFraction_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &cyNp1 = currentMassFraction_->field_of_state(stk::mesh::StateNP1);
  copy_mass_fraction(yNp1, k, cyNp1, 0);

  GenericFieldType &yN = massFraction_->field_of_state(stk::mesh::StateN);
  ScalarFieldType &cyN = currentMassFraction_->field_of_state(stk::mesh::StateN);
  copy_mass_fraction(yN, k, cyN, 0);

  if ( realm_.number_of_states() > 2 ) {
    GenericFieldType &yNm1 = massFraction_->field_of_state(stk::mesh::StateNM1);
    ScalarFieldType &cyNm1 = currentMassFraction_->field_of_state(stk::mesh::StateNM1);
    copy_mass_fraction(yNm1, k, cyNm1, 0);
  }

  // manage boundary data; copy kth value of bcField (from) to currentBcField (to)
  std::set<std::pair<stk::mesh::FieldBase*, stk::mesh::FieldBase*> >::iterator it;
  for ( it = bcMassFractionSet_.begin(); it != bcMassFractionSet_.end(); ++it) {
    const stk::mesh::FieldBase *fromBcField = (*it).first;
    const stk::mesh::FieldBase *toBcField = (*it).second;
    copy_mass_fraction(*fromBcField, k, *toBcField, 0);
  }
}

//--------------------------------------------------------------------------
//-------- copy_mass_fraction ----------------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::copy_mass_fraction(
  const stk::mesh::FieldBase &fromField,
  const int fromFieldIndex,
  const stk::mesh::FieldBase &toField,
  const int toFieldIndex)
{
  field_index_copy(realm_.meta_data(), realm_.bulk_data(), fromField, fromFieldIndex, toField, toFieldIndex, 
                   realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::solve_and_update()
{

  if ( isInit_ ) {
    // nothing to do now...
    isInit_ = false;
  }

  // compute effective viscosity; same for all mass fraction
  diffFluxCoeffAlgDriver_->execute();

  // we solve for n-1 mass fraction
  const int nm1MassFraction = numMassFraction_ - 1;
    
  for ( int i = 0; i < maxIterations_; ++i ) {

    NaluEnv::self().naluOutputP0() << " " << i+1 << "/" << maxIterations_ << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    double nonLinearResidualSum = 0.0;
    double linearResidualSum = 0.0;
    double linearIterationsSum = 0.0;

    for ( int k = 0; k < nm1MassFraction; ++k ) {

      // load np1, n and nm1 mass fraction to "current"; also populate "current" bc
      double timeA = NaluEnv::self().nalu_time();
      set_current_mass_fraction(k);
      double timeB = NaluEnv::self().nalu_time();
      timerMisc_ += (timeB-timeA);

      // compute nodal gradient
      assembleNodalGradAlgDriver_->execute();

      // mass fraction assemble, load_complete and solve
      assemble_and_solve(yTmp_);

      // update
      timeA = NaluEnv::self().nalu_time();
      field_axpby(
        realm_.meta_data(),
        realm_.bulk_data(),
        1.0, *yTmp_,
        1.0, *currentMassFraction_, 
        realm_.get_activate_aura());
      timeB = NaluEnv::self().nalu_time();
      timerAssemble_ += (timeB-timeA);

      // copy currentMassFraction back to mass fraction_k
      copy_mass_fraction(*currentMassFraction_, 0, *massFraction_, k);

      // increment solve counts and norms
      linearIterationsSum += linsys_->linearSolveIterations();
      nonLinearResidualSum += linsys_->nonLinearResidual();
      linearResidualSum += linsys_->linearResidual();

    }

    // compute nth mass fraction
    compute_nth_mass_fraction();

    // save total nonlinear residual
    nonLinearResidualSum_ = nonLinearResidualSum/double(nm1MassFraction);

    // save
    if ( realm_.currentNonlinearIteration_ == 1 )
      firstNonLinearResidualSum_ = nonLinearResidualSum_;

    // dump norm and averages
    std::string systemName = "MassFractionEQS";
    const int systemNameOffset = systemName.length()+8;
    NaluEnv::self().naluOutputP0()
      << std::setw(systemNameOffset) << std::right << systemName
      << std::setw(32-systemNameOffset)  << std::right << linearIterationsSum/double(nm1MassFraction)
      << std::setw(18) << std::right << linearResidualSum/double(nm1MassFraction)
      << std::setw(15) << std::right << nonLinearResidualSum/double(nm1MassFraction)
      << std::setw(14) << std::right << nonLinearResidualSum_/firstNonLinearResidualSum_ << std::endl;
  }

}

//--------------------------------------------------------------------------
//-------- compute_nth_mass_fraction ---------------------------------------
//--------------------------------------------------------------------------
void
MassFractionEquationSystem::compute_nth_mass_fraction()
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nm1MassFraction = numMassFraction_-1;
  const double lowerBound = 1.0e-16;
  const double upperBound = 1.0+lowerBound;

  // define some common selectors; select all nodes (locally and shared)
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*massFraction_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * yi = stk::mesh::field_data(*massFraction_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const int offSet = k*numMassFraction_;
      
      // find the sum; clip as well
      double sum = 0.0;
      for ( int i = 0; i < nm1MassFraction; ++i ) {
        const double yiClipped = std::min(upperBound, std::max(lowerBound, yi[offSet+i]));
        sum += yiClipped;
        yi[offSet+i] = yiClipped;
      }
      // set nth mass fraction
      yi[offSet+nm1MassFraction] = 1.0 - sum;
    }
  }
}

//--------------------------------------------------------------------------
//-------- system_is_converged ---------------------------------------------
//--------------------------------------------------------------------------
bool
MassFractionEquationSystem::system_is_converged()
{
  bool isConverged = true;
  if ( NULL != linsys_ ) {
    isConverged = (nonLinearResidualSum_/firstNonLinearResidualSum_ <  convergenceTolerance_ );
  }
  return isConverged;
}

//--------------------------------------------------------------------------
//-------- provide_scaled_norm ---------------------------------------------
//--------------------------------------------------------------------------
double
MassFractionEquationSystem::provide_scaled_norm()
{
  return nonLinearResidualSum_/firstNonLinearResidualSum_;
}

//--------------------------------------------------------------------------
//-------- provide_norm ---------------------------------------------
//--------------------------------------------------------------------------
double
MassFractionEquationSystem::provide_norm()
{
  return nonLinearResidualSum_;
}

} // namespace nalu
} // namespace Sierra
