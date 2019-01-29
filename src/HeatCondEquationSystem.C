/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <HeatCondEquationSystem.h>

#include <AssembleElemSolverAlgorithm.h>
#include <AssembleHeatCondWallSolverAlgorithm.h>
#include <AssembleHeatCondIrradWallSolverAlgorithm.h>
#include <AssembleScalarEdgeDiffSolverAlgorithm.h>
#include <AssembleScalarElemDiffSolverAlgorithm.h>
#include <AssembleScalarDiffNonConformalSolverAlgorithm.h>
#include <AssembleScalarFluxBCSolverAlgorithm.h>
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
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <Enums.h>
#include <ErrorIndicatorAlgorithmDriver.h>
#include <FieldFunctions.h>
#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSystem.h>
#include <master_element/MasterElement.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Realms.h>
#include <HeatCondMassBackwardEulerNodeSuppAlg.h>
#include <HeatCondMassBDF2NodeSuppAlg.h>
#include <ProjectedNodalGradientEquationSystem.h>
#include <PstabErrorIndicatorEdgeAlgorithm.h>
#include <PstabErrorIndicatorElemAlgorithm.h>
#include <SimpleErrorIndicatorScalarElemAlgorithm.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <TimeIntegrator.h>
#include <SolverAlgorithmDriver.h>

// template for kernels
#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

// kernels
#include <kernel/ScalarDiffElemKernel.h>
#include <kernel/ScalarDiffFemKernel.h>
#include <kernel/ScalarDiffHOElemKernel.h>
#include <kernel/ScalarMassHOElemKernel.h>

#include <element_promotion/ElementDescription.h>

#include <user_functions/SteadyThermalContactSrcHOElemKernel.h>

// bc kernels
#include <kernel/ScalarFluxPenaltyElemKernel.h>

// user functions
#include <user_functions/SteadyThermalContactAuxFunction.h>
#include <user_functions/SteadyThermalContactSrcNodeSuppAlg.h>
#include <user_functions/SteadyThermalContactSrcElemSuppAlg.h>
#include <user_functions/SteadyThermal3dContactAuxFunction.h>
#include <user_functions/SteadyThermalContact3DSrcNodeSuppAlg.h>
#include <user_functions/SteadyThermal3dContactSrcElemSuppAlgDep.h>
#include <user_functions/SteadyThermal3dContactSrcElemKernel.h>

#include <overset/UpdateOversetFringeAlgorithmDriver.h>

// Nalu utils
#include <utils/StkHelpers.h>

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

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// HeatCondEquationSystem - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
HeatCondEquationSystem::HeatCondEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "HeatCondEQS", "temperature"),
    managePNG_(realm_.get_consistent_mass_matrix_png("temperature")),
    temperature_(NULL),
    dtdx_(NULL),
    tTmp_(NULL),
    dualNodalVolume_(NULL),
    coordinates_(NULL),
    exact_temperature_(NULL),
    exact_dtdx_(NULL),
    exact_laplacian_(NULL),
    density_(NULL),
    specHeat_(NULL),
    thermalCond_(NULL),
    edgeAreaVec_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(realm_, "temperature", "dtdx")),
    isInit_(true),
    projectedNodalGradEqs_(NULL)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("temperature");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_TEMPERATURE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("temperature");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for temperature: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create projected nodal gradient equation system
  if ( managePNG_ ) {
    manage_png(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
HeatCondEquationSystem::~HeatCondEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- manage_png ------------------------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::manage_png(
  EquationSystems& eqSystems)
{
  projectedNodalGradEqs_ 
    = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG, "dtdx", "qTmp", "temperature", "PNGGradEQS");
  // fill the map; only require wall (which is the same name)...
  projectedNodalGradEqs_->set_data_map(WALL_BC, "temperature");
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  temperature_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature", numStates));
  stk::mesh::put_field_on_mesh(*temperature_, *part, nullptr);
  realm_.augment_restart_variable_list("temperature");

  dtdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dtdx"));
  stk::mesh::put_field_on_mesh(*dtdx_, *part, nDim, nullptr);

  // delta solution for linear solver
  tTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "tTmp"));
  stk::mesh::put_field_on_mesh(*tTmp_, *part, nullptr);

  dualNodalVolume_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume"));
  stk::mesh::put_field_on_mesh(*dualNodalVolume_, *part, nullptr);

  coordinates_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates"));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);

  // props
  density_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "density"));
  stk::mesh::put_field_on_mesh(*density_, *part, nullptr);

  specHeat_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_heat"));
  stk::mesh::put_field_on_mesh(*specHeat_, *part, nullptr);

  thermalCond_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "thermal_conductivity"));
  stk::mesh::put_field_on_mesh(*thermalCond_, *part, nullptr);

  // push to property list
  realm_.augment_property_map(DENSITY_ID, density_);
  realm_.augment_property_map(SPEC_HEAT_ID, specHeat_);
  realm_.augment_property_map(THERMAL_COND_ID, thermalCond_);

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &tempN = temperature_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &tempNp1 = temperature_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlgA
      = new CopyFieldAlgorithm(realm_, part,
                               &tempNp1, &tempN,
                               0, 1,
                               stk::topology::NODE_RANK);

    copyStateAlg_.push_back(theCopyAlgA);
  }
}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::register_edge_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  //====================================================
  // Register edge data
  //====================================================
  if ( realm_.realmUsesEdges_ ) {
    const int nDim = meta_data.spatial_dimension();
    edgeAreaVec_ = &(meta_data.declare_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector"));
    stk::mesh::put_field_on_mesh(*edgeAreaVec_, *part, nDim, nullptr);
  }

}

//--------------------------------------------------------------------------
//-------- register_element_fields -----------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::register_element_fields(
  stk::mesh::Part *part,
  const stk::topology & /* theTopo */)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  
  // deal with heat conduction error indicator; elemental field of size unity
  if ( realm_.solutionOptions_->activateAdaptivity_) {
    const int numIp = 1;
    GenericFieldType *pstabEI= &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "error_indicator"));
    stk::mesh::put_field_on_mesh(*pstabEI, *part, numIp, nullptr);
  }
  
  // register the intersected elemental field
  if ( realm_.query_for_overset() ) {
    const int sizeOfElemField = 1;
    GenericFieldType *intersectedElement
      = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "intersected_element"));
    stk::mesh::put_field_on_mesh(*intersectedElement, *part, sizeOfElemField, nullptr);
  }
}
//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &tempNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dtdxNone = dtdx_->field_of_state(stk::mesh::StateNone);


  // non-solver; contribution to projected nodal gradient; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( edgeNodalGradient_ && realm_.realmUsesEdges_ ) {
        theAlg = new AssembleNodalGradEdgeAlgorithm(realm_, part, &tempNp1, &dtdxNone);
      }
      else {
        theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, &tempNp1, &dtdxNone, edgeNodalGradient_);
      }
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // solver; interior edge/element contribution (diffusion)
  if ( !realm_.solutionOptions_->useConsolidatedSolverAlg_ ) {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theSolverAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theSolverAlg = new AssembleScalarEdgeDiffSolverAlgorithm(realm_, part, this,
                                                                 &tempNp1, &dtdxNone, thermalCond_);
      }
      else {
        theSolverAlg = new AssembleScalarElemDiffSolverAlgorithm(realm_, part, this,
                                                                 &tempNp1, &dtdxNone, thermalCond_);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theSolverAlg;
      
      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string> >::iterator isrc 
        = realm_.solutionOptions_->elemSrcTermsMap_.find("temperature");
      if ( isrc != realm_.solutionOptions_->elemSrcTermsMap_.end() ) {
        
        if ( realm_.realmUsesEdges_ )
          throw std::runtime_error("HeatCondElemSrcTerms::Error can not use element source terms for an edge-based scheme");
        
        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if (sourceName == "steady_2d_thermal" ) {
            suppAlg = new SteadyThermalContactSrcElemSuppAlg(realm_);
          }
          else if (sourceName == "steady_3d_thermal" ) {
            suppAlg = new SteadyThermal3dContactSrcElemSuppAlgDep(realm_);
          }
          else if (sourceName == "FEM" ) {
            throw std::runtime_error("HeatCondElemSrcTerms::Error FEM must use consolidated approach");
          }
          else {
            throw std::runtime_error("HeatCondElemSrcTerms::Error Source term is not supported: " + sourceName);
          }
          NaluEnv::self().naluOutputP0() << "HeatCondElemSrcTerms::added() " << sourceName << std::endl;
          theSolverAlg->supplementalAlg_.push_back(suppAlg);
        }
      }
    }
    else {
      itsi->second->partVec_.push_back(part);
    } 
  }
  else {
    if ( realm_.realmUsesEdges_ )
      throw std::runtime_error("HeatCondElem::Error can not use supplemental design for an edge-based scheme");

    KernelBuilder kb(*this, *part, solverAlgDriver_->solverAlgorithmMap_, realm_.using_tensor_product_kernels());

    kb.build_topo_kernel_if_requested<SteadyThermal3dContactSrcElemKernel>("steady_3d_thermal",
      realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs()
    );

    kb.build_topo_kernel_if_requested<ScalarDiffElemKernel>("CVFEM_DIFF",
      realm_.bulk_data(), *realm_.solutionOptions_, temperature_, thermalCond_, kb.data_prereqs()
    );

    kb.build_fem_kernel_if_requested<ScalarDiffFemKernel>("FEM_DIFF",
      realm_.bulk_data(), *realm_.solutionOptions_, temperature_, thermalCond_, kb.data_prereqs()
    );

    kb.build_sgl_kernel_if_requested<ScalarDiffHOElemKernel>("experimental_ho_cvfem_diffusion",
      realm_.bulk_data(),  *realm_.solutionOptions_, temperature_, thermalCond_, kb.data_prereqs_HO()
    );

    kb.build_sgl_kernel_if_requested<SteadyThermalContactSrcHOElemKernel>("experimental_ho_cvfem_mms_source",
      realm_.bulk_data(),  *realm_.solutionOptions_, kb.data_prereqs_HO()
    );

    kb.report();
  }


  // time term; nodally lumped
  const AlgorithmType algMass = MASS;
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsm
    = solverAlgDriver_->solverAlgMap_.find(algMass);
  if ( itsm == solverAlgDriver_->solverAlgMap_.end() ) {
    // create the solver alg
    AssembleNodeSolverAlgorithm *theAlg
      = new AssembleNodeSolverAlgorithm(realm_, part, this);
    solverAlgDriver_->solverAlgMap_[algMass] = theAlg;

    // now create the supplemental alg for mass term
    if ( realm_.number_of_states() == 2 ) {
      HeatCondMassBackwardEulerNodeSuppAlg *theMass
        = new HeatCondMassBackwardEulerNodeSuppAlg(realm_);
      theAlg->supplementalAlg_.push_back(theMass);
    }
    else {
      HeatCondMassBDF2NodeSuppAlg *theMass
        = new HeatCondMassBDF2NodeSuppAlg(realm_);
      theAlg->supplementalAlg_.push_back(theMass);
    }

    // Add src term supp alg...; limited number supported
    std::map<std::string, std::vector<std::string> >::iterator isrc 
      = realm_.solutionOptions_->srcTermsMap_.find("temperature");
    if ( isrc != realm_.solutionOptions_->srcTermsMap_.end() ) {
      std::vector<std::string> mapNameVec = isrc->second;
      for (size_t k = 0; k < mapNameVec.size(); ++k ) {
        std::string sourceName = mapNameVec[k];
        if (sourceName == "steady_2d_thermal" ) {
          SteadyThermalContactSrcNodeSuppAlg *theSrc
            = new SteadyThermalContactSrcNodeSuppAlg(realm_);
          NaluEnv::self().naluOutputP0() << "HeatCondNodalSrcTerms::added() " << sourceName << std::endl;

          theAlg->supplementalAlg_.push_back(theSrc);
        }
        else if (sourceName == "steady_3d_thermal" ) {
          SteadyThermalContact3DSrcNodeSuppAlg *theSrc
          = new SteadyThermalContact3DSrcNodeSuppAlg(realm_);
          theAlg->supplementalAlg_.push_back(theSrc);
        }
        else {
          throw std::runtime_error("HeatCondNodalSrcTerms::Error Source term is not supported: " + sourceName);
        }
      }
    }
  }
  else {
    itsm->second->partVec_.push_back(part);
  }

  // deal with adaptivity
  if ( realm_.solutionOptions_->activateAdaptivity_) {

    // non-solver alg
    std::map<AlgorithmType, Algorithm *>::iterator itEI
      = realm_.errorIndicatorAlgDriver_->algMap_.find(algType);
    if ( itEI == realm_.errorIndicatorAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( realm_.solutionOptions_->errorIndicatorType_ & EIT_PSTAB ) {
        if ( realm_.realmUsesEdges_)
          theAlg = new PstabErrorIndicatorEdgeAlgorithm(realm_, part, temperature_, dtdx_, true);
        else
          theAlg = new PstabErrorIndicatorElemAlgorithm(realm_, part, temperature_, dtdx_, true);
      }
      else if ( realm_.solutionOptions_->errorIndicatorType_ & EIT_SIMPLE_DUDX2 ) {
        theAlg = new SimpleErrorIndicatorScalarElemAlgorithm(realm_, part, temperature_, dtdx_);
      }
      else {
        throw std::runtime_error("Sorry, only zz-like EI for heat conduction is supported");
      }
      realm_.errorIndicatorAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      itEI->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const WallBoundaryConditionData &wallBCData)
{

  const AlgorithmType algType = WALL;

  // np1
  ScalarFieldType &tempNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dtdxNone = dtdx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // non-solver; dtdx; allow for element-based shifted; all bcs are of generic type "WALL"
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &tempNp1, &dtdxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // extract the value for user specified temperaure and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  std::string temperatureName = "temperature";
  UserDataType theDataType = get_bc_data_type(userData, temperatureName);

  if ( userData.tempSpec_ ||  FUNCTION_UD == theDataType ) {
 
    // register boundary data; temperature_bc
    ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

    AuxFunction *theAuxFunc = NULL;
    if ( CONSTANT_UD == theDataType ) {
      Temperature theTemp = userData.temperature_;
      std::vector<double> userSpec(1);
      userSpec[0] = theTemp.temperature_;
      // new it
      theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
    }
    else {
      // extract the name
      std::string fcnName = get_bc_function_name(userData, temperatureName);
      // switch on the name found...
      if ( fcnName == "steady_2d_thermal" ) {
        theAuxFunc = new SteadyThermalContactAuxFunction();
      }
      else if ( fcnName == "steady_3d_thermal" ) {
        theAuxFunc = new SteadyThermal3dContactAuxFunction();
      }
      else {
        throw std::runtime_error("Only steady_2d/3d_thermal user functions supported");

      }
    }
    
    // bc data alg
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);
    
    // copy temperature_bc to temperature np1...
    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               theBcField, &tempNp1,
                               0, 1,
                               stk::topology::NODE_RANK);
    bcDataMapAlg_.push_back(theCopyAlg);

    // wall specified temperature solver algorithm
    if ( realm_.solutionOptions_->useConsolidatedBcSolverAlg_ ) {
      // solver for weak wall
      auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

      stk::topology elemTopo = get_elem_topo(realm_, *part);
      
      AssembleFaceElemSolverAlgorithm* faceElemSolverAlg = nullptr;
      bool solverAlgWasBuilt = false;

      std::tie(faceElemSolverAlg, solverAlgWasBuilt) 
        = build_or_add_part_to_face_elem_solver_alg(algType, *this, *part, elemTopo, solverAlgMap, "wall");
      
      auto& activeKernels = faceElemSolverAlg->activeKernels_;
      
      if (solverAlgWasBuilt) {
        build_face_elem_topo_kernel_automatic<ScalarFluxPenaltyElemKernel>
          (partTopo, elemTopo, *this, activeKernels, "heat_cond_weak_wall",
           realm_.meta_data(), *realm_.solutionOptions_,
           temperature_, theBcField, thermalCond_,
           faceElemSolverAlg->faceDataNeeded_, faceElemSolverAlg->elemDataNeeded_);
      }
    }
    else {
      // Dirichlet bc
      std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
        solverAlgDriver_->solverDirichAlgMap_.find(algType);
      if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
        DirichletBC *theAlg
          = new DirichletBC(realm_, this, part, &tempNp1, theBcField, 0, 1);
        solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
      }
      else {
        itd->second->partVec_.push_back(part);
      }
    }
  }
  else if( userData.heatFluxSpec_ && !userData.robinParameterSpec_ ) {

    const AlgorithmType algTypeHF = WALL_HF;

    ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "heat_flux_bc"));
    stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

    NormalHeatFlux heatFlux = userData.q_;
    std::vector<double> userSpec(1);
    userSpec[0] = heatFlux.qn_;

    // new it
    ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

    // bc data alg
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 theBcField, theAuxFunc,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);

    // solver; lhs; same for edge and element-based scheme
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
        solverAlgDriver_->solverAlgMap_.find(algTypeHF);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleScalarFluxBCSolverAlgorithm *theAlg
        = new AssembleScalarFluxBCSolverAlgorithm(realm_, part, this,
            theBcField, realm_.realmUsesEdges_);
      solverAlgDriver_->solverAlgMap_[algTypeHF] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }

  }
  else if ( userData.irradSpec_ ) {
    
    const AlgorithmType algTypeRAD = WALL_RAD;

    // check for emissivity
    if ( !userData.emissSpec_)
      throw std::runtime_error("Sorry, irradiation was specified while emissivity was not");

    // register boundary data;
    ScalarFieldType *irradField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "irradiation"));
    stk::mesh::put_field_on_mesh(*irradField, *part, nullptr);
    ScalarFieldType *emissField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "emissivity"));
    stk::mesh::put_field_on_mesh(*emissField, *part, nullptr);

    // aux algs; irradiation
    Irradiation irrad = userData.irradiation_;
    std::vector<double> irradUserSpec(1);
    irradUserSpec[0] = irrad.irradiation_;
    AuxFunction *irradAuxFunc = new ConstantAuxFunction(0, 1, irradUserSpec);
    
    AuxFunctionAlgorithm *irradAuxAlg
      = new AuxFunctionAlgorithm(realm_, part,
          irradField, irradAuxFunc,
          stk::topology::NODE_RANK);
    
    // aux algs; emissivity
    Emissivity emiss = userData.emissivity_;
    std::vector<double> emissUserSpec(1);
    emissUserSpec[0] = emiss.emissivity_;
    AuxFunction *emissAuxFunc = new ConstantAuxFunction(0, 1, emissUserSpec);
    
    AuxFunctionAlgorithm *emissAuxAlg
      = new AuxFunctionAlgorithm(realm_, part,
          emissField, emissAuxFunc,
          stk::topology::NODE_RANK);
    
    // if this is a multi-physics coupling, only populate IC for irradiation (xfer will handle it)
    if ( userData.isInterface_ ) {
      // xfer will handle population; only need to populate the initial value
      realm_.initCondAlg_.push_back(irradAuxAlg);
    }
    else {
      // put it on bcData
      bcDataAlg_.push_back(irradAuxAlg);
    }
    
    // emissivity is placed on bc data (never via XFER)
    bcDataAlg_.push_back(emissAuxAlg);

    // solver; lhs
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algTypeRAD);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleHeatCondIrradWallSolverAlgorithm *theAlg
        = new AssembleHeatCondIrradWallSolverAlgorithm(realm_, part, this,
            realm_.realmUsesEdges_);
      solverAlgDriver_->solverAlgMap_[algTypeRAD] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
  }
  else if ( userData.htcSpec_ || userData.refTempSpec_ || userData.robinParameterSpec_ ) {
    
    const AlgorithmType algTypeCHT = WALL_CHT;

    // If the user specified a Robin parameter, this is a Robin-type CHT; otherwise, it's convection
    bool isRobinCHT = userData.robinParameterSpec_;
    bool isConvectionCHT = !isRobinCHT;

    // first make sure all appropriate variables were specified
    if (isConvectionCHT)
    {
      if ( !userData.refTempSpec_)
        throw std::runtime_error("Sorry, h was specified while Tref was not");
      if ( !userData.htcSpec_)
        throw std::runtime_error("Sorry, Tref was specified while h was not");
    }
    else
    {
      if ( !userData.refTempSpec_)
        throw std::runtime_error("Sorry, Robin parameter was specified while Tref was not");
      if ( !userData.heatFluxSpec_)
        throw std::runtime_error("Sorry, Robin parameter was specified while heat flux was not");
    }

    // register boundary data
    ScalarFieldType *normalHeatFluxField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "normal_heat_flux"));
    stk::mesh::put_field_on_mesh(*normalHeatFluxField, *part, nullptr);
    ScalarFieldType *tRefField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "reference_temperature"));
    stk::mesh::put_field_on_mesh(*tRefField, *part, nullptr);

    ScalarFieldType *alphaField = NULL;
    if (isConvectionCHT)
    {
      alphaField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "heat_transfer_coefficient"));
    }
    if (isRobinCHT)
    {
      alphaField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "robin_coupling_parameter"));
    }
    stk::mesh::put_field_on_mesh(*alphaField, *part, nullptr);
  
    // aux algs
    AuxFunctionAlgorithm *alphaAuxAlg = nullptr;
    if (isRobinCHT)
    {
      RobinCouplingParameter alpha = userData.robinCouplingParameter_;
      std::vector<double> alphaUserSpec(1);
      alphaUserSpec[0] = alpha.robinCouplingParameter_;
      AuxFunction *alphaAuxFunc = new ConstantAuxFunction(0, 1, alphaUserSpec);
      alphaAuxAlg = new AuxFunctionAlgorithm(realm_,
                                             part,
                                             alphaField,
                                             alphaAuxFunc,
                                             stk::topology::NODE_RANK);
    }

    AuxFunctionAlgorithm *htcAuxAlg = nullptr;
    if (isConvectionCHT)
    {
      HeatTransferCoefficient htc = userData.heatTransferCoefficient_;
      std::vector<double> htcUserSpec(1);
      htcUserSpec[0] = htc.heatTransferCoefficient_;
      AuxFunction *htcAuxFunc = new ConstantAuxFunction(0, 1, htcUserSpec);
      htcAuxAlg = new AuxFunctionAlgorithm(realm_, 
                                           part,
                                           alphaField, 
                                           htcAuxFunc,
                                           stk::topology::NODE_RANK);
    }

    NormalHeatFlux heatFlux = userData.q_;
    std::vector<double> qnUserSpec(1);
    // For convection, pass a zero heat flux field; for Robin, use specified value
    qnUserSpec[0] = (isRobinCHT ? heatFlux.qn_ : 0.0);
    AuxFunction *qnAuxFunc = new ConstantAuxFunction(0, 1, qnUserSpec);
    AuxFunctionAlgorithm *qnAuxAlg = new AuxFunctionAlgorithm(realm_,
                                                              part,
                                                              normalHeatFluxField,
                                                              qnAuxFunc,
                                                              stk::topology::NODE_RANK);
  
    ReferenceTemperature tRef = userData.referenceTemperature_;
    std::vector<double> tRefUserSpec(1);
    tRefUserSpec[0] = tRef.referenceTemperature_;
    AuxFunction *tRefAuxFunc = new ConstantAuxFunction(0, 1, tRefUserSpec);
    AuxFunctionAlgorithm *tRefAuxAlg = new AuxFunctionAlgorithm(realm_, 
                                                                part,
                                                                tRefField, 
                                                                tRefAuxFunc,
                                                                stk::topology::NODE_RANK);


    // decide where to put population of data
    // Normal heat flux, reference temperature, and coupling parameter
    // come from a transfer if this is an interface, so in that case
    // only need to populate the initial values
    if ( userData.isInterface_ ) {
      // xfer will handle population; only need to populate the initial value
      realm_.initCondAlg_.push_back(tRefAuxAlg);
      if (isRobinCHT) 
      {
        realm_.initCondAlg_.push_back(alphaAuxAlg);
        realm_.initCondAlg_.push_back(qnAuxAlg);
      }
      if (isConvectionCHT) realm_.initCondAlg_.push_back(htcAuxAlg);
    }
    else {
      // put it on bcData
      bcDataAlg_.push_back(tRefAuxAlg);
      if (isRobinCHT)
      {
        bcDataAlg_.push_back(alphaAuxAlg);
        bcDataAlg_.push_back(qnAuxAlg);
      }
      if (isConvectionCHT) bcDataAlg_.push_back(htcAuxAlg);
    }
    // For convection-type, normal heat flux remains zero -- just set at IC
    if (isConvectionCHT) realm_.initCondAlg_.push_back(qnAuxAlg);

    // solver contribution
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algTypeCHT);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      AssembleHeatCondWallSolverAlgorithm *theAlg
        = new AssembleHeatCondWallSolverAlgorithm(realm_, 
                                                  part, 
                                                  this,
                                                  tRefField,
                                                  alphaField,
                                                  normalHeatFluxField,
                                                  realm_.realmUsesEdges_);
      solverAlgDriver_->solverAlgMap_[algTypeCHT] = theAlg;
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
    
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{
  const AlgorithmType algType = NON_CONFORMAL;

  // np1
  ScalarFieldType &tempNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dtdxNone = dtdx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dtdx; DG algorithm decides on locations for integration points
  if ( !managePNG_ ) {
    if ( edgeNodalGradient_ ) {    
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        Algorithm *theAlg 
          = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &tempNp1, &dtdxNone, edgeNodalGradient_);
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
          = new AssembleNodalGradNonConformalAlgorithm(realm_, part, &tempNp1, &dtdxNone);
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
    AssembleScalarDiffNonConformalSolverAlgorithm *theAlg
      = new AssembleScalarDiffNonConformalSolverAlgorithm(realm_, part, this, temperature_, thermalCond_);
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
HeatCondEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(temperature_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(temperature_,1,1)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_TEMPERATURE;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("temperature");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_TEMPERATURE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

void
HeatCondEquationSystem::predict_state()
{
  // copy state n to state np1
  ScalarFieldType &tN = temperature_->field_of_state(stk::mesh::StateN);
  ScalarFieldType &tNp1 = temperature_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), tN, tNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::solve_and_update()
{
  // initialize fields
  if ( isInit_ ) {
    compute_projected_nodal_gradient();
    isInit_ = false;
  }

  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;
    
    // heat conduction assemble, load_complete and solve
    assemble_and_solve(tTmp_);

    // update
    double timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(),
      realm_.bulk_data(),
      1.0, *tTmp_,
      1.0, *temperature_, 
      realm_.get_activate_aura());
    double timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB-timeA);
   
    // projected nodal gradient
    timeA = NaluEnv::self().nalu_time();
    compute_projected_nodal_gradient();
    timeB = NaluEnv::self().nalu_time();
    timerMisc_ += (timeB-timeA);
  }  
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient --------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::compute_projected_nodal_gradient()
{
  if ( !managePNG_ ) {
    assembleNodalGradAlgDriver_->execute();
  }
  else {
    projectedNodalGradEqs_->solve_and_update_external();
  }
}

//--------------------------------------------------------------------------
//-------- register_initial_condition_fcn ----------------------------------
//--------------------------------------------------------------------------
void
HeatCondEquationSystem::register_initial_condition_fcn(
  stk::mesh::Part *part,
  const std::map<std::string, std::string> &theNames,
  const std::map<std::string, std::vector<double> > &/*theParams*/)
{
  // iterate map and check for name
  const std::string dofName = "temperature";
  std::map<std::string, std::string>::const_iterator iterName
    = theNames.find(dofName);
  if (iterName != theNames.end()) {
    std::string fcnName = (*iterName).second;
    AuxFunction *theAuxFunc = NULL;
    if ( fcnName == "steady_2d_thermal" ) {
      // create the function
      theAuxFunc = new SteadyThermalContactAuxFunction();      
    }
    else if ( fcnName == "steady_3d_thermal" ) {
      // create the function
      theAuxFunc = new SteadyThermal3dContactAuxFunction();      
    }
    else {
      throw std::runtime_error("HeatCondEquationSystem::register_initial_condition_fcn: steady_2d/3d_thermal only supported");
    }
    
    // create the algorithm
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
				 temperature_, theAuxFunc,
				 stk::topology::NODE_RANK);
    
    // push to ic
    realm_.initCondAlg_.push_back(auxAlg);
    
  }
}

} // namespace nalu
} // namespace Sierra
