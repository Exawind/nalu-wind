/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <pmr/RadiativeTransportEquationSystem.h>
#include <pmr/RadTransBlackBodyNodeSuppAlg.h>
#include <pmr/RadTransIsoScatteringNodeSuppAlg.h>
#include <pmr/AssembleRadTransEdgeSolverAlgorithm.h>
#include <pmr/AssembleRadTransEdgeUpwindSolverAlgorithm.h>
#include <pmr/AssembleRadTransWallSolverAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
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
#include <master_element/MasterElement.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>

// template for kernels
#include "AlgTraits.h"
#include "kernel/KernelBuilder.h"
#include "kernel/KernelBuilderLog.h"

// implemented kernels
#include "AssembleElemSolverAlgorithm.h"
#include "pmr/RadTransAdvectionSUCVElemKernel.h"
#include "pmr/RadTransAbsorptionBlackBodyElemKernel.h"
#include "pmr/RadTransIsotropicScatteringElemKernel.h"
#include "pmr/RadTransWallElemKernel.h"

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

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// RadiativeTransportEquationSystem - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
RadiativeTransportEquationSystem::RadiativeTransportEquationSystem(
  EquationSystems& eqSystems,
  const int quadratureOrder,
  const bool activateScattering,
  const bool activateUpwind,
  const bool deactivateSucv,
  const bool externalCoupling)
  : EquationSystem(eqSystems, "RadiativeTransportEQS", "intensity"),
    quadratureOrder_(quadratureOrder),
    activateScattering_(activateScattering),
    activateUpwind_(activateUpwind),
    deactivateSucv_(deactivateSucv),
    externalCoupling_(externalCoupling),
    intensity_(NULL),
    currentIntensity_(NULL),
    intensityBc_(NULL),
    emissivity_(NULL),
    transmissivity_(NULL),
    environmentalT_(NULL),
    iTmp_(NULL),
    dualNodalVolume_(NULL),
    coordinates_(NULL),
    temperature_(NULL),
    radiativeHeatFlux_(NULL),
    divRadiativeHeatFlux_(NULL),
    radiationSource_(NULL),
    scalarFlux_(NULL),
    scalarFluxOld_(NULL),
    absorptionCoeff_(NULL),
    scatteringCoeff_(NULL),
    edgeAreaVec_(NULL),
    irradiation_(NULL),
    bcTemperature_(NULL),
    assembledBoundaryArea_(NULL),
    isInit_(true),
    ordinateDirections_(0),
    currentWeight_(0),
    systemL2Norm_(0.0),
    nonLinearResidualSum_(0.0),
    firstNonLinearResidualSum_(0.0)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("intensity");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_INTENSITY);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);
  // turn off standard output
  linsys_->provideOutput_ = false;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  // extract quadrature weights
  const int thurgoodFac = ( nDim == 3 ) ? 8 : 2;
  ordinateDirections_ = thurgoodFac*quadratureOrder_*quadratureOrder_;

  // resize total set and current
  Sn_.resize(nDim*ordinateDirections_);
  weights_.resize(ordinateDirections_);
  currentSn_.resize(nDim);

  // create symmetric quad set
  create_quadrature_set();
  
  // tell the user scattering is or is not active
  NaluEnv::self().naluOutputP0() << "Scattering source term is active " << activateScattering_;

  // check for upwind option...
  if ( activateUpwind_ )
    if ( !realm_.realmUsesEdges_ )
      throw std::runtime_error("PMR upwind only supported when using the edge-based scheme. please switch");

}


//--------------------------------------------------------------------------
//-------- create_quadrature_set -------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::create_quadrature_set()
{

  // FIXME: deal with 2D
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  int j, k, m, n, noct;

  double xx, yy, zz, ds, ss;
  double arc1, arc2, arc3;
  double ang1, ang2, ang3;

  double pi = std::acos(-1.0);

  double qsgn[24] = {  1.0,  1.0,  1.0,  1.0,  1.0, -1.0,  1.0,
                       -1.0,  1.0,  1.0, -1.0, -1.0, -1.0,  1.0,
                       1.0, -1.0,  1.0, -1.0, -1.0, -1.0,  1.0,
                       -1.0, -1.0, -1.0 };

  //  compute quadrature set for the first octant
  double *xs = new double[ nDim*quadratureOrder_*quadratureOrder_ ];
  double *ys = new double[ nDim*quadratureOrder_*quadratureOrder_ ];
  double *zs = new double[ nDim*quadratureOrder_*quadratureOrder_ ];

  m = 0;
  ds = 1.0/double(quadratureOrder_);

  // compute ordinate directions for the first octant
  for ( j=0; j < quadratureOrder_; ++j ) {

    // get vertices for basal triangles pointing up in the z-direction
    // see Figure 2 in Thurgood, Pollard, and Becker

    xx = 1.0 - j*ds;
    yy = 0.0;
    zz = 0.0 + j*ds;

    for ( k=0; k < quadratureOrder_-j; ++k ) {

      xs[m] = xx - k*ds;
      ys[m] = yy + k*ds;
      zs[m] = zz;
      m++;

      xs[m] = xx - (k+1)*ds;
      ys[m] = yy + (k+1)*ds;
      zs[m] = zz;
      m++;

      xs[m] = xx - (k+1)*ds;
      ys[m] = yy + k*ds;
      zs[m] = zz + ds;
      m++;

    }
  }

  for ( j=0; j < quadratureOrder_-1; ++j ) {

    // get vertices for basal triangles pointing down in the z-direction
    // see Figure 2 in Thurgood, Pollard, and Becker

    xx = 1.0 - (j+1)*ds;
    yy = ds;
    zz = 0.0 + j*ds;

    for ( k=0; k < quadratureOrder_-j-1; ++k ) {

      xs[m] = xx - k*ds;
      ys[m] = yy + k*ds;
      zs[m] = zz;
      m++;

      xs[m] = xx - (k+1)*ds;
      ys[m] = yy + k*ds;
      zs[m] = zz + ds;;
      m++;

      xs[m] = xx - k*ds;
      ys[m] = yy + (k-1)*ds;
      zs[m] = zz + ds;
      m++;

    }

  }

  // compute quadrature weights for the first octant

  for ( k=0; k < quadratureOrder_*quadratureOrder_; ++k ) {

    // compute ordinate directions through basal triangle centroids
    // see Equation 4 in Thurgood, Pollard, and Becker

    xx = ( xs[nDim*k+0] + xs[nDim*k+1] + xs[nDim*k+2] )/3.0;
    yy = ( ys[nDim*k+0] + ys[nDim*k+1] + ys[nDim*k+2] )/3.0;
    zz = ( zs[nDim*k+0] + zs[nDim*k+1] + zs[nDim*k+2] )/3.0;

    ss = std::sqrt( xx*xx + yy*yy + zz*zz );

    Sn_[nDim*k+0] = xx/ss;
    Sn_[nDim*k+1] = yy/ss;
    Sn_[nDim*k+2] = zz/ss;

    // project basal triangle vertices to spherical triangle vertices

    ss = std::sqrt(xs[nDim*k+0]*xs[nDim*k+0]+ys[nDim*k+0]*ys[nDim*k+0]+zs[nDim*k+0]*zs[nDim*k+0]);
    xs[nDim*k+0] /= ss;
    ys[nDim*k+0] /= ss;
    zs[nDim*k+0] /= ss;

    ss = std::sqrt(xs[nDim*k+1]*xs[nDim*k+1]+ys[nDim*k+1]*ys[nDim*k+1]+zs[nDim*k+1]*zs[nDim*k+1]);
    xs[nDim*k+1] /= ss;
    ys[nDim*k+1] /= ss;
    zs[nDim*k+1] /= ss;

    ss = std::sqrt(xs[nDim*k+2]*xs[nDim*k+2]+ys[nDim*k+2]*ys[nDim*k+2]+zs[nDim*k+2]*zs[nDim*k+2]);
    xs[nDim*k+2] /= ss;
    ys[nDim*k+2] /= ss;
    zs[nDim*k+2] /= ss;

    // compute spherical triangle edge arc angles
    //  cos(arc) = R1 dot R2
    //  where R1 and R2 are the vectors from the sphere centroiD
    // to arc vertices

    arc1 = std::acos( xs[nDim*k+0]*xs[nDim*k+1]+ys[nDim*k+0]*ys[nDim*k+1]+zs[nDim*k+0]*zs[nDim*k+1] );

    arc2 = std::acos( xs[nDim*k+1]*xs[nDim*k+2]+ys[nDim*k+1]*ys[nDim*k+2]+zs[nDim*k+1]*zs[nDim*k+2] );

    arc3 = std::acos( xs[nDim*k+2]*xs[nDim*k+0]+ys[nDim*k+2]*ys[nDim*k+0]+zs[nDim*k+2]*zs[nDim*k+0] );

    /*
      compute spherical triangle interior angles
      spherical law of cosines:

      cos(arc3) = cos(arc1)*cos(arc2) + sin(arc1)*sin(arc2)*ang3

      arcN is the angle swept out by a spherical triangle segment
      relative to the sphere centroid

      angN is the interior spherical triangle angle opposite
      segment defined by arcN
    */

    ang1 = std::acos( ( std::cos(arc1) - std::cos(arc2)*std::cos(arc3) ) / ( std::sin(arc2)*std::sin(arc3) ) );

    ang2 = std::acos( ( std::cos(arc2) - std::cos(arc3)*std::cos(arc1) ) / ( std::sin(arc3)*std::sin(arc1) ) );

    ang3 = std::acos( ( std::cos(arc3) - std::cos(arc1)*std::cos(arc2) ) / ( std::sin(arc1)*std::sin(arc2) ) );

    // area of a spherical triangle is equal to its spherical excess

    weights_[k] = ang1 + ang2 + ang3 - pi;

  }

  //  set directions and weights for remaining octants

  noct= ordinateDirections_/8;

  for ( n=1; n<8; ++n ) {

    for ( k=0; k<noct; ++k ) {

      j = k + n*noct;
      weights_[j] = weights_[k];

      Sn_[nDim*j+0] = Sn_[nDim*k+0]*qsgn[nDim*n+0];
      Sn_[nDim*j+1] = Sn_[nDim*k+1]*qsgn[nDim*n+1];
      Sn_[nDim*j+2] = Sn_[nDim*k+2]*qsgn[nDim*n+2];

    }

  }

  // write quadratures
  double l_sum[4] = {0.0,0.0,0.0,0.0};
  for ( n=0; n<ordinateDirections_; ++n ) {
    l_sum[0] += weights_[n];
    l_sum[1] += weights_[n]*Sn_[nDim*n+0];
    l_sum[2] += weights_[n]*Sn_[nDim*n+1];
    l_sum[3] += weights_[n]*Sn_[nDim*n+2];
  }

  NaluEnv::self().naluOutputP0() << " Discrete Ordinate Directions and Weights "  << std::endl;
  NaluEnv::self().naluOutputP0() << "    Quadrature Order    = " << quadratureOrder_    << std::endl;
  NaluEnv::self().naluOutputP0() << "    Number of Ordinates = " << ordinateDirections_ << std::endl;
  NaluEnv::self().naluOutputP0() << "    Weights sum         = " << l_sum[0]      << std::endl;
  NaluEnv::self().naluOutputP0() << "    X-Ordinates sum     = " << l_sum[1]      << std::endl;
  NaluEnv::self().naluOutputP0() << "    Y-Ordinates sum     = " << l_sum[2]      << std::endl;
  NaluEnv::self().naluOutputP0() << "    Z-Ordinates sum     = " << l_sum[3]      << std::endl;

  for ( n=0; n<ordinateDirections_; ++n ) {
    NaluEnv::self().naluOutputP0() << n+1 << " " << weights_[n] << " "
                    << Sn_[nDim*n+0] << " " <<  Sn_[nDim*n+1] << " " << Sn_[nDim*n+2] << std::endl;
  }

  // delete dynamic memory
  delete [] xs;
  delete [] ys;
  delete [] zs;

}


//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
RadiativeTransportEquationSystem::~RadiativeTransportEquationSystem()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  // register all number of ordinates intensity; reserve intensity_ for "curent"
  intensity_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "intensity"));
  stk::mesh::put_field_on_mesh(*intensity_, *part, nullptr);

  // may not want all of these at production time...
  for ( int k = 0; k < ordinateDirections_; ++k ) {
    std::stringstream ss;
    ss << k;
    const std::string incrementName = ss.str();
    const std::string theName = "intensity_" + incrementName;
    ScalarFieldType *intensityK = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, theName));
    stk::mesh::put_field_on_mesh(*intensityK, *part, nullptr);
  }

  // delta solution for linear solver
  iTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "iTmp"));
  stk::mesh::put_field_on_mesh(*iTmp_, *part, nullptr);

  const int numVolStates = realm_.does_mesh_move() ? realm_.number_of_states() : 1;
  dualNodalVolume_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume", numVolStates));
  stk::mesh::put_field_on_mesh(*dualNodalVolume_, *part, nullptr);
  if (numVolStates > 1) realm_.augment_restart_variable_list("dual_nodal_volume");

  coordinates_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates"));
  stk::mesh::put_field_on_mesh(*coordinates_, *part, nDim, nullptr);

  temperature_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature"));
  stk::mesh::put_field_on_mesh(*temperature_, *part, nullptr);

  radiativeHeatFlux_ = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "radiative_heat_flux"));
  stk::mesh::put_field_on_mesh(*radiativeHeatFlux_, *part, nDim, nullptr);

  divRadiativeHeatFlux_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_radiative_heat_flux"));
  stk::mesh::put_field_on_mesh(*divRadiativeHeatFlux_, *part, nullptr);

  radiationSource_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "radiation_source"));
  stk::mesh::put_field_on_mesh(*radiationSource_, *part, nullptr);

  scalarFlux_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar_flux"));
  stk::mesh::put_field_on_mesh(*scalarFlux_, *part, nullptr);

  // for non-linear residual
  scalarFluxOld_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar_flux_old"));
  stk::mesh::put_field_on_mesh(*scalarFluxOld_, *part, nullptr);

  // props; register and push
  absorptionCoeff_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "absorption_coefficient"));
  stk::mesh::put_field_on_mesh(*absorptionCoeff_, *part, nullptr);
  // possibly provided by another coupling mechanism; if so, do not push to propery evaluation
  if (!externalCoupling_)
    realm_.augment_property_map(ABSORBTION_COEFF_ID, absorptionCoeff_);

  // always register, however, do not make the user provide a value (default to zero)
  scatteringCoeff_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scattering_coefficient"));
  stk::mesh::put_field_on_mesh(*scatteringCoeff_, *part, nullptr);
  if ( activateScattering_ )
    realm_.augment_property_map(SCATTERING_COEFF_ID, scatteringCoeff_);
  
  // make sure all states are properly populated (restart can handle this)
  if ( numVolStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &dualNdVolN = dualNodalVolume_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &dualNdVolNp1 = dualNodalVolume_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlgDlNdVol
      = new CopyFieldAlgorithm(realm_, part,
                               &dualNdVolNp1, &dualNdVolN,
                               0, 1,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlgDlNdVol);
  }

}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::register_edge_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

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
RadiativeTransportEquationSystem::register_element_fields(
  stk::mesh::Part * /* part */,
  const stk::topology & /* theTopo */)
{
  // n/a
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  // push back part vector
  interiorPartVec_.push_back(part);

  // solver; interior contribution
  if ( realm_.realmUsesEdges_ ) {
    // edge-based SUCV or upwind
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
      = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theSolverAlg = NULL;
      if ( activateUpwind_ ) // only supported for edge (constructor enforces this)
        theSolverAlg = new AssembleRadTransEdgeUpwindSolverAlgorithm(realm_, part, this);
      else
        theSolverAlg = new AssembleRadTransEdgeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algType] = theSolverAlg; 
    }
    else {
      itsi->second->partVec_.push_back(part);
    }

    // add in nodal-based source terms
    const AlgorithmType algMass = SRC;
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsrc =
      solverAlgDriver_->solverAlgMap_.find(algMass);
    if ( itsrc == solverAlgDriver_->solverAlgMap_.end() ) {
      // create the solver alg
      AssembleNodeSolverAlgorithm *theAlg
        = new AssembleNodeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algMass] = theAlg;
      
      // (mu+sc)*Ib
      RadTransBlackBodyNodeSuppAlg *bbSrc
        = new RadTransBlackBodyNodeSuppAlg(realm_, this);
      theAlg->supplementalAlg_.push_back(bbSrc);
      
      // isotropic scattering
      if ( activateScattering_ ) {
        RadTransIsoScatteringNodeSuppAlg *isSrc
          = new RadTransIsoScatteringNodeSuppAlg(realm_, this);
        theAlg->supplementalAlg_.push_back(isSrc);
      }
    }
    else {
      itsrc->second->partVec_.push_back(part);
    }
  }
  else {
    // element-based uses consolidated approach fully
    stk::topology partTopo = part->topology();
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;
    
    std::tie(solverAlg, solverAlgWasBuilt) = build_or_add_part_to_solver_alg(*this, *part, solverAlgMap);

    ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
    auto& activeKernels = solverAlg->activeKernels_;
    
    if (solverAlgWasBuilt) {

      // allow option to remove SUCV, likely, not useful
      build_topo_kernel_if_requested<RadTransAdvectionSUCVElemKernel>
        (partTopo, *this, activeKernels, "advection_sucv",
         realm_.bulk_data(), this, deactivateSucv_ ? 0.0 : 1.0, *realm_.solutionOptions_, dataPreReqs);

      // lumped mass set to true
      build_topo_kernel_if_requested<RadTransAbsorptionBlackBodyElemKernel>
        (partTopo, *this, activeKernels, "absorption_black_body",
         realm_.bulk_data(), true, dataPreReqs);

      // lumped mass set to true
      build_topo_kernel_if_requested<RadTransIsotropicScatteringElemKernel>
        (partTopo, *this, activeKernels, "isotropic_scattering",
         realm_.bulk_data(), true, dataPreReqs);

      report_invalid_supp_alg_names();
    }
  }
}
//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &partTopo,
  const WallBoundaryConditionData &wallBCData)
{

  const AlgorithmType algType = WALL;

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // extract the value for user specified temperature and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  if ( userData.tempSpec_ ) {

    // push back part vector
    bcPartVec_.push_back(part);

    // check if this is an interface bc
    const bool isInterface = userData.isInterface_;

    // register germane fields (boundary data)
    intensityBc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "intensity_bc"));
    stk::mesh::put_field_on_mesh(*intensityBc_, *part, nullptr);

    emissivity_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "emissivity"));
    stk::mesh::put_field_on_mesh(*emissivity_, *part, nullptr);

    transmissivity_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "transmissivity"));
    stk::mesh::put_field_on_mesh(*transmissivity_, *part, nullptr);

    environmentalT_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "environmental_temperature"));
    stk::mesh::put_field_on_mesh(*environmentalT_, *part, nullptr);

    irradiation_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "irradiation"));
    stk::mesh::put_field_on_mesh(*irradiation_, *part, nullptr);

    bcTemperature_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature_bc"));
    stk::mesh::put_field_on_mesh(*bcTemperature_, *part, nullptr);

    assembledBoundaryArea_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_boundary_area"));
    stk::mesh::put_field_on_mesh(*assembledBoundaryArea_, *part, nullptr);

    // interior temperature is not over written by boundary value; push to bcTemperature_
    Temperature theTemp = userData.temperature_;
    std::vector<double> userSpec(1);
    userSpec[0] = theTemp.temperature_;
    ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);
    AuxFunctionAlgorithm *auxAlg
      = new AuxFunctionAlgorithm(realm_, part,
                                 bcTemperature_, theAuxFunc,
                                 stk::topology::NODE_RANK);
    
    // interface bcs expect bc temperature from elsewhere; just push this wall bc as part of initial work
    if ( isInterface )
      realm_.initCondAlg_.push_back(auxAlg);
    else 
      bcDataAlg_.push_back(auxAlg);
      
    // emissivity
    Emissivity emiss = userData.emissivity_;
    std::vector<double> userSpecEmiss(1);
    userSpecEmiss[0] = emiss.emissivity_;
    ConstantAuxFunction *theAuxFuncEmiss = new ConstantAuxFunction(0, 1, userSpecEmiss);
    AuxFunctionAlgorithm *auxAlgEmiss
      = new AuxFunctionAlgorithm(realm_, part,
                                 emissivity_, theAuxFuncEmiss,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlgEmiss);

    // transmissivity
    Transmissivity tmiss = userData.transmissivity_;
    std::vector<double> userSpecTmiss(1);
    userSpecTmiss[0] = tmiss.transmissivity_;
    ConstantAuxFunction *theAuxFuncTmiss = new ConstantAuxFunction(0, 1, userSpecTmiss);
    AuxFunctionAlgorithm *auxAlgTmiss
      = new AuxFunctionAlgorithm(realm_, part,
                                 transmissivity_, theAuxFuncTmiss,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlgTmiss);

    // environmental t
    EnvironmentalT environT = userData.environmentalT_;
    std::vector<double> userSpecEnvironT(1);
    userSpecEnvironT[0] = environT.environmentalT_;
    ConstantAuxFunction *theAuxFuncEnvironT = new ConstantAuxFunction(0, 1, userSpecEnvironT);
    AuxFunctionAlgorithm *auxAlgEnvironT
      = new AuxFunctionAlgorithm(realm_, part,
                                 environmentalT_, theAuxFuncEnvironT,
                                 stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlgEnvironT);


    // copy intensity_bc to intensity
    /*
      CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
      iBc, intensity_,
      0, 1,
      stk::topology::NODE_RANK);
      bcDataMapAlg_.push_back(theCopyAlg);
    */

    // solver; lhs: weak bc flux implementation
    if ( realm_.realmUsesEdges_ ) {
      std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi
        = solverAlgDriver_->solverAlgMap_.find(algType);
      if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
        AssembleRadTransWallSolverAlgorithm *theAlg
          = new AssembleRadTransWallSolverAlgorithm(realm_, part, this, realm_.realmUsesEdges_);
        solverAlgDriver_->solverAlgMap_[algType] = theAlg;
      }
      else {
        itsi->second->partVec_.push_back(part);
      }
    }
    else {

      // element-based uses consolidated approach fully
      auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

      AssembleElemSolverAlgorithm* solverAlg = nullptr;
      bool solverAlgWasBuilt = false;
      
      std::tie(solverAlg, solverAlgWasBuilt) = build_or_add_part_to_face_bc_solver_alg(*this, *part, solverAlgMap, "wall");
      
      ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
      auto& activeKernels = solverAlg->activeKernels_;
      
      if (solverAlgWasBuilt) {
        build_face_topo_kernel_automatic<RadTransWallElemKernel>
          (partTopo, *this, activeKernels, "rad_trans_wall_bc",
           realm_.bulk_data(), this, false, dataPreReqs);

        report_built_supp_alg_names();
      } 
    }
  }
  else {
    throw std::runtime_error("Hmmm... Does it make sense to not specify a temperature?");
  }

}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

void
RadiativeTransportEquationSystem::predict_state()
{
  // Intensity has no state
}

//--------------------------------------------------------------------------
//-------- set_current_ordinate_info ---------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::set_current_ordinate_info(
  const int k)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();
  currentWeight_ = weights_[k];
  for ( int j = 0; j < nDim; ++j )
    currentSn_[j] = Sn_[k*nDim+j];

  // extract current intensity based on k passed in
  std::stringstream ss;
  ss << k;
  const std::string incrementName = ss.str();
  const std::string theName = "intensity_" + incrementName;

  // advertise current pointer
  currentIntensity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, theName);

  // copy intensity_k -> intensity_
  copy_ordinate_intensity(*currentIntensity_, *intensity_);

}

//--------------------------------------------------------------------------
//-------- copy_ordinate_intensity -----------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::copy_ordinate_intensity(
  const ScalarFieldType &fromField,
  const ScalarFieldType &toField)
{
  field_copy(realm_.meta_data(), realm_.bulk_data(), fromField, toField, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- get_current_ordinate --------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::get_current_ordinate(
  double *Sk) const
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();
  for ( int j = 0; j < nDim; ++j )
    Sk[j] = currentSn_[j];
}

//--------------------------------------------------------------------------
//-------- get_current_ordinate_info ---------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::get_current_ordinate_info(
  double &weight,
  double *Sk) const
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();
  weight = currentWeight_;
  for ( int j = 0; j < nDim; ++j )
    Sk[j] = currentSn_[j];
}


//--------------------------------------------------------------------------
//-------- get_stefan_boltzmann --------------------------------------------
//--------------------------------------------------------------------------
double
RadiativeTransportEquationSystem::get_stefan_boltzmann() const
{
  return realm_.get_stefan_boltzmann();
}

//--------------------------------------------------------------------------
//-------- get_intensity ---------------------------------------------------
//--------------------------------------------------------------------------
ScalarFieldType *
RadiativeTransportEquationSystem::get_intensity() const
{
  return intensity_;
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::solve_and_update()
{

  assemble_boundary_area();

  if ( isInit_ ) {
    initialize_intensity();
    compute_bc_intensity();
    isInit_ = false;
  }
  
  compute_radiation_source();
  
  // extract equation system status before first iteration or first ordinate solve
  bool firstTimeStepSolve = firstTimeStepSolve_;
  
  for ( int i = 0; i < maxIterations_; ++i ) {
    
    // zero out qj, G; irradiation
    zero_out_fields();
    zero_irradiation();
    
    NaluEnv::self().naluOutputP0() << "   "
                                   << userSuppliedName_ << " Iteration: " << i+1 << "/" << maxIterations_ << std::endl;
    
    double nonLinearResidualSum = 0.0;
    double linearIterationsSum = 0.0;
    for ( int k = 0; k < ordinateDirections_; ++k ) {
      
      // unload Sk and weight for this ordinate direction k
      set_current_ordinate_info(k);
      
      // intensity RTE assemble, load_complete and solve
      assemble_and_solve(iTmp_);
      
      // update
      double timeA = NaluEnv::self().nalu_time();
      field_axpby(
        realm_.meta_data(),
        realm_.bulk_data(),
        1.0, *iTmp_,
        1.0, *intensity_, 
        realm_.get_activate_aura());
      double timeB = NaluEnv::self().nalu_time();
      timerAssemble_ += (timeB-timeA);
      
      // assemble qj, G; operates on intensity_
      assemble_fields();
      
      assemble_irradiation();
      
      // copy intensity_ back to intensity_k
      copy_ordinate_intensity(*intensity_, *currentIntensity_);
      
      // increment solve counts and norms
      linearIterationsSum += linsys_->linearSolveIterations();
      nonLinearResidualSum += linsys_->nonLinearResidual();
      
    }
    
    // save total nonlinear residual
    nonLinearResidualSum_ = nonLinearResidualSum/double(ordinateDirections_);
    
    // save the very first nonlinear residual
    if ( firstTimeStepSolve  ) {
      firstNonLinearResidualSum_ = std::max(std::numeric_limits<double>::epsilon(), nonLinearResidualSum_);
      firstTimeStepSolve = false;
    }
    
    // normalize_irradiation
    normalize_irradiation();
    
    // compute boundary intensity
    compute_bc_intensity();
    
    // compute divRadFLux and norm
    compute_div_norm();
    copy_ordinate_intensity(*scalarFlux_, *scalarFluxOld_);
    
    // dump norm and averages
    NaluEnv::self().naluOutputP0()
      << "EqSystem Name:       " << userSuppliedName_ << std::endl
      << "   aver iters      = " << linearIterationsSum/double(ordinateDirections_) << std::endl
      << "nonlinearResidNrm  = " << nonLinearResidualSum_
      << " scaled: " << nonLinearResidualSum_/firstNonLinearResidualSum_ << std::endl
      << "Scalar flux norm   = " << systemL2Norm_ << std::endl;
    NaluEnv::self().naluOutputP0() << std::endl;
    
    // check for convergence
    if ( system_is_converged() ) {
      NaluEnv::self().naluOutputP0() << "Local Iteration Intensity Equation System Converged" << std::endl;
      break;
    }
  }
}

//--------------------------------------------------------------------------
//-------- system_is_converged ---------------------------------------------
//--------------------------------------------------------------------------
bool
RadiativeTransportEquationSystem::system_is_converged()
{
  bool isConverged = true;
  if ( NULL != linsys_ ) {
    const double bestConverged
      = std::min(nonLinearResidualSum_/firstNonLinearResidualSum_, systemL2Norm_);
    isConverged = (bestConverged <  convergenceTolerance_ );
  }
  return isConverged;
}

//--------------------------------------------------------------------------
//-------- provide_scaled_norm ---------------------------------------------
//--------------------------------------------------------------------------
double
RadiativeTransportEquationSystem::provide_scaled_norm()
{
  return nonLinearResidualSum_/firstNonLinearResidualSum_;
}

//--------------------------------------------------------------------------
//-------- provide_norm ---------------------------------------------
//--------------------------------------------------------------------------
double
RadiativeTransportEquationSystem::provide_norm()
{
  return nonLinearResidualSum_;
}

//--------------------------------------------------------------------------
//-------- initialize_intensity --------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::initialize_intensity()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const double inv_pi = 1.0/acos(-1.0);
  const double sb = get_stefan_boltzmann();

  // select all nodes
  stk::mesh::Selector s_all_nodes
     = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
     &stk::mesh::selectUnion(interiorPartVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();
    double *intensity = stk::mesh::field_data(*intensity_, b);
    const double *temperature = stk::mesh::field_data(*temperature_, b);

    for ( size_t k = 0 ; k < length ; ++k ) {
      const double T = temperature[k];
      intensity[k] = inv_pi*sb*T*T*T*T;
    }
  }

  // now copy to all set of intensity
  for ( int k = 0; k < ordinateDirections_; ++k ) {
     std::stringstream ss;
     ss << k;
     const std::string incrementName = ss.str();
     const std::string theName = "intensity_" + incrementName;
     ScalarFieldType *kthIntensity = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, theName);
     copy_ordinate_intensity(*intensity_, *kthIntensity);
   }

}

//--------------------------------------------------------------------------
//-------- compute_bc_intensity --------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::compute_bc_intensity()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const double inv_pi = 1.0/acos(-1.0);
  const double sb = get_stefan_boltzmann();

  // select all faces
  stk::mesh::Selector s_union = stk::mesh::selectUnion(bcPartVec_)
    &(meta_data.locally_owned_part() | meta_data.globally_shared_part());

  stk::mesh::BucketVector const& bc_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_union );
  for ( stk::mesh::BucketVector::const_iterator ib = bc_node_buckets.begin();
        ib != bc_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();
    double *intensityBc = stk::mesh::field_data(*intensityBc_, b);
    const double *temperature = stk::mesh::field_data(*bcTemperature_, b);
    const double *irradiation = stk::mesh::field_data(*irradiation_, b);
    const double *emissivity = stk::mesh::field_data(*emissivity_, b);
    const double *transmissivity = stk::mesh::field_data(*transmissivity_, b);
    const double *environmentalT = stk::mesh::field_data(*environmentalT_, b);

    for ( size_t k = 0 ; k < length ; ++k ) {
      const double T = temperature[k];
      const double eps = emissivity[k];
      const double tau = transmissivity[k];
      const double envT = environmentalT[k];

      intensityBc[k] = inv_pi*(tau*sb*envT*envT*envT*envT
                               + eps*sb*T*T*T*T
                               + (1.0-eps-tau)*irradiation[k]);
      
    }
  }
}

//--------------------------------------------------------------------------
//-------- compute_radiation_source ----------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::compute_radiation_source()
{
  // external coupling will provide radiation source through xfer
  if ( externalCoupling_ )
    return;

  // otherwise, proceed with computing source term based on what this realm knows
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const double inv_pi = 1.0/acos(-1.0);
  const double sb = get_stefan_boltzmann();

  // select all nodes
  stk::mesh::Selector s_all_nodes
     = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
     &stk::mesh::selectUnion(interiorPartVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();

    double *radiationSource = stk::mesh::field_data(*radiationSource_, b);
    const double *temperature = stk::mesh::field_data(*temperature_, b);
    const double * absorption = stk::mesh::field_data(*absorptionCoeff_, b);

    for ( size_t k = 0 ; k < length ; ++k ) {
      const double T = temperature[k];
      radiationSource[k] = absorption[k]*sb*T*T*T*T*inv_pi;
    }
  }
}
//--------------------------------------------------------------------------
//-------- zero_out_fields -------------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::zero_out_fields()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // define interior node selectors
  stk::mesh::Selector s_all_nodes_interior
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(interiorPartVec_);

  stk::mesh::BucketVector const& int_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes_interior );
  for ( stk::mesh::BucketVector::const_iterator ib = int_node_buckets.begin();
        ib != int_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();
    double * scalarFlux = stk::mesh::field_data(*scalarFlux_, b);
    double * radiativeHeatFlux = stk::mesh::field_data(*radiativeHeatFlux_, b);

    for ( size_t k = 0 ; k < length ; ++k ) {
      // scalars
      scalarFlux[k] = 0.0;
      // vectors
      const size_t offSet = k*nDim;
      for ( int j = 0; j < nDim; ++j ) {
        radiativeHeatFlux[offSet+j] = 0.0;
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- zero_irradiation ------------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::zero_irradiation()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // boundary
  stk::mesh::Selector s_all_nodes_bc
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(bcPartVec_);

  stk::mesh::BucketVector const& bc_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes_bc );
  for ( stk::mesh::BucketVector::const_iterator ib = bc_node_buckets.begin();
        ib != bc_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();
    double * irradiation = stk::mesh::field_data(*irradiation_, b);
    for ( size_t k = 0 ; k < length ; ++k ) {
      // scalars
      irradiation[k] = 0.0;
    }
  }
}

//--------------------------------------------------------------------------
//-------- assemble_boundary_area ------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::assemble_boundary_area()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  GenericFieldType *exposedAreaVec = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");

  // zero all nodes for assemble boundary area
  stk::mesh::Selector s_all_nodes_bc
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(bcPartVec_);

  stk::mesh::BucketVector const& bc_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes_bc );
  for ( stk::mesh::BucketVector::const_iterator ib = bc_node_buckets.begin();
        ib != bc_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();
    double * assembledBCA = stk::mesh::field_data(*assembledBoundaryArea_, b);
    for ( size_t k = 0 ; k < length ; ++k ) {
      // scalars
      assembledBCA[k] = 0.0;
    }
  }

  // setup for buckets; union parts and ask for locally owned
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(bcPartVec_);
  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec, b, k);

      // face node relations for nodal gather
      stk::mesh::Entity const * face_node_rels = b.begin_nodes(k);

      // number of nodes (equals ips) and face data
      int num_face_ip = b.num_nodes(k);

      for ( int ip = 0; ip < num_face_ip; ++ip ) {

        // nearest node maps to face ip...
        const int nn = ip;
        stk::mesh::Entity nodeNN = face_node_rels[nn];

        // pointer to fields to assemble
        double * assembledBCA = stk::mesh::field_data(*assembledBoundaryArea_, nodeNN);

        const int offSet = ip*nDim;
        double amag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          amag += areaVec[offSet+j]*areaVec[offSet+j];
        }
        amag = std::sqrt(amag);

        *assembledBCA += amag;
      }
    }
  }

  stk::mesh::parallel_sum(bulk_data, {assembledBoundaryArea_});

  if ( realm_.hasPeriodic_) {
    const bool bypassFieldCheck = false; // fields are not defined at all slave/master node pairs
    realm_.periodic_field_update(assembledBoundaryArea_, 1, bypassFieldCheck);
  }

}

//--------------------------------------------------------------------------
//-------- assemble_fields -------------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::assemble_fields()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // current weights
  double weight = currentWeight_;
  const double *p_Sk = &currentSn_[0];

  // define some common selectors
  stk::mesh::Selector s_all_nodes_interior
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(interiorPartVec_);

  stk::mesh::BucketVector const& int_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes_interior );
  for ( stk::mesh::BucketVector::const_iterator ib = int_node_buckets.begin();
        ib != int_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();
    double * radiativeHeatFlux = stk::mesh::field_data(*radiativeHeatFlux_, b);
    double * scalarFlux = stk::mesh::field_data(*scalarFlux_, b);
    const double * intensity = stk::mesh::field_data(*intensity_, b);
    for ( size_t k = 0 ; k < length ; ++k ) {
      const double I = intensity[k];
      // scalars
      scalarFlux[k] += I*weight;
      const size_t offSet = k*nDim;
      for ( int j = 0; j < nDim; ++j ) {
        radiativeHeatFlux[offSet+j] += I*p_Sk[j]*weight;
      }
    }
  }
}


//--------------------------------------------------------------------------
//-------- assemble_irradiation --------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::assemble_irradiation()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  GenericFieldType *exposedAreaVec = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");

  // current weights
  double weight = currentWeight_;
  const double *p_Sk = &currentSn_[0];

  // params
  const bool useShifted = realm_.realmUsesEdges_;

  // nodal fields to gather; gather everything other than what we are assembling
  std::vector<double> ws_intensity;

  // geometry related to populate
  std::vector<double> ws_shape_function;

  // setup for buckets; union parts and ask for locally owned
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(bcPartVec_);
  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // extract master element specifics
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsIp = meFC->numIntPoints_;

    // resize some things; algorithm related
    ws_intensity.resize(nodesPerFace);
    ws_shape_function.resize(numScsIp*nodesPerFace);

    // pointers
    double *p_intensity = &ws_intensity[0];
    double *p_shape_function = &ws_shape_function[0];

    if ( useShifted )
      meFC->shifted_shape_fcn(&p_shape_function[0]);
    else
      meFC->shape_fcn(&p_shape_function[0]);

    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec, b, k);

      // face node relations for nodal gather
      stk::mesh::Entity const * face_node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        // gather scalar
        p_intensity[ni] = *stk::mesh::field_data(*intensity_, face_node_rels[ni]);
      }

      // start the assembly
      for ( int ip = 0; ip < num_nodes; ++ip ) {

        // nearest node maps to face ip...
        const int nn = ip;
        stk::mesh::Entity nodeNN = face_node_rels[nn];

        // pointer to fields to assemble
        double *irrad = stk::mesh::field_data(*irradiation_, nodeNN);

        // interpolate to scs point; operate on saved off ws_field
        double iBc = 0.0;
        const int offSetSF = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_shape_function[offSetSF+ic];
          iBc += r*p_intensity[ic];
        }

        // offset to face area vector; compute area mag
        const int offSet = ip*nDim;
        double amag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          amag += areaVec[offSet+j]*areaVec[offSet+j];
        }
        amag = std::sqrt(amag);

        // see if this ordinate direction should count..
        double dot = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double nj = areaVec[offSet+j]/amag;
          dot += nj*p_Sk[j];
        }

        if ( dot > 0.0 )
          *irrad += weight*iBc*dot*amag;
      }
    }
  }

  // let's not parallel assemble until the normalization..
}

//--------------------------------------------------------------------------
//-------- normalize_irradiation -------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::normalize_irradiation()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  stk::mesh::parallel_sum(bulk_data, {irradiation_});

  if ( realm_.hasPeriodic_) {
    const bool bypassFieldCheck = false; // fields are not defined at all slave/master node pairs
    realm_.periodic_field_update(irradiation_, 1, bypassFieldCheck);
  }

  // boundary nodes
  stk::mesh::Selector s_all_nodes_bc
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(bcPartVec_);

  stk::mesh::BucketVector const& bc_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes_bc );
  for ( stk::mesh::BucketVector::const_iterator ib = bc_node_buckets.begin();
        ib != bc_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();
    const double * assembledBCA = stk::mesh::field_data(*assembledBoundaryArea_, b);
    double * irradiation = stk::mesh::field_data(*irradiation_, b);
    for ( size_t k = 0 ; k < length ; ++k ) {
      irradiation[k] /= assembledBCA[k];
    }
  }
}


//--------------------------------------------------------------------------
//-------- compute_div_norm ------------------------------------------------
//--------------------------------------------------------------------------
void
RadiativeTransportEquationSystem::compute_div_norm()
{

  const double sb = get_stefan_boltzmann();

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors
  stk::mesh::Selector s_all_nodes_interior
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(interiorPartVec_);

  stk::mesh::BucketVector const& int_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes_interior );

  double l2Norm = 0.0;
  for ( stk::mesh::BucketVector::const_iterator ib = int_node_buckets.begin();
        ib != int_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();
    double * divRadiativeHeatFlux = stk::mesh::field_data(*divRadiativeHeatFlux_, b);
    double * scalarFlux = stk::mesh::field_data(*scalarFlux_, b);
    double * scalarFluxOld = stk::mesh::field_data(*scalarFluxOld_, b);
    double * temperature = stk::mesh::field_data(*temperature_, b);
    double * absorption = stk::mesh::field_data(*absorptionCoeff_, b);

    for ( size_t k = 0 ; k < length ; ++k ) {
      const double T = temperature[k];
      const double Gold = scalarFluxOld[k];
      const double G = scalarFlux[k];

      divRadiativeHeatFlux[k] = absorption[k]*(4.0*sb*T*T*T*T-G);
      l2Norm += (G-Gold)*(G-Gold);
    }
  }

  // parallel assemble sqrt(l2 norm)
  l2Norm = std::sqrt(l2Norm);
  double g_l2Norm = 0.0;
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &l2Norm, &g_l2Norm, 1);
  systemL2Norm_ = g_l2Norm/realm_.l2Scaling_;

}

} // namespace nalu
} // namespace Sierra
