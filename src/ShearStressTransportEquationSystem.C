/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <ShearStressTransportEquationSystem.h>
#include <AlgorithmDriver.h>
#include <ComputeSSTMaxLengthScaleElemAlgorithm.h>
#include <FieldFunctions.h>
#include <master_element/MasterElement.h>
#include <NaluEnv.h>
#include <SpecificDissipationRateEquationSystem.h>
#include <SolutionOptions.h>
#include <TurbKineticEnergyEquationSystem.h>
#include <Realm.h>

// stk_util
#include <stk_util/parallel/Parallel.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// basic c++
#include <cmath>
#include <vector>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ShearStressTransportEquationSystem - manage SST
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ShearStressTransportEquationSystem::ShearStressTransportEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "ShearStressTransportWrap"),
    tkeEqSys_(NULL),
    sdrEqSys_(NULL),
    tke_(NULL),
    sdr_(NULL),
    minDistanceToWall_(NULL),
    fOneBlending_(NULL),
    maxLengthScale_(NULL),
    isInit_(true),
    sstMaxLengthScaleAlgDriver_(NULL)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create momentum and pressure
  tkeEqSys_= new TurbKineticEnergyEquationSystem(eqSystems);
  sdrEqSys_ = new SpecificDissipationRateEquationSystem(eqSystems);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ShearStressTransportEquationSystem::~ShearStressTransportEquationSystem()
{
  if ( NULL != sstMaxLengthScaleAlgDriver_ )
    delete sstMaxLengthScaleAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::initialize()
{
  // let equation systems that are owned some information
  tkeEqSys_->convergenceTolerance_ = convergenceTolerance_;
  sdrEqSys_->convergenceTolerance_ = convergenceTolerance_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int numStates = realm_.number_of_states();

  // re-register tke and sdr for convenience
  tke_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke", numStates));
  stk::mesh::put_field_on_mesh(*tke_, *part, nullptr);
  sdr_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_dissipation_rate", numStates));
  stk::mesh::put_field_on_mesh(*sdr_, *part, nullptr);

  // SST parameters that everyone needs
  minDistanceToWall_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*minDistanceToWall_, *part, nullptr);
  fOneBlending_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "sst_f_one_blending"));
  stk::mesh::put_field_on_mesh(*fOneBlending_, *part, nullptr);
  
  // DES model
  if ( SST_DES == realm_.solutionOptions_->turbulenceModel_ ) {
    maxLengthScale_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "sst_max_length_scale"));
    stk::mesh::put_field_on_mesh(*maxLengthScale_, *part, nullptr);
  }

  // add to restart field
  realm_.augment_restart_variable_list("minimum_distance_to_wall");
  realm_.augment_restart_variable_list("sst_f_one_blending");
}


//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;
  
  if ( SST_DES == realm_.solutionOptions_->turbulenceModel_ ) {

    if ( NULL == sstMaxLengthScaleAlgDriver_ )
      sstMaxLengthScaleAlgDriver_ = new AlgorithmDriver(realm_);

    // create edge algorithm
    std::map<AlgorithmType, Algorithm *>::iterator it =
      sstMaxLengthScaleAlgDriver_->algMap_.find(algType);

    if ( it == sstMaxLengthScaleAlgDriver_->algMap_.end() ) {
      ComputeSSTMaxLengthScaleElemAlgorithm *theAlg
        = new ComputeSSTMaxLengthScaleElemAlgorithm(realm_, part);
      sstMaxLengthScaleAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &/*wallBCData*/)
{
  // push mesh part
  wallBcPart_.push_back(part);
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::solve_and_update()
{
  // wrap timing
  // SST_FIXME: deal with timers; all on misc for SSTEqs double timeA, timeB;
  if ( isInit_ ) {
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    sdrEqSys_->assemble_nodal_gradient();
    clip_min_distance_to_wall();
    
    // deal with DES option
    if ( SST_DES == realm_.solutionOptions_->turbulenceModel_ )
      sstMaxLengthScaleAlgDriver_->execute();

    isInit_ = false;
  } else if (realm_.has_mesh_motion()) {
    if (realm_.currentNonlinearIteration_ == 1)
      clip_min_distance_to_wall();

    if (SST_DES == realm_.solutionOptions_->turbulenceModel_)
      sstMaxLengthScaleAlgDriver_->execute();
  }

  // compute blending for SST model
  compute_f_one_blending();

  // SST effective viscosity for k and omega
  tkeEqSys_->compute_effective_diff_flux_coeff();
  sdrEqSys_->compute_effective_diff_flux_coeff();

  // wall values
  tkeEqSys_->compute_wall_model_parameters();
  sdrEqSys_->compute_wall_model_parameters();

  // start the iteration loop
  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << name_ << std::endl;

    // tke and sdr assemble, load_complete and solve; Jacobi iteration
    tkeEqSys_->assemble_and_solve(tkeEqSys_->kTmp_);
    sdrEqSys_->assemble_and_solve(sdrEqSys_->wTmp_);

    // update each
    update_and_clip();

    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    sdrEqSys_->assemble_nodal_gradient();
  }

}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::initial_work()
{
  // do not lett he user specify a negative field
  const double clipValue = 1.0e-8;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // required fields
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  ScalarFieldType *viscosity = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");

  // required fields with state
  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*sdr_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    const double *visc = stk::mesh::field_data(*viscosity, b);
    const double *rho = stk::mesh::field_data(*density, b);
    double *tke = stk::mesh::field_data(tkeNp1, b);
    double *sdr = stk::mesh::field_data(sdrNp1, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      const double tkeNew = tke[k];
      const double sdrNew = sdr[k];
      
      if ( (tkeNew >= 0.0) && (sdrNew > 0.0) ) {
        // nothing
      }
      else if ( (tkeNew < 0.0) && (sdrNew < 0.0) ) {
        // both negative;
        tke[k] = clipValue;
        sdr[k] = rho[k]*clipValue/visc[k];
      }
      else if ( tkeNew < 0.0 ) {
        tke[k] = visc[k]*sdrNew/rho[k];
        sdr[k] = sdrNew;
      }
      else {
        sdr[k] = rho[k]*tkeNew/visc[k];
        tke[k] = tkeNew;
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- post_adapt_work -------------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::post_adapt_work()
{
  if ( realm_.process_adaptivity() ) {
    NaluEnv::self().naluOutputP0() << "--ShearStressTransportEquationSystem::post_adapt_work()" << std::endl;

    if ( SST_DES == realm_.solutionOptions_->turbulenceModel_ )
      sstMaxLengthScaleAlgDriver_->execute();

    // wall values
    tkeEqSys_->compute_wall_model_parameters();
    sdrEqSys_->compute_wall_model_parameters();
  }

}

//--------------------------------------------------------------------------
//-------- update_and_clip() -----------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::update_and_clip()
{
  const double clipValue = 1.0e-8;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // required fields
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  ScalarFieldType *viscosity = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  ScalarFieldType *turbViscosity = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");

  // required fields with state
  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*turbViscosity);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    const double *visc = stk::mesh::field_data(*viscosity, b);
    const double *rho = stk::mesh::field_data(*density, b);
    const double *kTmp = stk::mesh::field_data(*tkeEqSys_->kTmp_, b);
    const double *wTmp = stk::mesh::field_data(*sdrEqSys_->wTmp_, b);
    double *tke = stk::mesh::field_data(tkeNp1, b);
    double *sdr = stk::mesh::field_data(sdrNp1, b);
    double *tvisc = stk::mesh::field_data(*turbViscosity, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      const double tkeNew = tke[k] + kTmp[k];
      const double sdrNew = sdr[k] + wTmp[k];
      
      if ( (tkeNew >= 0.0) && (sdrNew > 0.0) ) {
        // if all is well
        tke[k] = tkeNew;
        sdr[k] = sdrNew;
      }
      else if ( (tkeNew < 0.0) && (sdrNew < 0.0) ) {
        // both negative; set k to small, tvisc to molecular visc and use Prandtl/Kolm for sdr
        tke[k] = clipValue;
        tvisc[k] = visc[k];
        sdr[k] = rho[k]*clipValue/visc[k];
      }
      else if ( tkeNew < 0.0 ) {
        // only tke is off; reset tvisc to molecular visc and compute new tke appropriately
        tvisc[k] = visc[k];
        tke[k] = visc[k]*sdrNew/rho[k];
        sdr[k] = sdrNew;
      }
      else {
        // only sdr if off; reset tvisc to molecular visc and compute new sdr appropriately
        tvisc[k] = visc[k];
        sdr[k] = rho[k]*tkeNew/visc[k];
        tke[k] = tkeNew;
      }
    }
  }

  // parallel assemble clipped value
  if (realm_.debug()) {
    NaluEnv::self().naluOutputP0() << "Add SST clipping diagnostic" << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- clip_min_distance_to_wall ---------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::clip_min_distance_to_wall()
{
  // if this is a restart, then min distance has already been clipped
  if (realm_.restarted_simulation())
    return;

  // okay, no restart: proceed with clipping of minimum wall distance
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract fields required
  GenericFieldType *exposedAreaVec = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // define vector of parent topos; should always be UNITY in size
  std::vector<stk::topology> parentTopo;

  // selector
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
      &stk::mesh::selectUnion(wallBcPart_);

   stk::mesh::BucketVector const& face_buckets =
     realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
   for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
         ib != face_buckets.end() ; ++ib ) {
     stk::mesh::Bucket & b = **ib ;

     // extract connected element topology
     b.parent_topology(stk::topology::ELEMENT_RANK, parentTopo);
     ThrowAssert ( parentTopo.size() == 1 );
     stk::topology theElemTopo = parentTopo[0];

     // extract master element
     MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theElemTopo);

     const stk::mesh::Bucket::size_type length   = b.size();

     for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

       // get face
       stk::mesh::Entity face = b[k];
       int num_face_nodes = bulk_data.num_nodes(face);

       // pointer to face data
       const double * areaVec = stk::mesh::field_data(*exposedAreaVec, face);

       // extract the connected element to this exposed face; should be single in size!
       const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
       ThrowAssert( bulk_data.num_elements(face) == 1 );

       // get element; its face ordinal number and populate face_node_ordinals
       stk::mesh::Entity element = face_elem_rels[0];
       const int face_ordinal = bulk_data.begin_element_ordinals(face)[0];
       const int *face_node_ordinals = meSCS->side_node_ordinals(face_ordinal);

       // get the relations off of element
       stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);

       // loop over face nodes
       for ( int ip = 0; ip < num_face_nodes; ++ip ) {

         const int offSetAveraVec = ip*nDim;

         const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);
         const int nearestNode = face_node_ordinals[ip];

         // left and right nodes; right is on the face; left is the opposing node
         stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
         stk::mesh::Entity nodeR = elem_node_rels[nearestNode];

         // extract nodal fields
         const double * coordL = stk::mesh::field_data(*coordinates, nodeL );
         const double * coordR = stk::mesh::field_data(*coordinates, nodeR );

         double aMag = 0.0;
         for ( int j = 0; j < nDim; ++j ) {
           const double axj = areaVec[offSetAveraVec+j];
           aMag += axj*axj;
         }
         aMag = std::sqrt(aMag);

         // form unit normal and determine yp (approximated by 1/4 distance along edge)
         double ypbip = 0.0;
         for ( int j = 0; j < nDim; ++j ) {
           const double nj = areaVec[offSetAveraVec+j]/aMag;
           const double ej = 0.25*(coordR[j] - coordL[j]);
           ypbip += nj*ej*nj*ej;
         }
         ypbip = std::sqrt(ypbip);

         // assemble to nodal quantities
         double *minD = stk::mesh::field_data(*minDistanceToWall_, nodeR );

         *minD = std::max(*minD, ypbip);
       }
     }
   }
}

//--------------------------------------------------------------------------
//-------- compute_f_one_blending ------------------------------------------
//--------------------------------------------------------------------------
void
ShearStressTransportEquationSystem::compute_f_one_blending()
{
  // compute fone with parameters appropriate for 2003 SST implementation
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // model parameters
  const double betaStar = realm_.get_turb_model_constant(TM_betaStar);
  const double sigmaWTwo = realm_.get_turb_model_constant(TM_sigmaWTwo);
  const double CDkwClip = 1.0e-10; // 2003 SST

  // required fields with state; min_distance is fine
  ScalarFieldType &sdrNp1 = sdr_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);

  // fields not saved off
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  ScalarFieldType *viscosity = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  VectorFieldType *dkdx = tkeEqSys_->dkdx_;
  VectorFieldType *dwdx = sdrEqSys_->dwdx_;

  //select all nodes (locally and shared)
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*fOneBlending_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // fields; supplemental and non-const fOne and ftwo
    const double * sdr = stk::mesh::field_data(sdrNp1, b);
    const double * tke = stk::mesh::field_data(tkeNp1, b);
    const double * minD = stk::mesh::field_data(*minDistanceToWall_, b);
    const double * rho = stk::mesh::field_data(*density, b);
    const double * mu = stk::mesh::field_data(*viscosity, b);
    const double * dk = stk::mesh::field_data(*dkdx, b);
    const double * dw = stk::mesh::field_data(*dwdx, b);
    double * fOne = stk::mesh::field_data(*fOneBlending_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // compute cross diff
      double crossDiff = 0.0;
      for ( int j = 0; j < nDim; ++j ) {
        crossDiff += dk[k*nDim+j]*dw[k*nDim+j];
      }

      // some temps
      const double minDSq = minD[k]*minD[k];
      const double trbDiss = std::sqrt(tke[k])/betaStar/sdr[k]/minD[k];
      const double lamDiss = 500.0*mu[k]/rho[k]/sdr[k]/minDSq;
      const double CDkw = std::max(2.0*rho[k]*sigmaWTwo*crossDiff/sdr[k], CDkwClip);

      // arguments
      const double fArgOne = std::min(std::max(trbDiss, lamDiss), 4.0*rho[k]*sigmaWTwo*tke[k]/CDkw/minDSq);

      // real deal
      fOne[k] = std::tanh(fArgOne*fArgOne*fArgOne*fArgOne);

    }
  }
}

} // namespace nalu
} // namespace Sierra
