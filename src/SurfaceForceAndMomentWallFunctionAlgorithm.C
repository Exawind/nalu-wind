/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <SurfaceForceAndMomentWallFunctionAlgorithm.h>
#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// basic c++
#include <cmath>
#include <fstream>
#include <iomanip>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// SurfaceForceAndMomentWallFunctionAlgorithm - post process sigma_ijnjdS and 
//                                  tau_wall via lumped nodal projection (area 
//                                  weighed) Driven by SurfaceForceAndMomentAlgDriver
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SurfaceForceAndMomentWallFunctionAlgorithm::SurfaceForceAndMomentWallFunctionAlgorithm(
  Realm &realm,
  stk::mesh::PartVector &partVec,
  const std::string &outputFileName,
  const int &frequency,
  const std::vector<double > &parameters,
  const bool &useShifted)
  : Algorithm(realm, partVec),
    outputFileName_(outputFileName),
    frequency_(frequency),
    parameters_(parameters),
    useShifted_(useShifted),
    yplusCrit_(11.63),
    elog_(9.8),
    kappa_(realm.get_turb_model_constant(TM_kappa)),
    coordinates_(NULL),
    velocity_(NULL),
    pressure_(NULL),
    pressureForce_(NULL),
    tauWall_(NULL),
    yplus_(NULL),
    bcVelocity_(NULL),
    density_(NULL),
    viscosity_(NULL),
    wallFrictionVelocityBip_(NULL),
    wallNormalDistanceBip_(NULL),
    exposedAreaVec_(NULL),
    assembledArea_(NULL),
    w_(16)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  pressure_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  pressureForce_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "pressure_force");
  tauWall_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "tau_wall");
  yplus_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "yplus");
  bcVelocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "wall_velocity_bc");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  wallFrictionVelocityBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_friction_velocity_bip");
  wallNormalDistanceBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_normal_distance_bip");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  assembledArea_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_area_force_moment_wf");

  // error check on params
  const size_t nDim = meta_data.spatial_dimension();
  if ( parameters_.size() > nDim )
    throw std::runtime_error("SurfaceForce: parameter length wrong; expect nDim");

  // make sure that the wall function params are registered
  if ( NULL == wallFrictionVelocityBip_ )
    throw std::runtime_error("SurfaceForce: wall friction velocity is not registered; wall bcs and post processing must be consistent");

  // deal with file name and banner
  if ( NaluEnv::self().parallel_rank() == 0 ) {
    std::ofstream myfile;
    myfile.open(outputFileName_.c_str());
    myfile << std::setw(w_) 
           << "Time" << std::setw(w_) 
           << "Fpx"  << std::setw(w_) << "Fpy" << std::setw(w_)  << "Fpz" << std::setw(w_) 
           << "Fvx"  << std::setw(w_) << "Fvy" << std::setw(w_)  << "Fvz" << std::setw(w_) 
           << "Mtx"  << std::setw(w_) << "Mty" << std::setw(w_)  << "Mtz" << std::setw(w_) 
           << "Y+min" << std::setw(w_) << "Y+max"<< std::endl;
    myfile.close();
  }
 }

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SurfaceForceAndMomentWallFunctionAlgorithm::~SurfaceForceAndMomentWallFunctionAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceForceAndMomentWallFunctionAlgorithm::execute()
{

  // check to see if this is a valid step to process output file
  const int timeStepCount = realm_.get_time_step_count();
  const bool processMe = (timeStepCount % frequency_) == 0 ? true : false;

  // do not waste time here
  if ( !processMe )
    return;

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // set min and max values
  double yplusMin = 1.0e8;
  double yplusMax = -1.0e8;

  // bip values
  std::vector<double> uBip(nDim);
  std::vector<double> uBcBip(nDim);
  std::vector<double> unitNormal(nDim);

  // tangential work array
  std::vector<double> uiTangential(nDim);
  std::vector<double> uiBcTangential(nDim);

  // pointers to fixed values
  double *p_uBip = &uBip[0];
  double *p_uBcBip = &uBcBip[0];
  double *p_unitNormal= &unitNormal[0];
  double *p_uiTangential = &uiTangential[0];
  double *p_uiBcTangential = &uiBcTangential[0];

  // nodal fields to gather
  std::vector<double> ws_velocityNp1;
  std::vector<double> ws_bcVelocity;
  std::vector<double> ws_pressure;
  std::vector<double> ws_density;
  std::vector<double> ws_viscosity;

  // master element
  std::vector<double> ws_face_shape_function;

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);
  const double currentTime = realm_.get_current_time();

  // local force and MomentWallFunction; i.e., to be assembled
  double l_force_moment[9] = {};

  // work force, MomentWallFunction and radius; i.e., to be pused to cross_product()
  double ws_p_force[3] = {};
  double ws_v_force[3] = {};
  double ws_t_force[3] = {};
  double ws_moment[3] = {};
  double ws_radius[3] = {};

  // centroid
  double centroid[3] = {};
  for ( size_t k = 0; k < parameters_.size(); ++k)
    centroid[k] = parameters_[k];

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->num_integration_points();

    // mapping from ip to nodes for this ordinal
    const int *faceIpNodeMap = meFC->ipNodeMap();

    // algorithm related; element
    ws_velocityNp1.resize(nodesPerFace*nDim);
    ws_bcVelocity.resize(nodesPerFace*nDim);
    ws_pressure.resize(nodesPerFace);
    ws_density.resize(nodesPerFace);
    ws_viscosity.resize(nodesPerFace);
    ws_face_shape_function.resize(numScsBip*nodesPerFace);

    // pointers
    double *p_velocityNp1 = &ws_velocityNp1[0];
    double *p_bcVelocity = &ws_bcVelocity[0];
    double *p_pressure = &ws_pressure[0];
    double *p_density = &ws_density[0];
    double *p_viscosity = &ws_viscosity[0];
    double *p_face_shape_function = &ws_face_shape_function[0];

    // shape functions
    if ( useShifted_ )
      meFC->shifted_shape_fcn(&p_face_shape_function[0]);
    else
      meFC->shape_fcn(&p_face_shape_function[0]);

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get face
      stk::mesh::Entity face = b[k];

      // face node relations
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);

      //======================================
      // gather nodal data off of face
      //======================================
      for ( int ni = 0; ni < nodesPerFace; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];

        // gather scalars
        p_pressure[ni]    = *stk::mesh::field_data(*pressure_, node);
        p_density[ni]    = *stk::mesh::field_data(densityNp1, node);
        p_viscosity[ni] = *stk::mesh::field_data(*viscosity_, node);

        // gather vectors
        double * uNp1 = stk::mesh::field_data(velocityNp1, node);
        double * uBc = stk::mesh::field_data(*bcVelocity_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_velocityNp1[offSet+j] = uNp1[j];
          p_bcVelocity[offSet+j] = uBc[j];
        }
      }

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
      const double *wallNormalDistanceBip = stk::mesh::field_data(*wallNormalDistanceBip_, face);
      const double *wallFrictionVelocityBip = stk::mesh::field_data(*wallFrictionVelocityBip_, face);

      for ( int ip = 0; ip < numScsBip; ++ip ) {

        // offsets
        const int offSetAveraVec = ip*nDim;
        const int offSetSF_face = ip*nodesPerFace;
        const int localFaceNode = faceIpNodeMap[ip];
        
        // zero out vector quantities; squeeze in aMag
        double aMag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          p_uBip[j] = 0.0;
          p_uBcBip[j] = 0.0;
          const double axj = areaVec[offSetAveraVec+j];
          aMag += axj*axj;
        }
        aMag = std::sqrt(aMag);

        // interpolate to bip
        double pBip = 0.0;
        double rhoBip = 0.0;
        double muBip = 0.0;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
          pBip += r*p_pressure[ic];
          rhoBip += r*p_density[ic];
          muBip += r*p_viscosity[ic];
          const int offSetFN = ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_uBip[j] += r*p_velocityNp1[offSetFN+j];
            p_uBcBip[j] += r*p_bcVelocity[offSetFN+j];
          }
        }

        // form unit normal
        for ( int j = 0; j < nDim; ++j ) {
          p_unitNormal[j] = areaVec[offSetAveraVec+j]/aMag;
        }

        // determine tangential velocity
        double uTangential = 0.0;
        for ( int i = 0; i < nDim; ++i ) {
          double uiTan = 0.0;
          double uiBcTan = 0.0;
          for ( int j = 0; j < nDim; ++j ) {
            const double ninj = p_unitNormal[i]*p_unitNormal[j];
            if ( i==j ) {
              const double om_nini = 1.0 - ninj;
              uiTan += om_nini*p_uBip[j];
              uiBcTan += om_nini*p_uBcBip[j];
            }
            else {
              uiTan -= ninj*p_uBip[j];
              uiBcTan -= ninj*p_uBcBip[j];
            }
          }
          // save off tangential components and augment magnitude
          p_uiTangential[i] = uiTan;
          p_uiBcTangential[i] = uiBcTan;
          uTangential += (uiTan-uiBcTan)*(uiTan-uiBcTan);
        }
        uTangential = std::sqrt(uTangential);

        // extract bip data
        const double yp = wallNormalDistanceBip[ip];
        const double utau= wallFrictionVelocityBip[ip];

        // determine yplus
        const double yplusBip = rhoBip*yp*utau/muBip;

        // min and max
        yplusMin = std::min(yplusMin, yplusBip);
        yplusMax = std::max(yplusMax, yplusBip);

        double lambda = muBip/yp*aMag;
        if ( yplusBip > yplusCrit_)
          lambda = rhoBip*kappa_*utau/std::log(elog_*yplusBip)*aMag;

        // extract nodal fields
        stk::mesh::Entity node = face_node_rels[localFaceNode];
        const double * coord = stk::mesh::field_data(*coordinates_, node );
        double *pressureForce = stk::mesh::field_data(*pressureForce_, node );
        double *tauWall = stk::mesh::field_data(*tauWall_, node );
        double *yplus = stk::mesh::field_data(*yplus_, node );
        const double assembledArea = *stk::mesh::field_data(*assembledArea_, node );

        // load radius; assemble force -sigma_ij*njdS
        double uParallel = 0.0;
        for ( int i = 0; i < nDim; ++i ) {
          const double ai = areaVec[offSetAveraVec+i];
          ws_radius[i] = coord[i] - centroid[i];
          const double uDiff = p_uiTangential[i] - p_uiBcTangential[i];
          ws_p_force[i] = pBip*ai;
          ws_v_force[i] = lambda*uDiff;
          ws_t_force[i] = ws_p_force[i] + ws_v_force[i];
          pressureForce[i] += ws_p_force[i];
          uParallel += uDiff*uDiff;
        }

        cross_product(&ws_t_force[0], &ws_moment[0], &ws_radius[0]);

        // assemble for and moment
        for ( int j = 0; j < 3; ++j ) {
          l_force_moment[j] += ws_p_force[j];
          l_force_moment[j+3] += ws_v_force[j];
          l_force_moment[j+6] += ws_moment[j];
        }

        // assemble tauWall; area weighting is hiding in lambda/assembledArea
        *tauWall += lambda*std::sqrt(uParallel)/assembledArea;

        // deal with yplus
        *yplus += yplusBip*aMag/assembledArea;

      }
    }
  }

  if ( processMe ) {
    // parallel assemble and output
    double g_force_moment[9] = {};
    stk::ParallelMachine comm = NaluEnv::self().parallel_comm();

    // Parallel assembly of L2
    stk::all_reduce_sum(comm, &l_force_moment[0], &g_force_moment[0], 9);

    // min/max
    double g_yplusMin = 0.0, g_yplusMax = 0.0;
    stk::all_reduce_min(comm, &yplusMin, &g_yplusMin, 1);
    stk::all_reduce_max(comm, &yplusMax, &g_yplusMax, 1);

    // deal with file name and banner
    if ( NaluEnv::self().parallel_rank() == 0 ) {
      std::ofstream myfile;
      myfile.open(outputFileName_.c_str(), std::ios_base::app);
      myfile << std::setprecision(6) 
             << std::setw(w_) 
             << currentTime << std::setw(w_) 
             << g_force_moment[0] << std::setw(w_) << g_force_moment[1] << std::setw(w_) << g_force_moment[2] << std::setw(w_)
             << g_force_moment[3] << std::setw(w_) << g_force_moment[4] << std::setw(w_) << g_force_moment[5] <<  std::setw(w_)
             << g_force_moment[6] << std::setw(w_) << g_force_moment[7] << std::setw(w_) << g_force_moment[8] <<  std::setw(w_)
             << g_yplusMin << std::setw(w_) << g_yplusMax << std::endl;
      myfile.close();
    }
  }

}

//--------------------------------------------------------------------------
//-------- pre_work --------------------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceForceAndMomentWallFunctionAlgorithm::pre_work()
{

  // common
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  //======================
  // assemble area
  //======================

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
      &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets
    = realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
           ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int numScsBip = meFC->num_integration_points();

    // mapping from ip to nodes for this ordinal
    const int *faceIpNodeMap = meFC->ipNodeMap();

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get face
      stk::mesh::Entity face = b[k];

      // face node relations
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);

      for ( int ip = 0; ip < numScsBip; ++ip ) {
        // offsets
        const int offSetAveraVec = ip*nDim;

        // nearest node mapping to this ip
        const int localFaceNode = faceIpNodeMap[ip];

        // extract nodal fields
        stk::mesh::Entity node = face_node_rels[localFaceNode];
        double *assembledArea = stk::mesh::field_data(*assembledArea_, node );

        // aMag
        double aMag = 0.0;
        for ( int j = 0; j < nDim; ++j)
          aMag += areaVec[offSetAveraVec+j]*areaVec[offSetAveraVec+j];
         aMag = std::sqrt(aMag);

         // assemble nodal quantities
         *assembledArea += aMag;
       }
     }
   }
}

//--------------------------------------------------------------------------
//-------- cross_product ----------------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceForceAndMomentWallFunctionAlgorithm::cross_product(
  double *force, double *cross, double *rad)
{
  cross[0] =   rad[1]*force[2] - rad[2]*force[1];
  cross[1] = -(rad[0]*force[2] - rad[2]*force[0]);
  cross[2] =   rad[0]*force[1] - rad[1]*force[0];
}


} // namespace nalu
} // namespace Sierra
