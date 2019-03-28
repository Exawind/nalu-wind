/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeWallFrictionVelocityAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// basic c++
#include <cmath>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ComputeWallFrictionVelocityAlgorithm - utau at wall bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeWallFrictionVelocityAlgorithm::ComputeWallFrictionVelocityAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  const bool &useShifted)
  : Algorithm(realm, part),
    useShifted_(useShifted),
    yplusCrit_(11.63),
    elog_(9.8),
    kappa_(realm.get_turb_model_constant(TM_kappa)),
    maxIteration_(20),
    tolerance_(1.0e-6)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  bcVelocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "wall_velocity_bc");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  wallFrictionVelocityBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_friction_velocity_bip");
  wallNormalDistanceBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_normal_distance_bip");
  assembledWallArea_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_wf");
  assembledWallNormalDistance_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_normal_distance");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeWallFrictionVelocityAlgorithm::~ComputeWallFrictionVelocityAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeWallFrictionVelocityAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // zero out assembled nodal quantities
  zero_nodal_fields();

  // bip values
  std::vector<double> uBip(nDim);
  std::vector<double> uBcBip(nDim);
  std::vector<double> unitNormal(nDim);

  // pointers to fixed values
  double *p_uBip = &uBip[0];
  double *p_uBcBip = &uBcBip[0];
  double *p_unitNormal= &unitNormal[0];

  // nodal fields to gather
  std::vector<double> ws_velocityNp1;
  std::vector<double> ws_bcVelocity;
  std::vector<double> ws_density;
  std::vector<double> ws_viscosity;

  // master element
  std::vector<double> ws_shape_function;
  std::vector<double> ws_face_shape_function;

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define vector of parent topos; should always be UNITY in size
  std::vector<stk::topology> parentTopo;

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

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

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = b.topology().num_nodes();
    const int numScsBip = meFC->num_integration_points();

    // mapping from ip to nodes for this ordinal; face perspective (use with face_node_relations)
    const int *faceIpNodeMap = meFC->ipNodeMap();

    // algorithm related; element
    ws_velocityNp1.resize(nodesPerFace*nDim);
    ws_bcVelocity.resize(nodesPerFace*nDim);
    ws_density.resize(nodesPerFace);
    ws_viscosity.resize(nodesPerFace);
    ws_face_shape_function.resize(numScsBip*nodesPerFace);

    // pointers
    double *p_velocityNp1 = &ws_velocityNp1[0];
    double *p_bcVelocity = &ws_bcVelocity[0];
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

      //======================================
      // gather nodal data off of face
      //======================================
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);
      int num_face_nodes = bulk_data.num_nodes(face);
      // sanity check on num nodes
      ThrowAssert( num_face_nodes == nodesPerFace );
      for ( int ni = 0; ni < num_face_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];

        // gather scalars
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
      double *wallNormalDistanceBip = stk::mesh::field_data(*wallNormalDistanceBip_, face);
      double *wallFrictionVelocityBip = stk::mesh::field_data(*wallFrictionVelocityBip_, face);

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = bulk_data.begin_element_ordinals(face)[0];

      // get the relations off of element
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);

      // loop over face nodes
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int offSetAveraVec = ip*nDim;

        const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);
        const int localFaceNode = faceIpNodeMap[ip];

        // left and right nodes; right is on the face; left is the opposing node
        stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
        stk::mesh::Entity nodeR = face_node_rels[localFaceNode];

        // extract nodal fields
        const double * coordL = stk::mesh::field_data(*coordinates_, nodeL );
        const double * coordR = stk::mesh::field_data(*coordinates_, nodeR );

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
        double rhoBip = 0.0;
        double muBip = 0.0;
        const int offSetSF_face = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
          rhoBip += r*p_density[ic];
          muBip += r*p_viscosity[ic];
          const int offSetFN = ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_uBip[j] += r*p_velocityNp1[offSetFN+j];
            p_uBcBip[j] += r*p_bcVelocity[offSetFN+j];
          }
        }

        // form unit normal and determine yp (approximated by 1/4 distance along edge)
        double ypBip = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double nj = areaVec[offSetAveraVec+j]/aMag;
          const double ej = 0.25*(coordR[j] - coordL[j]);
          ypBip += nj*ej*nj*ej;
          p_unitNormal[j] = nj;
        }
        ypBip = std::sqrt(ypBip);
        wallNormalDistanceBip[ip] = ypBip;

        // assemble to nodal quantities
        double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, nodeR );
        double * assembledWallNormalDistance = stk::mesh::field_data(*assembledWallNormalDistance_, nodeR );

        *assembledWallArea += aMag;
        *assembledWallNormalDistance += aMag*ypBip;

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
          uTangential += (uiTan-uiBcTan)*(uiTan-uiBcTan);
        }
        uTangential = std::sqrt(uTangential);

        // provide an initial guess based on yplusCrit_ (more robust than a pure guess on utau)
        double utauGuess = yplusCrit_*muBip/rhoBip/ypBip;
  
        compute_utau(uTangential, ypBip, rhoBip, muBip, utauGuess);
        wallFrictionVelocityBip[ip] = utauGuess;
      }
    }
  }

  // parallel assemble and normalize
  normalize_nodal_fields();
}

//--------------------------------------------------------------------------
//-------- zero_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeWallFrictionVelocityAlgorithm::zero_nodal_fields()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length  = b.size();
    double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, b);
    double * assembledWallNormalDistance = stk::mesh::field_data(*assembledWallNormalDistance_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      assembledWallArea[k] = 0.0;
      assembledWallNormalDistance[k] = 0.0;
    }
  }
}

//--------------------------------------------------------------------------
//-------- compute_utau ----------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeWallFrictionVelocityAlgorithm::compute_utau(
    const double &up, const double &yp,
    const double &density, const double &viscosity,
    double &utau )
{
  bool converged = false;

  const double A = elog_*density*yp/viscosity;

  for ( int k = 0; k < maxIteration_; ++k ) {

    const double wrk = std::log(A*utau);

    // evaluate F'

    const double fPrime = -(1.0+wrk);

    // evaluate function
    const double f = kappa_*up- utau*wrk;

    // update variable
    const double df = f/fPrime;

    utau -= df;
    if ( std::abs(df) < tolerance_ ) {
      converged = true;
      break;
    }
  }

  // report trouble
  if (!converged ) {
    NaluEnv::self().naluOutputP0() << "Issue with utau; not converged " << std::endl;
    NaluEnv::self().naluOutputP0() << up << " " << yp << " " << utau << std::endl;
  }

}

//--------------------------------------------------------------------------
//-------- normalize_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeWallFrictionVelocityAlgorithm::normalize_nodal_fields()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  stk::mesh::parallel_sum(bulk_data, {assembledWallArea_, assembledWallNormalDistance_});

  // periodic assemble
  if ( realm_.hasPeriodic_) {
    const unsigned fieldSize = 1;
    const bool bypassFieldCheck = false; // fields are not defined at all slave/master node pairs
    realm_.periodic_field_update(assembledWallArea_, fieldSize, bypassFieldCheck);
    realm_.periodic_field_update(assembledWallNormalDistance_, fieldSize, bypassFieldCheck);
  }

  // normalize
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length  = b.size();
    const double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, b);
    double * assembledWallNormalDistance = stk::mesh::field_data(*assembledWallNormalDistance_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      assembledWallNormalDistance[k] /= assembledWallArea[k];
    }
  }
}

} // namespace nalu
} // namespace Sierra
