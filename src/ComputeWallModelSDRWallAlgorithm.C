/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeWallModelSDRWallAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>

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
// ComputeWallModelSDRWallAlgorithm - wall function omega at wall bc;
//                                    utau/sqrt(betaStar*kapa*yp)
// compute yp locally; gather utau
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeWallModelSDRWallAlgorithm::ComputeWallModelSDRWallAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  const bool &useShifted)
  : Algorithm(realm, part),
    useShifted_(useShifted),
    sqrtBetaStar_(std::sqrt(realm.get_turb_model_constant(TM_betaStar))),
    kappa_(realm.get_turb_model_constant(TM_kappa))
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  wallFrictionVelocityBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_friction_velocity_bip");
  sdrBc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "wall_model_sdr_bc");
  assembledWallArea_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_sdr");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeWallModelSDRWallAlgorithm::~ComputeWallModelSDRWallAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeWallModelSDRWallAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

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

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get face
      stk::mesh::Entity face = b[k];

      //======================================
      // gather nodal data off of face; n/a
      //======================================
      int num_face_nodes = bulk_data.num_nodes(face);

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
      double *wallFrictionVelocityBip = stk::mesh::field_data(*wallFrictionVelocityBip_, face);

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
        const double * coordL = stk::mesh::field_data(*coordinates_, nodeL );
        const double * coordR = stk::mesh::field_data(*coordinates_, nodeR );

        // aMag
        double aMag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[offSetAveraVec+j];
          aMag += axj*axj;
        }
        aMag = std::sqrt(aMag);

        // determine yp (approximated by 1/4 distance along edge)
        double ypbip = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double nj = areaVec[offSetAveraVec+j]/aMag;
          const double ej = 0.25*(coordR[j] - coordL[j]);
          ypbip += nj*ej*nj*ej;
        }
        ypbip = std::sqrt(ypbip);

        const double utau = wallFrictionVelocityBip[ip];

        // compute wall function wall sdr
        const double wallFuncSdr = utau/sqrtBetaStar_/kappa_/ypbip;

        // assemble to nodal quantities; will normalize and assemble in driver
        double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, nodeR );
        double * sdrBc = stk::mesh::field_data(*sdrBc_, nodeR );

        *assembledWallArea += aMag;
        *sdrBc += wallFuncSdr*aMag;
      }
    }
  }
}


} // namespace nalu
} // namespace Sierra
