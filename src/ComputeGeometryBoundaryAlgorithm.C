/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeGeometryBoundaryAlgorithm.h>

#include <Realm.h>
#include <FieldTypeDef.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ComputeGeometryBoundaryAlgorithm - compute area_bip
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeGeometryBoundaryAlgorithm::ComputeGeometryBoundaryAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeGeometryBoundaryAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract fields
  GenericFieldType *exposedAreaVec = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // setup for buckets; union parts and ask for locally owned
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);
  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // extract master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meFC->nodesPerElement_;
    const int numScsIp = meFC->num_integration_points();

    // define scratch field
    std::vector<double > ws_coordinates(nodesPerElement*nDim);
    std::vector<double > ws_scs_areav(numScsIp*nDim);

    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // face data
      double * areaVec = stk::mesh::field_data(*exposedAreaVec, b, k);

      // face node relations for nodal gather
      stk::mesh::Entity const * face_node_rels = b.begin_nodes(k);

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      int num_nodes = b.num_nodes(k);
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];
        double * coords = stk::mesh::field_data(*coordinates, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          ws_coordinates[offSet+j] = coords[j];
        }
      }

      // compute scs integration point areavec
      double scs_error = 0.0;
      meFC->determinant(1, &ws_coordinates[0], &ws_scs_areav[0], &scs_error);

      // scarrer to area vector
      for ( int ip = 0; ip < numScsIp; ++ip ) {
        const int offSet = ip*nDim;
        for ( int j=0; j < nDim; ++j ) {
          areaVec[offSet+j] = ws_scs_areav[offSet+j];
        }
      }
    }
  }
}

} // namespace nalu
} // namespace Sierra
