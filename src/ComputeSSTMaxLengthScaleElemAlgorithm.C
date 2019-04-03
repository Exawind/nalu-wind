/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeSSTMaxLengthScaleElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ComputeSSTMaxLengthScaleElemAlgorithm - Max SST length scale for DES
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeSSTMaxLengthScaleElemAlgorithm::ComputeSSTMaxLengthScaleElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    coordinates_(NULL),
    maxLengthScale_(NULL)
{
  // save off data
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  maxLengthScale_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "sst_max_length_scale");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeSSTMaxLengthScaleElemAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double largelyNegative = -1.0e16;

  // define some common selectors
  stk::mesh::Selector s_all_nodes = stk::mesh::selectUnion(partVec_);

  // initialize to something largely negative
  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    
    double * maxLengthScale = stk::mesh::field_data(*maxLengthScale_, b );
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      maxLengthScale[k] = largelyNegative;
    }
  }

  // fill in nodal values
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union);
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // extract master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());

    // extract master element specifics
    const int numScsIp = meSCS->num_integration_points();
    const int *lrscv = meSCS->adjacentNodes();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get elem
      stk::mesh::Entity elem = b[k];

      // node relations
      stk::mesh::Entity const * node_rels = bulk_data.begin_nodes(elem);

      // compute max edge length
      for ( int ip = 0; ip < numScsIp; ++ip ) {
	
        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];
	
	// extract L/R nodes; no bother in gathering
        stk::mesh::Entity nodeL = node_rels[il];
        stk::mesh::Entity nodeR = node_rels[ir];

	// pointers to real data
        const double * coordsL = stk::mesh::field_data(*coordinates_, nodeL );
        const double * coordsR = stk::mesh::field_data(*coordinates_, nodeR );

	// compute distance
        double dx = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          double dxj = coordsR[j] - coordsL[j];
          dx += dxj*dxj;
        }
	dx = std::sqrt(dx);

	// extract L/R nodal values
	double * maxLengthL = stk::mesh::field_data(*maxLengthScale_, nodeL );
        double * maxLengthR = stk::mesh::field_data(*maxLengthScale_, nodeR );
	
	// populate with max...
	*maxLengthL = std::max(*maxLengthL, dx);
	*maxLengthR = std::max(*maxLengthR, dx);
      }
    }
  }  

  // parallel reduce; worry about periodic?
  std::vector<const stk::mesh::FieldBase *> fieldVec;
  fieldVec.push_back(maxLengthScale_);
  stk::mesh::parallel_max(bulk_data, fieldVec);
  
  // deal with periodicity
  if ( realm_.hasPeriodic_) {
    realm_.periodic_field_update(maxLengthScale_, 1);
  }
 
}

} // namespace nalu
} // namespace Sierra
