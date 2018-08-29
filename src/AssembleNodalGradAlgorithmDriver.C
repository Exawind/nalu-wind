/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <AssembleNodalGradAlgorithmDriver.h>
#include <Algorithm.h>
#include <AlgorithmDriver.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradAlgorithmDriver - Drives nodal grad algorithms
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradAlgorithmDriver::AssembleNodalGradAlgorithmDriver(
  Realm &realm,
  const std::string & scalarQName,
  const std::string & dqdxName)
  : AlgorithmDriver(realm),
    scalarQName_(scalarQName),
    dqdxName_(dqdxName)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradAlgorithmDriver::~AssembleNodalGradAlgorithmDriver()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- pre_work --------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradAlgorithmDriver::pre_work()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract fields
  VectorFieldType *dqdx = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, dqdxName_);

  // define some common selectors; select all nodes (locally and shared)
  // where dqdx is defined
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*dqdx);

  //===========================================================
  // zero out nodal gradient
  //===========================================================

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    const stk::mesh::Bucket::size_type length   = b.size();
    double * gq = stk::mesh::field_data(*dqdx, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const int offSet = k*nDim;

      for ( int j = 0; j < nDim; ++j ) {
        gq[offSet+j] = 0.0;
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- post_work -------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradAlgorithmDriver::post_work()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // extract fields
  VectorFieldType *dqdx = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, dqdxName_);
  stk::mesh::parallel_sum(bulk_data, {dqdx});

  if ( realm_.hasPeriodic_) {
    const unsigned nDim = meta_data.spatial_dimension();
    realm_.periodic_field_update(dqdx, nDim);
  }

  if ( realm_.hasOverset_ ) {
    // this is a vector
    const unsigned nDim = meta_data.spatial_dimension();
    realm_.overset_orphan_node_field_update(dqdx, 1, nDim);
  }
}

} // namespace nalu
} // namespace Sierra
