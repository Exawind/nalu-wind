// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



// nalu
#include <AssembleNodalGradEdgeAlgorithm.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradEdgeAlgorithm - base class equation system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradEdgeAlgorithm::AssembleNodalGradEdgeAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx)
  : Algorithm(realm, part),
    scalarQ_(scalarQ),
    dqdx_(dqdx)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  edgeAreaVec_ = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradEdgeAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  //===========================================================
  // assemble edge-based gradient operator to the node
  //===========================================================

  stk::mesh::BucketVector const& edge_buckets =
    realm_.get_buckets( stk::topology::EDGE_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
        ib != edge_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // pointer to edge area vector
    double * av = stk::mesh::field_data(*edgeAreaVec_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      stk::mesh::Entity const * edge_node_rels = b.begin_nodes(k);

      // sanity check on number or nodes
      ThrowAssert( b.num_nodes(k) == 2 );

      // left and right nodes
      stk::mesh::Entity nodeL = edge_node_rels[0];
      stk::mesh::Entity nodeR = edge_node_rels[1];

      // grad phi at nodes
      double * gradQL = stk::mesh::field_data( *dqdx_, nodeL);
      double * gradQR = stk::mesh::field_data( *dqdx_, nodeR);

      // dual volume at nodes
      const double volL = *stk::mesh::field_data( *dualNodalVolume_, nodeL);
      const double volR = *stk::mesh::field_data( *dualNodalVolume_, nodeR);

      // phi at nodes
      const double qL = *stk::mesh::field_data( *scalarQ_, nodeL);
      const double qR = *stk::mesh::field_data( *scalarQ_, nodeR);

      // start the work...
      const double qip = 0.5*(qL + qR);
      const double invVolL = 1.0/volL;
      const double invVolR = 1.0/volR;

      const size_t offSet = k*nDim;
      for ( int j = 0; j < nDim; ++j ) {
        const double aj = av[offSet+j];
        const double ajQip = aj*qip;
        gradQL[j] += ajQip*invVolL;
        gradQR[j] -= ajQip*invVolR;
      }
    }
  }

}

} // namespace nalu
} // namespace Sierra
