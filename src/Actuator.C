/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <Actuator.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Simulation.h>

// master elements
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// basic c++
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <cmath>

namespace sierra{
namespace nalu{

void
Actuator::populate_candidate_elements()
{
  stk::mesh::MetaData & metaData = realm_.meta_data();
  stk::mesh::BulkData & bulkData = realm_.bulk_data();

  const int nDim = metaData.spatial_dimension();

  // fields
  VectorFieldType *coordinates = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // point data structures
  Point minCorner, maxCorner;

  // extract part
  stk::mesh::PartVector searchParts;
  for ( size_t k = 0; k < searchTargetNames_.size(); ++k ) {
    stk::mesh::Part *thePart = metaData.get_part(searchTargetNames_[k]);
    if ( NULL != thePart )
      searchParts.push_back(thePart);
    else
      throw std::runtime_error(get_class_name() + ": Part is null" + searchTargetNames_[k]);
  }

  // selector and bucket loop
  stk::mesh::Selector s_locally_owned = metaData.locally_owned_part()
    &stk::mesh::selectUnion(searchParts);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned );

  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {

    stk::mesh::Bucket & b = **ib;

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get element
      stk::mesh::Entity elem = b[k];

      // initialize max and min
      for (int j = 0; j < nDim; ++j ) {
        minCorner[j] = +1.0e16;
        maxCorner[j] = -1.0e16;
      }

      // extract elem_node_relations
      stk::mesh::Entity const* elem_node_rels = bulkData.begin_nodes(elem);
      const int num_nodes = bulkData.num_nodes(elem);

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates, node );

        // check max/min
        for ( int j = 0; j < nDim; ++j ) {
          minCorner[j] = std::min(minCorner[j], coords[j]);
          maxCorner[j] = std::max(maxCorner[j], coords[j]);
        }
      }

      // setup ident
      stk::search::IdentProc<uint64_t,int> theIdent(bulkData.identifier(elem), NaluEnv::self().parallel_rank());

      // create the bounding point box and push back
      boundingElementBox theBox(Box(minCorner,maxCorner), theIdent);
      boundingElementBoxVec_.push_back(theBox);
    }
  }
}

}
}
