// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef EntityExposedFaceSorter_h
#define EntityExposedFaceSorter_h

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/EntitySorterBase.hpp>

namespace sierra {
namespace nalu {

//=============================================================================
// Class Definition
//=============================================================================
// EntityExposedFaceSorter
//=============================================================================
/**
 * * @par Description:
 * - Class that sorts by exposed face ordinal.
 *
 * @par Design Considerations:
 * - We want to sort based on exposed face ordinal - all exposed faces
 */
//=============================================================================

class EntityExposedFaceSorter : public stk::mesh::EntitySorterBase {

  public:
    
  virtual void sort(stk::mesh::BulkData &bulk, stk::mesh::EntityVector& entityVector) const
  {
    stk::mesh::EntityRank entityVecRank = bulk.entity_rank(entityVector[0]);    
    if ( entityVecRank == bulk.mesh_meta_data().side_rank() ) {
      std::sort(entityVector.begin(), entityVector.end(),
        [&bulk](stk::mesh::Entity a, stk::mesh::Entity b) {
        return bulk.begin_element_ordinals(a)[0] > bulk.begin_element_ordinals(b)[0]; });
    }
  }
};
  
} // end sierra namespace
} // end nalu namespace

#endif
