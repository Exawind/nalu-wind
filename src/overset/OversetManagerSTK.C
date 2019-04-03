/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <NaluEnv.h>
#include <NaluParsing.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"
#include <utils/StkHelpers.h>

// overset
#include <overset/OversetInfo.h>
#include <overset/OversetManagerSTK.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/SkinBoundary.hpp>

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// stk_io
#include <stk_io/StkMeshIoBroker.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_util/environment/CPUTime.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// OversetManagerSTK
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
OversetManagerSTK::OversetManagerSTK(
  Realm &realm, 
  const OversetUserData &oversetUserData)
: OversetManager(realm ),
  oversetUserData_(oversetUserData),
  searchMethod_(stk::search::KDTREE),
  nDim_(realm.spatialDimension_),
  oversetAlgDetailedOutput_(oversetUserData.detailedOutput_),
  needToGhostCount_(0),
  firstInitialization_(true)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
OversetManagerSTK::~OversetManagerSTK()
{}

void
OversetManagerSTK::setup()
{
  // Add all overset BC parts to the BC vector so that check missing BCs can
  // ignore these parts
  std::string targetName(oversetUserData_.oversetSurface_);
  stk::mesh::Part* part = metaData_->get_part(targetName);
  if (nullptr == part)
    throw std::runtime_error("OversetManagerSTK:: Invalid overset surface provided");
  realm_.bcPartVec_.push_back(part);
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::initialize()
{
  const double timeA = NaluEnv::self().nalu_time();

  // initialize all ghosting data structures
  initialize_ghosting();

  if ( firstInitialization_ ) {
    // declare the part that represents the intersected elements/nodes;
    declare_inactive_part();

    // declare create the part for the surface of the intersected elements/nodes
    declare_background_surface_part();
  }

  // remove current set of elements/faces in parts; only required in mesh motion is active
  if ( realm_.has_mesh_motion() )
    clear_parts();

  // define overset bounding box for cutting
  define_overset_bounding_box();

  // define overset bounding boxes
  define_overset_bounding_boxes();

  // define background bounding boxes
  define_background_bounding_boxes();

  // perform the coarse search to find the intersected elements
  determine_intersected_elements();

  // create a part that holds the intersected elements that should be inactive
  populate_inactive_part();

  // skin the inActivePart_ and, therefore, populate the backgroundSurfacePart_
  skin_exposed_surface_on_inactive_part();
    
  // define surfaces that include orphan nodes
  set_orphan_surface_part_vec();
  
  // define OversetInfo object for each node on the exposed parts
  create_overset_info_vec();

  // search for nodes in elements
  orphan_node_search();

  // set elemental data on inactive part
  set_data_on_inactive_part();

  // set flag for the next possible time we are through the initializtion method
  firstInitialization_ = false;

  // end time
  const double timeB = NaluEnv::self().nalu_time();
  realm_.timerNonconformal_ += (timeB-timeA);
}

//--------------------------------------------------------------------------
//-------- initialize_ghosting ---------------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::initialize_ghosting()
{
  // initialize need to ghost and elems to ghost
  needToGhostCount_ = 0;
  elemsToGhost_.clear();

  bulkData_->modification_begin();  
  if ( oversetGhosting_ == NULL) {
    // create new ghosting
    std::string theGhostName = "nalu_overset_ghosting";
    oversetGhosting_ = &(bulkData_->create_ghosting( theGhostName ));
  }
  else {
    bulkData_->destroy_ghosting(*oversetGhosting_);
  }
  bulkData_->modification_end();
}

//--------------------------------------------------------------------------
//-------- declare_inactive_part -------------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::declare_inactive_part()
{
  // not sure where this needs to be in order for the block to show up in the output file?
  std::string partName = oversetUserData_.backgroundCutBlock_;
  inActivePart_ =  &metaData_->declare_part(partName, stk::topology::ELEMENT_RANK);
}
  
//--------------------------------------------------------------------------
//-------- declare_background_surface_part ---------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::declare_background_surface_part()
{
  // not sure where this needs to be in order for the block to show up in the output file?
  std::string partName = oversetUserData_.backgroundSurface_;
  backgroundSurfacePart_ =  &metaData_->declare_part(partName, metaData_->side_rank());
}
  
//--------------------------------------------------------------------------
//-------- define_overset_bounding_box -------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::define_overset_bounding_box()
{
  // extract coordinates
  VectorFieldType *coordinates
  = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  
  // obtained via block_2 max/min coords
  std::vector<double> minOversetCorner(nDim_);
  std::vector<double> maxOversetCorner(nDim_);

  // initialize max and min
  for (int j = 0; j < nDim_; ++j ) {
    minOversetCorner[j] = +1.0e16;
    maxOversetCorner[j] = -1.0e16;
  }
    
  // selector and bucket vector for overset part; first extract the part name
  stk::mesh::PartVector targetPartOversetVec;
  for ( size_t k = 0; k < oversetUserData_.oversetBlockVec_.size(); ++k ) {
    std::string targetNameOverset(oversetUserData_.oversetBlockVec_[k]);
    stk::mesh::Part *targetPartOverset = metaData_->get_part(targetNameOverset);
    if ( NULL == targetPartOverset )
      throw std::runtime_error("Null pointer for overset block vec name");
    targetPartOversetVec.push_back(targetPartOverset);
  }

  stk::mesh::Selector s_locally_owned_union_overset
    = metaData_->locally_owned_part() &stk::mesh::selectUnion(targetPartOversetVec);
  stk::mesh::BucketVector const &locally_owned_elem_buckets_overset =
      bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union_overset );

  //======================================================================
  // determine the min/max bounding box by iterating all overset elements
  //======================================================================
  for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_elem_buckets_overset.begin();
        ib != locally_owned_elem_buckets_overset.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;

    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get element
      stk::mesh::Entity element = b[k];

      // extract elem_node_relations
      stk::mesh::Entity const* elem_node_rels = bulkData_->begin_nodes(element);
      const int num_nodes = bulkData_->num_nodes(element);

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates, node );

        // check max/min
        for ( int j = 0; j < nDim_; ++j ) {
          minOversetCorner[j] = std::min(minOversetCorner[j], coords[j]);
          maxOversetCorner[j] = std::max(maxOversetCorner[j], coords[j]);
        }
      }
    }
  }

  // parallel reduce max/min
  std::vector<double> g_minOversetCorner(nDim_);
  std::vector<double> g_maxOversetCorner(nDim_);

  stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
  stk::all_reduce_min(comm, &minOversetCorner[0], &g_minOversetCorner[0], nDim_);
  stk::all_reduce_max(comm, &maxOversetCorner[0], &g_maxOversetCorner[0], nDim_);

  // copy to the point; with reduction below, be very deliberate since these are cheap loops
  Point minOverset;
  Point maxOverset;
  
  // determine translation distance to provide minimum origin at (0,0,0)
  std::vector<double> translateDistance(nDim_);
  for ( int i = 0; i < nDim_; ++i ) {
    translateDistance[i] = -g_minOversetCorner[i];
  }

  // translate
  for ( int i = 0; i < nDim_; ++i ) {
    g_minOversetCorner[i] += translateDistance[i];
    g_maxOversetCorner[i] += translateDistance[i];
  }

  // reduce the translated box
  const double percentOverlap = oversetUserData_.percentOverlap_;
  for ( int i = 0; i < nDim_; ++i ) {
    const double distance = (g_maxOversetCorner[i] - g_minOversetCorner[i])*percentOverlap/100.0;
    g_minOversetCorner[i] += distance;
    g_maxOversetCorner[i] -= distance;
  }

  // translate back to original
  for ( int i = 0; i < nDim_; ++i ) {
    g_minOversetCorner[i] -= translateDistance[i];
    g_maxOversetCorner[i] -= translateDistance[i];
  }

  // inform the user and copy into points
  NaluEnv::self().naluOutputP0() << "Min/Max coords for overset bounding box" << std::endl;
  for ( int i = 0; i < nDim_; ++i ) {
    minOverset[i] = g_minOversetCorner[i];
    maxOverset[i] = g_maxOversetCorner[i];
    NaluEnv::self().naluOutputP0() << "component: " << i << " " << minOverset[i] << " " << maxOverset[i] << std::endl;
  }

  // set up the processor info for this bounding box; attach it to local rank with unique id (zero)
  const size_t overSetBoundingBoxIdent = 0;
  const int parallelRankForBoundingBox = NaluEnv::self().parallel_rank();
  stk::search::IdentProc<uint64_t,int> theIdent(overSetBoundingBoxIdent, parallelRankForBoundingBox);

  // bounding box for all of the overset mesh
  boundingElementBox oversetBox(Box(minOverset,maxOverset), theIdent);
  boundingElementOversetBoxVec_.push_back(oversetBox);
}

//--------------------------------------------------------------------------
//-------- define_overset_bounding_boxes -----------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::define_overset_bounding_boxes()
{
  // extract coordinates
  VectorFieldType *coordinates
    = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  
  // setup data structures for search
  Point minBackgroundCorner, maxBackgroundCorner;

  // selector and bucket vector for overset part; first extract the part name (s)
  stk::mesh::PartVector targetPartOversetVec;
  for ( size_t k = 0; k < oversetUserData_.oversetBlockVec_.size(); ++k ) {
    std::string targetNameOverset(oversetUserData_.oversetBlockVec_[k]);
    stk::mesh::Part *targetPartOverset = metaData_->get_part(targetNameOverset);
    if ( NULL == targetPartOverset )
      throw std::runtime_error("Null pointer for overset block vec name");
    targetPartOversetVec.push_back(targetPartOverset);
  }

  stk::mesh::Selector s_locally_owned_union_overset
    = metaData_->locally_owned_part() &stk::mesh::selectUnion(targetPartOversetVec);
  stk::mesh::BucketVector const &locally_owned_elem_buckets_overset 
    = bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union_overset );

  for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_elem_buckets_overset.begin();
        ib != locally_owned_elem_buckets_overset.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get element
      stk::mesh::Entity element = b[k];

      // initialize max and min
      for (int j = 0; j < nDim_; ++j ) {
        minBackgroundCorner[j] = +1.0e16;
        maxBackgroundCorner[j] = -1.0e16;
      }
        
      // extract elem_node_relations
      stk::mesh::Entity const* elem_node_rels = bulkData_->begin_nodes(element);
      const int num_nodes = bulkData_->num_nodes(element);
        
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates, node );

        // check max/min
        for ( int j = 0; j < nDim_; ++j ) {
          minBackgroundCorner[j] = std::min(minBackgroundCorner[j], coords[j]);
          maxBackgroundCorner[j] = std::max(maxBackgroundCorner[j], coords[j]);
        }
      }

      // setup ident
      stk::search::IdentProc<uint64_t,int> theIdent(bulkData_->identifier(element), NaluEnv::self().parallel_rank());
      
      // create the bounding point box and push back
      boundingElementBox theBox(Box(minBackgroundCorner,maxBackgroundCorner), theIdent);
      boundingElementOversetBoxesVec_.push_back(theBox);
    }
  }
}

//--------------------------------------------------------------------------
//-------- define_background_bounding_boxes --------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::define_background_bounding_boxes()
{
  // extract coordinates
  VectorFieldType *coordinates
    = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  
  // setup data structures for search
  Point minBackgroundCorner, maxBackgroundCorner;

  // selector for background part; first extract the part name
  std::string targetNameBackground(oversetUserData_.backgroundBlock_);
  stk::mesh::Part *targetPartBackground = metaData_->get_part(targetNameBackground);

  stk::mesh::Selector s_locally_owned_union_back = metaData_->locally_owned_part()
            &stk::mesh::Selector(*targetPartBackground);

  stk::mesh::BucketVector const& locally_owned_elem_buckets_back =
      bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union_back );

  for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_elem_buckets_back.begin();
      ib != locally_owned_elem_buckets_back.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get element
      stk::mesh::Entity element = b[k];

      // initialize max and min
      for (int j = 0; j < nDim_; ++j ) {
        minBackgroundCorner[j] = +1.0e16;
        maxBackgroundCorner[j] = -1.0e16;
      }
        
      // extract elem_node_relations
      stk::mesh::Entity const* elem_node_rels = bulkData_->begin_nodes(element);
      const int num_nodes = bulkData_->num_nodes(element);
        
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates, node );

        // check max/min
        for ( int j = 0; j < nDim_; ++j ) {
          minBackgroundCorner[j] = std::min(minBackgroundCorner[j], coords[j]);
          maxBackgroundCorner[j] = std::max(maxBackgroundCorner[j], coords[j]);
        }
      }

      // setup ident
      stk::search::IdentProc<uint64_t,int> theIdent(bulkData_->identifier(element), NaluEnv::self().parallel_rank());

      // populate map for later intersection
      searchIntersectedElementMap_[bulkData_->identifier(element)] = element;

      // create the bounding point box and push back
      boundingElementBox theBox(Box(minBackgroundCorner,maxBackgroundCorner), theIdent);
      boundingElementBackgroundBoxesVec_.push_back(theBox);
    }
  }
}
  
//--------------------------------------------------------------------------
//-------- determine_intersected_elements ----------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::determine_intersected_elements()
{
  // use local searchKeyPair since we do not need to save this off
  std::vector<std::pair<theKey, theKey> > searchKeyPair;
  
  // proceed with coarse search
  stk::search::coarse_search(boundingElementOversetBoxVec_, boundingElementBackgroundBoxesVec_, searchMethod_, NaluEnv::self().parallel_comm(), searchKeyPair);

  // iterate search key; extract found elements and push to vector
  std::vector<std::pair<theKey, theKey> >::const_iterator ii;
  for( ii=searchKeyPair.begin(); ii!=searchKeyPair.end(); ++ii ) {

    const uint64_t theBox = ii->second.id();
    unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned box_proc = ii->second.proc();

    // if this box in on-rank, extract the element; otherwise, do not worry about it
    if ( box_proc == theRank ) {
      // find the element
      std::map<uint64_t, stk::mesh::Entity>::iterator iterEM;
      iterEM=searchIntersectedElementMap_.find(theBox);
      if ( iterEM == searchIntersectedElementMap_.end() )
        throw std::runtime_error("No entry in searchElementMap found");
      stk::mesh::Entity theElemMeshObj = iterEM->second;
      intersectedElementVec_.push_back(theElemMeshObj);
    }
  }
}

//--------------------------------------------------------------------------
//-------- clear_parts -----------------------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::clear_parts()
{  
  // clear some internal data structures
  boundingElementOversetBoxVec_.clear();
  boundingElementOversetBoxesVec_.clear();
  boundingElementBackgroundBoxesVec_.clear();
  boundingPointVecBackground_.clear();
  boundingPointVecOverset_.clear();
  searchIntersectedElementMap_.clear();
  searchKeyPairBackground_.clear();
  searchKeyPairOverset_.clear();
  orphanPointSurfaceVecOverset_.clear();
  orphanPointSurfaceVecBackground_.clear();

  // delete info vec before clear
  delete_info_vec();
  oversetInfoVec_.clear();

  oversetInfoMapOverset_.clear();
  oversetInfoMapBackground_.clear();

  // remove elements from current parts; at this point, do not try to figure out the delta...
  stk::mesh::PartVector noneSkin, noneInactive, inactivePartVector(1,inActivePart_), backgroundPartVector(1,backgroundSurfacePart_);
  
  // load up elements within skinned mesh
  std::vector<stk::mesh::Entity > backgroundSurfaceVec;
  stk::mesh::Selector s_background = stk::mesh::Selector(*backgroundSurfacePart_);
  stk::mesh::BucketVector const& bucket_surface =
    bulkData_->get_buckets( stk::topology::ELEM_RANK, s_background );
  for ( stk::mesh::BucketVector::const_iterator ib = bucket_surface.begin();
        ib != bucket_surface.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity theElement = b[k];
      backgroundSurfaceVec.push_back(theElement);
    }
  }

  stk::mesh::Selector s_inactive = stk::mesh::Selector(*inActivePart_);

  // clear the elements from the two parts in question
  bulkData_->modification_begin();

  // first intersected elements
  for ( size_t k = 0; k < intersectedElementVec_.size(); ++k ) {
    stk::mesh::Entity theElement = intersectedElementVec_[k];
    if (s_inactive(bulkData_->bucket(theElement)))
      bulkData_->change_entity_parts(theElement, noneInactive, inactivePartVector);
  }
  
  // second background elements
  for ( size_t k = 0; k < backgroundSurfaceVec.size(); ++k ) {
    stk::mesh::Entity theElement = backgroundSurfaceVec[k];
    if (s_background(bulkData_->bucket(theElement)))
      bulkData_->change_entity_parts(theElement, noneSkin, backgroundPartVector);
  }
  
  bulkData_->modification_end();

  // clear
  intersectedElementVec_.clear();

}

//--------------------------------------------------------------------------
//-------- populate_inactive_part ------------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::populate_inactive_part()
{
  // push all elements intersected to a new part
  bulkData_->modification_begin();
  
  stk::mesh::PartVector thePartVector;
  thePartVector.push_back(inActivePart_);
  for ( size_t k = 0; k < intersectedElementVec_.size(); ++k ) {
    stk::mesh::Entity theElement = intersectedElementVec_[k];
    bulkData_->change_entity_parts( theElement, thePartVector);
  }
  
  bulkData_->modification_end();
}
  
//--------------------------------------------------------------------------
//-------- skin_exposed_surface_on_inactive_part -------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::skin_exposed_surface_on_inactive_part()
{
  double start_time = NaluEnv::self().nalu_time();

  NaluEnv::self().naluOutputP0() << "OversetManagerSTK::skin_exposed_surface_on_inactive_part(): Begin" << std::endl;

  // skin the inactive part to obtain all exposed surface
  stk::mesh::Selector s_inactive = stk::mesh::Selector(*inActivePart_);
  stk::mesh::Selector s_active = !s_inactive;
  stk::mesh::PartVector partToSkinVec;
  stk::mesh::PartVector partToPopulateVec;
  partToSkinVec.push_back(inActivePart_); // e.g. block_3
  partToPopulateVec.push_back(backgroundSurfacePart_); // e.g. surface_101

  stk::mesh::create_exposed_block_boundary_sides(*bulkData_, s_inactive, partToPopulateVec, s_active);

  const double end_time = NaluEnv::self().nalu_time();

  // set mesh reading
  realm_.timerSkinMesh_ = (end_time - start_time);

  NaluEnv::self().naluOutputP0() << "OversetManagerSTK::skin_exposed_surface_on_inactive_part(): End" << std::endl;
}
  
//--------------------------------------------------------------------------
//-------- set_orphan_surface_part_vec -------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::set_orphan_surface_part_vec()
{
  std::string targetNameOverset(oversetUserData_.oversetSurface_);
  std::string targetNameBackground(oversetUserData_.backgroundSurface_);

  // extract overset part
  stk::mesh::Part *targetPartOverset = metaData_->get_part(targetNameOverset);
  
  // place the orphans on the overset into orphanPointSurfaceVecOverset
  orphanPointSurfaceVecOverset_.push_back(targetPartOverset);
  
  // extract background part
  stk::mesh::Part *targetPartBackground = metaData_->get_part(targetNameBackground);
  
  // place the orphans on the background into orphanPointSurfaceVecBackground
  orphanPointSurfaceVecBackground_.push_back(targetPartBackground);
}
  
//--------------------------------------------------------------------------
//-------- create_overset_info_vec -----------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::create_overset_info_vec()
{
  // extract coordinates
  VectorFieldType *coordinates
    = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  
  Point localNodalCoords;
  
  // first populate overset info on overset mesh
  stk::mesh::Selector s_locally_owned_overset = (metaData_->locally_owned_part() | metaData_->globally_shared_part())
            &stk::mesh::selectUnion(orphanPointSurfaceVecOverset_);

  stk::mesh::BucketVector const& locally_owned_node_bucket_overset =
      bulkData_->get_buckets( stk::topology::NODE_RANK, s_locally_owned_overset );

  for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_node_bucket_overset.begin();
      ib != locally_owned_node_bucket_overset.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;

    const stk::mesh::Bucket::size_type length   = b.size();

    // point to data
    const double * coords = stk::mesh::field_data(*coordinates, b);
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get node
      stk::mesh::Entity node = b[k];

      // create the info
      OversetInfo *theInfo = new OversetInfo(node, nDim_);
      
      // fill map
      stk::search::IdentProc<uint64_t,int> theIdent(bulkData_->identifier(node), NaluEnv::self().parallel_rank());
      oversetInfoMapOverset_[bulkData_->identifier(node)] = theInfo;
        
      // push it back
      oversetInfoVec_.push_back(theInfo);
        
      // pointers to real data
      const size_t offSet = k*nDim_;
      
      // fill in nodal coordinates
      for (int j = 0; j < nDim_; ++j ) {
        const double xj = coords[offSet+j];
        theInfo->nodalCoords_[j] = xj;
        localNodalCoords[j] = xj;
      }
      
      // create the bounding point box and push back
      boundingPoint thePt(localNodalCoords, theIdent);
      boundingPointVecOverset_.push_back(thePt);
    }
  }
  
  // populate background overset info
  stk::mesh::Selector s_locally_owned_background = (metaData_->locally_owned_part() | metaData_->globally_shared_part())
    &stk::mesh::selectUnion(orphanPointSurfaceVecBackground_);
  
  stk::mesh::BucketVector const& locally_owned_node_bucket_background =
      bulkData_->get_buckets( stk::topology::NODE_RANK, s_locally_owned_background );
    
  for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_node_bucket_background.begin();
      ib != locally_owned_node_bucket_background.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
      
    const stk::mesh::Bucket::size_type length   = b.size();

    // point to data
    const double * coords = stk::mesh::field_data(*coordinates, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get node
      stk::mesh::Entity node = b[k];
      
      // create the info
      OversetInfo *theInfo = new OversetInfo(node, nDim_);

      // fill map
      stk::search::IdentProc<uint64_t,int> theIdent(bulkData_->identifier(node), NaluEnv::self().parallel_rank());
      oversetInfoMapBackground_[bulkData_->identifier(node)] = theInfo;
      
      // push it back
      oversetInfoVec_.push_back(theInfo);
        
      // pointers to real data
      const size_t offSet = k*nDim_;
      
      // fill in nodal coordinates
      for (int j = 0; j < nDim_; ++j ) {
        const double xj = coords[offSet+j];
        theInfo->nodalCoords_[j] = xj;
        localNodalCoords[j] = xj;
      }
      
      // create the bounding point box and push back
      boundingPoint thePt(localNodalCoords, theIdent);
      boundingPointVecBackground_.push_back(thePt);
    }
  }
}

//--------------------------------------------------------------------------
//-------- orphan_node_search ---------------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::orphan_node_search()
{
  // coarse search 
  coarse_search(
    boundingPointVecOverset_, boundingElementBackgroundBoxesVec_, searchKeyPairOverset_);
  coarse_search(
    boundingPointVecBackground_, boundingElementOversetBoxesVec_, searchKeyPairBackground_);

  // deal with ghosting so that fine search isInElement has all of the data that it needs
  manage_ghosting();

  // fine search
  complete_search(searchKeyPairOverset_, oversetInfoMapOverset_);  
  complete_search(searchKeyPairBackground_, oversetInfoMapBackground_);
}

//--------------------------------------------------------------------------
//-------- set_data_on_inactive_part ---------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::set_data_on_inactive_part()
{
  // extract elemental field
  GenericFieldType *intersectedElement
    = metaData_->get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "intersected_element");

  // first, initialize element field to zero everywhere
  stk::mesh::Selector s_everywhere = stk::mesh::selectField(*intersectedElement);
  stk::mesh::BucketVector const& elem_buckets = bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_everywhere);
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin() ;
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * interElem = stk::mesh::field_data(*intersectedElement, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      interElem[k] = 0.0;
    }
  }
  
  // now, set inactive field on inactive bucket
  stk::mesh::Selector s_inactive = stk::mesh::Selector(*inActivePart_);
  stk::mesh::BucketVector const& inactive_elem_buckets = bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_inactive );
  for ( stk::mesh::BucketVector::const_iterator ib = inactive_elem_buckets.begin() ;
        ib != inactive_elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * interElem = stk::mesh::field_data(*intersectedElement, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      interElem[k] = 1.0;
    }
  }
}

//--------------------------------------------------------------------------
//-------- coarse_search ---------------------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::coarse_search(
  std::vector<boundingPoint> &boundingPointVec,
  std::vector<boundingElementBox> &boundingElementVec,
  std::vector<std::pair<theKey, theKey> > &searchKeyPair)
{
  // first coarse search for potential donors for the orphans on the overset
  searchKeyPair.clear();
  stk::search::coarse_search(boundingPointVec, boundingElementVec,
    searchMethod_, NaluEnv::self().parallel_comm(), searchKeyPair);

  // now determine elements to ghost
  std::vector<std::pair<boundingPoint::second_type, boundingElementBox::second_type> >::const_iterator ii;  
  for ( ii=searchKeyPair.begin(); ii!=searchKeyPair.end(); ++ii ) {
    const uint64_t theBox = ii->second.id();
    unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned pt_proc = ii->first.proc();
    const unsigned box_proc = ii->second.proc();
    if ( (box_proc == theRank) && (pt_proc != theRank) ) {
      
      // Send box to pt proc

      // find the element
      stk::mesh::Entity theElemMeshObj = bulkData_->get_entity(stk::topology::ELEMENT_RANK, theBox);
      if ( !(bulkData_->is_valid(theElemMeshObj)) )
        throw std::runtime_error("no valid entry for element");

      // new element to ghost counter
      needToGhostCount_++;

      // deal with elements to push back to be ghosted
      stk::mesh::EntityProc theElemPair(theElemMeshObj, pt_proc);
      elemsToGhost_.push_back(theElemPair);
    }
  }
}

//--------------------------------------------------------------------------
//-------- manage_ghosting -------------------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::manage_ghosting()
{  
  // check for ghosting need
  uint64_t g_needToGhostCount = 0;
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &needToGhostCount_, &g_needToGhostCount, 1);
  if (g_needToGhostCount > 0) {
    
    NaluEnv::self().naluOutputP0() << "Overset alg will ghost a number of entities: "
                    << g_needToGhostCount  << std::endl;
    
    bulkData_->modification_begin();
    bulkData_->change_ghosting( *oversetGhosting_, elemsToGhost_);
    bulkData_->modification_end();

    populate_ghost_comm_procs(*bulkData_, *oversetGhosting_, ghostCommProcs_);
  }
  else {
    NaluEnv::self().naluOutputP0() << "Overset alg will NOT ghost entities: " << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- complete_search -------------------------------------------------
//--------------------------------------------------------------------------
void
OversetManagerSTK::complete_search(
  std::vector<std::pair<theKey, theKey> > searchKeyPair,
  std::map<uint64_t, OversetInfo *> &oversetInfoMap)
{
  // extract coordinates
  VectorFieldType *coordinates
    = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // define vectors; fixed size
  std::vector<double> isoParCoords(nDim_);
  std::vector<double> orphanCoords(nDim_);
  std::vector<double> orphanCoordCheck(nDim_);
  // variable size (resize later)
  std::vector<double> elementCoords;
  
  std::vector<std::pair<boundingPoint::second_type, boundingElementBox::second_type> >::const_iterator ii;  
  for ( ii=searchKeyPair.begin(); ii!=searchKeyPair.end(); ++ii ) {
    
    const uint64_t thePt = ii->first.id();
    const uint64_t theBox = ii->second.id();
    const unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned pt_proc = ii->first.proc();

    // check if I own the point...
    if ( theRank == pt_proc ) {
      // extract element from global ID
      stk::mesh::Entity elem = bulkData_->get_entity(stk::topology::ELEMENT_RANK, theBox);

      // extract the topo from this element...
      const stk::topology elementTopo = bulkData_->bucket(elem).topology();

      if ( !(bulkData_->is_valid(elem)) )
        throw std::runtime_error("no valid entry for element");

      // proceed as required; all elements should have already been ghosted via the coarse search
      int elemIsGhosted = bulkData_->bucket(elem).owned() ? 0 : 1;
      
      // find the point
      std::map<uint64_t, OversetInfo *>::iterator iterInfo;
      iterInfo=oversetInfoMap.find(thePt);
      
      if ( iterInfo == oversetInfoMap.end() )
        throw std::runtime_error("no valid entry for oversetInfoMap");
      
      // extract overset info
      OversetInfo *theInfo = iterInfo->second;
      
      // extract orphan node coords
      orphanCoords = theInfo->nodalCoords_;
      
      // now load the elemental nodal coords
      stk::mesh::Entity const * elem_node_rels = bulkData_->begin_nodes(elem);
      int num_nodes = bulkData_->num_nodes(elem);
      
      // resize
      elementCoords.resize(nDim_*num_nodes);
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];
        const double * coords = stk::mesh::field_data(*coordinates, node );
        for ( int j = 0; j < nDim_; ++j ) {
          const int offSet = j*num_nodes +ni;
          elementCoords[offSet] = coords[j];
        }
      }
      
      // extract master element
      MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(elementTopo);
      const double nearestDistance = meSCS->isInElement(&elementCoords[0],
        &(orphanCoords[0]),
        &(isoParCoords[0]));

      if ( nearestDistance < theInfo->bestX_ ) {
        theInfo->owningElement_ = elem;
        theInfo->meSCS_ = meSCS;
        theInfo->isoParCoords_ = isoParCoords;
        theInfo->bestX_ = nearestDistance;
        theInfo->elemIsGhosted_ = elemIsGhosted;
      }
    }
    else {
      // not this proc's issue
    }
  }

  // check to see that all orphan coords have a home...
  if ( oversetAlgDetailedOutput_ ) {
    const double tol = 1.0e-6;
    const double maxTol = 1.0+tol;
    std::map<uint64_t, OversetInfo *>::iterator iterOrphan;
    for (iterOrphan =  oversetInfoMap.begin();
         iterOrphan != oversetInfoMap.end();
         ++iterOrphan) {
      
      OversetInfo * infoObject = (*iterOrphan).second;
      stk::mesh::Entity elem = infoObject->owningElement_;   
    
      if ( infoObject->bestX_ > maxTol || !(bulkData_->is_valid(elem)) ) {
        NaluEnv::self().naluOutputP0() << "Sorry, orphan node for node " << bulkData_->identifier(infoObject->orphanNode_)
            << " does not have an ideal bestX; consider clipping "
            << infoObject->bestX_ << std::endl;
        if ( !(bulkData_->is_valid(elem)) )
          NaluEnv::self().naluOutputP0() << "In fact, the owning master element is null" << std::endl;
        else
          NaluEnv::self().naluOutputP0() << " The best element is "
            << bulkData_->identifier(infoObject->owningElement_)
            << " consider clipping "<< std::endl;

        // clip to isoPar min/max... 
        if ( infoObject->bestX_ > maxTol && oversetUserData_.clipIsoParametricCoords_) {
          NaluEnv::self().naluOutputP0()
            << "Will clip the isoParametricCoords for node id: " << bulkData_->identifier(infoObject->orphanNode_) << std::endl;
          const double minTol = -1.0-tol;
          for ( int j = 0; j < nDim_; ++j ) {
            if ( infoObject->isoParCoords_[j] > maxTol )
              infoObject->isoParCoords_[j] = maxTol;
            if ( infoObject->isoParCoords_[j] < minTol)
              infoObject->isoParCoords_[j] = minTol;
          }
        }
      }
      else {
        NaluEnv::self().naluOutputP0() << "Orphan node (all is well): " << bulkData_->identifier(infoObject->orphanNode_)
                                       << " has the following best X " << infoObject->bestX_ << std::endl;
        NaluEnv::self().naluOutputP0() << " with owning element: " << bulkData_->identifier(elem) << std::endl;
        NaluEnv::self().naluOutputP0() << " elem nodes ";
        stk::mesh::Entity const* elem_node_rels = bulkData_->begin_nodes(elem);
        const int num_nodes = bulkData_->num_nodes(elem);
        for ( int en = 0; en < num_nodes; ++en )
          NaluEnv::self().naluOutputP0() << bulkData_->identifier(elem_node_rels[en]) << " ";
        NaluEnv::self().naluOutputP0() << std::endl;
      }
    }
  }  
}

} // namespace nalu
} // namespace Sierra
