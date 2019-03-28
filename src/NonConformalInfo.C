/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <DgInfo.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <Realm.h>
#include <NaluEnv.h>

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

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

// vector, pair and find
#include <vector>
#include <utility>
#include <algorithm>

namespace sierra{
namespace nalu{

class DgInfo;

// compare operator 
struct compareGaussPoint {
  compareGaussPoint()  {}
  bool operator () (const std::pair<theKey, theKey> &p, const uint64_t i) {
    return (p.first.id() < i);
  }
  bool operator () (const uint64_t i, const std::pair<theKey, theKey> &p) {
    return (i < p.first.id());
  }
};

// Sortintlowhigh
struct sortIntLowHigh {
  sortIntLowHigh() {}
  bool operator () (const std::pair<theKey, theKey> &p1, const std::pair<theKey, theKey> &p2) {
    return (p1.first.id() < p2.first.id());
  }
};

  
//==========================================================================
// Class Definition
//==========================================================================
// Acon_NonConformalInfo - contains virtual edge data
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
NonConformalInfo::NonConformalInfo(
   Realm &realm,
   const stk::mesh::PartVector currentPartVec,
   const stk::mesh::PartVector opposingPartVec,
   const double expandBoxPercentage,
   const std::string &searchMethodName,
   const bool clipIsoParametricCoords,
   const double searchTolerance,
   const bool   dynamicSearchTolAlg,
   const std::string debugName)
  : realm_(realm ),
    name_(debugName),
    currentPartVec_(currentPartVec),
    opposingPartVec_(opposingPartVec),
    expandBoxPercentage_(expandBoxPercentage),
    searchMethod_(stk::search::BOOST_RTREE),
    clipIsoParametricCoords_(clipIsoParametricCoords),
    searchTolerance_(searchTolerance),
    dynamicSearchTolAlg_(dynamicSearchTolAlg),
    meshMotion_(realm_.has_mesh_motion()),
    canReuse_(false)
{
  // determine search method for this pair
  if ( searchMethodName == "boost_rtree" )
    searchMethod_ = stk::search::BOOST_RTREE;
  else if ( searchMethodName == "stk_kdtree" )
    searchMethod_ = stk::search::KDTREE;
  else
    NaluEnv::self().naluOutputP0() << "NonConformalInfo::search method not declared; will use boost_rtree" << std::endl;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
NonConformalInfo::~NonConformalInfo()
{
  delete_dgInfo();
}

//--------------------------------------------------------------------------
//-------- delete_dgInfo ---------------------------------------------------
//--------------------------------------------------------------------------
void
NonConformalInfo::delete_dgInfo()
{
  std::vector<std::vector<DgInfo*> >::iterator ii;
  for( ii=dgInfoVec_.begin(); ii!=dgInfoVec_.end(); ++ii ) {
    std::vector<DgInfo *> &faceDgInfoVec = (*ii);
    for ( size_t k = 0; k < faceDgInfoVec.size(); ++k )
      delete faceDgInfoVec[k];
  }
  dgInfoVec_.clear();
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
NonConformalInfo::initialize()
{

  // clear some of the search info
  boundingSphereVec_.clear();
  boundingFaceElementBoxVec_.clear();
  searchKeyPair_.clear();

  // delete info only if adaptivity is active
  if ( realm_.mesh_changed() ) {
    delete_dgInfo();
  }
  
  // construct if the size is zero; reset always
  if ( dgInfoVec_.size() == 0 )
    construct_dgInfo();
  reset_dgInfo();
  
  // construct the points and boxes required for the search
  construct_bounding_points();
  construct_bounding_boxes();

  // ghosting
  determine_elems_to_ghost();
}

//--------------------------------------------------------------------------
//-------- construct_dgInfo ------------------------------------------------
//--------------------------------------------------------------------------
void
NonConformalInfo::construct_dgInfo()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  const int nDim = meta_data.spatial_dimension();

  // define vector of parent topos; should always be UNITY in size
  std::vector<stk::topology> parentTopo;
  
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(currentPartVec_);
  
  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  
  // need to keep track of some sort of local id for each gauss point...
  uint64_t localGaussPointId = 0;
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    
    stk::mesh::Bucket & b = **ib;
    
    const stk::mesh::Bucket::size_type length   = b.size();
    
    // extract connected element topology
    b.parent_topology(stk::topology::ELEMENT_RANK, parentTopo);
    ThrowAssert ( parentTopo.size() == 1 );
    stk::topology currentElemTopo = parentTopo[0];

    // volume and surface master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(currentElemTopo);
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    
    // master element-specific values
    const int numScsBip = meFC->num_integration_points();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get face, global and local id
      stk::mesh::Entity face = b[k];
      uint64_t globalFaceId = bulk_data.identifier(face);
      
      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number
      stk::mesh::Entity element = face_elem_rels[0];
      const stk::mesh::ConnectivityOrdinal* face_elem_ords = bulk_data.begin_element_ordinals(face);
      const int currentFaceOrdinal = face_elem_ords[0];
      
      std::vector<DgInfo *> faceDgInfoVec(numScsBip);
      for ( int ip = 0; ip < numScsBip; ++ip ) { 
        DgInfo *dgInfo = new DgInfo(NaluEnv::self().parallel_rank(), globalFaceId, localGaussPointId++, ip, 
                                    face, element, currentFaceOrdinal, meFC, meSCS, currentElemTopo, nDim, searchTolerance_); 
        faceDgInfoVec[ip] = dgInfo;
      }
      
      // push them all back
      dgInfoVec_.push_back(faceDgInfoVec);
    }
  }
}

//--------------------------------------------------------------------------
//-------- reset_dgInfo ----------------------------------------------------
//--------------------------------------------------------------------------
void
NonConformalInfo::reset_dgInfo()
{
  std::vector<std::vector<DgInfo*> >::iterator ii;
  for( ii=dgInfoVec_.begin(); ii!=dgInfoVec_.end(); ++ii ) {
    std::vector<DgInfo *> &theVec = (*ii);    
    for ( size_t k = 0; k < theVec.size(); ++k ) {
      DgInfo *dgInfo = theVec[k];
      if ( !canReuse_ ) {
        dgInfo->allOpposingFaceIdsOld_.clear();
        dgInfo->allOpposingFaceIdsOld_ = dgInfo->allOpposingFaceIds_;
      }
      // always reset bestX and opposing faceIDs for the upcoming search
      dgInfo->bestX_ = dgInfo->bestXRef_;
      dgInfo->allOpposingFaceIds_.clear();
    }
  }
}
  
//--------------------------------------------------------------------------
//-------- construct_bounding_points ---------------------------------------
//--------------------------------------------------------------------------
void
NonConformalInfo::construct_bounding_points()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  const int nDim = meta_data.spatial_dimension();

  // dial in Gauss-labato points
  const bool useShifted = realm_.has_nc_gauss_labatto_quadrature();

  // hold the point location for integration points
  Point currentIpCoords;

  // nodal fields to gather
  std::vector<double> ws_face_coordinates;
  // master element
  std::vector<double> ws_face_shape_function;

  // fields
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  
  std::vector<std::vector<DgInfo*> >::iterator ii;
  for( ii=dgInfoVec_.begin(); ii!=dgInfoVec_.end(); ++ii ) {
    std::vector<DgInfo *> &theVec = (*ii);

    //=======================================================
    // all ips on this face use a common face master element 
    //            gather common operations once 
    //=======================================================
    const DgInfo *firstDgInfo = theVec[0];
    MasterElement *meFC = firstDgInfo->meFCCurrent_;
    
    // master element-specific values
    const int numScsBip = meFC->num_integration_points();
    const int nodesPerFace = meFC->nodesPerElement_;
   
    // algorithm related; face
    ws_face_coordinates.resize(nodesPerFace*nDim);  
    ws_face_shape_function.resize(numScsBip*nodesPerFace);
    
    // pointers
    double *p_face_coordinates = &ws_face_coordinates[0];
    double *p_face_shape_function = &ws_face_shape_function[0];
    
    // populate shape function
    if ( useShifted )
      meFC->shifted_shape_fcn(&p_face_shape_function[0]);
    else
      meFC->shape_fcn(&p_face_shape_function[0]);
  
    // gather nodal data off of face
    stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(firstDgInfo->currentFace_);
    const int num_face_nodes = bulk_data.num_nodes(firstDgInfo->currentFace_);
    
    // sanity check on num nodes
    ThrowAssert( num_face_nodes == nodesPerFace );
    for ( int ni = 0; ni < num_face_nodes; ++ni ) {
      stk::mesh::Entity node = face_node_rels[ni];
      double * coords = stk::mesh::field_data(*coordinates, node);
      for ( int j =0; j < nDim; ++j ) {
        p_face_coordinates[ni*nDim+j] = coords[j];
      }
    }
    
    // now loop over all ips on this face
    for ( size_t k = 0; k < theVec.size(); ++k ) {
      
      DgInfo *dgInfo = theVec[k];

      // extract point radius; set to small if dynamic alg is not activated
      const double pointRadius = dynamicSearchTolAlg_ ? dgInfo->nearestDistance_*dgInfo->nearestDistanceSafety_ : 1.0e-16;
      
      // local and current ip
      const uint64_t localIp  = dgInfo->localGaussPointId_; 
      const int currentFaceIp = dgInfo->currentGaussPointId_;

      // compute coordinates
      for ( int j = 0; j < nDim; ++j )
        currentIpCoords[j] = 0.0;
        
      // interpolate to gauss point
      for ( int ic = 0; ic < nodesPerFace; ++ic ) {
        const double r = p_face_shape_function[currentFaceIp*nodesPerFace+ic];
        for ( int j = 0; j < nDim; ++j ) {
          currentIpCoords[j] += r*p_face_coordinates[ic*nDim+j];
        }
      }

      // extract isoparametric coords on current face from meFC
      const double *intgLoc = useShifted ? meFC->integration_location_shift() : meFC->integration_locations();
      
      // copy these coordinates
      for ( int j = 0; j < nDim; ++j ) {
        dgInfo->currentGaussPointCoords_[j] = currentIpCoords[j];
      }
      
      // save face iso-parametric coordinates; extract conversion factor from CVFEM to isInElement
      const double conversionFac = meFC->scal_to_standard_iso_factor();
      for ( int j = 0; j < nDim-1; ++j ) {
        dgInfo->currentIsoParCoords_[j] = conversionFac*intgLoc[currentFaceIp*(nDim-1)+j]; 
      }
      
      // setup ident for this point; use local integration point id
      stk::search::IdentProc<uint64_t,int> theIdent(localIp, NaluEnv::self().parallel_rank());
      
      // create the bounding sphere and push back
      boundingSphere theSphere(Sphere(currentIpCoords, pointRadius), theIdent);
      boundingSphereVec_.push_back(theSphere);
    } 
  }
}

void
NonConformalInfo::delete_range_points_found(std::vector<boundingSphere>                  &SphereVec,
                                            const std::vector<std::pair<theKey, theKey>> &searchKeyPair) const {

  struct compare {
    bool operator()(const boundingSphere &a, const theKey  &b) const {return a.second < b;}   
    bool operator()(const theKey  &a, const boundingSphere &b) const {return a < b.second;}   
    bool operator()(const boundingSphere &a, const boundingSphere &b) const {return a.second.id() < b.second.id();}   
  }; 
  if (!std::is_sorted(SphereVec.begin(), SphereVec.end(), compare()))
     std::sort (SphereVec.begin(), SphereVec.end(), compare());

  std::vector<theKey> keys_found;
  keys_found.reserve(searchKeyPair.size());
  for (const auto &ii : searchKeyPair) {
    keys_found.push_back(ii.first);
  }
  {
    std::sort(keys_found.begin(), keys_found.end());
    const auto it = std::unique(keys_found.begin(), keys_found.end());
    keys_found.resize(it-keys_found.begin());
  }
  std::vector<boundingSphere> difference(SphereVec.size());
  {
    const auto it =
      std::set_difference(
        SphereVec.begin(),  SphereVec.end(),
        keys_found.begin(), keys_found.end(),
        difference.begin(), compare());
    difference.resize(it-difference.begin());
  }
  swap(difference, SphereVec);
}

void
NonConformalInfo::repeat_search_if_needed(const std::vector<boundingSphere>      &boundingSphereVec,
                                          std::vector<std::pair<theKey, theKey>> &searchKeyPair) const {
  unsigned num_iterations = 0;

  std::vector<boundingSphere> SphereVec(boundingSphereVec);
  delete_range_points_found(SphereVec,searchKeyPair);

  int any_not_empty, not_empty = !SphereVec.empty();
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &not_empty, &any_not_empty, 1);

  while (any_not_empty) {
    for (auto &ii : SphereVec) ii.first.set_radius(2*ii.first.radius());
    std::vector<std::pair<theKey, theKey>> KeyPair;
    stk::search::coarse_search(SphereVec, boundingFaceElementBoxVec_, searchMethod_, NaluEnv::self().parallel_comm(), KeyPair);

    searchKeyPair.reserve(searchKeyPair.size() + KeyPair.size()); 
    searchKeyPair.insert(searchKeyPair.end(), KeyPair.begin(), KeyPair.end());

    delete_range_points_found(SphereVec,searchKeyPair);
    not_empty = !SphereVec.empty();
    stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &not_empty, &any_not_empty, 1);
    ++num_iterations;

    if (10<num_iterations) {
      NaluEnv::self().naluOutputP0() << "NonConformalInfo::repeat_search_if_needed issue with " << name_ << std::endl; 
      NaluEnv::self().naluOutputP0() << "Increased search tolerance 10 times and still failed to find match." << std::endl; 
      throw std::runtime_error("Could be an internal logic error. Try turning off dynamic search tolerance algorithm...");
    }
  } 
}

//--------------------------------------------------------------------------
//-------- determine_elems_to_ghost ----------------------------------------
//--------------------------------------------------------------------------
void
NonConformalInfo::determine_elems_to_ghost()
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // perform the coarse search
  stk::search::coarse_search(boundingSphereVec_, boundingFaceElementBoxVec_, searchMethod_, NaluEnv::self().parallel_comm(), searchKeyPair_);
    
  if (dynamicSearchTolAlg_) repeat_search_if_needed(boundingSphereVec_, searchKeyPair_);

  // sort based on local gauss point
  std::sort (searchKeyPair_.begin(), searchKeyPair_.end(), sortIntLowHigh());

  std::vector<std::pair<theKey, theKey> >::const_iterator ii;
  for( ii=searchKeyPair_.begin(); ii!=searchKeyPair_.end(); ++ii ) {

    const uint64_t theBox = ii->second.id();
    unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned pt_proc = ii->first.proc();
    const unsigned box_proc = ii->second.proc();
    if ( (box_proc == theRank) && (pt_proc != theRank) ) {

      // Send box to pt proc

      // find the face element
      stk::mesh::Entity face = bulk_data.get_entity(meta_data.side_rank(), theBox);
      if ( !(bulk_data.is_valid(face)) )
        throw std::runtime_error("no valid entry for face");
      
      // extract the connected element
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      ThrowAssert( bulk_data.num_elements(face) == 1 );
      stk::mesh::Entity element = face_elem_rels[0];
          
      // deal with elements to push back to be ghosted; downward relations come for the ride...
      stk::mesh::EntityProc theElemPair(element, pt_proc);
      realm_.nonConformalManager_->elemsToGhost_.push_back(theElemPair);
    }
  }
}

//--------------------------------------------------------------------------
//-------- complete_search -------------------------------------------------
//--------------------------------------------------------------------------
void
NonConformalInfo::complete_search()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  const int nDim = meta_data.spatial_dimension();

  // dynamic algorithm requires normal distance between point and ip
  double bestElemIpCoords[3];

  // fields
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  std::vector<double> currentGaussPointCoords(nDim);
  std::vector<double> opposingIsoParCoords(nDim);

  // invert the process... Loop over dgInfoVec_ and query searchKeyPair_ for this information
  std::vector<DgInfo *> problemDgInfoVec;
  std::vector<std::vector<DgInfo*> >::iterator ii;
  for( ii=dgInfoVec_.begin(); ii!=dgInfoVec_.end(); ++ii ) {
    std::vector<DgInfo *> &theVec = (*ii);
    for ( size_t k = 0; k < theVec.size(); ++k ) {
        
      DgInfo *dgInfo = theVec[k];
      const uint64_t localGaussPointId  = dgInfo->localGaussPointId_; 

      // set initial nearestDistance and save off nearest distance under dgInfo
      double nearestDistance = std::numeric_limits<double>::max();
      const double nearestDistanceSaved = dgInfo->nearestDistance_;
        
      std::pair <std::vector<std::pair<theKey, theKey> >::const_iterator, std::vector<std::pair<theKey, theKey> >::const_iterator > 
        p2 = std::equal_range(searchKeyPair_.begin(), searchKeyPair_.end(), localGaussPointId, compareGaussPoint());

      if ( p2.first == p2.second ) {
        problemDgInfoVec.push_back(dgInfo);        
      }
      else {
        for (std::vector<std::pair<theKey, theKey> >::const_iterator jj = p2.first; jj != p2.second; ++jj ) {
          
          const uint64_t theBox = jj->second.id();
          const unsigned theRank = NaluEnv::self().parallel_rank();
          const unsigned pt_proc = jj->first.proc();

          // check if I own the point...
          if ( theRank == pt_proc ) {

            // yes, I own the point... However, what about the face element? Who owns that?

            // proceed as required; all elements should have already been ghosted via the coarse search
            stk::mesh::Entity opposingFace = bulk_data.get_entity(meta_data.side_rank(), theBox);
            if ( !(bulk_data.is_valid(opposingFace)) )
              throw std::runtime_error("no valid entry for face element");

            int opposingFaceIsGhosted = bulk_data.bucket(opposingFace).owned() ? 0 : 1;
            
            // extract the gauss point coordinates
            currentGaussPointCoords = dgInfo->currentGaussPointCoords_;
            
            // now load the face elemental nodal coords
            stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(opposingFace);
            int num_nodes = bulk_data.num_nodes(opposingFace);
            
            std::vector<double> theElementCoords(nDim*num_nodes);
            
            for ( int ni = 0; ni < num_nodes; ++ni ) {
              stk::mesh::Entity node = face_node_rels[ni];
              const double * coords =  stk::mesh::field_data(*coordinates, node);
              for ( int j = 0; j < nDim; ++j ) {
                const int offSet = j*num_nodes +ni;
                theElementCoords[offSet] = coords[j];
              }
            }
            
            // extract the topo from this face element...
            const stk::topology theFaceTopo = bulk_data.bucket(opposingFace).topology();
            MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(theFaceTopo);

            // extract the connected element to the opposing face
            const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(opposingFace);
            ThrowAssert( bulk_data.num_elements(opposingFace) == 1 );
            stk::mesh::Entity opposingElement = face_elem_rels[0];
            
            // extract the opposing element topo and associated master element
            const stk::topology theOpposingElementTopo = bulk_data.bucket(opposingElement).topology();
            MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theOpposingElementTopo);
            
            // possible reuse            
            dgInfo->allOpposingFaceIds_.push_back(bulk_data.identifier(opposingFace));
            
            // find distance between true current gauss point coords (the point) and the candidate bounding box
            const double nearDistance = meFC->isInElement(&theElementCoords[0],
                                                             &(currentGaussPointCoords[0]),
                                                             &(opposingIsoParCoords[0]));
            
            // check is this is the best candidate
            if ( nearDistance < dgInfo->bestX_ ) {
              // save the opposing face element and master element
              dgInfo->opposingFace_ = opposingFace;
              dgInfo->meFCOpposing_ = meFC;
             
              if ( dynamicSearchTolAlg_ ) {
                // find the projected normal distance between point and centroid; all we need is an approximation
                meFC->interpolatePoint(nDim, &opposingIsoParCoords[0], &theElementCoords[0], &bestElemIpCoords[0]);
                double theDistance = 0.0;
                for ( int j = 0; j < nDim; ++j ) {
                  double dxj = currentGaussPointCoords[j] - bestElemIpCoords[j];
                  theDistance += dxj*dxj;
                }
                theDistance = std::sqrt(theDistance);
                nearestDistance = std::min(nearestDistance,theDistance);
                  
                // If the nearest distance between the surfaces at this point is smaller then the current
                // distance can be reduced a bit.  Otherwise make sure the current distance is increased as needed.
                if (nearestDistance < dgInfo->nearestDistance_) {
                  const double relax = 0.8;
                  dgInfo->nearestDistance_ = relax*nearestDistanceSaved + (1.0-relax)*nearestDistance;
                }
                else {
                  dgInfo->nearestDistance_ = nearestDistance;
                }
              }
              
              // save off ordinal for opposing face
              const stk::mesh::ConnectivityOrdinal* face_elem_ords = bulk_data.begin_element_ordinals(opposingFace);
              dgInfo->opposingFaceOrdinal_ = face_elem_ords[0];

              // save off all required opposing information
              dgInfo->opposingElement_ = opposingElement;
              dgInfo->meSCSOpposing_ = meSCS;
              dgInfo->opposingElementTopo_ = theOpposingElementTopo;
              dgInfo->opposingIsoParCoords_ = opposingIsoParCoords;
              dgInfo->bestX_ = nearDistance;
              dgInfo->opposingFaceIsGhosted_ = opposingFaceIsGhosted;
            }
          }
          else {
            // not this proc's issue
          }
        }
      }
    }
  }
  
  // check for problems... will want to be more pro-active in the near future, e.g., expand and search...
  if ( problemDgInfoVec.size() > 0 ) {
    NaluEnv::self().naluOutputP0() << "NonConformalInfo::complete_search issue with " << name_ 
                                   << " Size of issue is " << problemDgInfoVec.size() << std::endl; 
    NaluEnv::self().naluOutputP0() << "Problem ips are as follows: " << std::endl; 
    for ( size_t k = 0; k < problemDgInfoVec.size(); ++k ) {
      problemDgInfoVec[k]->dump_info(); 
    }
    NaluEnv::self().naluOutputP0() << std::endl;
    throw std::runtime_error("Try to adjust the search tolerance and re-submit...");
  }

  // check for reuse and also provide diagnostics on sizes for opposing surface set
  size_t totalOpposingFaceSize = 0;
  size_t totalDgInfoSize = 0;
  size_t maxOpposingSize = 0;
  size_t minOpposingSize = 1e6;
    
  size_t numberOfFacesMissing = 0;
  for( size_t iv = 0; iv < dgInfoVec_.size(); ++iv ) {
    std::vector<DgInfo *> &theVec = dgInfoVec_[iv];
    for ( size_t k = 0; k < theVec.size(); ++k ) {
      
      // extract the info object; new and old
      DgInfo *dgInfo = theVec[k];
      
      // counts
      size_t opposingCount = dgInfo->allOpposingFaceIds_.size();
      totalDgInfoSize++;
      totalOpposingFaceSize += opposingCount;
      maxOpposingSize = std::max(maxOpposingSize, opposingCount);
      minOpposingSize = std::min(minOpposingSize, opposingCount);
        
      // extract the bestX opposing face id
      const size_t bestOpposingId = bulk_data.identifier(dgInfo->opposingFace_);
      
      // is the required active stencil opposing id within the vector of face 
      // ids returned in the search? (no need to sort given the size)
      auto itF = std::find(dgInfo->allOpposingFaceIdsOld_.begin(), 
                           dgInfo->allOpposingFaceIdsOld_.end(), bestOpposingId);
      
      // increment missing faces if NOT found
      if (itF == dgInfo->allOpposingFaceIdsOld_.end()) {
        numberOfFacesMissing++;
      }
    }
  }
  
  // global sum
  NaluEnv::self().naluOutputP0() << "DgInfo size overview for name: " << name_ << std::endl;
  size_t g_numberOfFacesMissing;
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &numberOfFacesMissing, &g_numberOfFacesMissing, 1);
  if ( g_numberOfFacesMissing > 0 ) {
    NaluEnv::self().naluOutputP0() << "  Ghosted search entries ARE NOT sufficient for re-use " << std::endl;
    canReuse_ = false;
  }
  else {
    NaluEnv::self().naluOutputP0() << "  Ghosted search entries ARE sufficient for re-use " << std::endl;
    canReuse_ = true;
  }
    
 // finally, provide mean opposing face count
 size_t g_total[2] = {};
 size_t g_minOpposingSize; size_t g_maxOpposingSize;
 size_t l_total[2] = {totalDgInfoSize, totalOpposingFaceSize};
 stk::all_reduce_sum(NaluEnv::self().parallel_comm(), l_total, g_total, 2);
 stk::all_reduce_min(NaluEnv::self().parallel_comm(), &minOpposingSize, &g_minOpposingSize, 1);
 stk::all_reduce_max(NaluEnv::self().parallel_comm(), &maxOpposingSize, &g_maxOpposingSize, 1);
 NaluEnv::self().naluOutputP0() << "  Min/Max/Average opposing face size: " << g_minOpposingSize << "/"
                                << g_maxOpposingSize << "/" << g_total[1]/g_total[0] << std::endl;
}
  
//--------------------------------------------------------------------------
//-------- construct_bounding_boxes ----------------------------------------
//--------------------------------------------------------------------------
void
NonConformalInfo::construct_bounding_boxes()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  const int nDim = meta_data.spatial_dimension();

  // specify dynamic tolerance algorithm factor
  const double dynamicFac = dynamicSearchTolAlg_ ? 0.0 : 1.0;

  // fields
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // points
  Point minCorner, maxCorner;

  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(opposingPartVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {

    stk::mesh::Bucket & b = **ib;

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get face
      stk::mesh::Entity face = b[k];

      // initialize max and min
      for (int j = 0; j < nDim; ++j ) {
        minCorner[j] = +1.0e16;
        maxCorner[j] = -1.0e16;
      }

      // extract elem_node_relations
      stk::mesh::Entity const* face_node_rels = bulk_data.begin_nodes(face);
      const int num_nodes = bulk_data.num_nodes(face);

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];

        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates, node );

        // check max/min
        for ( int j = 0; j < nDim; ++j ) {
          minCorner[j] = std::min(minCorner[j], coords[j]);
          maxCorner[j] = std::max(maxCorner[j], coords[j]);
        }
      }
      
      // setup ident
      stk::search::IdentProc<uint64_t,int> theIdent(bulk_data.identifier(face), NaluEnv::self().parallel_rank());

      // expand the box by both % and search tolerance
      for ( int i = 0; i < nDim; ++i ) {
        const double theMin = minCorner[i];
        const double theMax = maxCorner[i];
        const double increment = expandBoxPercentage_*(theMax - theMin) + searchTolerance_*dynamicFac;
        minCorner[i] -= increment;
        maxCorner[i] += increment;
      }

      // create the bounding point box and push back
      boundingElementBox theBox(Box(minCorner,maxCorner), theIdent);
      boundingFaceElementBoxVec_.push_back(theBox);
    }
  }
}  

//--------------------------------------------------------------------------
//-------- provide_diagnosis -----------------------------------------------
//--------------------------------------------------------------------------
void
NonConformalInfo::provide_diagnosis()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  const int nDim = meta_data.spatial_dimension();

  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  std::vector<double> currentGaussPointCoords(nDim);
  std::vector<double> opposingGaussPointCoords(nDim);
 
  std::vector<double> currentIsoParCoords(nDim);
  std::vector<double> opposingIsoParCoords(nDim);

  NaluEnv::self().naluOutput() << std::endl;
  NaluEnv::self().naluOutput() << "Non Conformal Alg review for surface: " << name_ << std::endl;
  NaluEnv::self().naluOutput() << "===================================== " << std::endl;
  std::vector<std::vector<DgInfo*> >::iterator ii;
  for( ii=dgInfoVec_.begin(); ii!=dgInfoVec_.end(); ++ii ) {
    std::vector<DgInfo *> &theVec = (*ii);
    for ( size_t k = 0; k < theVec.size(); ++k ) {
      DgInfo *dgInfo = theVec[k];

      // first, dump info
      dgInfo->dump_info();

      // now proceed to detailed face/element current/opposing checks
      const uint64_t localGaussPointId  = dgInfo->localGaussPointId_; 
      const uint64_t currentGaussPointId  = dgInfo->currentGaussPointId_; 

      // extract current face
      stk::mesh::Entity currentFace = dgInfo->currentFace_;

      // extract the gauss point isopar/geometric coordinates for current
      currentGaussPointCoords = dgInfo->currentGaussPointCoords_;
      currentIsoParCoords = dgInfo->currentIsoParCoords_;

      // extract the master element for current; with npe
      MasterElement *meFCCurrent = dgInfo->meFCCurrent_;      
      const int currentNodesPerFace = meFCCurrent->nodesPerElement_;

      // face:node relations
      stk::mesh::Entity const * current_face_node_rels = bulk_data.begin_nodes(currentFace);
      int current_face_num_nodes = bulk_data.num_nodes(currentFace); 

      // gather nodal coordinates
      std::vector <double > currentFaceNodalCoords(nDim*currentNodesPerFace);
      for ( int ni = 0; ni < current_face_num_nodes; ++ni ) {
        stk::mesh::Entity node = current_face_node_rels[ni];
        const double * coords = stk::mesh::field_data(*coordinates, node);
        for ( int j=0; j < nDim; ++j ) {
          currentFaceNodalCoords[j*currentNodesPerFace+ni] = coords[j];
        }
      }

      // interpolate to current GP
      std::vector<double> checkCurrentFaceGaussPointCoords(nDim);
      meFCCurrent->interpolatePoint(
        nDim,
        &currentIsoParCoords[0],
        &currentFaceNodalCoords[0],
        &checkCurrentFaceGaussPointCoords[0]);
      
      // extract current element
      stk::mesh::Entity currentElement = dgInfo->currentElement_;
      
      // best X
      const double bX = dgInfo->bestX_ ;
      
      // best opposing face
      stk::mesh::Entity theBestFace = dgInfo->opposingFace_;
      
      // extract the gauss point isopar coordiantes for opposing
      opposingIsoParCoords = dgInfo->opposingIsoParCoords_;

      // extract the master element for opposing; with npe
      MasterElement *meFCOpposing = dgInfo->meFCOpposing_;      
      const int opposingNodesPerFace = meFCOpposing->nodesPerElement_;

      // face:node relations
      stk::mesh::Entity const * opposing_face_node_rels = bulk_data.begin_nodes(theBestFace);
      int opposing_face_num_nodes = bulk_data.num_nodes(theBestFace); 

      // gather nodal coordinates
      std::vector <double > opposingFaceNodalCoords(nDim*opposingNodesPerFace);
      for ( int ni = 0; ni < opposing_face_num_nodes; ++ni ) {
        stk::mesh::Entity node = opposing_face_node_rels[ni];
        const double * coords = stk::mesh::field_data(*coordinates, node);
        for ( int j=0; j < nDim; ++j ) {
          opposingFaceNodalCoords[j*opposingNodesPerFace+ni] = coords[j];
        }
      }
      
      // interpolate to opposing GP
      std::vector<double> checkOpposingFaceGaussPointCoords(nDim);
      meFCOpposing->interpolatePoint(
        nDim,
        &opposingIsoParCoords[0],
        &opposingFaceNodalCoords[0],
        &(checkOpposingFaceGaussPointCoords[0]));

      // global id for opposing element
      const uint64_t opElemId = bulk_data.identifier(dgInfo->opposingElement_);

      // compute a norm between the curent nd opposing coordinate checks
      double distanceNorm = 0.0;
      for ( int j = 0; j < nDim; ++j ) 
        distanceNorm += std::pow(checkCurrentFaceGaussPointCoords[j] -checkOpposingFaceGaussPointCoords[j], 2);
      distanceNorm = std::sqrt(distanceNorm);

      // provide output...
      NaluEnv::self().naluOutput() << "Gauss Point Lid: " << localGaussPointId << " Review " << std::endl;
      NaluEnv::self().naluOutput() << "  encapsulated by Gid: (";  
      for ( int ni = 0; ni < current_face_num_nodes; ++ni ) {
        stk::mesh::Entity node = current_face_node_rels[ni];
        NaluEnv::self().naluOutput() << bulk_data.identifier(node) << " ";
      }
      NaluEnv::self().naluOutput() << ")" << std::endl;
      NaluEnv::self().naluOutput() << "Current Gauss Point id: " << currentGaussPointId 
                                   << " (nearest node) " << bulk_data.identifier(current_face_node_rels[currentGaussPointId])
                                   << std::endl;
      
      NaluEnv::self().naluOutput() << "  Current element Gid: " << bulk_data.identifier(currentElement) 
                                   << " (face ordinal: " << dgInfo->currentFaceOrdinal_ << ")" << std::endl;  
      
      NaluEnv::self().naluOutput() << "  has Gp coordinates: " ;
      for ( int i = 0; i < nDim; ++i )
        NaluEnv::self().naluOutput() << currentGaussPointCoords[i] << " ";
      NaluEnv::self().naluOutput() << std::endl;
      NaluEnv::self().naluOutput() << "  The best X is: " << bX << std::endl;
      NaluEnv::self().naluOutput() << "  Opposing element Gid: " << opElemId 
                                   << " (face ordinal: " << dgInfo->opposingFaceOrdinal_ << ")" << std::endl;  
      NaluEnv::self().naluOutput() << "  encapsulated by Gid: (";  
      for ( int ni = 0; ni < opposing_face_num_nodes; ++ni ) {
        stk::mesh::Entity node = opposing_face_node_rels[ni];
        NaluEnv::self().naluOutput() << bulk_data.identifier(node) << " ";
      }
      NaluEnv::self().naluOutput() << ")" << std::endl;
      NaluEnv::self().naluOutput() << "  INTERNAL CHECK.... does current Gp and opposing found Gp match coordiantes? What error?" << std::endl;
      NaluEnv::self().naluOutput() << "  current and opposing Gp coordinates:        " << std::endl;
      for ( int i = 0; i < nDim; ++i )
        NaluEnv::self().naluOutput() << "      " << i << " " << checkCurrentFaceGaussPointCoords[i] << " " << checkOpposingFaceGaussPointCoords[i] << std::endl;
      NaluEnv::self().naluOutput() << "  current and opposing Gp isoPar coordinates: " << std::endl;
      for ( int i = 0; i < nDim-1; ++i )
        NaluEnv::self().naluOutput() << "      " << i << " " << currentIsoParCoords[i] << " " << opposingIsoParCoords[i] << std::endl;
      NaluEnv::self().naluOutput() << std::endl;
      NaluEnv::self().naluOutput() << " in the end, the Error Distance Norm is: " << distanceNorm << std::endl;
      NaluEnv::self().naluOutput() << "-------------------------------------------------------------------" << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- error_check -----------------------------------------------------
//--------------------------------------------------------------------------
size_t
NonConformalInfo::error_check()
{
  // check for coincident nodes via intersection of parts provided
  std::vector<stk::mesh::EntityId> coindidentNodesVec;

  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  stk::mesh::Selector s_locally_owned_intersected = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(currentPartVec_) 
    &stk::mesh::selectUnion(opposingPartVec_);
  
  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_locally_owned_intersected );
  
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      coindidentNodesVec.push_back(bulk_data.identifier(b[k]));
    }
  } 
   
  // report the data if problem nodes were found
  if ( coindidentNodesVec.size() > 0 ) {
    NaluEnv::self().naluOutput() << std::endl;
    NaluEnv::self().naluOutput() << "Non Conformal Alg (P" << NaluEnv::self().parallel_rank() << ") error found on surface: " 
                                 << name_ << std::endl;
    NaluEnv::self().naluOutput() << "========================================= " << std::endl;
    for ( size_t k = 0; k < coindidentNodesVec.size(); ++k )
      NaluEnv::self().naluOutput() << "coincident nodeId found: " << coindidentNodesVec[k] << std::endl;
  }

  return coindidentNodesVec.size();
}

} // namespace nalu
} // namespace sierra
