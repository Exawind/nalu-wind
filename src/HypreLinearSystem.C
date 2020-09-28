// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "HypreLinearSystem.h"

namespace sierra {
namespace nalu {

HypreLinearSystem::HypreLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver)
  : LinearSystem(realm, numDof, eqSys, linearSolver), name_(eqSys->name_)
{
  rank_ = realm_.bulk_data().parallel_rank();
  columnsOwned_.clear();
  rowCountOwned_.clear();
  columnsShared_.clear();
  rowCountShared_.clear();
}

HypreLinearSystem::~HypreLinearSystem()
{
  if (systemInitialized_) {
    HYPRE_IJMatrixDestroy(mat_);
    HYPRE_IJVectorDestroy(rhs_);
    HYPRE_IJVectorDestroy(sln_);
    systemInitialized_ = false;
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  if (_nHypreAssembles>0) {
    printf("\tMean HYPRE_IJMatrix/VectorAssemble Time (%d samples)=%1.5f   Total=%1.5f\n",
	   _nHypreAssembles, _hypreAssembleTime/_nHypreAssembles, _hypreAssembleTime);
  }
#endif
}

void
HypreLinearSystem::beginLinearSystemConstruction()
{
  if (inConstruction_) return;
  inConstruction_ = true;

#ifndef HYPRE_BIGINT
  // Make sure that HYPRE is compiled with 64-bit integer support when running
  // O(~1B) linear systems.
  uint64_t totalRows = (static_cast<uint64_t>(realm_.hypreNumNodes_) *
                        static_cast<uint64_t>(numDof_));
  uint64_t maxHypreSize = static_cast<uint64_t>(std::numeric_limits<HypreIntType>::max());

  if (totalRows > maxHypreSize)
    throw std::runtime_error(
      "The linear system size is greater than what HYPRE is compiled for. "
      "Please recompile with bigint support and link to Nalu");
#endif

  if (rank_ == 0) {
    iLower_ = realm_.hypreILower_;
  } else {
    iLower_ = realm_.hypreILower_ * numDof_ ;
  }

  iUpper_ = realm_.hypreIUpper_  * numDof_ - 1;
  // For now set column indices the same as row indices
  jLower_ = iLower_;
  jUpper_ = iUpper_;

  // The total number of rows handled by this MPI rank for Hypre
  numRows_ = (iUpper_ - iLower_ + 1);
  // Total number of global rows in the system
  maxRowID_ = realm_.hypreNumNodes_ * numDof_ - 1;
  globalNumRows_ = maxRowID_ + 1;


  rowCountOwned_.resize(numRows_);
  std::fill(rowCountOwned_.begin(), rowCountOwned_.end(), 0);

  columnsOwned_.resize(numRows_);
  for (unsigned i=0; i<columnsOwned_.size(); ++i)
    columnsOwned_[i].resize(0);

  rowCountShared_.clear();
  columnsShared_.clear();

#if 0
  if (numDof_ > 0)
    std::cerr << rank_ << "\t" << numDof_ << "\t"
              << realm_.hypreILower_ << "\t" << realm_.hypreIUpper_ << "\t"
                << iLower_ << "\t" << iUpper_ << "\t"
                << numRows_ << "\t" << maxRowID_ << std::endl;
#endif
  
  // Allocate memory for the arrays used to track row types and row filled status.
  skippedRows_.clear();
  oversetRows_.clear();

  auto& bulk = realm_.bulk_data();
  std::vector<const stk::mesh::FieldBase*> fVec{realm_.hypreGlobalId_};

  stk::mesh::copy_owned_to_shared(bulk, fVec);
  stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);

  if (realm_.oversetManager_ != nullptr &&
      realm_.oversetManager_->oversetGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.oversetManager_->oversetGhosting_, fVec);

  if (realm_.nonConformalManager_ != nullptr &&
      realm_.nonConformalManager_->nonConformalGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, fVec);
  
  if (realm_.periodicManager_ != nullptr &&
      realm_.periodicManager_->periodicGhosting_ != nullptr) {
    realm_.periodicManager_->parallel_communicate_field(realm_.hypreGlobalId_);
    realm_.periodicManager_->periodic_parallel_communicate_field(
      realm_.hypreGlobalId_);
  }
}


void
HypreLinearSystem::fill_owned_shared_data_structures_1DoF(const unsigned numNodes,
							  std::vector<HypreIntType>& hids) {
  for (unsigned i=0; i<numNodes; ++i) {
    HypreIntType hid = hids[i];
    if (hid>=iLower_ && hid<=iUpper_) {
      HypreIntType lid = hid-iLower_;
      rowCountOwned_[lid]++;	      
      columnsOwned_[lid].insert(columnsOwned_[lid].end(), hids.begin(), hids.end());
    } else {
      if (rowCountShared_.find(hid)!=rowCountShared_.end()) {
	rowCountShared_.at(hid)++;
	columnsShared_.at(hid).insert(columnsShared_.at(hid).end(), hids.begin(), hids.end());
      } else {
	std::pair<HypreIntType, unsigned> x = std::make_pair(hid,1);
	rowCountShared_.insert(x);
	std::pair<HypreIntType, std::vector<HypreIntType> > y = std::make_pair(hid,hids);
	columnsShared_.insert(y);
      }	      
    }
  }
}
  
void
HypreLinearSystem::fill_owned_shared_data_structures(const unsigned numNodes,
						     std::vector<HypreIntType>& hids,
						     std::vector<HypreIntType>& columns) {
  for (unsigned i=0; i<numNodes; ++i) {
    /* hid = hid*numDof + d */
    HypreIntType hid = hids[i];
    for (unsigned d=0; d<numDof_; ++d) {
      HypreIntType HID = hid*numDof_+d;
      if (HID>=iLower_ && HID<=iUpper_) {
	HypreIntType lid = HID-iLower_;
	rowCountOwned_[lid]++;
	columnsOwned_[lid].insert(columnsOwned_[lid].end(), columns.begin(), columns.end());
      } else {
	HypreIntType lid = HID;
	if (rowCountShared_.find(lid)!=rowCountShared_.end()) {
	  rowCountShared_.at(lid)++;
	  columnsShared_.at(lid).insert(columnsShared_.at(lid).end(), columns.begin(), columns.end());
	} else {
	  std::pair<HypreIntType, unsigned> x = std::make_pair(lid,1);
	  rowCountShared_.insert(x);
	  std::pair<HypreIntType, std::vector<HypreIntType> > y = std::make_pair(lid,columns);
	  columnsShared_.insert(y);
	}
      }	  
    }
  }
}

void
HypreLinearSystem::fill_hids_columns(const unsigned numNodes, 
				     stk::mesh::Entity const * nodes,
				     std::vector<HypreIntType> & hids,
				     std::vector<HypreIntType> & columns) {
  for (unsigned i=0; i<numNodes; ++i) {
    hids[i] = get_entity_hypre_id(nodes[i]);
    for (unsigned d=0; d<numDof_; ++d)
      columns[i*numDof_+d] = hids[i]*numDof_+d;
  }
}

void
HypreLinearSystem::buildNodeGraph(
  const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_owned );

  if (numDof_==1) {
    std::vector<HypreIntType> hids(1);
    for(size_t ib=0; ib<buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *buckets[ib];
      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
	stk::mesh::Entity node = b[k];
	hids[0] = get_entity_hypre_id(node);
	
	/* fill owned/shared 1 Dof */
	fill_owned_shared_data_structures_1DoF(1, hids);
      }
    }
  } else {
    std::vector<HypreIntType> hids(1);
    std::vector<HypreIntType> columns(numDof_);

    for(size_t ib=0; ib<buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *buckets[ib];
      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
	stk::mesh::Entity node = b[k];
	hids[0] = get_entity_hypre_id(node);
	for (unsigned d=0; d<numDof_; ++d) columns[d] = hids[0]*numDof_+d;

	  /* fill owned/shared for more than 1 Dof */
	fill_owned_shared_data_structures(1, hids, columns);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());
  stk::mesh::BucketVector const& buckets = realm_.get_buckets(realm_.meta_data().side_rank(), s_owned);

  if (numDof_==1) {
    std::vector<HypreIntType> hids(0);

    for(size_t ib=0; ib<buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);

      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
	
	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	  
	/* save the hypre ids */
	for (unsigned i=0; i<numNodes; ++i) hids[i] = get_entity_hypre_id(nodes[i]);
	
	/* fill owned/shared 1 Dof */
	fill_owned_shared_data_structures_1DoF(numNodes, hids);
      }
    }
  } else {
    std::vector<HypreIntType> hids(0);
    std::vector<HypreIntType> columns(0);
    
    for(size_t ib=0; ib<buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);
      columns.resize(numNodes*numDof_);

      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
	
	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	
	/* save the hids and columns */
	fill_hids_columns(numNodes, nodes, hids, columns);
	
	/* fill owned/shared for more than 1 Dof */
	fill_owned_shared_data_structures(numNodes, hids, columns);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());
  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::EDGE_RANK, s_owned);

  if (numDof_==1) {
    std::vector<HypreIntType> hids(0);

    for(size_t ib=0; ib<buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *buckets[ib];
      
      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);

      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {

	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	  
	/* save the hypre ids */
	for (unsigned i=0; i<numNodes; ++i) hids[i] = get_entity_hypre_id(nodes[i]);
	
	/* fill owned/shared 1 Dof */
	fill_owned_shared_data_structures_1DoF(numNodes, hids);
      }
    }
  } else {
    std::vector<HypreIntType> hids(0);
    std::vector<HypreIntType> columns(0);

    for(size_t ib=0; ib<buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);
      columns.resize(numNodes*numDof_);

      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {

	stk::mesh::Entity const * nodes = b.begin_nodes(k);
	
          /* save the hids and columns */
	fill_hids_columns(numNodes, nodes, hids, columns);
	
	/* fill owned/shared for more than 1 Dof */
	fill_owned_shared_data_structures(numNodes, hids, columns);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());
  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::ELEM_RANK, s_owned);

  if (numDof_==1) {
    std::vector<HypreIntType> hids(0);

    for(size_t ib=0; ib<buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);

      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {

	stk::mesh::Entity const * nodes = b.begin_nodes(k);

	/* save the hypre ids */
	for (unsigned i=0; i<numNodes; ++i) hids[i] = get_entity_hypre_id(nodes[i]);
	
	/* fill owned/shared 1 Dof */
	fill_owned_shared_data_structures_1DoF(numNodes, hids);
      }
    }
  } else {
    std::vector<HypreIntType> hids(0);
    std::vector<HypreIntType> columns(0);

    for(size_t ib=0; ib<buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);
      columns.resize(numNodes*numDof_);

      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {

	stk::mesh::Entity const * nodes = b.begin_nodes(k);

	/* save the hids and columns */
	fill_hids_columns(numNodes, nodes, hids, columns);
	
	/* fill owned/shared for more than 1 Dof */
	fill_owned_shared_data_structures(numNodes, hids, columns);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreLinearSystem::buildFaceElemToNodeGraph(
  const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( metaData.side_rank(), s_owned );

  if (numDof_==1) {
    for(size_t ib=0; ib<face_buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *face_buckets[ib];
      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
	const stk::mesh::Entity face = b[k];
	
	// extract the connected element to this exposed face; should be single in size!
	const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
	ThrowAssert( bulkData.num_elements(face) == 1 );

	// get connected element and nodal relations
	stk::mesh::Entity element = face_elem_rels[0];
	const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);
	
	// figure out the global dof ids for each dof on each node
	const unsigned numNodes = (unsigned)bulkData.num_nodes(element);
	
	if (numNodes) {
	  /* save the hypre ids */
	  std::vector<HypreIntType> hids(numNodes);
	  for (unsigned i=0; i<numNodes; ++i) hids[i] = get_entity_hypre_id(elem_nodes[i]);
	  
	  /* fill owned/shared 1 Dof */
	  fill_owned_shared_data_structures_1DoF(numNodes, hids);
	}
      }
    }
  } else {
    for(size_t ib=0; ib<face_buckets.size(); ++ib) {
      const stk::mesh::Bucket & b = *face_buckets[ib];
      for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
	const stk::mesh::Entity face = b[k];
	
	// extract the connected element to this exposed face; should be single in size!
	const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
	ThrowAssert( bulkData.num_elements(face) == 1 );

	// get connected element and nodal relations
	stk::mesh::Entity element = face_elem_rels[0];
	const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);
	
	// figure out the global dof ids for each dof on each node
	const unsigned numNodes = (unsigned)bulkData.num_nodes(element);
	
	if (numNodes) {	  
	  std::vector<HypreIntType> columns(numNodes*numDof_);
	  std::vector<HypreIntType> hids(numNodes);

          /* save the hids and columns */
	  fill_hids_columns(numNodes, elem_nodes, hids, columns);

	  /* fill owned/shared for more than 1 Dof */
	  fill_owned_shared_data_structures(numNodes, hids, columns);
	}
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreLinearSystem::buildReducedElemToNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}


void
HypreLinearSystem::buildNonConformalNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}


void
HypreLinearSystem::buildOversetNodeGraph(
  const stk::mesh::PartVector&)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  // extract the rank
  const int theRank = NaluEnv::self().parallel_rank();

  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  beginLinearSystemConstruction();

  std::vector<stk::mesh::Entity> entities;
  std::vector<HypreIntType> hids;

  // Mark all the fringe nodes as skipped so that sumInto doesn't add into these
  // rows during assembly process
  for( const OversetInfo* oversetInfo : realm_.oversetManager_->oversetInfoVec_) {

    // extract element mesh object and orphan node
    stk::mesh::Entity owningElement = oversetInfo->owningElement_;
    stk::mesh::Entity orphanNode = oversetInfo->orphanNode_;

    // extract the owning rank for this node
    const int nodeRank = bulkData.parallel_owner_rank(orphanNode);

    const bool nodeIsLocallyOwned = (theRank == nodeRank);
    if ( !nodeIsLocallyOwned )
      continue;

    // relations
    stk::mesh::Entity const* elem_nodes = bulkData.begin_nodes(owningElement);
    const size_t numNodes = bulkData.num_nodes(owningElement);
    const size_t numEntities = numNodes+1;
    entities.resize(numEntities);
    hids.resize(numEntities);

    entities[0] = orphanNode;
    hids[0] = get_entity_hypre_id(entities[0]);
    for(size_t n=0; n < numNodes; ++n) {
      entities[n+1] = elem_nodes[n];
      hids[n+1] = get_entity_hypre_id(entities[n+1]);
    }

    /* save the hypre ids */
    for (unsigned d=0; d<numDof_; ++d) {
      HypreIntType hid = hids[0] * numDof_ + d;
      skippedRows_.insert(hid);
      oversetRows_.insert(hid);
      if (hid>=iLower_ && hid<=iUpper_) {
	HypreIntType lid = hid-iLower_;
	rowCountOwned_[lid]++;
	columnsOwned_[lid].resize(0);
	columnsOwned_[lid].insert(columnsOwned_[lid].end(), hids.begin(), hids.end());
      }
    } 
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf, oversetRows_ size=%d\n",
	 rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec,(int)oversetRows_.size());
#endif

}


void
HypreLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  // Grab nodes regardless of whether they are owned or shared
  const stk::mesh::Selector sel = stk::mesh::selectUnion(parts);
  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
      for (unsigned d=0; d<numDof_; ++d) {
	HypreIntType lid = hid * numDof_ + d;
	skippedRows_.insert(lid);
	if (lid>=iLower_ && lid<=iUpper_) {
	  rowCountOwned_[lid-iLower_]++;
	  columnsOwned_[lid-iLower_].push_back(lid);
	}
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  for (const auto& node: nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);
    for (unsigned d=0; d<numDof_; ++d) {
      HypreIntType lid = hid * numDof_ + d;
      skippedRows_.insert(lid);
      if (lid>=iLower_ && lid<=iUpper_) {
	rowCountOwned_[lid-iLower_]++;
	columnsOwned_[lid-iLower_].push_back(lid);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void 
HypreLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::NgpMesh::ConnectedNodes nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  for (unsigned i=0; i<nodeList.size(); ++i) {
    HypreIntType hid = get_entity_hypre_id(nodeList[i]);
    for (unsigned d=0; d<numDof_; ++d) {
      HypreIntType lid = hid * numDof_ + d;
      skippedRows_.insert(lid);
      if (lid>=iLower_ && lid<=iUpper_) {
	rowCountOwned_[lid-iLower_]++;
	columnsOwned_[lid-iLower_].push_back(lid);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreLinearSystem::finalizeLinearSystem()
{
  ThrowRequire(inConstruction_);
  inConstruction_ = false;

  finalizeSolver();

  /* create these mappings */
  fill_entity_to_row_mapping();

  /* fill the various device data structures need in device coeff applier */
  fill_device_data_structures();

  /**********************************************************************************/
  /* Build the coeff applier ... host data structure for building the linear system */
  if (!hostCoeffApplier) {
    HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);
    bool ensureReproducible = solver->getConfig()->ensureReproducible();
    bool useNativeCudaSort = solver->getConfig()->useNativeCudaSort();
    
    hostCoeffApplier.reset(new HypreLinSysCoeffApplier(useNativeCudaSort, ensureReproducible, numDof_, 1, 
						       globalNumRows_, rank_, iLower_, iUpper_, jLower_, jUpper_,
						       map_shared_, mat_elem_cols_owned_, mat_elem_cols_shared_,
						       mat_elem_start_owned_, mat_elem_start_shared_,
						       mat_row_start_owned_, mat_row_start_shared_,							  
						       rhs_row_start_owned_, rhs_row_start_shared_,
						       row_indices_owned_, row_indices_shared_, 
						       row_counts_owned_, row_counts_shared_,
						       num_mat_pts_to_assemble_total_owned_,
						       num_mat_pts_to_assemble_total_shared_,
						       num_rhs_pts_to_assemble_total_owned_,
						       num_rhs_pts_to_assemble_total_shared_,
						       periodic_bc_rows_owned_, entityToLID_, entityToLIDHost_,
						       skippedRowsMap_, skippedRowsMapHost_,
						       oversetRowsMap_, oversetRowsMapHost_,
						       num_mat_overset_pts_owned_, num_rhs_overset_pts_owned_));
    deviceCoeffApplier = hostCoeffApplier->device_pointer();
  }


  // At this stage the LHS and RHS data structures are ready for
  // sumInto/assembly.
  systemInitialized_ = true;
}

void
HypreLinearSystem::finalizeSolver()
{
  MPI_Comm comm = realm_.bulk_data().parallel();
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_);
  HYPRE_IJVectorSetObjectType(rhs_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_);
  HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));
}


void HypreLinearSystem::fill_entity_to_row_mapping()
{
  const auto& meta = realm_.meta_data();
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  stk::mesh::Selector selector =
    (meta.locally_owned_part() | meta.globally_shared_part())
      & stk::mesh::selectField(*realm_.naluGlobalId_)
      & !(realm_.get_inactive_selector());
  entityToLIDHost_ = HypreIntTypeViewHost("entityToRowLIDHost",bulk.get_size_of_entity_index_space());
  entityToLID_ = HypreIntTypeView("entityToRowLID",bulk.get_size_of_entity_index_space());

  const stk::mesh::BucketVector& nodeBuckets = realm_.get_buckets(stk::topology::NODE_RANK, selector);
  for(const stk::mesh::Bucket* bptr : nodeBuckets) {
    const stk::mesh::Bucket& b = *bptr;
    for(size_t i=0; i<b.size(); ++i) {
      stk::mesh::Entity node = b[i];
      const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
      const auto mnode = (naluId == bulk.identifier(node)) ? node : bulk.get_entity(stk::topology::NODE_RANK, naluId);
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);
      entityToLIDHost_[node.local_offset()] = hid;
    }
  }
  Kokkos::deep_copy(entityToLID_, entityToLIDHost_);
}


void HypreLinearSystem::fill_device_data_structures()
{
  /***************************************************************************/
  /* Construct the device data structures for where to  write into the lists */
  /***************************************************************************/

  /**********************************/
  /* Construct the owned part first */
  std::vector<HypreIntType> matElemColsOwned(0);
  std::vector<unsigned> matElemStartOwned(0);
  matElemStartOwned.push_back(0);
  std::vector<unsigned> rhsGlobalCountOwned(0);
  std::vector<HypreIntType> matColumnsPerRowCountOwned(0);
  std::vector<HypreIntType> periodicBCsOwned(0);
  std::vector<HypreIntType> validRowsOwned(0);
  num_mat_pts_to_assemble_total_owned_ = 0;
  num_rhs_pts_to_assemble_total_owned_ = 0;
  num_mat_overset_pts_owned_=0;
  num_rhs_overset_pts_owned_=0;

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  for (HypreIntType j=iLower_; j<=iUpper_; ++j) {
    unsigned matRowCount = 1;
    unsigned rhsRowCount = 1;
    HypreIntType matRowColumnCount = 1;
    
    HypreIntType jShift = j-iLower_;

    if (oversetRows_.find(j) != oversetRows_.end()) {
      std::sort(columnsOwned_[jShift].begin(), columnsOwned_[jShift].end());
      unsigned t=1;
      matRowColumnCount = 1;
      HypreIntType col = columnsOwned_[jShift][0];
      
      for (unsigned i=1; i<columnsOwned_[jShift].size(); ++i) {
	if (columnsOwned_[jShift][i]==col) {
	  t++;
	} else {
	  matElemColsOwned.push_back(col);
	  matElemStartOwned.push_back(matElemStartOwned.back() + t);
	  t = 1;
	  col = columnsOwned_[jShift][i];
	  matRowColumnCount++;
	}
      }
      matElemColsOwned.push_back(col);
      matElemStartOwned.push_back(matElemStartOwned.back() + t);
      matRowCount = columnsOwned_[jShift].size();
      rhsRowCount = rowCountOwned_[jShift];
      num_mat_overset_pts_owned_+=matRowCount;
      num_rhs_overset_pts_owned_++;

    } else if (skippedRows_.find(j) != skippedRows_.end()) {
      /* Deal with dirichlet BCs */
      matElemStartOwned.push_back(matElemStartOwned.back() + 1);
      matElemColsOwned.push_back(j);
    } else {      
      /* check periodic BC first */
      if (columnsOwned_[jShift].size()==0) {
	matElemStartOwned.push_back(matElemStartOwned.back() + 1);
	matElemColsOwned.push_back(j);
	periodicBCsOwned.push_back(j);
      } else if (columnsOwned_[jShift].size()==1) {
	matElemStartOwned.push_back(matElemStartOwned.back() + 1);
	matElemColsOwned.push_back(j);
      } else {
	std::sort(columnsOwned_[jShift].begin(), columnsOwned_[jShift].end());
	unsigned t=1;
	matRowColumnCount = 1;
	HypreIntType col = columnsOwned_[jShift][0];
	for (unsigned i=1; i<columnsOwned_[jShift].size(); ++i) {
	  if (columnsOwned_[jShift][i]==col) {
	    t++;
	  } else {
	    matElemColsOwned.push_back(col);
	    matElemStartOwned.push_back(matElemStartOwned.back() + t);
	    t = 1;
	    col = columnsOwned_[jShift][i];
	    matRowColumnCount++;
	  }
	}
	matElemColsOwned.push_back(col);
	matElemStartOwned.push_back(matElemStartOwned.back() + t);
	matRowCount = columnsOwned_[jShift].size();
	rhsRowCount = rowCountOwned_[jShift];
      } 
    }

    /* Figure out the row indices affected by this rank */
    if (matRowCount>0) {
      validRowsOwned.push_back(j);
      matColumnsPerRowCountOwned.push_back(matRowColumnCount);
      rhsGlobalCountOwned.push_back(rhsRowCount);
      num_mat_pts_to_assemble_total_owned_ += matRowCount;
      num_rhs_pts_to_assemble_total_owned_ += rhsRowCount;
    }
  }  

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
  gettimeofday(&_start, NULL);
#endif

  /**********************************/
  /* Construct the shared part next */
  std::vector<HypreIntType> matElemColsShared(0);
  std::vector<unsigned> matElemStartShared(0);
  matElemStartShared.push_back(0);
  std::vector<unsigned> rhsGlobalCountShared(0);
  std::vector<HypreIntType> matColumnsPerRowCountShared(0);
  std::vector<HypreIntType> validRowsShared(0);
  num_mat_pts_to_assemble_total_shared_ = 0;
  num_rhs_pts_to_assemble_total_shared_ = 0;

  for (auto it = rowCountShared_.begin(); it != rowCountShared_.end(); it++) {
    unsigned matRowCount = 1;
    unsigned rhsRowCount = 1;
    HypreIntType matRowColumnCount = 1;
    
    HypreIntType hid = it->first;
    rhsRowCount = it->second;
    std::vector<HypreIntType> columns = columnsShared_[hid];
    if (skippedRows_.find(hid) != skippedRows_.end()) {
      continue;
    } else if (columns.size()==1) {
      matElemStartShared.push_back(matElemStartShared.back() + 1);
      matElemColsShared.push_back(hid);
    } else if (columns.size()>1) {
      std::sort(columns.begin(), columns.end());
      unsigned t=1;
      matRowColumnCount = 1;
      HypreIntType col = columns[0];

      for (unsigned i=1; i<columns.size(); ++i) {
	if (columns[i]==col) {
	  t++;
	} else {
	  matElemColsShared.push_back(col);
	  matElemStartShared.push_back(matElemStartShared.back() + t);
	  t = 1;
	  col = columns[i];
	  matRowColumnCount++;
	}
      }
      matElemColsShared.push_back(col);
      matElemStartShared.push_back(matElemStartShared.back() + t);
      matRowCount = columns.size();
    } else {
      matRowCount = 0;
      rhsRowCount = 0;
      matRowColumnCount = 0;
    }
    /* Figure out the row indices affected by this rank */
    if (matRowCount>0) {
      validRowsShared.push_back(hid);
      matColumnsPerRowCountShared.push_back(matRowColumnCount);
      rhsGlobalCountShared.push_back(rhsRowCount);
      num_mat_pts_to_assemble_total_shared_ += matRowCount;
      num_rhs_pts_to_assemble_total_shared_ += rhsRowCount;
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
  gettimeofday(&_start, NULL);
#endif

  /********************************************/
  /* Matrix element data structures ... owned */
  /********************************************/
  unsigned n1 = matElemStartOwned.size();
  unsigned n2 = matElemColsOwned.size();

  mat_elem_start_owned_ = UnsignedView("mat_elem_start_owned",n1);
  UnsignedViewHost mat_elem_start_owned_host = Kokkos::create_mirror_view(mat_elem_start_owned_);

  mat_elem_cols_owned_ = HypreIntTypeView("mat_elem_cols_owned",n2);
  HypreIntTypeViewHost mat_elem_cols_owned_host = Kokkos::create_mirror_view(mat_elem_cols_owned_);

  for (unsigned i=0; i<n1; ++i) mat_elem_start_owned_host(i) = matElemStartOwned[i];
  for (unsigned i=0; i<n2; ++i) mat_elem_cols_owned_host(i) = matElemColsOwned[i];

  Kokkos::deep_copy(mat_elem_start_owned_, mat_elem_start_owned_host);
  Kokkos::deep_copy(mat_elem_cols_owned_, mat_elem_cols_owned_host);

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
  gettimeofday(&_start, NULL);
#endif

  /*********************************************/
  /* Matrix element data structures ... shared */
  /*********************************************/
  n1 = matElemStartShared.size();
  n2 = matElemColsShared.size();

  mat_elem_start_shared_ = UnsignedView("mat_elem_start_shared",n1);
  UnsignedViewHost mat_elem_start_shared_host = Kokkos::create_mirror_view(mat_elem_start_shared_);

  mat_elem_cols_shared_ = HypreIntTypeView("mat_elem_cols_shared",n2);
  HypreIntTypeViewHost mat_elem_cols_shared_host = Kokkos::create_mirror_view(mat_elem_cols_shared_);

  for (unsigned i=0; i<n1; ++i) mat_elem_start_shared_host(i) = matElemStartShared[i];
  for (unsigned i=0; i<n2; ++i) mat_elem_cols_shared_host(i) = matElemColsShared[i];

  Kokkos::deep_copy(mat_elem_start_shared_, mat_elem_start_shared_host);
  Kokkos::deep_copy(mat_elem_cols_shared_, mat_elem_cols_shared_host);

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
  gettimeofday(&_start, NULL);
#endif

  /***********************************/
  /* Other data structures ... owned */
  /***********************************/
  unsigned numRows = validRowsOwned.size();
  row_indices_owned_ = HypreIntTypeView("row_indices_owned",numRows);
  HypreIntTypeViewHost row_indices_owned_host = Kokkos::create_mirror_view(row_indices_owned_);

  row_counts_owned_ = HypreIntTypeView("row_counts_owned",numRows);
  HypreIntTypeViewHost row_counts_owned_host = Kokkos::create_mirror_view(row_counts_owned_);

  rhs_row_start_owned_ = UnsignedView("rhs_row_start_owned",numRows+1);
  UnsignedViewHost rhs_row_start_owned_host = Kokkos::create_mirror_view(rhs_row_start_owned_);

  mat_row_start_owned_ = UnsignedView("mat_row_start_owned",numRows+1);
  UnsignedViewHost mat_row_start_owned_host = Kokkos::create_mirror_view(mat_row_start_owned_);

  /* create the maps */
  rhs_row_start_owned_host(0) = 0;
  mat_row_start_owned_host(0) = 0;
  for (unsigned i=0; i<numRows; ++i) {
    row_indices_owned_host(i) = validRowsOwned[i];
    row_counts_owned_host(i) = matColumnsPerRowCountOwned[i];
    rhs_row_start_owned_host(i+1) = rhs_row_start_owned_host(i) + rhsGlobalCountOwned[i];
    mat_row_start_owned_host(i+1) = mat_row_start_owned_host(i) + matColumnsPerRowCountOwned[i];
  }
  Kokkos::deep_copy(row_counts_owned_, row_counts_owned_host);
  Kokkos::deep_copy(row_indices_owned_, row_indices_owned_host);
  Kokkos::deep_copy(rhs_row_start_owned_, rhs_row_start_owned_host);
  Kokkos::deep_copy(mat_row_start_owned_, mat_row_start_owned_host);

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
  gettimeofday(&_start, NULL);
#endif

  /************************************/
  /* Other data structures ... shared */
  /************************************/
  numRows = validRowsShared.size();
  row_indices_shared_ = HypreIntTypeView("row_indices_shared",numRows);
  HypreIntTypeViewHost row_indices_shared_host = Kokkos::create_mirror_view(row_indices_shared_);

  row_counts_shared_ = HypreIntTypeView("row_counts_shared",numRows);
  HypreIntTypeViewHost row_counts_shared_host = Kokkos::create_mirror_view(row_counts_shared_);

  rhs_row_start_shared_ = UnsignedView("rhs_row_start_shared",numRows+1);
  UnsignedViewHost rhs_row_start_shared_host = Kokkos::create_mirror_view(rhs_row_start_shared_);

  mat_row_start_shared_ = UnsignedView("mat_row_start_shared",numRows+1);
  UnsignedViewHost mat_row_start_shared_host = Kokkos::create_mirror_view(mat_row_start_shared_);

  /* create the maps */
  rhs_row_start_shared_host(0) = 0;
  mat_row_start_shared_host(0) = 0;
  for (unsigned i=0; i<numRows; ++i) {
    row_indices_shared_host(i) = validRowsShared[i];
    row_counts_shared_host(i) = matColumnsPerRowCountShared[i];
    rhs_row_start_shared_host(i+1) = rhs_row_start_shared_host(i) + rhsGlobalCountShared[i];
    mat_row_start_shared_host(i+1) = mat_row_start_shared_host(i) + matColumnsPerRowCountShared[i];
  }

  Kokkos::deep_copy(row_counts_shared_, row_counts_shared_host);
  Kokkos::deep_copy(row_indices_shared_, row_indices_shared_host);
  Kokkos::deep_copy(rhs_row_start_shared_, rhs_row_start_shared_host);
  Kokkos::deep_copy(mat_row_start_shared_, mat_row_start_shared_host);

  /* Create the map on device */
  map_shared_ = MemoryMap(numRows);
  auto ms = map_shared_;
  auto ris = row_indices_shared_;
  Kokkos::parallel_for("init_shared_map", numRows, KOKKOS_LAMBDA(const unsigned& i) {
      ms.insert(ris(i),i);
    });

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
  gettimeofday(&_start, NULL);
#endif

  /* Handle periodic boundary conditions */
  periodic_bc_rows_owned_ = HypreIntTypeView("periodic_bc_rows",periodicBCsOwned.size());
  HypreIntTypeViewHost periodic_bc_rows_owned_host = Kokkos::create_mirror_view(periodic_bc_rows_owned_);
  for (unsigned i=0; i<periodicBCsOwned.size(); ++i) periodic_bc_rows_owned_host(i) = periodicBCsOwned[i];
  Kokkos::deep_copy(periodic_bc_rows_owned_, periodic_bc_rows_owned_host);

  /* skipped rows data structure */
  skippedRowsMap_ = HypreIntTypeUnorderedMap(skippedRows_.size());
  skippedRowsMapHost_ = HypreIntTypeUnorderedMapHost(skippedRows_.size());
  for (auto t : skippedRows_) skippedRowsMapHost_.insert(t);
  Kokkos::deep_copy(skippedRowsMap_, skippedRowsMapHost_);

  /* overset rows data structure */
  oversetRowsMap_ = HypreIntTypeUnorderedMap(oversetRows_.size());
  oversetRowsMapHost_ = HypreIntTypeUnorderedMapHost(oversetRows_.size());
  for (auto t : oversetRows_) oversetRowsMapHost_.insert(t);
  Kokkos::deep_copy(oversetRowsMap_, oversetRowsMapHost_);


#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif

  /* clear this data so that the next time a coeffApplier is built, these get rebuilt from scratch */
  rowCountOwned_.resize(numRows_);
  std::fill(rowCountOwned_.begin(), rowCountOwned_.end(), 0);

  columnsOwned_.resize(numRows_);
  for (unsigned i=0; i<columnsOwned_.size(); ++i) columnsOwned_[i].resize(0);

  rowCountShared_.clear();
  columnsShared_.clear();

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  printf("rank=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif

}


void
HypreLinearSystem::loadComplete()
{
  HypreLinSysCoeffApplier* hcApplier = dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());
  std::vector<HYPRE_IJVector> rhs(1);
  rhs[0] = rhs_;
  hcApplier->finishAssembly(mat_, rhs);
  loadCompleteSolver();
}


void
HypreLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorAssemble(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorAssemble(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  _hypreAssembleTime+=msec;
  _nHypreAssembles++;
#endif

  solver->comm_ = realm_.bulk_data().parallel();

  // Set flag to indicate zeroSystem that the matrix must be reinitialized
  // during the next invocation.
  matrixAssembled_ = true;
}

void
HypreLinearSystem::zeroSystem()
{
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  // It is unsafe to call IJMatrixInitialize multiple times without intervening
  // call to IJMatrixAssemble. This occurs during the first outer iteration (of
  // first timestep in static application and every timestep in moving mesh
  // applications) when the data structures have been created but never used and
  // zeroSystem is called for a reset. Include a check to ensure we only
  // initialize if it was previously assembled.
  if (matrixAssembled_) {
    HYPRE_IJMatrixInitialize(mat_);
    HYPRE_IJVectorInitialize(rhs_);
    HYPRE_IJVectorInitialize(sln_);

    // Set flag to false until next invocation of IJMatrixAssemble in loadComplete
    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parRhs_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parSln_, 0.0);

}

sierra::nalu::CoeffApplier* HypreLinearSystem::get_coeff_applier()
{
  /* reset the internal data */
  HypreLinSysCoeffApplier* hcApplier = dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());
  hcApplier->resetInternalData();
  return deviceCoeffApplier;
}

/********************************************************************************************************/
/*                     Beginning of HypreLinSysCoeffApplier implementations                             */
/********************************************************************************************************/
HypreLinearSystem::HypreLinSysCoeffApplier::HypreLinSysCoeffApplier(bool useNativeCudaSort, bool ensureReproducible, unsigned numDof,
								    unsigned nDim, HypreIntType globalNumRows, int rank, 
								    HypreIntType iLower, HypreIntType iUpper,
								    HypreIntType jLower, HypreIntType jUpper, MemoryMap map_shared,
								    HypreIntTypeView mat_elem_cols_owned, HypreIntTypeView mat_elem_cols_shared,
								    UnsignedView mat_elem_start_owned, UnsignedView mat_elem_start_shared,
								    UnsignedView mat_row_start_owned, UnsignedView mat_row_start_shared,
								    UnsignedView rhs_row_start_owned, UnsignedView rhs_row_start_shared,
								    HypreIntTypeView row_indices_owned, HypreIntTypeView row_indices_shared, 
								    HypreIntTypeView row_counts_owned, HypreIntTypeView row_counts_shared,
								    HypreIntType num_mat_pts_to_assemble_total_owned,
								    HypreIntType num_mat_pts_to_assemble_total_shared,
								    HypreIntType num_rhs_pts_to_assemble_total_owned,
								    HypreIntType num_rhs_pts_to_assemble_total_shared,
								    HypreIntTypeView periodic_bc_rows_owned,
								    HypreIntTypeView entityToLID, HypreIntTypeViewHost entityToLIDHost,
								    HypreIntTypeUnorderedMap skippedRowsMap, HypreIntTypeUnorderedMapHost skippedRowsMapHost,
								    HypreIntTypeUnorderedMap oversetRowsMap, HypreIntTypeUnorderedMapHost oversetRowsMapHost,
								    HypreIntType num_mat_overset_pts_owned, HypreIntType num_rhs_overset_pts_owned)
  : useNativeCudaSort_(useNativeCudaSort), ensureReproducible_(ensureReproducible), numDof_(numDof), nDim_(nDim),
    globalNumRows_(globalNumRows), rank_(rank), iLower_(iLower), iUpper_(iUpper), jLower_(jLower), jUpper_(jUpper),
    map_shared_(map_shared), mat_elem_cols_owned_(mat_elem_cols_owned), mat_elem_cols_shared_(mat_elem_cols_shared),
    mat_elem_start_owned_(mat_elem_start_owned), mat_elem_start_shared_(mat_elem_start_shared),
    mat_row_start_owned_(mat_row_start_owned), mat_row_start_shared_(mat_row_start_shared),  
    rhs_row_start_owned_(rhs_row_start_owned), rhs_row_start_shared_(rhs_row_start_shared),  
    row_indices_owned_(row_indices_owned), row_indices_shared_(row_indices_shared),
    row_counts_owned_(row_counts_owned), row_counts_shared_(row_counts_shared),
    num_mat_pts_to_assemble_total_owned_(num_mat_pts_to_assemble_total_owned),
    num_mat_pts_to_assemble_total_shared_(num_mat_pts_to_assemble_total_shared),
    num_rhs_pts_to_assemble_total_owned_(num_rhs_pts_to_assemble_total_owned),
    num_rhs_pts_to_assemble_total_shared_(num_rhs_pts_to_assemble_total_shared),
    periodic_bc_rows_owned_(periodic_bc_rows_owned), entityToLID_(entityToLID), entityToLIDHost_(entityToLIDHost),
    skippedRowsMap_(skippedRowsMap), skippedRowsMapHost_(skippedRowsMapHost),
    oversetRowsMap_(oversetRowsMap), oversetRowsMapHost_(oversetRowsMapHost), 
    num_mat_overset_pts_owned_(num_mat_overset_pts_owned), num_rhs_overset_pts_owned_(num_rhs_overset_pts_owned),
    devicePointer_(nullptr)
{
  /* The total number of rows handled by this MPI rank for Hypre */
  num_rows_owned_ = row_indices_owned_.extent(0);
  num_rows_shared_ = row_indices_shared_.extent(0);
  num_rows_ = num_rows_owned_ + num_rows_shared_;

  /* The total number of nonzeors handled by this MPI rank for Hypre */
  num_nonzeros_owned_ = mat_elem_start_owned_.extent(0)-1;
  num_nonzeros_shared_ = mat_elem_start_shared_.extent(0)-1;
  num_nonzeros_ = num_nonzeros_owned_ + num_nonzeros_shared_;
  
  num_nonzeros_shared_ = num_nonzeros_shared_ < 0 ? 0 : num_nonzeros_shared_;
  num_nonzeros_ = num_nonzeros_owned_ + num_nonzeros_shared_;

  /*************************************/
  /* ALLOCATE Space for the CSR Matrix */
  /*************************************/

  /* values */
  d_values_owned_ = DoubleView("values_owned",num_nonzeros_owned_);
  h_values_owned_ = Kokkos::create_mirror_view(d_values_owned_);

  d_values_shared_ = DoubleView("values_shared",num_nonzeros_shared_);
  h_values_shared_ = Kokkos::create_mirror_view(d_values_shared_);

  /* col_indices */
  d_col_indices_owned_ = HypreIntTypeView("col_indices_owned",num_nonzeros_owned_);
  h_col_indices_owned_ = Kokkos::create_mirror_view(d_col_indices_owned_);

  d_col_indices_shared_ = HypreIntTypeView("col_indices_shared",num_nonzeros_shared_);
  h_col_indices_shared_ = Kokkos::create_mirror_view(d_col_indices_shared_);

  /* row_indices */
  h_row_indices_owned_ = Kokkos::create_mirror_view(row_indices_owned_);
  Kokkos::deep_copy(h_row_indices_owned_, row_indices_owned_);

  h_row_indices_shared_ = Kokkos::create_mirror_view(row_indices_shared_);
  Kokkos::deep_copy(h_row_indices_shared_, row_indices_shared_);

  /* row_counts */
  h_row_counts_owned_ = Kokkos::create_mirror_view(row_counts_owned_);
  Kokkos::deep_copy(h_row_counts_owned_, row_counts_owned_);

  h_row_counts_shared_ = Kokkos::create_mirror_view(row_counts_shared_);
  Kokkos::deep_copy(h_row_counts_shared_, row_counts_shared_);

  /*************************************/
  /* ALLOCATE Space for the Rhs Vector */
  /*************************************/
  d_rhs_owned_ = DoubleView2D("rhs_owned", num_rows_owned_, nDim_);
  h_rhs_owned_ = Kokkos::create_mirror_view(d_rhs_owned_);

  d_rhs_shared_ = DoubleView2D("rhs_shared", num_rows_shared_, nDim_);
  h_rhs_shared_ = Kokkos::create_mirror_view(d_rhs_shared_);

  /***************************************/
  /* ALLOCATE Space for the temporaries  */
  /***************************************/
  mat_counter_owned_ = UnsignedView("mat_counter_owned", mat_elem_start_owned_.extent(0));
  rhs_counter_owned_ = UnsignedView("rhs_counter_owned", rhs_row_start_owned_.extent(0));

  mat_counter_shared_ = UnsignedView("mat_counter_shared", mat_elem_start_shared_.extent(0));
  rhs_counter_shared_ = UnsignedView("rhs_counter_shared", rhs_row_start_shared_.extent(0));

  /* storage for the matrix lists */
  cols_owned_ = HypreIntTypeView("cols_owned",num_mat_pts_to_assemble_total_owned_);
  vals_owned_ = DoubleView("vals_owned",num_mat_pts_to_assemble_total_owned_);

  cols_shared_ = HypreIntTypeView("cols_shared",num_mat_pts_to_assemble_total_shared_);
  vals_shared_ = DoubleView("vals_shared",num_mat_pts_to_assemble_total_shared_);

  /* storage for the rhs lists */
  rhs_vals_owned_ = DoubleView2D("rhs_vals_owned", num_rhs_pts_to_assemble_total_owned_, nDim_);
  rhs_vals_shared_ = DoubleView2D("rhs_vals_shared", num_rhs_pts_to_assemble_total_shared_, nDim_);

  /***************************************/
  /* ALLOCATE Space for work vectors     */
  /***************************************/
  HypreIntType N1 = num_mat_pts_to_assemble_total_owned_ > num_mat_pts_to_assemble_total_shared_ ? 
    num_mat_pts_to_assemble_total_owned_ : num_mat_pts_to_assemble_total_shared_;
  HypreIntType N2 = num_rhs_pts_to_assemble_total_owned_ > num_rhs_pts_to_assemble_total_shared_ ? 
    num_rhs_pts_to_assemble_total_owned_ : num_rhs_pts_to_assemble_total_shared_;
  HypreIntType N3 = N1 > N2 ? N1 : N2;
  
  iwork_ = HypreIntTypeView("iwork",N3);
  dwork_ = DoubleView("dwork",N3);

  /* check skipped rows */
  checkSkippedRows_ = HypreIntTypeViewScalar("checkSkippedRows_");
  Kokkos::deep_copy(checkSkippedRows_, 1);

  /* Work space for overset. These are used to accumulate data from legacy, non-NGP sumInto calls */
  d_overset_row_indices_ = HypreIntTypeView("overset_row_indices",num_rhs_overset_pts_owned_);
  h_overset_row_indices_ = Kokkos::create_mirror_view(d_overset_row_indices_);

  d_overset_rhs_vals_ = DoubleView("overset_rhs_vals",num_rhs_overset_pts_owned_);
  h_overset_rhs_vals_ = Kokkos::create_mirror_view(d_overset_rhs_vals_);

  d_overset_rows_ = HypreIntTypeView("overset_rows",num_mat_overset_pts_owned_);
  h_overset_rows_ = Kokkos::create_mirror_view(d_overset_rows_);

  d_overset_cols_ = HypreIntTypeView("overset_cols",num_mat_overset_pts_owned_);
  h_overset_cols_ = Kokkos::create_mirror_view(d_overset_cols_);

  d_overset_vals_ = DoubleView("overset_vals",num_mat_overset_pts_owned_);
  h_overset_vals_ = Kokkos::create_mirror_view(d_overset_vals_);

  overset_mat_counter_=0;
  overset_rhs_counter_=0;
}


KOKKOS_FUNCTION
void 
HypreLinearSystem::HypreLinSysCoeffApplier::binarySearchOwned(unsigned l, unsigned r, HypreIntType x, unsigned& result) { 
  if (r >= l) { 
    unsigned mid = l + (r - l) / 2; 
    if (mat_elem_cols_owned_(mid) == x) {
      result = mid;
      return; 
    } 
    if (mat_elem_cols_owned_(mid) > x) {
      binarySearchOwned(l, mid - 1, x, result);
      return;
    }
    binarySearchOwned(mid + 1, r, x, result); 
  } 
}


KOKKOS_FUNCTION
void 
HypreLinearSystem::HypreLinSysCoeffApplier::binarySearchShared(unsigned l, unsigned r, HypreIntType x, unsigned& result) { 
  if (r >= l) { 
    unsigned mid = l + (r - l) / 2; 
    if (mat_elem_cols_shared_(mid) == x) {
      result = mid;
      return; 
    } 
    if (mat_elem_cols_shared_(mid) > x) {
      binarySearchShared(l, mid - 1, x, result);
      return;
    }
    binarySearchShared(mid + 1, r, x, result); 
  } 
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::sum_into(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const HypreIntType& iLower, const HypreIntType& iUpper,
  unsigned numDof) {
  
  unsigned numRows = numEntities * numDof;

  for(unsigned i=0; i<numEntities; ++i) {
    HypreIntType hid = entityToLID_[entities[i].local_offset()];
    for(unsigned d=0; d<numDof; ++d) {
      unsigned lid = i*numDof + d;
      localIds[lid] = hid*numDof + d;
    }
  }

  for (unsigned i=0; i<numEntities; ++i) {
    int ix = i * numDof;
    HypreIntType hid = localIds[ix];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid)) continue;
    }

    if (hid>=iLower && hid<=iUpper) {

      for (unsigned d=0; d<numDof; ++d) {
	unsigned ir = ix + d;
	HypreIntType hid = (HypreIntType)localIds[ir];
	const double* cur_lhs = &lhs(ir, 0);
	unsigned lower = mat_row_start_owned_(hid-iLower);
	unsigned upper = mat_row_start_owned_(hid-iLower+1)-1;

	/* fill the matrix values */
	for (unsigned k=0; k<numRows; ++k) {
	  /* binary search subrange rather than a map.find */
	  HypreIntType col = localIds[k];
	  unsigned matIndex;
	  binarySearchOwned(lower,upper,col,matIndex);	  
	  /* Find the matrix element memory location */
	  matIndex = Kokkos::atomic_fetch_add(&mat_counter_owned_(matIndex), (unsigned)1);
	  /* write to the lists */
	  cols_owned_(matIndex) = col;
	  vals_owned_(matIndex) = cur_lhs[k];
	}
	/* fill the right hand side values */
	unsigned rhsIndex = hid-iLower;
	rhsIndex = Kokkos::atomic_fetch_add(&rhs_counter_owned_(rhsIndex), (unsigned)1);
	rhs_vals_owned_(rhsIndex,0) = rhs[ir];
      }

    } else {

      for (unsigned d=0; d<numDof; ++d) {
	unsigned ir = ix + d;
	const double* cur_lhs = &lhs(ir, 0);
	hid = localIds[ir];
	
	if (!map_shared_.exists(hid)) continue;

	/* Find the index of the row */
	unsigned index = map_shared_.value_at(map_shared_.find(hid));
	unsigned lower = mat_row_start_shared_(index);
	unsigned upper = mat_row_start_shared_(index+1)-1;

	/* fill the matrix values */
	for (unsigned k=0; k<numRows; ++k) {
	  /* binary search subrange rather than a map.find */
	  HypreIntType col = localIds[k];
	  unsigned matIndex;
	  binarySearchShared(lower,upper,col,matIndex);	  
	  /* Find the matrix element memory location */
	  matIndex = Kokkos::atomic_fetch_add(&mat_counter_shared_(matIndex), (unsigned)1);
	  /* write to the lists */
	  cols_shared_(matIndex) = col;
	  vals_shared_(matIndex) = cur_lhs[k];
	}
	/* fill the right hand side values */
	unsigned rhsIndex = Kokkos::atomic_fetch_add(&rhs_counter_shared_(index), (unsigned)1);
	rhs_vals_shared_(rhsIndex,0) = rhs[ir];
      }
    }
  }
}


KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::sum_into_1DoF(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const HypreIntType& iLower, const HypreIntType& iUpper) {
  
  for(unsigned i=0; i<numEntities; ++i)
    localIds[i] = entityToLID_[entities[i].local_offset()];

  for (unsigned i=0; i<numEntities; ++i) {
    HypreIntType hid = localIds[i];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid)) continue;
    }

    const double* cur_lhs = &lhs(i, 0);

    if (hid>=iLower && hid<=iUpper) {
      /* fill the matrix values */
      unsigned lower = mat_row_start_owned_(hid-iLower);
      unsigned upper = mat_row_start_owned_(hid-iLower+1)-1;
      for (unsigned k=0; k<numEntities; ++k) {
	/* binary search subrange rather than a map.find */
	HypreIntType col = localIds[k];
 	unsigned matIndex;
	binarySearchOwned(lower,upper,col,matIndex);	  
	/* Find the matrix element memory location */
	matIndex = Kokkos::atomic_fetch_add(&mat_counter_owned_(matIndex), (unsigned)1);
	/* write to the lists */
	cols_owned_(matIndex) = col;
	vals_owned_(matIndex) = cur_lhs[k];
      }

      /* fill the right hand side values */
      unsigned rhsIndex = hid-iLower;
      rhsIndex = Kokkos::atomic_fetch_add(&rhs_counter_owned_(rhsIndex), (unsigned)1);
      rhs_vals_owned_(rhsIndex,0) = rhs[i];
    } else {
      if (!map_shared_.exists(hid)) continue;
      /* Find the index of the row */
      unsigned index = map_shared_.value_at(map_shared_.find(hid));
      unsigned lower = mat_row_start_shared_(index);
      unsigned upper = mat_row_start_shared_(index+1)-1;
      
      /* fill the matrix values */
      for (unsigned k=0; k<numEntities; ++k) {
	/* binary search subrange rather than a map.find */
	HypreIntType col = localIds[k];
	unsigned matIndex;
	binarySearchShared(lower,upper,col,matIndex);	  
	/* Find the matrix element memory location */
	matIndex = Kokkos::atomic_fetch_add(&mat_counter_shared_(matIndex), (unsigned)1);
	/* write to the lists */
	cols_shared_(matIndex) = col;
	vals_shared_(matIndex) = cur_lhs[k];
      }

      /* fill the right hand side values */
      unsigned rhsIndex = Kokkos::atomic_fetch_add(&rhs_counter_shared_(index), (unsigned)1);
      rhs_vals_shared_(rhsIndex,0) = rhs[i];
    }
  }
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  if (numDof_==1)
    sum_into_1DoF(numEntities,entities,localIds,rhs,lhs,iLower_,iUpper_);
  else
    sum_into(numEntities,entities,localIds,rhs,lhs,iLower_,iUpper_,numDof_);
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::sum_into_nonNGP(const std::vector<stk::mesh::Entity>& entities,
							    const std::vector<double>& rhs,
							    const std::vector<double>& lhs) {

  /* Pure host implementation */
  const size_t numEntities = entities.size();

  HypreIntType hid0 = entityToLIDHost_[entities[0].local_offset()];
  if (oversetRowsMapHost_.exists(hid0)) {
    if (numDof_==1) {
      if (hid0>=iLower_ && hid0<=iUpper_) {
	for (size_t i=0; i<numEntities; ++i) {
	  h_overset_rows_(overset_mat_counter_) = hid0;
	  h_overset_cols_(overset_mat_counter_) = entityToLIDHost_[entities[i].local_offset()];
	  h_overset_vals_(overset_mat_counter_) = lhs[i];
	  overset_mat_counter_++;
      	}
	h_overset_row_indices_(overset_rhs_counter_) = hid0;
	h_overset_rhs_vals_(overset_rhs_counter_) = rhs[0];
	overset_rhs_counter_++;
      }
    } else {
      throw std::runtime_error("HypreLinSysCoeffApplier::sum_into_nonNGP not yet implemented for numDof>1. Exiting.");
    }
  } else {
    throw std::runtime_error("HypreLinSysCoeffApplier::sum_into_nonNGP not yet implemented for (NON) overset constaint algorithms. Exiting.");
  }
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::applyDirichletBCs(Realm & realm, 
							      stk::mesh::FieldBase * solutionField,
							      stk::mesh::FieldBase * bcValuesField,
							      const stk::mesh::PartVector& parts) {

  resetInternalData();

  /************************************************************/
  /* this is a hack to get dirichlet bcs working consistently */

  /* Step 1: execute the old CPU code */
  auto& meta = realm.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm.get_inactive_selector()));

  const auto& bkts = realm.get_buckets(
    stk::topology::NODE_RANK, sel);

  double diag_value = 1.0;
  std::vector<HypreIntType> tCols(0);
  std::vector<double> tVals(0);
  std::vector<double> trhsVals(0);

  NGPDoubleFieldType ngpSolutionField = realm.ngp_field_manager().get_field<double>(solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField = realm.ngp_field_manager().get_field<double>(bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_host();
  ngpBCValuesField.sync_to_host();

  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (unsigned in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm.hypreGlobalId_, node);

      for (unsigned d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        double bcval = bcValues[in*numDof_ + d] - solution[in*numDof_ + d];
	
	/* fill these temp values */
	tCols.push_back(lid);
	tVals.push_back(diag_value);
	trhsVals.push_back(bcval);
      }
    }
  }

  /* Step 2 : allocate space in which to push the temporaries */
  HypreIntTypeView c = HypreIntTypeView("c",tCols.size());
  HypreIntTypeViewHost ch = Kokkos::create_mirror_view(c);

  DoubleView v = DoubleView("v",tVals.size());
  DoubleViewHost vh = Kokkos::create_mirror_view(v);

  DoubleView rv = DoubleView("rv",trhsVals.size());
  DoubleViewHost rvh = Kokkos::create_mirror_view(rv);

  /* Step 3 : next copy the std::vectors into the host mirrors */
  for (unsigned int i=0; i<tCols.size(); ++i) {
    ch(i) = tCols[i];
    vh(i) = tVals[i];
    rvh(i) = trhsVals[i];
  }
  /* Step 4 : deep copy this to device */
  Kokkos::deep_copy(c,ch);
  Kokkos::deep_copy(v,vh);
  Kokkos::deep_copy(rv,rvh);

  /* Step 5 : append this to the existing data structure */

  /* For device capture */
  auto mat_row_start_owned = mat_row_start_owned_;
  auto mat_counter = mat_counter_owned_;
  auto rhs_counter = rhs_counter_owned_;
  auto cols = cols_owned_;
  auto vals = vals_owned_;
  auto rhs_vals = rhs_vals_owned_;
  auto iLower = iLower_;

  int N = (int) tCols.size();
  Kokkos::parallel_for("dirichlet_bcs", N, KOKKOS_LAMBDA(const unsigned& i) {
      HypreIntType hid = c(i);
      unsigned matIndex = mat_row_start_owned(hid-iLower);
      matIndex = mat_counter(matIndex);

      unsigned rhsIndex = hid-iLower;
      rhsIndex = rhs_counter(rhsIndex);

      cols(matIndex)=c(i);
      vals(matIndex)=v(i);
      rhs_vals(rhsIndex,0) = rv(i);
    });
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::fillRhsVector(const HypreIntType nrows, const HypreIntType N, const int index,
							  const UnsignedView & rhs_row_start,
							  const DoubleView2D & values_in,
							  DoubleView2D & values_out) {


  int team_size, threads_per_row, rows_per_team;

  threads_per_row = nextPowerOfTwo((N + nrows - 1)/nrows, 32);
  threads_per_row/=2;
  threads_per_row = threads_per_row < 1 ? 1 : threads_per_row;

#ifdef KOKKOS_ENABLE_CUDA  
  int blockDim_x = 128;
  rows_per_team = blockDim_x/threads_per_row;
  team_size = rows_per_team;
#else
  threads_per_row=1;
  rows_per_team=1;
  team_size=1;
#endif

  int nteams = (nrows + rows_per_team - 1)/rows_per_team;
  Kokkos::TeamPolicy<>team_exec(nteams,team_size,threads_per_row);
  Kokkos::parallel_for("fillRhsVector",team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {
      auto rowId = team.league_rank()*rows_per_team + team.team_rank();
      if(rowId>=nrows) return;
      
      /* should be fine to read the same value with each vector lane */
      auto begin = rhs_row_start(rowId);
      auto end = rhs_row_start(rowId+1);
      auto rowLen = end - begin;
      double sum = 0.;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, rowLen),
			      [=](int i, double& lsum) {
				lsum += values_in(begin+i, index);
			      }, sum);
      /* its fine to write this duplicated */
      values_out(rowId,index) = sum;
    });
}

#ifdef KOKKOS_ENABLE_CUDA  

void
HypreLinearSystem::HypreLinSysCoeffApplier::sortRhsElementBins(const HypreIntType nrows, const HypreIntType N, const unsigned index,
							       const HypreIntTypeView & row_indices, const UnsignedView & rhs_row_start,
							       HypreIntTypeView & iwork, DoubleView2D & values_in_out) 
{
  if (ensureReproducible_) {
    
    int team_size = 1;
    int threads_per_row = 1;
    int rows_per_team = team_size/threads_per_row;
#ifdef KOKKOS_ENABLE_CUDA  
    int t = (N + nrows - 1)/nrows;
    threads_per_row = nextPowerOfTwo(t, 32);
    threads_per_row = max(threads_per_row, 1);
    int blockDim_x = 128;
    rows_per_team = blockDim_x/threads_per_row;
    team_size = rows_per_team;
#endif

    int nteams = (nrows + rows_per_team - 1)/rows_per_team;
    auto team_exec = Kokkos::TeamPolicy<>(nteams,team_size,threads_per_row);
    Kokkos::parallel_for("fill_rows",team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {
	auto rowId = team.league_rank()*rows_per_team + team.team_rank();
	if(rowId>=nrows) return;
	/* should be fine to read the same value with each vector lane */
	auto begin = rhs_row_start(rowId);
	auto end = rhs_row_start(rowId+1);
	auto row = row_indices(rowId);
	auto rowLen = end - begin;
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, rowLen), [&] (int i) {
	    iwork(begin+i) = row;
	});
      });

    if (useNativeCudaSort_) {
      thrust::device_ptr<HypreIntType> rows_ptr = thrust::device_pointer_cast(iwork.data());
      thrust::device_ptr<HypreIntType> rows_ptr_end = thrust::device_pointer_cast(iwork.data() + N);
      double * ptr = values_in_out.data()+index*N;
      thrust::device_ptr<double> data_ptr = thrust::device_pointer_cast(ptr);
      thrust::device_ptr<double> data_ptr_end = thrust::device_pointer_cast(ptr + N);
      
      typedef thrust::device_vector<HypreIntType>::iterator IntIterator;
      typedef thrust::device_vector<double>::iterator DoubleIterator;
      typedef thrust::tuple<IntIterator, DoubleIterator> IteratorTuple;
      typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
      ZipIterator iter_begin(thrust::make_tuple(rows_ptr, data_ptr));
      ZipIterator iter_end(thrust::make_tuple(rows_ptr_end, data_ptr_end));
      thrust::stable_sort(thrust::device, iter_begin, iter_end, absAscendingOrdering<HypreIntType>());      
    } else {
      throw std::runtime_error("use_native_cuda_sort=no, ensure_reproducible=yes in input file. Pure kokkos version not yet supported. Exiting");
    }
  }
}

#else 

/* This is added to get rid of compiler warnings */
void
HypreLinearSystem::HypreLinSysCoeffApplier::sortRhsElementBins(const HypreIntType /* nrows */, const HypreIntType /* N */, const unsigned /* index */,
							       const HypreIntTypeView & /* row_indices */, const UnsignedView & /* rhs_row_start */,
							       HypreIntTypeView & /* iwork */, DoubleView2D & /* values_in_out */) { } 

#endif



void
HypreLinearSystem::HypreLinSysCoeffApplier::fillCSRMatrix(const HypreIntType nnz, const HypreIntType N,
							  const UnsignedView & mat_elem_start,
							  const HypreIntTypeView & col_indices_in,
							  const DoubleView & values_in,
							  HypreIntTypeView & col_indices_out,
							  DoubleView & values_out) {
  
  int team_size,  threads_per_nnz, nnz_per_team;

  threads_per_nnz = nextPowerOfTwo((N + nnz - 1)/nnz, 32);
  threads_per_nnz/=2;
  threads_per_nnz = threads_per_nnz < 1 ? 1 : threads_per_nnz;

#ifdef KOKKOS_ENABLE_CUDA  
  int blockDim_x = 128;
  nnz_per_team = blockDim_x/threads_per_nnz;
  team_size = nnz_per_team;
#else
  threads_per_nnz=1;
  nnz_per_team=1;
  team_size=1;
#endif
 
  int nteams = (nnz + nnz_per_team - 1)/nnz_per_team;
  auto team_exec = Kokkos::TeamPolicy<>(nteams,team_size,threads_per_nnz);
  
  /* Fill the CSR Matrix */
  Kokkos::parallel_for("fillCSRMatrix",team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {
      auto elemId = team.league_rank()*nnz_per_team + team.team_rank();
      if(elemId>=nnz) return;
  
      /* read in these values by 1 thread only */
      auto begin = mat_elem_start(elemId);
      auto end = mat_elem_start(elemId+1);
      auto nnzLen = end - begin;

      /* Inner reduction to create a matrix element */
      double sum = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, nnzLen),
			      [=] (const HypreIntType& i, double& update) { 
				update += values_in(begin+i); 
			      }, sum);
      
      col_indices_out(elemId) = col_indices_in(begin);
      values_out(elemId) = sum;
    });
}

#ifdef KOKKOS_ENABLE_CUDA  

void
HypreLinearSystem::HypreLinSysCoeffApplier::sortMatrixElementBins(const HypreIntType nrows, const HypreIntType N,
								  const HypreIntType global_num_cols, 
								  const UnsignedView & mat_row_start,
								  const HypreIntTypeView & row_indices,
								  HypreIntTypeView & iwork,
								  HypreIntTypeView & col_indices_in_out,
								  DoubleView & values_in_out)
{
  if (ensureReproducible_) {
    int team_size = 1;
    int threads_per_row = 1;
    int rows_per_team = team_size/threads_per_row;
    int t = (N + nrows - 1)/nrows;
    threads_per_row = nextPowerOfTwo(t, 32);
    threads_per_row = max(threads_per_row, 1);
    int blockDim_x = 128;
    rows_per_team = blockDim_x/threads_per_row;
    team_size = rows_per_team;

    int nteams = (nrows + rows_per_team - 1)/rows_per_team;
    auto team_exec = Kokkos::TeamPolicy<>(nteams,team_size,threads_per_row);
    Kokkos::parallel_for("compute_dense_index",team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {
	auto rowId = team.league_rank()*rows_per_team + team.team_rank();
	if(rowId>=nrows) return;
	/* should be fine to read the same value with each vector lane */
	auto begin = mat_row_start(rowId);
	auto end = mat_row_start(rowId+1);
	auto row = row_indices(rowId);
	auto rowLen = end - begin;
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, rowLen), [&] (int i) {
	    iwork(begin+i) = row * global_num_cols + col_indices_in_out(begin+i);
	});
      });

    if (useNativeCudaSort_) {
      thrust::device_ptr<HypreIntType> keys_ptr = thrust::device_pointer_cast(iwork.data());
      thrust::device_ptr<HypreIntType> keys_ptr_end = thrust::device_pointer_cast(iwork.data() + N);
      thrust::device_ptr<double> data_ptr = thrust::device_pointer_cast(values_in_out.data());
      thrust::device_ptr<double> data_ptr_end = thrust::device_pointer_cast(values_in_out.data() + N);
      thrust::device_ptr<HypreIntType> cols_ptr = thrust::device_pointer_cast(col_indices_in_out.data());
      typedef thrust::device_vector<HypreIntType>::iterator IntIterator;
      typedef thrust::device_vector<double>::iterator DoubleIterator;
      typedef thrust::tuple<IntIterator, DoubleIterator> IteratorTuple;
      typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
      ZipIterator iter_begin(thrust::make_tuple(keys_ptr, data_ptr));
      ZipIterator iter_end(thrust::make_tuple(keys_ptr_end, data_ptr_end));
      thrust::stable_sort_by_key(thrust::device, iter_begin, iter_end, cols_ptr, absAscendingOrdering<HypreIntType>());      
    } else {
      throw std::runtime_error("use_native_cuda_sort=no, ensure_reproducible=yes in input file. Pure kokkos version not yet supported. Exiting");
    }
  }
}

#else

/* This is added to get rid of compiler warnings */

void
HypreLinearSystem::HypreLinSysCoeffApplier::sortMatrixElementBins(const HypreIntType /* nrows */, const HypreIntType /* N */,
								  const HypreIntType /* global_num_cols */, 
								  const UnsignedView & /* mat_row_start */,
								  const HypreIntTypeView & /* row_indices */,
								  HypreIntTypeView & /* iwork */,
								  HypreIntTypeView & /* col_indices_in_out */,
								  DoubleView & /* values_in_out */) { }

#endif


void
HypreLinearSystem::HypreLinSysCoeffApplier::finishAssembly(HYPRE_IJMatrix hypreMat, std::vector<HYPRE_IJVector> hypreRhs) {
  
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  /*******************/
  /* Overset Cleanup */
  /*******************/    

  if (overset_mat_counter_) {
    /* Matrix */
    /* Fill the "Device" views */
    Kokkos::deep_copy(d_overset_rows_, h_overset_rows_);
    Kokkos::deep_copy(d_overset_cols_, h_overset_cols_);
    Kokkos::deep_copy(d_overset_vals_, h_overset_vals_);
    
    unsigned N = d_overset_rows_.extent(0);
    auto orows = d_overset_rows_;
    auto ocols = d_overset_cols_;
    auto ovals = d_overset_vals_;
    auto iLower = iLower_;
    auto mat_counter = mat_counter_owned_;
    auto mat_row_start = mat_row_start_owned_;
    auto mat_elem_keys = mat_elem_cols_owned_;
    auto cols = cols_owned_;
    auto vals = vals_owned_;
    /* write to the matrix */
    Kokkos::parallel_for("fillOversetMatrixRows", N, KOKKOS_LAMBDA(const unsigned& i) {
	HypreIntType row = orows(i);
	HypreIntType col = ocols(i);
	/* binary search subrange rather than a map.find */
	unsigned lower = mat_row_start(row-iLower);
	unsigned upper = mat_row_start(row-iLower+1)-1;
	unsigned matIndex;
	binarySearchOwned(lower,upper,col,matIndex);	  
	matIndex = Kokkos::atomic_fetch_add(&mat_counter(matIndex), (unsigned)1);
	cols(matIndex) = col;
	vals(matIndex) = ovals(i);
      });

    /* RHS */
    /* Fill the "Device" views */
    Kokkos::deep_copy(d_overset_row_indices_, h_overset_row_indices_);
    Kokkos::deep_copy(d_overset_rhs_vals_, h_overset_rhs_vals_);
    
    N = d_overset_rhs_vals_.extent(0);
    auto orow_indices = d_overset_row_indices_;
    auto orvals = d_overset_rhs_vals_;
    auto rhs_counter = rhs_counter_owned_;
    auto rhs_vals = rhs_vals_owned_;
    /* write to the rhs */
    Kokkos::parallel_for("fillOversetRhsVector", N, KOKKOS_LAMBDA(const unsigned& i) {
	HypreIntType row = orow_indices(i);
	unsigned rhsIndex = row-iLower;
	rhsIndex = Kokkos::atomic_fetch_add(&rhs_counter(rhsIndex), (unsigned)1);
	rhs_vals(rhsIndex,0) = orvals(i);
      });
  }

  /**********/
  /* Matrix */
  /**********/

  if (num_nonzeros_owned_) {
    /* Sort ... if chosen */
    sortMatrixElementBins(num_rows_owned_, num_mat_pts_to_assemble_total_owned_, globalNumRows_,
			  mat_row_start_owned_, row_indices_owned_, iwork_,
			  cols_owned_, vals_owned_);
  
    /* Fill the owned matrix */
    fillCSRMatrix(num_nonzeros_owned_, num_mat_pts_to_assemble_total_owned_, mat_elem_start_owned_,
		  cols_owned_, vals_owned_, d_col_indices_owned_, d_values_owned_);

    /* Copy the data to the host */
    Kokkos::deep_copy(h_values_owned_, d_values_owned_);
    Kokkos::deep_copy(h_col_indices_owned_, d_col_indices_owned_);
    
    /* Set the owned part */
    HYPRE_IJMatrixSetValues(hypreMat, num_rows_owned_,
			    h_row_counts_owned_.data(), h_row_indices_owned_.data(),
			    h_col_indices_owned_.data(), h_values_owned_.data());  
  }
  
  if (num_nonzeros_shared_) { 
    /* Sort ... if chosen */
    sortMatrixElementBins(num_rows_shared_, num_mat_pts_to_assemble_total_shared_, globalNumRows_,
			  mat_row_start_shared_, row_indices_shared_, iwork_,
			  cols_shared_, vals_shared_);

    /* Fill the shared matrix */
    fillCSRMatrix(num_nonzeros_shared_, num_mat_pts_to_assemble_total_shared_, mat_elem_start_shared_,
		  cols_shared_, vals_shared_, d_col_indices_shared_, d_values_shared_);

    /* Copy the data to the host */
    Kokkos::deep_copy(h_values_shared_, d_values_shared_);
    Kokkos::deep_copy(h_col_indices_shared_, d_col_indices_shared_);

    /* Add the shared part */
    HYPRE_IJMatrixAddToValues(hypreMat, num_rows_shared_,
			      h_row_counts_shared_.data(), h_row_indices_shared_.data(), 
			      h_col_indices_shared_.data(), h_values_shared_.data());
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the stop time */
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  _assembleMatTime+=msec;
#endif
  _nAssembleMat++;

  /********/
  /* Rhs */
  /********/
  
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  for (unsigned i=0; i<hypreRhs.size(); ++i) {
    if (num_rows_owned_) {
      /* Sort ... if chosen */
      sortRhsElementBins(num_rows_owned_, num_rhs_pts_to_assemble_total_owned_, i, 
			 row_indices_owned_, rhs_row_start_owned_, iwork_, rhs_vals_owned_);
      
      /* Fill the owned rhs */
      fillRhsVector(num_rows_owned_, num_rhs_pts_to_assemble_total_owned_, i, 
		    rhs_row_start_owned_, rhs_vals_owned_, d_rhs_owned_);
    }
    
    if (num_rows_shared_) {
      /* Sort ... if chosen */
      sortRhsElementBins(num_rows_shared_, num_rhs_pts_to_assemble_total_shared_, i, 
			 row_indices_shared_, rhs_row_start_shared_, iwork_, rhs_vals_shared_);

      /* Fill the shared rhs */
      fillRhsVector(num_rows_shared_, num_rhs_pts_to_assemble_total_shared_, i, 
		    rhs_row_start_shared_, rhs_vals_shared_, d_rhs_shared_);
    }
  }

  /* Copy the data to the host */
  Kokkos::deep_copy(h_rhs_owned_, d_rhs_owned_);
  if (num_rows_shared_) Kokkos::deep_copy(h_rhs_shared_, d_rhs_shared_);

  for (unsigned i=0; i<hypreRhs.size(); ++i) {
    if (num_rows_owned_) {
      /* Set the owned part */
      HYPRE_IJVectorSetValues(hypreRhs[i], num_rows_owned_, h_row_indices_owned_.data(), h_rhs_owned_.data()+i*num_rows_owned_);
    }
    
    if (num_rows_shared_) {
      /* Add the shared part */
      HYPRE_IJVectorAddToValues(hypreRhs[i], num_rows_shared_, h_row_indices_shared_.data(), h_rhs_shared_.data()+i*num_rows_shared_);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the stop time */
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  _assembleRhsTime+=msec;
#endif
  _nAssembleRhs++;

  /* Reset after assembly */
  reinitialize_= true;
}


void
HypreLinearSystem::HypreLinSysCoeffApplier::resetInternalData() {

  Kokkos::deep_copy(checkSkippedRows_, 1);
  if (reinitialize_) {
    reinitialize_ = false;

    /* reset overset counters */
    overset_mat_counter_=0;
    overset_rhs_counter_=0;

    /* These seem slightly faster than deep copies */
    Kokkos::deep_copy(mat_counter_owned_, mat_elem_start_owned_);
    Kokkos::deep_copy(rhs_counter_owned_, rhs_row_start_owned_);
    Kokkos::deep_copy(mat_counter_shared_, mat_elem_start_shared_);
    Kokkos::deep_copy(rhs_counter_shared_, rhs_row_start_shared_);
    
    /* These seem slightly faster than deep copies */
    Kokkos::deep_copy(cols_owned_, -1);
    Kokkos::deep_copy(cols_shared_, -1);
    Kokkos::deep_copy(vals_owned_, 0);
    Kokkos::deep_copy(vals_shared_, 0);
    Kokkos::deep_copy(rhs_vals_owned_, 0);
    Kokkos::deep_copy(rhs_vals_shared_, 0);

    /* Apply periodic boundary conditions */
    int N = periodic_bc_rows_owned_.extent(0); 

    /* For device capture */
    auto periodic_bc_rows = periodic_bc_rows_owned_;
    auto mat_row_start_owned = mat_row_start_owned_;
    auto mat_counter = mat_counter_owned_;
    auto rhs_counter = rhs_counter_owned_;
    auto cols = cols_owned_;
    auto vals = vals_owned_;
    auto rhs_vals = rhs_vals_owned_;
    auto nDim = nDim_;
    auto iLower = iLower_;
    Kokkos::parallel_for("periodic_bcs", N, KOKKOS_LAMBDA(const unsigned& i) {
	HypreIntType hid = periodic_bc_rows(i);
	unsigned matIndex = mat_row_start_owned(hid-iLower);
	matIndex = mat_counter(matIndex);
	cols(matIndex) = hid;
	vals(matIndex) = 1.0;

	unsigned rhsIndex = hid-iLower;
	rhsIndex = rhs_counter(rhsIndex);
	for (unsigned d=0; d<nDim; ++d)
	  rhs_vals(rhsIndex,d) = 0.0;
      }); 
  }
}

void HypreLinearSystem::HypreLinSysCoeffApplier::free_device_pointer()
{
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
}

sierra::nalu::CoeffApplier* HypreLinearSystem::HypreLinSysCoeffApplier::device_pointer()
{
  if (devicePointer_ != nullptr) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
  devicePointer_ = sierra::nalu::create_device_expression(*this);
  return devicePointer_;
}

/********************************************************************************************************/
/*                           End of HypreLinSysCoeffApplier implementations                             */
/********************************************************************************************************/


void
HypreLinearSystem::sumInto(
  unsigned,
  const stk::mesh::NgpMesh::ConnectedNodes&,
  const SharedMemView<const double*, DeviceShmem>&,
  const SharedMemView<const double**, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const char*  /* trace_tag */)
{
}


void
HypreLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& /* scratchVals */,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char*  /* trace_tag */)
{
  HypreLinSysCoeffApplier* hcApplier = dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());
  hcApplier->sum_into_nonNGP(entities, rhs, lhs);
}

void
HypreLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned,
  const unsigned)
{
  HypreLinSysCoeffApplier* hcApplier = dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());
  hcApplier->applyDirichletBCs(realm_, solutionField, bcValuesField, parts);
}

HypreIntType
HypreLinearSystem::get_entity_hypre_id(const stk::mesh::Entity& node)
{
  auto& bulk = realm_.bulk_data();
  const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
  const auto mnode = (naluId == bulk.identifier(node)) ? node : bulk.get_entity(stk::topology::NODE_RANK, naluId);
#ifndef NDEBUG
  if (!bulk.is_valid(node))
    throw std::runtime_error("BAD STK NODE");
#endif
  HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);

#ifndef NDEBUG
  HypreIntType chk = ((hid+1) * numDof_ - 1);
  if ((hid < 0) || (chk > maxRowID_)) {
    std::cerr << bulk.parallel_rank() << "\t"
              << hid << "\t" << iLower_ << "\t" << iUpper_ << std::endl;
    throw std::runtime_error("BAD STK to hypre conversion");
  }
#endif

  return hid;
}

int
HypreLinearSystem::solve(stk::mesh::FieldBase* linearSolutionField)
{
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(
    linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFile = eqSysName_ + ".IJM." + writeCounter + ".mat";
    const std::string rhsFile = eqSysName_ + ".IJV." + writeCounter + ".rhs";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());
    HYPRE_IJVectorPrint(rhs_, rhsFile.c_str());
  }

  int iters = 0;
  double finalResidNorm = 0.0;

  // Call solve
  int status = 0;

  status = solver->solve(iters, finalResidNorm, realm_.isFinalOuterIter_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string slnFile = eqSysName_ + ".IJV." + writeCounter + ".sln";
    HYPRE_IJVectorPrint(sln_, slnFile.c_str());
    ++eqSys_->linsysWriteCounter_;
  }

  double norm2 = copy_hypre_to_stk(linearSolutionField);
  sync_field(linearSolutionField);

  linearSolveIterations_ = iters;
  // Hypre provides relative residuals not the final residual, so multiply by
  // the non-linear residual to obtain a final residual that is comparable to
  // what is reported by TpetraLinearSystem. Note that this assumes the initial
  // solution vector is set to 0 at the start of linear iterations.
  linearResidual_ = finalResidNorm * norm2;
  nonLinearResidual_ = realm_.l2Scaling_ * norm2;

  if (eqSys_->firstTimeStepSolve_)
    firstNonLinearResidual_ = nonLinearResidual_;

  scaledNonLinearResidual_ =
    nonLinearResidual_ /
    std::max(std::numeric_limits<double>::epsilon(), firstNonLinearResidual_);

  if (provideOutput_) {
    const int nameOffset = eqSysName_.length() + 8;
    NaluEnv::self().naluOutputP0()
      << std::setw(nameOffset) << std::right << eqSysName_
      << std::setw(32 - nameOffset) << std::right << iters << std::setw(18)
      << std::right << linearResidual_ << std::setw(15) << std::right
      << nonLinearResidual_ << std::setw(14) << std::right
      << scaledNonLinearResidual_ << std::endl;
  }

  eqSys_->firstTimeStepSolve_ = false;
  return status;
}

double
HypreLinearSystem::copy_hypre_to_stk(
  stk::mesh::FieldBase* stkField)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const auto sel = stk::mesh::selectField(*stkField)
    & meta.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, sel);

  double lclnorm2 = 0.0;
  double rhsVal = 0.0;

  for (auto b: bkts) {
    double* field = (double*) stk::mesh::field_data(*stkField, *b);
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = get_entity_hypre_id(node);

      for (size_t d=0; d < numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        int sid = in * numDof_ + d;
        HYPRE_IJVectorGetValues(sln_, 1, &lid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_, 1, &lid, &rhsVal);
        lclnorm2 += rhsVal * rhsVal;
      }
    }
  }
  NGPDoubleFieldType ngpField = realm_.ngp_field_manager().get_field<double>(stkField->mesh_meta_data_ordinal());  
  ngpField.modify_on_host();
  ngpField.sync_to_device();

  double gblnorm2 = 0.0;
  stk::all_reduce_sum(bulk.parallel(), &lclnorm2, &gblnorm2, 1);

  return std::sqrt(gblnorm2);
}

}  // nalu
}  // sierra
