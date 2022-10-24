// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "HypreLinearSystem.h"

#include <iostream>
#include <fstream>

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
  globalMatSharedRowCounts_.clear();
  localMatSharedRowCounts_.clear();
  globalRhsSharedRowCounts_.clear();
  localRhsSharedRowCounts_.clear();
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  sprintf(oname_, "debug_out_%d.txt", rank_);
  output_ = fopen(oname_, "wt");
  fprintf(
    output_, "rank_=%d EqnName=%s : %s %s %d\n", rank_, name_.c_str(), __FILE__,
    __FUNCTION__, __LINE__);
  fclose(output_);
#endif
}

HypreLinearSystem::~HypreLinearSystem()
{
  if (hypreMatrixVectorsCreated_) {
    HYPRE_IJMatrixDestroy(mat_);
    HYPRE_IJVectorDestroy(rhs_);
    HYPRE_IJVectorDestroy(sln_);
    hypreMatrixVectorsCreated_ = false;
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  if (buildBeginLinSysConstTimer_.size() > 0)
    printTimings(buildBeginLinSysConstTimer_, "\nbuildBeginLinSysConst");
  if (buildNodeGraphTimer_.size() > 0)
    printTimings(buildNodeGraphTimer_, "buildNodeGraph");
  if (buildFaceToNodeGraphTimer_.size() > 0)
    printTimings(buildFaceToNodeGraphTimer_, "buildFaceToNodeGraph");
  if (buildEdgeToNodeGraphTimer_.size() > 0)
    printTimings(buildEdgeToNodeGraphTimer_, "buildEdgeToNodeGraph");
  if (buildElemToNodeGraphTimer_.size() > 0)
    printTimings(buildElemToNodeGraphTimer_, "buildElemToNodeGraph");
  if (buildFaceElemToNodeGraphTimer_.size() > 0)
    printTimings(buildFaceElemToNodeGraphTimer_, "buildFaceElemToNodeGraph");
  if (buildOversetNodeGraphTimer_.size() > 0)
    printTimings(buildOversetNodeGraphTimer_, "buildOversetNodeGraph");
  if (buildDirichletNodeGraphTimer_.size() > 0)
    printTimings(buildDirichletNodeGraphTimer_, "buildDirichletNodeGraph");
  if (buildGraphTimer_.size() > 0)
    printTimings(buildGraphTimer_, "buildGraphTimer");
  if (finalizeLinearSystemTimer_.size() > 0)
    printTimings(finalizeLinearSystemTimer_, "finalizeLinearSystemTimer");
  if (hypreMatAssemblyTimer_.size() > 0)
    printTimings(hypreMatAssemblyTimer_, "hypreMatAssemblyTimer");
  if (hypreRhsAssemblyTimer_.size() > 0)
    printTimings(hypreRhsAssemblyTimer_, "hypreRhsAssemblyTimer");
#endif
}

void
HypreLinearSystem::printTimings(std::vector<double>& time, const char* name)
{
  double max = *std::max_element(time.begin(), time.end());
  double min = *std::min_element(time.begin(), time.end());
  double mean = 0;
  for (unsigned i = 0; i < time.size(); ++i)
    mean += time[i];
  mean /= time.size();
  if (rank_ == 0)
    printf(
      "%s %s : samples=%d mean=%1.5g, min=%1.5g, max=%1.5g\n", name,
      name_.c_str(), (int)time.size(), mean, min, max);
}

void
HypreLinearSystem::beginLinearSystemConstruction()
{
  if (inConstruction_)
    return;
  inConstruction_ = true;

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  buildBeginLinSysConstTimer_.resize(0);
  buildNodeGraphTimer_.resize(0);
  buildFaceToNodeGraphTimer_.resize(0);
  buildEdgeToNodeGraphTimer_.resize(0);
  buildElemToNodeGraphTimer_.resize(0);
  buildFaceElemToNodeGraphTimer_.resize(0);
  buildOversetNodeGraphTimer_.resize(0);
  buildDirichletNodeGraphTimer_.resize(0);
  buildGraphTimer_.resize(0);
  finalizeLinearSystemTimer_.resize(0);
  hypreMatAssemblyTimer_.resize(0);
  hypreRhsAssemblyTimer_.resize(0);
#endif

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

#ifndef HYPRE_BIGINT
  // Make sure that HYPRE is compiled with 64-bit integer support when running
  // O(~1B) linear systems.
  uint64_t totalRows =
    (static_cast<uint64_t>(realm_.hypreNumNodes_) *
     static_cast<uint64_t>(numDof_));
  uint64_t maxHypreSize =
    static_cast<uint64_t>(std::numeric_limits<HypreIntType>::max());

  if (totalRows > maxHypreSize)
    throw std::runtime_error(
      "The linear system size is greater than what HYPRE is compiled for. "
      "Please recompile with bigint support and link to Nalu");
#endif

  if (rank_ == 0) {
    iLower_ = realm_.hypreILower_;
  } else {
    iLower_ = realm_.hypreILower_ * numDof_;
  }

  iUpper_ = realm_.hypreIUpper_ * numDof_ - 1;
  // For now set column indices the same as row indices
  jLower_ = iLower_;
  jUpper_ = iUpper_;

  // The total number of rows handled by this MPI rank for Hypre
  numRows_ = (iUpper_ - iLower_ + 1);
  // Total number of global rows in the system
  maxRowID_ = realm_.hypreNumNodes_ * numDof_ - 1;
  globalNumRows_ = maxRowID_ + 1;

#if 0
  if (numDof_ > 0)
    std::cerr << rank_ << "\t" << numDof_ << "\t"
              << realm_.hypreILower_ << "\t" << realm_.hypreIUpper_ << "\t"
                << iLower_ << "\t" << iUpper_ << "\t"
                << numRows_ << "\t" << maxRowID_ << std::endl;
#endif

  rowCountOwned_.resize(numRows_);
  std::fill(rowCountOwned_.begin(), rowCountOwned_.end(), 0);

  columnsOwned_.resize(numRows_);
  for (unsigned i = 0; i < columnsOwned_.size(); ++i)
    columnsOwned_[i].resize(0);

  rowCountShared_.clear();
  columnsShared_.clear();

  int nprocs = realm_.bulk_data().parallel_size();
  globalMatSharedRowCounts_.resize(nprocs);
  localMatSharedRowCounts_.resize(nprocs);
  globalRhsSharedRowCounts_.resize(nprocs);
  localRhsSharedRowCounts_.resize(nprocs);
  offProcNNZToSend_ = 0;
  offProcNNZToRecv_ = 0;
  offProcRhsToSend_ = 0;
  offProcRhsToRecv_ = 0;

  // Allocate memory for the arrays used to track row types and row filled
  // status.
  skippedRows_.clear();
  oversetRows_.clear();

  std::vector<const stk::mesh::FieldBase*> fVec{realm_.hypreGlobalId_};

  if (
    realm_.oversetManager_ != nullptr &&
    realm_.oversetManager_->oversetGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.oversetManager_->oversetGhosting_, fVec);

  if (
    realm_.nonConformalManager_ != nullptr &&
    realm_.nonConformalManager_->nonConformalGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, fVec);

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildBeginLinSysConstTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::fill_owned_shared_data_structures_1DoF(
  const unsigned numNodes, std::vector<HypreIntType>& hids)
{
  for (unsigned i = 0; i < numNodes; ++i) {
    HypreIntType hid = hids[i];
    if (hid >= iLower_ && hid <= iUpper_) {
      HypreIntType lid = hid - iLower_;
      rowCountOwned_[lid]++;
      columnsOwned_[lid].insert(
        columnsOwned_[lid].end(), hids.begin(), hids.end());
    } else {
      if (rowCountShared_.find(hid) != rowCountShared_.end()) {
        rowCountShared_.at(hid)++;
        columnsShared_.at(hid).insert(
          columnsShared_.at(hid).end(), hids.begin(), hids.end());
      } else {
        std::pair<HypreIntType, unsigned> x = std::make_pair(hid, 1);
        rowCountShared_.insert(x);
        std::pair<HypreIntType, std::vector<HypreIntType>> y =
          std::make_pair(hid, hids);
        columnsShared_.insert(y);
      }
    }
  }
}

void
HypreLinearSystem::fill_owned_shared_data_structures(
  const unsigned numNodes,
  std::vector<HypreIntType>& hids,
  std::vector<HypreIntType>& columns)
{
  for (unsigned i = 0; i < numNodes; ++i) {
    /* hid = hid*numDof + d */
    HypreIntType hid = hids[i];
    for (unsigned d = 0; d < numDof_; ++d) {
      HypreIntType HID = hid * numDof_ + d;
      if (HID >= iLower_ && HID <= iUpper_) {
        HypreIntType lid = HID - iLower_;
        rowCountOwned_[lid]++;
        columnsOwned_[lid].insert(
          columnsOwned_[lid].end(), columns.begin(), columns.end());
      } else {
        HypreIntType lid = HID;
        if (rowCountShared_.find(lid) != rowCountShared_.end()) {
          rowCountShared_.at(lid)++;
          columnsShared_.at(lid).insert(
            columnsShared_.at(lid).end(), columns.begin(), columns.end());
        } else {
          std::pair<HypreIntType, unsigned> x = std::make_pair(lid, 1);
          rowCountShared_.insert(x);
          std::pair<HypreIntType, std::vector<HypreIntType>> y =
            std::make_pair(lid, columns);
          columnsShared_.insert(y);
        }
      }
    }
  }
}

void
HypreLinearSystem::fill_hids_columns(
  const unsigned numNodes,
  stk::mesh::Entity const* nodes,
  std::vector<HypreIntType>& hids,
  std::vector<HypreIntType>& columns)
{
  for (unsigned i = 0; i < numNodes; ++i) {
    hids[i] = get_entity_hypre_id(nodes[i]);
    for (unsigned d = 0; d < numDof_; ++d)
      columns[i * numDof_ + d] = hids[i] * numDof_ + d;
  }
}

void
HypreLinearSystem::buildNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned =
    metaData.locally_owned_part() & stk::mesh::selectUnion(parts) &
    !(stk::mesh::selectUnion(realm_.get_slave_part_vector())) &
    !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_owned);

  if (numDof_ == 1) {
    std::vector<HypreIntType> hids(1);
    for (size_t ib = 0; ib < buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *buckets[ib];
      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {
        stk::mesh::Entity node = b[k];
        hids[0] = get_entity_hypre_id(node);

        /* fill owned/shared 1 Dof */
        fill_owned_shared_data_structures_1DoF(1, hids);
      }
    }
  } else {
    std::vector<HypreIntType> hids(1);
    std::vector<HypreIntType> columns(numDof_);

    for (size_t ib = 0; ib < buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *buckets[ib];
      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {
        stk::mesh::Entity node = b[k];
        hids[0] = get_entity_hypre_id(node);
        for (unsigned d = 0; d < numDof_; ++d)
          columns[d] = hids[0] * numDof_ + d;

        /* fill owned/shared for more than 1 Dof */
        fill_owned_shared_data_structures(1, hids, columns);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());
  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(realm_.meta_data().side_rank(), s_owned);

  if (numDof_ == 1) {
    std::vector<HypreIntType> hids(0);

    for (size_t ib = 0; ib < buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);

      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

        stk::mesh::Entity const* nodes = b.begin_nodes(k);

        /* save the hypre ids */
        for (unsigned i = 0; i < numNodes; ++i)
          hids[i] = get_entity_hypre_id(nodes[i]);

        /* fill owned/shared 1 Dof */
        fill_owned_shared_data_structures_1DoF(numNodes, hids);
      }
    }
  } else {
    std::vector<HypreIntType> hids(0);
    std::vector<HypreIntType> columns(0);

    for (size_t ib = 0; ib < buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);
      columns.resize(numNodes * numDof_);

      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

        stk::mesh::Entity const* nodes = b.begin_nodes(k);

        /* save the hids and columns */
        fill_hids_columns(numNodes, nodes, hids, columns);

        /* fill owned/shared for more than 1 Dof */
        fill_owned_shared_data_structures(numNodes, hids, columns);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildFaceToNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());
  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::EDGE_RANK, s_owned);

  if (numDof_ == 1) {
    std::vector<HypreIntType> hids(0);

    for (size_t ib = 0; ib < buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);

      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

        stk::mesh::Entity const* nodes = b.begin_nodes(k);

        /* save the hypre ids */
        for (unsigned i = 0; i < numNodes; ++i)
          hids[i] = get_entity_hypre_id(nodes[i]);

        /* fill owned/shared 1 Dof */
        fill_owned_shared_data_structures_1DoF(numNodes, hids);
      }
    }
  } else {
    std::vector<HypreIntType> hids(0);
    std::vector<HypreIntType> columns(0);

    for (size_t ib = 0; ib < buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);
      columns.resize(numNodes * numDof_);

      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

        stk::mesh::Entity const* nodes = b.begin_nodes(k);

        /* save the hids and columns */
        fill_hids_columns(numNodes, nodes, hids, columns);

        /* fill owned/shared for more than 1 Dof */
        fill_owned_shared_data_structures(numNodes, hids, columns);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildEdgeToNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());
  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::ELEM_RANK, s_owned);

  if (numDof_ == 1) {
    std::vector<HypreIntType> hids(0);

    for (size_t ib = 0; ib < buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);

      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

        stk::mesh::Entity const* nodes = b.begin_nodes(k);

        /* save the hypre ids */
        for (unsigned i = 0; i < numNodes; ++i)
          hids[i] = get_entity_hypre_id(nodes[i]);

        /* fill owned/shared 1 Dof */
        fill_owned_shared_data_structures_1DoF(numNodes, hids);
      }
    }
  } else {
    std::vector<HypreIntType> hids(0);
    std::vector<HypreIntType> columns(0);

    for (size_t ib = 0; ib < buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *buckets[ib];

      auto numNodes = b.topology().num_nodes();
      hids.resize(numNodes);
      columns.resize(numNodes * numDof_);

      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

        stk::mesh::Entity const* nodes = b.begin_nodes(k);

        /* save the hids and columns */
        fill_hids_columns(numNodes, nodes, hids, columns);

        /* fill owned/shared for more than 1 Dof */
        fill_owned_shared_data_structures(numNodes, hids, columns);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildElemToNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::buildFaceElemToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets(metaData.side_rank(), s_owned);

  if (numDof_ == 1) {
    for (size_t ib = 0; ib < face_buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *face_buckets[ib];
      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {
        const stk::mesh::Entity face = b[k];

        // extract the connected element to this exposed face; should be single
        // in size!
        const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
        ThrowAssert(bulkData.num_elements(face) == 1);

        // get connected element and nodal relations
        stk::mesh::Entity element = face_elem_rels[0];
        const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);

        // figure out the global dof ids for each dof on each node
        const unsigned numNodes = (unsigned)bulkData.num_nodes(element);

        if (numNodes) {
          /* save the hypre ids */
          std::vector<HypreIntType> hids(numNodes);
          for (unsigned i = 0; i < numNodes; ++i)
            hids[i] = get_entity_hypre_id(elem_nodes[i]);

          /* fill owned/shared 1 Dof */
          fill_owned_shared_data_structures_1DoF(numNodes, hids);
        }
      }
    }
  } else {
    for (size_t ib = 0; ib < face_buckets.size(); ++ib) {
      const stk::mesh::Bucket& b = *face_buckets[ib];
      for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {
        const stk::mesh::Entity face = b[k];

        // extract the connected element to this exposed face; should be single
        // in size!
        const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
        ThrowAssert(bulkData.num_elements(face) == 1);

        // get connected element and nodal relations
        stk::mesh::Entity element = face_elem_rels[0];
        const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);

        // figure out the global dof ids for each dof on each node
        const unsigned numNodes = (unsigned)bulkData.num_nodes(element);

        if (numNodes) {
          std::vector<HypreIntType> columns(numNodes * numDof_);
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
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildFaceElemToNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::buildReducedElemToNodeGraph(const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildNonConformalNodeGraph(const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildOversetNodeGraph(const stk::mesh::PartVector&)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  // extract the rank
  const int theRank = NaluEnv::self().parallel_rank();

  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  beginLinearSystemConstruction();

  std::vector<stk::mesh::Entity> entities;
  std::vector<HypreIntType> hids;

  // Mark all the fringe nodes as skipped so that sumInto doesn't add into these
  // rows during assembly process
  for (const OversetInfo* oversetInfo :
       realm_.oversetManager_->oversetInfoVec_) {

    // extract element mesh object and orphan node
    stk::mesh::Entity owningElement = oversetInfo->owningElement_;
    stk::mesh::Entity orphanNode = oversetInfo->orphanNode_;

    // extract the owning rank for this node
    const int nodeRank = bulkData.parallel_owner_rank(orphanNode);

    const bool nodeIsLocallyOwned = (theRank == nodeRank);
    if (!nodeIsLocallyOwned)
      continue;

    // relations
    stk::mesh::Entity const* elem_nodes = bulkData.begin_nodes(owningElement);
    const size_t numNodes = bulkData.num_nodes(owningElement);
    const size_t numEntities = numNodes + 1;
    entities.resize(numEntities);
    hids.resize(numEntities);

    entities[0] = orphanNode;
    hids[0] = get_entity_hypre_id(entities[0]);
    for (size_t n = 0; n < numNodes; ++n) {
      entities[n + 1] = elem_nodes[n];
      hids[n + 1] = get_entity_hypre_id(entities[n + 1]);
    }

    /* save the hypre ids */
    for (unsigned d = 0; d < numDof_; ++d) {
      HypreIntType hid = hids[0] * numDof_ + d;
      skippedRows_.insert(hid);
      oversetRows_.insert(hid);
      if (hid >= iLower_ && hid <= iUpper_) {
        HypreIntType lid = hid - iLower_;
        rowCountOwned_[lid]++;
        columnsOwned_[lid].resize(0);
        columnsOwned_[lid].insert(
          columnsOwned_[lid].end(), hids.begin(), hids.end());
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildOversetNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::buildDirichletNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  // Grab nodes regardless of whether they are owned or shared
  const stk::mesh::Selector sel = stk::mesh::selectUnion(parts);
  const auto& bkts = realm_.get_buckets(stk::topology::NODE_RANK, sel);

  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
      for (unsigned d = 0; d < numDof_; ++d) {
        HypreIntType lid = hid * numDof_ + d;
        skippedRows_.insert(lid);
        if (lid >= iLower_ && lid <= iUpper_) {
          rowCountOwned_[lid - iLower_]++;
          columnsOwned_[lid - iLower_].push_back(lid);
        }
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildDirichletNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  for (const auto& node : nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);
    for (unsigned d = 0; d < numDof_; ++d) {
      HypreIntType lid = hid * numDof_ + d;
      skippedRows_.insert(lid);
      if (lid >= iLower_ && lid <= iUpper_) {
        rowCountOwned_[lid - iLower_]++;
        columnsOwned_[lid - iLower_].push_back(lid);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildDirichletNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::NgpMesh::ConnectedNodes nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  for (unsigned i = 0; i < nodeList.size(); ++i) {
    HypreIntType hid = get_entity_hypre_id(nodeList[i]);
    for (unsigned d = 0; d < numDof_; ++d) {
      HypreIntType lid = hid * numDof_ + d;
      skippedRows_.insert(lid);
      if (lid >= iLower_ && lid <= iUpper_) {
        rowCountOwned_[lid - iLower_]++;
        columnsOwned_[lid - iLower_].push_back(lid);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildDirichletNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::finalizeLinearSystem()
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  ThrowRequire(inConstruction_);
  inConstruction_ = false;

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  size_t used1 = 0, free1 = 0;
  stk::get_gpu_memory_info(used1, free1);
#endif

  /**********************************************************************************/
  /* Build the coeff applier ... host data structure for building the linear
   * system */
  if (!hostCoeffApplier)
    hostCoeffApplier.reset(
      new HypreLinSysCoeffApplier(numDof_, 1, iLower_, iUpper_));

  /* make the periodic node maps */
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  hcApplier->ngpMesh_ = realm_.ngp_mesh();
  hcApplier->ngpHypreGlobalId_ =
    realm_.ngp_field_manager().get_field<HypreIntType>(
      realm_.hypreGlobalId_->mesh_meta_data_ordinal());

  /* create these mappings */
  buildCoeffApplierPeriodicNodeToHIDMapping();

  /* fill the various device data structures need in device coeff applier */
  buildCoeffApplierDeviceDataStructures();

  /* compute the exact row sizes by reducing row counts at row indices across
   * all ranks */
  computeRowSizes();

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  size_t used2 = 0, free2 = 0;
  stk::get_gpu_memory_info(used2, free2);
  size_t total = used2 + free2;
  output_ = fopen(oname_, "at");
  fprintf(
    output_,
    "rank_=%d EqnName=%s : %s %s %d : usedMem before=%1.5g, usedMem "
    "after=%1.5g, total=%1.5g\n",
    rank_, name_.c_str(), __FILE__, __FUNCTION__, __LINE__, used1 / 1.e9,
    used2 / 1.e9, total / 1.e9);
  fclose(output_);
#endif

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  finalizeLinearSystemTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::computeRowSizes()
{
  MPI_Comm comm = realm_.bulk_data().parallel();
  int nprocs = realm_.bulk_data().parallel_size();
  int iproc = realm_.bulk_data().parallel_rank();

  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  std::fill(
    globalMatSharedRowCounts_.begin(), globalMatSharedRowCounts_.end(), 0);
  std::fill(
    localMatSharedRowCounts_.begin(), localMatSharedRowCounts_.end(), 0);

  std::fill(
    globalRhsSharedRowCounts_.begin(), globalRhsSharedRowCounts_.end(), 0);
  std::fill(
    localRhsSharedRowCounts_.begin(), localRhsSharedRowCounts_.end(), 0);

  /* set the send NNZ per row on this rank */
  for (unsigned i = 0; i < row_indices_shared_host_.extent(0); ++i) {
    HypreIntType shared_row = row_indices_shared_host_(i);
    HypreIntType shared_count = row_counts_shared_host_(i);
    for (int j = 0; j < nprocs; ++j) {
      HypreIntType lower = (HypreIntType)realm_.hypreOffsets_[j] * numDof_;
      HypreIntType upper = (HypreIntType)realm_.hypreOffsets_[j + 1] * numDof_;
      if (shared_row >= lower && shared_row < upper) {
        localMatSharedRowCounts_[j] += shared_count;
        localRhsSharedRowCounts_[j] += 1;
      }
    }
  }

  /* reduce the shared NNZ per row across all ranks */
  MPI_Allreduce(
    localMatSharedRowCounts_.data(), globalMatSharedRowCounts_.data(), nprocs,
    HYPRE_MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(
    localRhsSharedRowCounts_.data(), globalRhsSharedRowCounts_.data(), nprocs,
    HYPRE_MPI_INT, MPI_SUM, comm);

  /* compute the receive NNZ per row from all other ranks */
  offProcNNZToRecv_ = globalMatSharedRowCounts_[iproc];
  offProcRhsToRecv_ = globalRhsSharedRowCounts_[iproc];

  HypreIntType totalMatElmts = hcApplier->num_nonzeros_owned_;
  HypreIntType totalRhsElmts = hcApplier->num_rows_owned_;

  HypreDirectSolver* solver =
    reinterpret_cast<HypreDirectSolver*>(linearSolver_);
  HypreLinearSolverConfig* config =
    reinterpret_cast<HypreLinearSolverConfig*>(solver->getConfig());

  /* set the key hypre parameters */
  offProcNNZToSend_ = hcApplier->num_nonzeros_shared_;
  offProcRhsToSend_ = hcApplier->num_rows_shared_;

  if (config->simpleHypreMatrixAssemble()) {
    totalMatElmts += std::max(offProcNNZToSend_, offProcNNZToRecv_);
    totalRhsElmts += std::max(offProcRhsToSend_, offProcRhsToRecv_);
  } else {
    totalMatElmts += hcApplier->num_nonzeros_shared_;
    totalRhsElmts += hcApplier->num_rows_shared_;
  }

  /* Make big monolithic data structures for values and columns */
  hcApplier->values_dev_ = DoubleView("values_dev", totalMatElmts);
  hcApplier->rhs_dev_ =
    DoubleView2D("values_dev", totalRhsElmts, hcApplier->nDim_);

  cols_host_ = HypreIntTypeViewHost("cols_host", totalMatElmts);
  for (HypreIntType i = 0; i < hcApplier->num_nonzeros_owned_; ++i)
    cols_host_(i) = cols_owned_host_(i);
  for (HypreIntType i = 0; i < hcApplier->num_nonzeros_shared_; ++i)
    cols_host_(i + hcApplier->num_nonzeros_owned_) = cols_shared_host_(i);

  hcApplier->cols_dev_ = HypreIntTypeView("cols_dev", totalMatElmts);
  Kokkos::deep_copy(hcApplier->cols_dev_, cols_host_);

  /* Creat the rows for the mat (rows_host_ and rows_dev_) and rhs
   * (rhs_rows_host_ and rhs_rows_dev_)*/
  rows_host_ = HypreIntTypeViewHost("rows_host", totalMatElmts);
  rhs_rows_host_ =
    HypreIntTypeView2DHost("rhs_rows_host", totalRhsElmts, hcApplier->nDim_);
  HypreIntType k = 0;
  for (HypreIntType i = 0; i < hcApplier->num_rows_owned_; ++i) {
    HypreIntType row = row_indices_owned_host_(i);
    for (unsigned j = 0; j < hcApplier->nDim_; ++j)
      rhs_rows_host_(i, j) = row;
    for (HypreIntType j = 0; j < row_counts_owned_host_(i); ++j) {
      rows_host_(k) = row;
      ++k;
    }
  }
  k = hcApplier->num_nonzeros_owned_;
  for (HypreIntType i = 0; i < hcApplier->num_rows_shared_; ++i) {
    HypreIntType row = row_indices_shared_host_(i);
    for (unsigned j = 0; j < hcApplier->nDim_; ++j)
      rhs_rows_host_(i + hcApplier->num_rows_owned_, j) = row;
    for (HypreIntType j = 0; j < row_counts_shared_host_(i); ++j) {
      rows_host_(k) = row;
      ++k;
    }
  }
  rows_dev_ = HypreIntTypeView("rows_dev", totalMatElmts);
  Kokkos::deep_copy(rows_dev_, rows_host_);

  rhs_rows_dev_ =
    HypreIntTypeView2D("rhs_rows_dev", totalRhsElmts, hcApplier->nDim_);
  Kokkos::deep_copy(rhs_rows_dev_, rhs_rows_host_);
}

/**************************************************************/
/* Fill/Allocate Matrix/Rhs element data structures ... owned */
/**************************************************************/
void
HypreLinearSystem::buildCoeffApplierDeviceOwnedDataStructures()
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  std::vector<HypreIntType> matElemColsOwned(0);
  std::vector<HypreIntType> matColumnsPerRowCountOwned(0);
  std::vector<HypreIntType> periodicBCsOwned(0);
  std::vector<HypreIntType> validRowsOwned(0);
  hcApplier->num_mat_overset_pts_owned_ = 0;
  hcApplier->num_rhs_overset_pts_owned_ = 0;
  for (HypreIntType j = iLower_; j <= iUpper_; ++j) {
    HypreIntType jShift = j - iLower_;
    HypreIntType matRowColumnCount = 1;
    std::vector<HypreIntType> columns = columnsOwned_[jShift];

    if (oversetRows_.find(j) != oversetRows_.end()) {
      /* Overset */
      std::sort(columns.begin(), columns.end());

      /* scan the sorted list */
      HypreIntType col = columns[0];
      for (unsigned i = 1; i < columns.size(); ++i) {
        if (columns[i] != col) {
          matElemColsOwned.push_back(col);
          col = columns[i];
          matRowColumnCount++;
        }
      }
      matElemColsOwned.push_back(col);
      hcApplier->num_mat_overset_pts_owned_ += matRowColumnCount;
      hcApplier->num_rhs_overset_pts_owned_++;

    } else if (skippedRows_.find(j) != skippedRows_.end()) {
      /* Deal with dirichlet BCs */
      matElemColsOwned.push_back(j);
    } else {
      /* check periodic BC first */
      if (columns.size() == 0) {
        matElemColsOwned.push_back(j);
        periodicBCsOwned.push_back(j);
      } else if (columns.size() == 1) {
        matElemColsOwned.push_back(j);
      } else {
        /* Normal Row */
        std::sort(columns.begin(), columns.end());
        /* scan the sorted list */
        HypreIntType col = columns[0];
        for (unsigned i = 1; i < columns.size(); ++i) {
          if (columns[i] != col) {
            matElemColsOwned.push_back(col);
            col = columns[i];
            matRowColumnCount++;
          }
        }
        matElemColsOwned.push_back(col);
      }
    }
    validRowsOwned.push_back(j);
    matColumnsPerRowCountOwned.push_back(matRowColumnCount);
  }

  /* Set key meta data */
  hcApplier->num_rows_owned_ = validRowsOwned.size();
  hcApplier->num_nonzeros_owned_ = matElemColsOwned.size();

  hcApplier->mat_row_start_owned_ =
    UnsignedView("mat_row_start_owned", hcApplier->num_rows_owned_ + 1);

  cols_owned_host_ =
    HypreIntTypeViewHost("cols_owned_host", hcApplier->num_nonzeros_owned_);
  for (auto i = 0; i < hcApplier->num_nonzeros_owned_; ++i)
    cols_owned_host_(i) = matElemColsOwned[i];

  /***********************************/
  /* Other data structures ... owned */
  /***********************************/
  row_indices_owned_host_ =
    HypreIntTypeViewHost("row_indices_owned_host", hcApplier->num_rows_owned_);
  row_counts_owned_host_ =
    HypreIntTypeViewHost("row_counts_owned_host", hcApplier->num_rows_owned_);
  UnsignedViewHost mat_row_start_owned_host =
    Kokkos::create_mirror_view(hcApplier->mat_row_start_owned_);

  /* create the maps */
  mat_row_start_owned_host(0) = 0;
  for (auto i = 0; i < hcApplier->num_rows_owned_; ++i) {
    row_indices_owned_host_(i) = validRowsOwned[i];
    row_counts_owned_host_(i) = matColumnsPerRowCountOwned[i];
    mat_row_start_owned_host(i + 1) =
      mat_row_start_owned_host(i) + matColumnsPerRowCountOwned[i];
  }
  Kokkos::deep_copy(hcApplier->mat_row_start_owned_, mat_row_start_owned_host);

  /* Handle periodic boundary conditions */
  hcApplier->periodic_bc_rows_owned_ =
    HypreIntTypeView("periodic_bc_rows", periodicBCsOwned.size());
  HypreIntTypeViewHost periodic_bc_rows_owned_host =
    Kokkos::create_mirror_view(hcApplier->periodic_bc_rows_owned_);
  for (unsigned i = 0; i < periodicBCsOwned.size(); ++i)
    periodic_bc_rows_owned_host(i) = periodicBCsOwned[i];
  Kokkos::deep_copy(
    hcApplier->periodic_bc_rows_owned_, periodic_bc_rows_owned_host);

  /* Work space for overset. These are used to accumulate data from legacy,
   * non-NGP sumInto calls */
  /* these are used for coupled overset solves */
  if (
    hcApplier->num_rhs_overset_pts_owned_ &&
    hcApplier->num_mat_overset_pts_owned_) {
    hcApplier->d_overset_row_indices_ = HypreIntTypeView(
      "overset_row_indices", hcApplier->num_rhs_overset_pts_owned_);
    hcApplier->h_overset_row_indices_ =
      Kokkos::create_mirror_view(hcApplier->d_overset_row_indices_);

    hcApplier->d_overset_rhs_vals_ =
      DoubleView("overset_rhs_vals", hcApplier->num_rhs_overset_pts_owned_);
    hcApplier->h_overset_rhs_vals_ =
      Kokkos::create_mirror_view(hcApplier->d_overset_rhs_vals_);

    hcApplier->d_overset_rows_ =
      HypreIntTypeView("overset_rows", hcApplier->num_mat_overset_pts_owned_);
    hcApplier->h_overset_rows_ =
      Kokkos::create_mirror_view(hcApplier->d_overset_rows_);

    hcApplier->d_overset_cols_ =
      HypreIntTypeView("overset_cols", hcApplier->num_mat_overset_pts_owned_);
    hcApplier->h_overset_cols_ =
      Kokkos::create_mirror_view(hcApplier->d_overset_cols_);

    hcApplier->d_overset_vals_ =
      DoubleView("overset_vals", hcApplier->num_mat_overset_pts_owned_);
    hcApplier->h_overset_vals_ =
      Kokkos::create_mirror_view(hcApplier->d_overset_vals_);
  }

  hcApplier->overset_mat_counter_ = 0;
  hcApplier->overset_rhs_counter_ = 0;
}

/***************************************************************/
/* Fill/Allocate Matrix/Rhs element data structures ... shared */
/***************************************************************/
void
HypreLinearSystem::buildCoeffApplierDeviceSharedDataStructures()
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  std::vector<HypreIntType> matElemColsShared(0);
  std::vector<HypreIntType> matColumnsPerRowCountShared(0);
  std::vector<HypreIntType> validRowsShared(0);

  for (auto it = rowCountShared_.begin(); it != rowCountShared_.end(); it++) {
    HypreIntType matRowColumnCount = 1;
    HypreIntType hid = it->first;
    std::vector<HypreIntType> columns = columnsShared_[hid];

    if (skippedRows_.find(hid) != skippedRows_.end()) {
      continue;
    } else if (columns.size() == 1) {
      matElemColsShared.push_back(hid);
    } else if (columns.size() > 1) {
      /* Normal Row */
      std::sort(columns.begin(), columns.end());
      HypreIntType col = columns[0];
      for (unsigned i = 1; i < columns.size(); ++i) {
        if (columns[i] != col) {
          matElemColsShared.push_back(col);
          col = columns[i];
          matRowColumnCount++;
        }
      }
      matElemColsShared.push_back(col);
    } else
      continue;

    validRowsShared.push_back(hid);
    matColumnsPerRowCountShared.push_back(matRowColumnCount);
  }

  /* Set key meta data */
  hcApplier->num_rows_shared_ = validRowsShared.size();
  hcApplier->num_nonzeros_shared_ = matElemColsShared.size();

  hcApplier->rhs_row_start_shared_ =
    UnsignedView("rhs_row_start_shared", hcApplier->num_rows_shared_ + 1);
  hcApplier->mat_row_start_shared_ =
    UnsignedView("mat_row_start_shared", hcApplier->num_rows_shared_ + 1);

  /*********************************************/
  /* Matrix element data structures ... shared */
  /*********************************************/
  cols_shared_host_ =
    HypreIntTypeViewHost("cols_shared_host_", hcApplier->num_nonzeros_shared_);
  for (auto i = 0; i < hcApplier->num_nonzeros_shared_; ++i)
    cols_shared_host_(i) = matElemColsShared[i];

  /************************************/
  /* Other data structures ... shared */
  /************************************/
  row_indices_shared_host_ = HypreIntTypeViewHost(
    "row_indices_shared_host", hcApplier->num_rows_shared_);
  row_counts_shared_host_ =
    HypreIntTypeViewHost("row_counts_shared_host", hcApplier->num_rows_shared_);
  UnsignedViewHost rhs_row_start_shared_host =
    Kokkos::create_mirror_view(hcApplier->rhs_row_start_shared_);
  UnsignedViewHost mat_row_start_shared_host =
    Kokkos::create_mirror_view(hcApplier->mat_row_start_shared_);

  /* create the maps */
  rhs_row_start_shared_host(0) = 0;
  mat_row_start_shared_host(0) = 0;
  for (auto i = 0; i < hcApplier->num_rows_shared_; ++i) {
    row_indices_shared_host_(i) = validRowsShared[i];
    row_counts_shared_host_(i) = matColumnsPerRowCountShared[i];
    rhs_row_start_shared_host(i + 1) = rhs_row_start_shared_host(i) + 1;
    mat_row_start_shared_host(i + 1) =
      mat_row_start_shared_host(i) + matColumnsPerRowCountShared[i];
  }

  Kokkos::deep_copy(
    hcApplier->rhs_row_start_shared_, rhs_row_start_shared_host);
  Kokkos::deep_copy(
    hcApplier->mat_row_start_shared_, mat_row_start_shared_host);

  /* Create the map on device */
  HypreIntTypeView row_indices_shared =
    HypreIntTypeView("row_indices_shared", hcApplier->num_rows_shared_);
  Kokkos::deep_copy(row_indices_shared, row_indices_shared_host_);

  hcApplier->map_shared_ = MemoryMap(hcApplier->num_rows_shared_);
  auto ms = hcApplier->map_shared_;
  auto ris = row_indices_shared;
  Kokkos::parallel_for(
    "init_shared_map", hcApplier->num_rows_shared_,
    KOKKOS_LAMBDA(const HypreIntType i) { ms.insert(ris(i), i); });
}

/*************************************************************/
/* Fill/Allocate Matrix/Rhs element data structures          */
/*************************************************************/
void
HypreLinearSystem::buildCoeffApplierDeviceDataStructures()
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  /* Linear System data structures ... owned */
  buildCoeffApplierDeviceOwnedDataStructures();

  /* Linear System data structures ... owned */
  buildCoeffApplierDeviceSharedDataStructures();

  /* skipped rows data structure */
  hcApplier->skippedRowsMap_ = HypreIntTypeUnorderedMap(skippedRows_.size());
  hcApplier->skippedRowsMapHost_ =
    HypreIntTypeUnorderedMapHost(skippedRows_.size());
  for (auto t : skippedRows_)
    hcApplier->skippedRowsMapHost_.insert(t);
  Kokkos::deep_copy(hcApplier->skippedRowsMap_, hcApplier->skippedRowsMapHost_);

  /* overset rows data structure */
  hcApplier->oversetRowsMap_ = HypreIntTypeUnorderedMap(oversetRows_.size());
  hcApplier->oversetRowsMapHost_ =
    HypreIntTypeUnorderedMapHost(oversetRows_.size());
  for (auto t : oversetRows_)
    hcApplier->oversetRowsMapHost_.insert(t);
  Kokkos::deep_copy(hcApplier->oversetRowsMap_, hcApplier->oversetRowsMapHost_);

  /* check skipped rows */
  hcApplier->checkSkippedRows_ = HypreIntTypeViewScalar("checkSkippedRows_");
  Kokkos::deep_copy(hcApplier->checkSkippedRows_, 1);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  auto num_rows = hcApplier->num_rows_owned_ + hcApplier->num_rows_shared_;
  auto num_nonzeros =
    hcApplier->num_nonzeros_owned_ + hcApplier->num_nonzeros_shared_;

  // passed in as an arugment to this class
  size_t totalMemDevice =
    (hcApplier->mat_row_start_owned_.extent(0) +
     hcApplier->mat_row_start_shared_.extent(0) +
     hcApplier->rhs_row_start_shared_.extent(0)) *
      sizeof(unsigned) +
    sizeof(double) * (num_nonzeros + hcApplier->nDim_ * num_rows) +
    (hcApplier->cols_dev_.extent(0)) * sizeof(HypreIntType);
  totalMemDevice +=
    hcApplier->periodic_bc_rows_owned_.extent(0) * sizeof(HypreIntType);
  totalMemDevice +=
    hcApplier->skippedRowsMap_.size() * 2 * sizeof(HypreIntType);
  totalMemDevice +=
    hcApplier->oversetRowsMap_.size() * 2 * sizeof(HypreIntType);
  totalMemDevice +=
    hcApplier->map_shared_.size() * (sizeof(HypreIntType) + sizeof(unsigned));
  totalMemDevice += hcApplier->periodic_node_to_hypre_id_.size() *
                    (sizeof(HypreIntType) + sizeof(unsigned));
  totalMemDevice +=
    hcApplier->d_overset_row_indices_.extent(0) * sizeof(HypreIntType);
  totalMemDevice += hcApplier->d_overset_rhs_vals_.extent(0) * sizeof(double);
  totalMemDevice += hcApplier->d_overset_rows_.extent(0) * sizeof(HypreIntType);
  totalMemDevice += hcApplier->d_overset_cols_.extent(0) * sizeof(HypreIntType);
  totalMemDevice += hcApplier->d_overset_vals_.extent(0) * sizeof(double);

  // size_t used = 0, free = 0;
  // stk::get_gpu_memory_info(used, free);
  output_ = fopen(oname_, "at");
  fprintf(
    output_, "rank_=%d : %s %s %d : totalMemDevice=%1.5g\n", rank_, __FILE__,
    __FUNCTION__, __LINE__, totalMemDevice / 1.e9);
  fclose(output_);
#endif

  /* clear this data so that the next time a coeffApplier is built, these get
   * rebuilt from scratch */
  rowCountOwned_.resize(numRows_);
  std::fill(rowCountOwned_.begin(), rowCountOwned_.end(), 0);

  columnsOwned_.resize(numRows_);
  for (unsigned i = 0; i < columnsOwned_.size(); ++i)
    columnsOwned_[i].resize(0);

  rowCountShared_.clear();
  columnsShared_.clear();

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildGraphTimer_.push_back(msec);
#endif
}

void
HypreLinearSystem::buildCoeffApplierPeriodicNodeToHIDMapping()
{
  const auto& meta = realm_.meta_data();
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  stk::mesh::Selector selector = meta.universal_part() &
                                 stk::mesh::selectField(*realm_.naluGlobalId_) &
                                 !(realm_.get_inactive_selector());

  std::vector<HypreIntType> periodic_node(0);
  std::vector<HypreIntType> periodic_node_hypre_id(0);

  const stk::mesh::BucketVector& nodeBuckets =
    realm_.get_buckets(stk::topology::NODE_RANK, selector);
  for (const stk::mesh::Bucket* bptr : nodeBuckets) {
    const stk::mesh::Bucket& b = *bptr;
    for (size_t i = 0; i < b.size(); ++i) {
      stk::mesh::Entity node = b[i];
      const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
      const auto mnode = (naluId == bulk.identifier(node))
                           ? node
                           : bulk.get_entity(stk::topology::NODE_RANK, naluId);
      if (!bulk.is_valid(mnode)) {
        continue;
      }
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);
      if (naluId != bulk.identifier(node)) {
        periodic_node.push_back(node.local_offset());
        periodic_node_hypre_id.push_back(hid);
      }
    }
  }

  /* make the periodic node maps */
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  hcApplier->periodic_node_to_hypre_id_ = PeriodicNodeMap(periodic_node.size());
  PeriodicNodeMapHost periodic_node_to_hypre_id_host =
    PeriodicNodeMapHost(periodic_node.size());
  for (unsigned i = 0; i < periodic_node.size(); ++i) {
    periodic_node_to_hypre_id_host.insert(
      periodic_node[i], periodic_node_hypre_id[i]);
  }
  Kokkos::deep_copy(
    hcApplier->periodic_node_to_hypre_id_, periodic_node_to_hypre_id_host);
}

void
HypreLinearSystem::resetCoeffApplierData()
{
  /* reset the internal data */
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  Kokkos::deep_copy(hcApplier->checkSkippedRows_, 1);

  if (hcApplier->reinitialize_) {
    hcApplier->reinitialize_ = false;

    /* reset overset counters */
    hcApplier->overset_mat_counter_ = 0;
    hcApplier->overset_rhs_counter_ = 0;

    Kokkos::deep_copy(hcApplier->cols_dev_, cols_host_);
    Kokkos::deep_copy(rows_dev_, rows_host_);
    Kokkos::deep_copy(rhs_rows_dev_, rhs_rows_host_);
    Kokkos::deep_copy(hcApplier->values_dev_, 0);
    Kokkos::deep_copy(hcApplier->rhs_dev_, 0);

    // set the random access memory textures
    hcApplier->mat_row_start_owned_ra_ = hcApplier->mat_row_start_owned_;
    hcApplier->mat_row_start_shared_ra_ = hcApplier->mat_row_start_shared_;
    hcApplier->cols_dev_ra_ = hcApplier->cols_dev_;

    auto N = hcApplier->periodic_bc_rows_owned_.extent(0);
    auto periodic_bc_rows = hcApplier->periodic_bc_rows_owned_;
    auto mat_row_start_owned = hcApplier->mat_row_start_owned_ra_;
    auto vals = hcApplier->values_dev_;
    auto rhs_vals = hcApplier->rhs_dev_;
    auto nDim = hcApplier->nDim_;

    auto iLower = iLower_;
    Kokkos::parallel_for(
      "HypreLinearSystem::resetCoeffApplierData::periodic_bcs", N,
      KOKKOS_LAMBDA(const unsigned& i) {
        HypreIntType hid = periodic_bc_rows(i);
        unsigned matIndex = mat_row_start_owned(hid - iLower);
        vals(matIndex) = 1.0;
        for (unsigned d = 0; d < nDim; ++d)
          rhs_vals(hid - iLower, d) = 0.0;
      });
  }
}

void
HypreLinearSystem::finishCoupledOversetAssembly()
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  /*******************/
  /* Overset Cleanup */
  /*******************/

  if (hcApplier->overset_mat_counter_) {
    /* Matrix */
    /* Fill the "Device" views */
    Kokkos::deep_copy(hcApplier->d_overset_rows_, hcApplier->h_overset_rows_);
    Kokkos::deep_copy(hcApplier->d_overset_cols_, hcApplier->h_overset_cols_);
    Kokkos::deep_copy(hcApplier->d_overset_vals_, hcApplier->h_overset_vals_);

    unsigned N = hcApplier->d_overset_rows_.extent(0);
    auto orows = hcApplier->d_overset_rows_;
    auto ocols = hcApplier->d_overset_cols_;
    auto ovals = hcApplier->d_overset_vals_;
    auto iLower = iLower_;
    auto mat_row_start = hcApplier->mat_row_start_owned_ra_;
    auto cols_dev = hcApplier->cols_dev_ra_;
    auto vals = hcApplier->values_dev_;
    /* write to the matrix */
    Kokkos::parallel_for(
      "fillOversetMatrixRows", N, KOKKOS_LAMBDA(const unsigned& i) {
        HypreIntType row = orows(i);
        HypreIntType col = ocols(i);
        /* binary search subrange rather than a map.find */
        unsigned lower = mat_row_start(row - iLower);
        unsigned upper = mat_row_start(row - iLower + 1) - 1;
        unsigned matIndex = lower;
        for (matIndex = lower; matIndex <= upper; ++matIndex) {
          if (cols_dev(matIndex) == col)
            break;
        }
        vals(matIndex) = ovals(i);
      });

    /* RHS */
    /* Fill the "Device" views */
    Kokkos::deep_copy(
      hcApplier->d_overset_row_indices_, hcApplier->h_overset_row_indices_);
    Kokkos::deep_copy(
      hcApplier->d_overset_rhs_vals_, hcApplier->h_overset_rhs_vals_);

    N = hcApplier->d_overset_rhs_vals_.extent(0);
    auto orow_indices = hcApplier->d_overset_row_indices_;
    auto orvals = hcApplier->d_overset_rhs_vals_;
    auto rhs_vals = hcApplier->rhs_dev_;
    /* write to the rhs */
    Kokkos::parallel_for(
      "fillOversetRhsVector", N, KOKKOS_LAMBDA(const unsigned& i) {
        HypreIntType row = orow_indices(i);
        rhs_vals(row - iLower, 0) = orvals(i);
      });
  }
}

void
HypreLinearSystem::hypreIJMatrixSetAddToValues()
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  auto num_nonzeros_owned = hcApplier->num_nonzeros_owned_;
  auto num_nonzeros_shared = hcApplier->num_nonzeros_shared_;

  HypreDirectSolver* solver =
    reinterpret_cast<HypreDirectSolver*>(linearSolver_);
  HypreLinearSolverConfig* config =
    reinterpret_cast<HypreLinearSolverConfig*>(solver->getConfig());
  if (config->simpleHypreMatrixAssemble()) {
#if 0
    /* set the key hypre parameters */
    HYPRE_IJMatrixSetMaxOnProcElmts(mat_, hcApplier->num_nonzeros_owned_);
    HYPRE_IJMatrixSetOffProcSendElmts(mat_, offProcNNZToSend_);
    HYPRE_IJMatrixSetOffProcRecvElmts(mat_, offProcNNZToRecv_);
#endif
  }

  if (config->getWritePreassemblyMatrixFiles()) {
    MPI_Barrier(realm_.bulk_data().parallel());

    char rank_str[8];
    sprintf(rank_str, "%05d", rank_);
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFileRows = eqSysName_ + ".IJM." + writeCounter +
                                    ".mat." + std::string(rank_str) +
                                    ".preassem.i";
    const std::string matFileCols = eqSysName_ + ".IJM." + writeCounter +
                                    ".mat." + std::string(rank_str) +
                                    ".preassem.j";
    const std::string matFileVals = eqSysName_ + ".IJM." + writeCounter +
                                    ".mat." + std::string(rank_str) +
                                    ".preassem.v";
    const std::string matFileMeta = eqSysName_ + ".IJM." + writeCounter +
                                    ".mat." + std::string(rank_str) +
                                    ".preassem.meta";

    FILE* fid = fopen(matFileRows.c_str(), "wb");
    fwrite(
      rows_host_.data(), sizeof(HypreIntType),
      num_nonzeros_owned + num_nonzeros_shared, fid);
    fclose(fid);

    fid = fopen(matFileCols.c_str(), "wb");
    fwrite(
      cols_host_.data(), sizeof(HypreIntType),
      num_nonzeros_owned + num_nonzeros_shared, fid);
    fclose(fid);

    DoubleViewHost temp("temp", hcApplier->values_dev_.extent(0));
    Kokkos::deep_copy(temp, hcApplier->values_dev_);
    fid = fopen(matFileVals.c_str(), "wb");
    fwrite(
      temp.data(), sizeof(double), num_nonzeros_owned + num_nonzeros_shared,
      fid);
    fclose(fid);

    fid = fopen(matFileMeta.c_str(), "wb");
    HypreIntType meta[6] = {
      globalNumRows_,
      iLower_,
      iUpper_,
      num_nonzeros_owned,
      num_nonzeros_shared,
      (HypreIntType)rows_dev_.extent(0)};
    fwrite(meta, sizeof(HypreIntType), 6, fid);
    fclose(fid);

    MPI_Barrier(realm_.bulk_data().parallel());
  }

  if (num_nonzeros_owned) {
    /* Set the owned part */
    HYPRE_IJMatrixSetValues2(
      mat_, num_nonzeros_owned, NULL, rows_dev_.data(), NULL,
      hcApplier->cols_dev_.data(), hcApplier->values_dev_.data());
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
    scanBufferForBadValues(
      hcApplier->values_dev_.data(), num_nonzeros_owned, __FILE__, __FUNCTION__,
      __LINE__, "Owned Matrix");
    scanOwnedIndicesForBadValues(
      rows_dev_.data(), hcApplier->cols_dev_.data(), num_nonzeros_owned,
      __FILE__, __FUNCTION__, __LINE__);
#endif
  }

  if (num_nonzeros_shared) {
    /* Add the shared part */
    HYPRE_IJMatrixAddToValues2(
      mat_, num_nonzeros_shared, NULL, rows_dev_.data() + num_nonzeros_owned,
      NULL, hcApplier->cols_dev_.data() + num_nonzeros_owned,
      hcApplier->values_dev_.data() + num_nonzeros_owned);
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
    scanBufferForBadValues(
      hcApplier->values_dev_.data() + num_nonzeros_owned, num_nonzeros_shared,
      __FILE__, __FUNCTION__, __LINE__, "Shared Matrix");
    scanSharedIndicesForBadValues(
      rows_dev_.data() + num_nonzeros_owned,
      hcApplier->cols_dev_.data() + num_nonzeros_owned, num_nonzeros_shared,
      __FILE__, __FUNCTION__, __LINE__);
#endif
  }
}

void
HypreLinearSystem::hypreIJVectorSetAddToValues()
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  auto num_rows_owned = hcApplier->num_rows_owned_;
  auto num_rows_shared = hcApplier->num_rows_shared_;

  HypreDirectSolver* solver =
    reinterpret_cast<HypreDirectSolver*>(linearSolver_);
  HypreLinearSolverConfig* config =
    reinterpret_cast<HypreLinearSolverConfig*>(solver->getConfig());
  if (config->simpleHypreMatrixAssemble()) {
#if 0
    /* set the key hypre parameters */
    HYPRE_IJVectorSetMaxOnProcElmts(rhs_, num_rows_owned);
    HYPRE_IJVectorSetOffProcSendElmts(rhs_, offProcRhsToSend_);
    HYPRE_IJVectorSetOffProcRecvElmts(rhs_, offProcRhsToRecv_);
#endif
  }

  if (config->getWritePreassemblyMatrixFiles()) {
    MPI_Barrier(realm_.bulk_data().parallel());

    char rank_str[8];
    sprintf(rank_str, "%05d", rank_);
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string rhsFileRows = eqSysName_ + ".IJV." + writeCounter +
                                    ".rhs." + std::string(rank_str) +
                                    ".preassem.i";
    const std::string rhsFileVals = eqSysName_ + ".IJV." + writeCounter +
                                    ".rhs." + std::string(rank_str) +
                                    ".preassem.v";
    const std::string rhsFileMeta = eqSysName_ + ".IJV." + writeCounter +
                                    ".rhs." + std::string(rank_str) +
                                    ".preassem.meta";

    FILE* fid = fopen(rhsFileRows.c_str(), "wb");
    fwrite(
      rhs_rows_host_.data(), sizeof(HypreIntType),
      num_rows_owned + num_rows_shared, fid);
    fclose(fid);

    DoubleView2DHost temp(
      "temp", hcApplier->rhs_dev_.extent(0), hcApplier->nDim_);
    Kokkos::deep_copy(temp, hcApplier->rhs_dev_);
    fid = fopen(rhsFileVals.c_str(), "wb");
    fwrite(temp.data(), sizeof(double), num_rows_owned + num_rows_shared, fid);
    fclose(fid);

    fid = fopen(rhsFileMeta.c_str(), "wb");
    HypreIntType meta[3] = {
      num_rows_owned, num_rows_shared, (HypreIntType)rhs_rows_dev_.extent(0)};
    fwrite(meta, sizeof(HypreIntType), 3, fid);
    fclose(fid);

    MPI_Barrier(realm_.bulk_data().parallel());
  }

  if (num_rows_owned) {
    /* Set the owned part */
    HYPRE_IJVectorSetValues(
      rhs_, num_rows_owned, rhs_rows_dev_.data(), hcApplier->rhs_dev_.data());
  }

  if (num_rows_shared) {
    /* Add the shared part */
    HYPRE_IJVectorAddToValues(
      rhs_, num_rows_shared, rhs_rows_dev_.data() + num_rows_owned,
      hcApplier->rhs_dev_.data() + num_rows_owned);
  }
}

void
HypreLinearSystem::dumpMatrixStats()
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());
  HypreIntType totalMatElmts = hcApplier->num_nonzeros_owned_;
  HypreIntType totalRhsElmts = hcApplier->num_rows_owned_;

  /* parallel info */
  int nprocs = realm_.bulk_data().parallel_size();
  int iproc = realm_.bulk_data().parallel_rank();

  std::vector<HypreIntType> rows(numRows_);
  for (HypreIntType i = 0; i < numRows_; ++i)
    rows[i] = iLower_ + i;
  std::vector<HypreIntType> cols(numRows_);
  std::fill(cols.begin(), cols.end(), 0);

  /* retrive nnz per row from Hypre */
  HYPRE_IJMatrixGetRowCounts(mat_, numRows_, rows.data(), cols.data());

  /* compute nnz for this rank */
  HypreIntType nnz = 0;
  for (HypreIntType i = 0; i < numRows_; ++i)
    nnz += cols[i];

  /* NNZ from Hypre row counts .. after assembly */
  std::vector<HypreIntType> tmp(nprocs);
  std::fill(tmp.begin(), tmp.end(), 0);
  std::vector<HypreIntType> globalNNZPerProc(nprocs);
  std::fill(globalNNZPerProc.begin(), globalNNZPerProc.end(), 0);
  tmp[iproc] = nnz;
  MPI_Reduce(
    tmp.data(), globalNNZPerProc.data(), nprocs, HYPRE_MPI_INT, MPI_SUM, 0,
    realm_.bulk_data().parallel());

  /* NNZ owned ... before assembly */
  std::fill(tmp.begin(), tmp.end(), 0);
  tmp[iproc] = totalMatElmts;
  std::vector<HypreIntType> nnz_owned(nprocs);
  std::fill(nnz_owned.begin(), nnz_owned.end(), 0);
  MPI_Reduce(
    tmp.data(), nnz_owned.data(), nprocs, HYPRE_MPI_INT, MPI_SUM, 0,
    realm_.bulk_data().parallel());

  /* NNZ send ... before assembly */
  std::fill(tmp.begin(), tmp.end(), 0);
  tmp[iproc] = offProcNNZToSend_;
  std::vector<HypreIntType> nnz_send(nprocs);
  std::fill(nnz_send.begin(), nnz_send.end(), 0);
  MPI_Reduce(
    tmp.data(), nnz_send.data(), nprocs, HYPRE_MPI_INT, MPI_SUM, 0,
    realm_.bulk_data().parallel());

  /* NNZ recv ... before assembly */
  std::fill(tmp.begin(), tmp.end(), 0);
  tmp[iproc] = offProcNNZToRecv_;
  std::vector<HypreIntType> nnz_recv(nprocs);
  std::fill(nnz_recv.begin(), nnz_recv.end(), 0);
  MPI_Reduce(
    tmp.data(), nnz_recv.data(), nprocs, HYPRE_MPI_INT, MPI_SUM, 0,
    realm_.bulk_data().parallel());

  /* num rows */
  std::fill(tmp.begin(), tmp.end(), 0);
  tmp[iproc] = totalRhsElmts;
  std::vector<HypreIntType> nrows(nprocs);
  std::fill(nrows.begin(), nrows.end(), 0);
  MPI_Reduce(
    tmp.data(), nrows.data(), nprocs, HYPRE_MPI_INT, MPI_SUM, 0,
    realm_.bulk_data().parallel());

  /* Write to a file from rank 0 */
  if (iproc == 0) {
    char fname[1000];
#if defined(KOKKOS_ENABLE_GPU)
    sprintf(fname, "%s_decomp_%dGPUs.txt", name_.c_str(), nprocs);
#else
    sprintf(fname, "%s_decomp_%dCPUs.txt", name_.c_str(), nprocs);
#endif

    std::ofstream myfile;
    myfile.open(fname);
    myfile << "rank"
           << ",lower"
           << ",upper"
           << ",num_rows"
           << ",nnz"
           << ",nnz_owned"
           << ",nnz_send"
           << ",nnz_recv" << std::endl;
    for (int i = 0; i < nprocs; ++i) {
      myfile << i << "," << realm_.hypreOffsets_[i] << ","
             << realm_.hypreOffsets_[i + 1] << "," << nrows[i] << ","
             << globalNNZPerProc[i] << "," << nnz_owned[i] << "," << nnz_send[i]
             << "," << nnz_recv[i] << std::endl;
    }
    myfile.close();
  }
}

void
HypreLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver =
    reinterpret_cast<HypreDirectSolver*>(linearSolver_);

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  hypre_CSRMatrix* diag =
    hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix*)hypre_IJMatrixObject(mat_));
  hypre_CSRMatrix* offd =
    hypre_ParCSRMatrixOffd((hypre_ParCSRMatrix*)hypre_IJMatrixObject(mat_));
  HYPRE_Int nnz_diag = hypre_CSRMatrixNumNonzeros(diag);
  HYPRE_Int nnz_offd = hypre_CSRMatrixNumNonzeros(offd);
  double* ptr_diag = hypre_CSRMatrixData(diag);
  double* ptr_offd = hypre_CSRMatrixData(offd);
  scanBufferForBadValues(
    ptr_diag, nnz_diag, __FILE__, __FUNCTION__, __LINE__, "Diag Matrix");
  scanBufferForBadValues(
    ptr_offd, nnz_offd, __FILE__, __FUNCTION__, __LINE__, "Offd Matrix");
  output_ = fopen(oname_, "at");
  fprintf(
    output_,
    "rank=%d : diag num_rows=%d, num_cols=%d, offd num_rows=%d, num_cols=%d\n",
    rank_, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag),
    hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd));
  fclose(output_);
#endif

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  hypreMatAssemblyTimer_.back() += msec;
  gettimeofday(&_start, NULL);
#endif

  HYPRE_IJVectorAssemble(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorAssemble(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
         1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  hypreRhsAssemblyTimer_.back() += msec;
#endif

  solver->comm_ = realm_.bulk_data().parallel();

  HypreLinearSolverConfig* config =
    reinterpret_cast<HypreLinearSolverConfig*>(solver->getConfig());
  if (config->dumpHypreMatrixStats() && !matrixStatsDumped_) {
    dumpMatrixStats();
    matrixStatsDumped_ = true;
  }
}

void
HypreLinearSystem::loadComplete()
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  /* finish assembly for the coupled overset case */
  finishCoupledOversetAssembly();

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  /* Matrix */
  hypreIJMatrixSetAddToValues();

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the stop time */
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  hypreMatAssemblyTimer_.push_back(msec);
  gettimeofday(&_start, NULL);
#endif

  /* Rhs */
  hypreIJVectorSetAddToValues();

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the stop time */
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
         1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  hypreRhsAssemblyTimer_.push_back(msec);
#endif

  /* Reset after assembly */
  hcApplier->reinitialize_ = true;

  /* call IJMatrix/IJVectorAssemble */
  loadCompleteSolver();
}

void
HypreLinearSystem::zeroSystem()
{
  HypreDirectSolver* solver =
    reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  MPI_Comm comm = realm_.bulk_data().parallel();

  if (hypreMatrixVectorsCreated_) {
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
    sprintf(oname_, "debug_out_%d.txt", rank_);
    output_ = fopen(oname_, "wt");
    fprintf(
      output_, "rank_=%d EqnName=%s : %s %s %d\n", rank_, name_.c_str(),
      __FILE__, __FUNCTION__, __LINE__);
    fclose(output_);
#endif
    HYPRE_IJMatrixDestroy(mat_);
    HYPRE_IJVectorDestroy(rhs_);
    HYPRE_IJVectorDestroy(sln_);
    hypreMatrixVectorsCreated_ = false;
  }

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));
  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_);
  HYPRE_IJVectorSetObjectType(rhs_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));
  HYPRE_ParVectorSetConstantValues(solver->parRhs_, 0.0);

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_);
  HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));
  HYPRE_ParVectorSetConstantValues(solver->parSln_, 0.0);

  hypreMatrixVectorsCreated_ = true;
}

sierra::nalu::CoeffApplier*
HypreLinearSystem::get_coeff_applier()
{
  /* call this before getting the device coeff applier
     Do NOT move this!
   */
  resetCoeffApplierData();
  return hostCoeffApplier->device_pointer();
}

/********************************************************************************************************/
/*                     Beginning of HypreLinSysCoeffApplier implementations */
/********************************************************************************************************/
HypreLinearSystem::HypreLinSysCoeffApplier::HypreLinSysCoeffApplier(
  unsigned numDof, unsigned nDim, HypreIntType iLower, HypreIntType iUpper)
  : numDof_(numDof),
    nDim_(nDim),
    iLower_(iLower),
    iUpper_(iUpper),
    devicePointer_(nullptr)
{
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::sort(
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  unsigned N)
{
  if (N == 2) {
    int tmp;
    if (localIds[0] > localIds[1]) {
      tmp = localIds[0];
      localIds[0] = localIds[1];
      localIds[1] = tmp;
      tmp = sortPermutation[0];
      sortPermutation[0] = sortPermutation[1];
      sortPermutation[1] = tmp;
    }
  } else if (N == 3) {
    int tmp;
    if (localIds[0] > localIds[1]) {
      tmp = localIds[0];
      localIds[0] = localIds[1];
      localIds[1] = tmp;
      tmp = sortPermutation[0];
      sortPermutation[0] = sortPermutation[1];
      sortPermutation[1] = tmp;
    }
    if (localIds[0] > localIds[2]) {
      tmp = localIds[0];
      localIds[0] = localIds[2];
      localIds[2] = tmp;
      tmp = sortPermutation[0];
      sortPermutation[0] = sortPermutation[2];
      sortPermutation[2] = tmp;
    }
    if (localIds[1] > localIds[2]) {
      tmp = localIds[1];
      localIds[1] = localIds[2];
      localIds[2] = tmp;
      tmp = sortPermutation[1];
      sortPermutation[1] = sortPermutation[2];
      sortPermutation[2] = tmp;
    }
  } else if (N == 4) {
    int tmp;
    if (localIds[0] > localIds[1]) {
      tmp = localIds[0];
      localIds[0] = localIds[1];
      localIds[1] = tmp;
      tmp = sortPermutation[0];
      sortPermutation[0] = sortPermutation[1];
      sortPermutation[1] = tmp;
    }
    if (localIds[2] > localIds[3]) {
      tmp = localIds[2];
      localIds[2] = localIds[3];
      localIds[3] = tmp;
      tmp = sortPermutation[2];
      sortPermutation[2] = sortPermutation[3];
      sortPermutation[3] = tmp;
    }
    if (localIds[0] > localIds[2]) {
      tmp = localIds[0];
      localIds[0] = localIds[2];
      localIds[2] = tmp;
      tmp = sortPermutation[0];
      sortPermutation[0] = sortPermutation[2];
      sortPermutation[2] = tmp;
    }
    if (localIds[1] > localIds[3]) {
      tmp = localIds[1];
      localIds[1] = localIds[3];
      localIds[3] = tmp;
      tmp = sortPermutation[1];
      sortPermutation[1] = sortPermutation[3];
      sortPermutation[3] = tmp;
    }
    if (localIds[1] > localIds[2]) {
      tmp = localIds[1];
      localIds[1] = localIds[2];
      localIds[2] = tmp;
      tmp = sortPermutation[1];
      sortPermutation[1] = sortPermutation[2];
      sortPermutation[2] = tmp;
    }
  } else {
    for (unsigned i = 0; i < N - 1; ++i)
      for (unsigned j = 0; j < N - i - 1; ++j)
        if (localIds[j] > localIds[j + 1]) {
          int t = localIds[j];
          localIds[j] = localIds[j + 1];
          localIds[j + 1] = t;
          t = sortPermutation[j];
          sortPermutation[j] = sortPermutation[j + 1];
          sortPermutation[j + 1] = t;
        }
  }
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::sum_into(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const HypreIntType& iLower,
  const HypreIntType& iUpper,
  unsigned numDof,
  HypreIntType memShift)
{

  unsigned numRows = numEntities * numDof;

  for (unsigned i = 0; i < numEntities; ++i) {
    auto node = entities[i];
    HypreIntType hid;
    if (periodic_node_to_hypre_id_.exists(node.local_offset()))
      hid = periodic_node_to_hypre_id_.value_at(
        periodic_node_to_hypre_id_.find(node.local_offset()));
    else
      hid = ngpHypreGlobalId_.get(ngpMesh_, node, 0);

    for (unsigned d = 0; d < numDof; ++d) {
      unsigned lid = i * numDof + d;
      localIds[lid] = hid * numDof + d;
      sortPermutation[lid] = lid;
    }
  }

  // sort the local ids
  sort(localIds, sortPermutation, numEntities * numDof);

  for (unsigned i = 0; i < numEntities; ++i) {
    int ix = i * numDof;
    HypreIntType hid = localIds[ix];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid))
        continue;
    }

    if (hid >= iLower && hid <= iUpper) {

      for (unsigned d = 0; d < numDof; ++d) {
        unsigned ir = ix + d;
        hid = localIds[ir];

        int ii = sortPermutation[ir];
        const double* cur_lhs = &lhs(ii, 0);

        HypreIntType index = hid - iLower;

        /* fill the matrix values */
        unsigned matIndex = mat_row_start_owned_ra_(index);
        for (unsigned k = 0; k < numRows; ++k) {
          /* binary search subrange rather than a map.find */
          HypreIntType col = localIds[k];
          while (cols_dev_ra_(matIndex) < col)
            matIndex++;
          int kk = sortPermutation[k];

          /* write the matrix element */
          Kokkos::atomic_add(&values_dev_(matIndex), cur_lhs[kk]);
        }
        /* fill the right hand side values */
        Kokkos::atomic_add(&rhs_dev_(index, 0), rhs[ii]);
      }

    } else {

      for (unsigned d = 0; d < numDof; ++d) {
        unsigned ir = ix + d;
        hid = localIds[ir];

        int ii = sortPermutation[ir];
        const double* cur_lhs = &lhs(ii, 0);

        if (!map_shared_.exists(hid))
          continue;

        /* Find the index of the row */
        unsigned index = map_shared_.value_at(map_shared_.find(hid));
        unsigned matIndex = mat_row_start_shared_ra_(index) + memShift;

        /* fill the matrix values */
        for (unsigned k = 0; k < numRows; ++k) {
          /* binary search subrange rather than a map.find */
          HypreIntType col = localIds[k];
          while (cols_dev_ra_(matIndex) < col)
            matIndex++;
          int kk = sortPermutation[k];
          /* write the matrix element */
          Kokkos::atomic_add(&values_dev_(matIndex), cur_lhs[kk]);
        }
        /* fill the right hand side values */
        unsigned rhsIndex =
          rhs_row_start_shared_(index) + (iUpper - iLower + 1);
        Kokkos::atomic_add(&rhs_dev_(rhsIndex, 0), rhs[ii]);
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
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const HypreIntType& iLower,
  const HypreIntType& iUpper,
  HypreIntType memShift)
{

  for (unsigned i = 0; i < numEntities; ++i) {
    auto node = entities[i];
    if (periodic_node_to_hypre_id_.exists(node.local_offset()))
      localIds[i] = periodic_node_to_hypre_id_.value_at(
        periodic_node_to_hypre_id_.find(node.local_offset()));
    else
      localIds[i] = ngpHypreGlobalId_.get(ngpMesh_, node, 0);
    sortPermutation[i] = i;
  }

  // sort the local ids
  sort(localIds, sortPermutation, numEntities);

  for (unsigned i = 0; i < numEntities; ++i) {
    HypreIntType hid = localIds[i];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid))
        continue;
    }

    int ii = sortPermutation[i];
    const double* cur_lhs = &lhs(ii, 0);

    if (hid >= iLower && hid <= iUpper) {
      /* fill the matrix values */
      HypreIntType index = hid - iLower;
      unsigned matIndex = mat_row_start_owned_ra_(index);
      for (unsigned k = 0; k < numEntities; ++k) {
        /* binary search subrange rather than a map.find */
        HypreIntType col = localIds[k];
        while (cols_dev_ra_(matIndex) < col)
          matIndex++;
        /* write the matrix element */
        int kk = sortPermutation[k];
        Kokkos::atomic_add(&values_dev_(matIndex), cur_lhs[kk]);
        matIndex++;
      }
      /* fill the right hand side values */
      Kokkos::atomic_add(&rhs_dev_(index, 0), rhs[ii]);

    } else {

      if (!map_shared_.exists(hid))
        continue;
      /* Find the index of the row */
      unsigned index = map_shared_.value_at(map_shared_.find(hid));
      unsigned matIndex = mat_row_start_shared_ra_(index) + memShift;
      for (unsigned k = 0; k < numEntities; ++k) {
        /* binary search subrange rather than a map.find */
        HypreIntType col = localIds[k];
        while (cols_dev_ra_(matIndex) < col)
          matIndex++;
        /* write the matrix element */
        int kk = sortPermutation[k];
        Kokkos::atomic_add(&values_dev_(matIndex), cur_lhs[kk]);
        matIndex++;
      }
      /* fill the right hand side values */
      unsigned rhsIndex = rhs_row_start_shared_(index) + (iUpper - iLower + 1);
      Kokkos::atomic_add(&rhs_dev_(rhsIndex, 0), rhs[ii]);
    }
  }
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  if (numDof_ == 1)
    sum_into_1DoF(
      numEntities, entities, localIds, sortPermutation, rhs, lhs, iLower_,
      iUpper_, num_nonzeros_owned_);
  else
    sum_into(
      numEntities, entities, localIds, sortPermutation, rhs, lhs, iLower_,
      iUpper_, numDof_, num_nonzeros_owned_);
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::reset_rows(
  unsigned numNodes,
  const stk::mesh::Entity* nodeList,
  const double diag_value,
  const double rhs_residual,
  const HypreIntType iLower,
  const HypreIntType iUpper,
  const unsigned numDof,
  HypreIntType memShift)
{
  for (unsigned i = 0; i < numNodes; ++i) {
    HypreIntType lid;
    auto node = nodeList[i];
    if (periodic_node_to_hypre_id_.exists(node.local_offset()))
      lid = periodic_node_to_hypre_id_.value_at(
        periodic_node_to_hypre_id_.find(node.local_offset()));
    else
      lid = ngpHypreGlobalId_.get(ngpMesh_, node, 0);

    for (unsigned d = 0; d < numDof; ++d) {
      HypreIntType hid = lid * numDof + d;

      if (hid >= iLower && hid <= iUpper) {
        HypreIntType index = hid - iLower;
        unsigned lower = mat_row_start_owned_ra_(index);
        unsigned upper = mat_row_start_owned_ra_(index + 1);
        for (unsigned k = lower; k < upper; ++k) {
          values_dev_(k) = 0.0;
          if (cols_dev_ra_(k) == hid)
            values_dev_(k) = diag_value;
        }
        rhs_dev_(hid - iLower, 0) = rhs_residual;

      } else {
        if (!map_shared_.exists(hid))
          continue;
        unsigned index = map_shared_.value_at(map_shared_.find(hid));
        unsigned lower = mat_row_start_shared_ra_(index) + memShift;
        unsigned upper = mat_row_start_shared_ra_(index + 1) + memShift;
        for (unsigned k = lower; k < upper; ++k) {
          values_dev_(k) = 0.0;
          if (cols_dev_ra_(k) == hid)
            values_dev_(k) = diag_value;
        }
        unsigned rhsIndex =
          rhs_row_start_shared_(index) + (iUpper - iLower + 1);
        rhs_dev_(rhsIndex, 0) = rhs_residual;
      }
    }
  }
}

KOKKOS_FUNCTION
void
HypreLinearSystem::HypreLinSysCoeffApplier::resetRows(
  unsigned numNodes,
  const stk::mesh::Entity* nodeList,
  const unsigned,
  const unsigned,
  const double diag_value,
  const double rhs_residual)
{
  checkSkippedRows_() = 0;
  reset_rows(
    numNodes, nodeList, diag_value, rhs_residual, iLower_, iUpper_, numDof_,
    num_nonzeros_owned_);
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::free_device_pointer()
{
#if defined(KOKKOS_ENABLE_GPU)
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier*
HypreLinearSystem::HypreLinSysCoeffApplier::device_pointer()
{
#if defined(KOKKOS_ENABLE_GPU)
  if (devicePointer_ != nullptr) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
  devicePointer_ = sierra::nalu::create_device_expression(*this);
  return devicePointer_;
#else
  return this;
#endif
}

/********************************************************************************************************/
/*                           End of HypreLinSysCoeffApplier implementations */
/********************************************************************************************************/

void
HypreLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>& /* scratchIds */,
  std::vector<double>& /* scratchVals */,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char* /* trace_tag */)
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  /* Pure host implementation */
  const size_t numEntities = entities.size();
  HypreIntType hid0 =
    *stk::mesh::field_data(*realm_.hypreGlobalId_, entities[0]);

  if (hcApplier->oversetRowsMapHost_.exists(hid0)) {
    if (numDof_ == 1) {
      if (hid0 >= iLower_ && hid0 <= iUpper_) {
        for (size_t i = 0; i < numEntities; ++i) {
          hcApplier->h_overset_rows_(hcApplier->overset_mat_counter_) = hid0;
          hcApplier->h_overset_cols_(hcApplier->overset_mat_counter_) =
            *stk::mesh::field_data(*realm_.hypreGlobalId_, entities[i]);
          hcApplier->h_overset_vals_(hcApplier->overset_mat_counter_) = lhs[i];
          hcApplier->overset_mat_counter_++;
        }
        hcApplier->h_overset_row_indices_(hcApplier->overset_rhs_counter_) =
          hid0;
        hcApplier->h_overset_rhs_vals_(hcApplier->overset_rhs_counter_) =
          rhs[0];
        hcApplier->overset_rhs_counter_++;
      }
    } else {
      throw std::runtime_error("HypreLinearSystem::sumInto not "
                               "yet implemented for numDof>1. Exiting.");
    }
  } else {
    throw std::runtime_error(
      "HypreLinearSystem::sumInto not yet implemented for (NON) "
      "overset constaint algorithms. Exiting.");
  }
}

void
HypreLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned,
  const unsigned)
{
  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  /* Step 1: execute the old CPU code */
  auto& meta = realm_.meta_data();

  const stk::mesh::Selector selector =
    (meta.locally_owned_part() & stk::mesh::selectUnion(parts) &
     stk::mesh::selectField(*solutionField) &
     !(realm_.get_inactive_selector()));

  NGPDoubleFieldType ngpSolutionField =
    realm_.ngp_field_manager().get_field<double>(
      solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField =
    realm_.ngp_field_manager().get_field<double>(
      bcValuesField->mesh_meta_data_ordinal());

  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  /* data from hcApplier */
  const auto& ngpMesh = hcApplier->ngpMesh_;
  const auto hypreGID = hcApplier->ngpHypreGlobalId_;
  auto mat_row_start_owned = hcApplier->mat_row_start_owned_ra_;
  auto vals = hcApplier->values_dev_;
  auto rhs_vals = hcApplier->rhs_dev_;

  auto numDof = numDof_;
  auto iLower = iLower_;

  nalu_ngp::run_entity_algorithm(
    "HypreLinearSystem::applyDirichletBCs", ngpMesh, stk::topology::NODE_RANK,
    selector, KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
      const auto node = (*mi.bucket)[mi.bucketOrd];
      HypreIntType hid = hypreGID.get(ngpMesh, node, 0);
      for (unsigned d = 0; d < numDof; ++d) {
        HypreIntType lid = hid * numDof + d;
        unsigned matIndex = mat_row_start_owned(lid - iLower);
        vals(matIndex) = 1.0;
        rhs_vals(lid - iLower, 0) = ngpBCValuesField.get(ngpMesh, node, d) -
                                    ngpSolutionField.get(ngpMesh, node, d);
      }
    });
}

HypreIntType
HypreLinearSystem::get_entity_hypre_id(const stk::mesh::Entity& node)
{
  auto& bulk = realm_.bulk_data();
  const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
  const auto mnode = (naluId == bulk.identifier(node))
                       ? node
                       : bulk.get_entity(stk::topology::NODE_RANK, naluId);
  HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);
  return hid;
}

int
HypreLinearSystem::solve(stk::mesh::FieldBase* linearSolutionField)
{
  HypreDirectSolver* solver =
    reinterpret_cast<HypreDirectSolver*>(linearSolver_);

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

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  output_ = fopen(oname_, "at");
  fprintf(
    output_, "%s %s %d %s : rank=%d\n", __FILE__, __FUNCTION__, __LINE__,
    eqSysName_.c_str(), rank_);
#endif

  status = solver->solve(iters, finalResidNorm, realm_.isFinalOuterIter_);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  output_ = fopen(oname_, "at");
  fprintf(
    output_, "%s %s %d %s : rank=%d\n", __FILE__, __FUNCTION__, __LINE__,
    eqSysName_.c_str(), rank_);
#endif

  /* set this after the solve calls */
  solver->set_initialize_solver_flag();

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string slnFile = eqSysName_ + ".IJV." + writeCounter + ".sln";
    HYPRE_IJVectorPrint(sln_, slnFile.c_str());
  }

  HypreLinearSolverConfig* config =
    reinterpret_cast<HypreLinearSolverConfig*>(solver->getConfig());
  if (
    solver->getConfig()->getWriteMatrixFiles() ||
    config->getWritePreassemblyMatrixFiles()) {
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
HypreLinearSystem::copy_hypre_to_stk(stk::mesh::FieldBase* stkField)
{
  auto& meta = realm_.meta_data();
  const auto selector =
    stk::mesh::selectField(*stkField) & meta.locally_owned_part() &
    !(stk::mesh::selectUnion(realm_.get_slave_part_vector())) &
    !(realm_.get_inactive_selector());

  HypreLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreLinSysCoeffApplier*>(hostCoeffApplier.get());

  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  auto ngpField = realm_.ngp_field_manager().get_field<double>(
    stkField->mesh_meta_data_ordinal());
  auto ngpHypreGlobalId = hcApplier->ngpHypreGlobalId_;
  const auto& ngpMesh = hcApplier->ngpMesh_;
  const auto periodic_node_to_hypre_id = hcApplier->periodic_node_to_hypre_id_;

  auto iLower = iLower_;
  auto iUpper = iUpper_;
  auto numDof = numDof_;

  /******************************/
  /* Move solution to stk field */

  /* use internal hypre APIs to get directly at the pointer to the owned SLN
   * vector */
  double* sln_data = hypre_VectorData(
    hypre_ParVectorLocalVector((hypre_ParVector*)hypre_IJVectorObject(sln_)));
  nalu_ngp::run_entity_algorithm(
    "HypreLinearSystem::copy_hypre_to_stk", ngpMesh, stk::topology::NODE_RANK,
    selector, KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
      const auto node = (*mi.bucket)[mi.bucketOrd];
      HypreIntType hid;
      if (periodic_node_to_hypre_id.exists(node.local_offset()))
        hid = periodic_node_to_hypre_id.value_at(
          periodic_node_to_hypre_id.find(node.local_offset()));
      else
        hid = ngpHypreGlobalId.get(ngpMesh, node, 0);

      for (unsigned d = 0; d < numDof; ++d) {
        HypreIntType lid = hid * numDof + d;
        if (lid >= iLower && lid <= iUpper) {
          ngpField.get(mi, d) = sln_data[lid - iLower];
        }
      }
    });
  ngpField.modify_on_device();

  /********************/
  /* Compute RHS norm */
#if 1
  /* Use Hypre internal APIs */
  double gblnorm2 = hypre_ParVectorInnerProd(
    (hypre_ParVector*)hypre_IJVectorObject(rhs_),
    (hypre_ParVector*)hypre_IJVectorObject(rhs_));
#else
  double rhsnorm2 = 0.0;

  /* use internal hypre APIs to get directly at the pointer to the owned RHS
   * vector */
  double* rhs_data = hypre_VectorData(
    hypre_ParVectorLocalVector((hypre_ParVector*)hypre_IJVectorObject(rhs_)));
  Kokkos::parallel_reduce(
    "HypreLinearSystem::Reduction", N,
    KOKKOS_LAMBDA(const int i, double& update) {
      double t = rhs_data[i];
      update += t * t;
    },
    rhsnorm2);

  double gblnorm2 = 0.0;
  stk::all_reduce_sum(bulk.parallel(), &rhsnorm2, &gblnorm2, 1);
#endif

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  scanBufferForBadValues(rhs_data, N, __FILE__, __FUNCTION__, __LINE__, "RHS");
  scanBufferForBadValues(sln_data, N, __FILE__, __FUNCTION__, __LINE__, "SLN");
#endif

  return std::sqrt(gblnorm2);
}

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
void
HypreLinearSystem::scanBufferForBadValues(
  double* ptr,
  int N,
  const char* file,
  const char* func,
  int line,
  char* bufferName)
{
  output_ = fopen(oname_, "at");
  bool foundBadValue = false;
  int index = -1;
  double value = 0;
  for (int i = 0; i < N; ++i)
    if (!std::isfinite(ptr[i])) {
      foundBadValue = true;
      index = i;
      value = ptr[i];
      break;
    }
  if (foundBadValue)
    fprintf(
      output_, "%s %s %d %s : Found Bad %s value %1.15g at %d on rank %d\n",
      file, func, line, eqSysName_.c_str(), bufferName, value, index, rank_);
  else
    fprintf(
      output_, "%s %s %d %s : All %s values good on rank %d\n", file, func,
      line, eqSysName_.c_str(), bufferName, rank_);
  fclose(output_);
  return;
}

void
HypreLinearSystem::scanOwnedIndicesForBadValues(
  HypreIntType* rows,
  HypreIntType* cols,
  int N,
  const char* file,
  const char* func,
  int line)
{
  output_ = fopen(oname_, "at");
  for (int i = 0; i < N; ++i) {
    if (
      rows[i] < 0 || rows[i] >= globalNumRows_ || cols[i] < 0 ||
      cols[i] >= globalNumRows_) {
      fprintf(
        output_,
        "Very Bad : %s %s %d %s, Owned Matrix : Found Row/Column Index (%d,%d) "
        "outside of (%d, %d) at %d on rank %d\n",
        file, func, line, eqSysName_.c_str(), rows[i], cols[i], 0,
        globalNumRows_, i, rank_);
      return;
    } else if (rows[i] < iLower_ || rows[i] > iUpper_) {
      fprintf(
        output_,
        "Bad : %s %s %d %s, Owned Matrix : Found Row/Column Index (%d,%d) "
        "outside range (%d, %d) at %d on rank %d\n",
        file, func, line, eqSysName_.c_str(), rows[i], cols[i], iLower_,
        iUpper_, i, rank_);
      return;
    }
  }
  fprintf(
    output_, "%s %s %d %s : All Owned Indices good on rank %d\n", file, func,
    line, eqSysName_.c_str(), rank_);
  fclose(output_);
  return;
}

void
HypreLinearSystem::scanSharedIndicesForBadValues(
  HypreIntType* rows,
  HypreIntType* cols,
  int N,
  const char* file,
  const char* func,
  int line)
{
  output_ = fopen(oname_, "at");
  for (int i = 0; i < N; ++i) {
    if (
      rows[i] < 0 || rows[i] >= globalNumRows_ || cols[i] < 0 ||
      cols[i] >= globalNumRows_) {
      fprintf(
        output_,
        "Very Bad : %s %s %d %s, Shared Matrix : Found Row/Column Index "
        "(%d,%d) outside of (%d, %d) at %d on rank %d\n",
        file, func, line, eqSysName_.c_str(), rows[i], cols[i], 0,
        globalNumRows_, i, rank_);
      return;
    } else if (rows[i] >= iLower_ && rows[i] <= iUpper_) {
      fprintf(
        output_,
        "Bad : %s %s %d %s, Shared Matrix : Found Row/Column Index (%d,%d) "
        "inside range (%d, %d) at %d on rank %d\n",
        file, func, line, eqSysName_.c_str(), rows[i], cols[i], iLower_,
        iUpper_, i, rank_);
      return;
    }
  }
  fprintf(
    output_, "%s %s %d %s : All Shared Indices good on rank %d\n", file, func,
    line, eqSysName_.c_str(), rank_);
  fclose(output_);
  return;
}
#endif

} // namespace nalu
} // namespace sierra
