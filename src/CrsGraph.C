/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <CrsGraphHelpers.h>
#include <CrsGraph.h>
#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <FieldTypeDef.h>
#include <DgInfo.h>
#include <Realm.h>
#include <PeriodicManager.h>
#include <Simulation.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>
#include <utils/StkHelpers.h>
#include <utils/CreateDeviceExpression.h>
#include <ngp_utils/NgpLoopUtils.h>

#include <KokkosInterface.h>

// overset
#include <overset/OversetManager.h>
#include <overset/OversetInfo.h>

#include <stk_util/parallel/CommNeighbors.hpp>
#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/environment/WallTime.hpp>
#include <stk_util/util/SortAndUnique.hpp>

#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_topology/topology.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

// For Tpetra support
#include <Kokkos_Serial.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Details_shortSort.hpp>
#include <Tpetra_Details_makeOptimizedColMap.hpp>

#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_FancyOStream.hpp>

#include <set>
#include <limits>
#include <type_traits>

#include <sstream>
#define KK_MAP
namespace sierra{
namespace nalu{


///=========================================================================
///======== T P E T R A ====================================================
///=========================================================================

//==========================================================================
// Class Definition
//==========================================================================
CrsGraph::CrsGraph(
  Realm &realm,
  const unsigned numDof)
  : realm_(realm), numDof_(numDof), inConstruction_(false), isFinalized_(false)
{}

CrsGraph::~CrsGraph() {}

struct CompareEntityEqualById
{
  const stk::mesh::BulkData &m_mesh;
  const GlobalIdFieldType *m_naluGlobalId;

  CompareEntityEqualById(
    const stk::mesh::BulkData &mesh, const GlobalIdFieldType *naluGlobalId)
    : m_mesh(mesh),
      m_naluGlobalId(naluGlobalId) {}

  bool operator() (const stk::mesh::Entity& e0, const stk::mesh::Entity& e1)
  {
    const stk::mesh::EntityId e0Id = *stk::mesh::field_data(*m_naluGlobalId, e0);
    const stk::mesh::EntityId e1Id = *stk::mesh::field_data(*m_naluGlobalId, e1);
    return e0Id == e1Id ;
  }
};

struct CompareEntityById
{
  const stk::mesh::BulkData &m_mesh;
  const GlobalIdFieldType *m_naluGlobalId;

  CompareEntityById(
    const stk::mesh::BulkData &mesh, const GlobalIdFieldType *naluGlobalId)
    : m_mesh(mesh),
      m_naluGlobalId(naluGlobalId) {}

  bool operator() (const stk::mesh::Entity& e0, const stk::mesh::Entity& e1)
  {
    const stk::mesh::EntityId e0Id = *stk::mesh::field_data(*m_naluGlobalId, e0);
    const stk::mesh::EntityId e1Id = *stk::mesh::field_data(*m_naluGlobalId, e1);
    return e0Id < e1Id ;
  }
  bool operator() (const Connection& c0, const Connection& c1)
  {
    const stk::mesh::EntityId c0firstId = *stk::mesh::field_data(*m_naluGlobalId, c0.first);
    const stk::mesh::EntityId c1firstId = *stk::mesh::field_data(*m_naluGlobalId, c1.first);
    if (c0firstId != c1firstId) {
      return c0firstId < c1firstId;
    }
    const stk::mesh::EntityId c0secondId = *stk::mesh::field_data(*m_naluGlobalId, c0.second);
    const stk::mesh::EntityId c1secondId = *stk::mesh::field_data(*m_naluGlobalId, c1.second);
    return c0secondId < c1secondId;
  }
};

// determines whether the node is to be put into which map/graph/matrix
// FIXME - note that the DOFStatus enum can be Or'd together if need be to
//   distinguish ever more complicated situations, for example, a DOF that
//   is both owned and ghosted: OwnedDOF | GhostedDOF
int CrsGraph::getDofStatus(stk::mesh::Entity node)
{
    return getDofStatus_impl(node, realm_);
}

void CrsGraph::beginConstruction()
{
  if(inConstruction_) return;

  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::beginConstruction"));
  inConstruction_ = true;
  ThrowRequire(ownedGraph_.is_null());
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  // create a localID for all active nodes in the mesh...
  const stk::mesh::Selector s_universal = metaData.universal_part()
      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
      realm_.get_buckets( stk::topology::NODE_RANK, s_universal );

  // we allow for ghosted nodes when nonconformal is active. When periodic is active, we may
  // also have ghosted nodes due to the periodicGhosting. However, we want to exclude these
  // nodes

  LocalOrdinal numGhostNodes = 0;
  LocalOrdinal numOwnedNodes = 0;
  LocalOrdinal numNodes = 0;
  LocalOrdinal numSharedNotOwnedNotLocallyOwned = 0; // these are nodes on other procs
  // First, get the number of owned and sharedNotOwned (or num_sharedNotOwned_nodes = num_nodes - num_owned_nodes)
  //KOKKOS: BucketLoop parallel "reduce" is accumulating 4 sums
  kokkos_parallel_for("Nalu::CrsGraph::beginConstructionA", buckets.size(), [&] (const int& ib) {
    stk::mesh::Bucket & b = *buckets[ib];
    const stk::mesh::Bucket::size_type length = b.size();
    //KOKKOS: intra BucketLoop parallel reduce
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get node
      stk::mesh::Entity node = b[k];
      int status = getDofStatus(node);

      if (status & DS_SkippedDOF)
        continue;

      if (status & DS_OwnedDOF) {
        numNodes++;
        numOwnedNodes++;
      }

      if (status & DS_SharedNotOwnedDOF) {
        numNodes++;
        numSharedNotOwnedNotLocallyOwned++;
      }

      if (status & DS_GhostedDOF) {
        numGhostNodes++;
      }
    }
  });

  maxOwnedRowId_ = numOwnedNodes * numDof_;
  maxSharedNotOwnedRowId_ = numNodes * numDof_;

  // Next, grab all the global ids, owned first, then sharedNotOwned.

  // Also, we'll build up our own local id map. Note: first we number
  // the owned nodes then we number the sharedNotOwned nodes.

  // make separate arrays that hold the owned and sharedNotOwned gids
  std::vector<stk::mesh::Entity> owned_nodes, shared_not_owned_nodes;
  owned_nodes.reserve(numOwnedNodes);
  shared_not_owned_nodes.reserve(numSharedNotOwnedNotLocallyOwned);

  std::vector<GlobalOrdinal> ownedGids, sharedNotOwnedGids;
  ownedGids.reserve(maxOwnedRowId_);
  sharedNotOwnedGids.reserve(numSharedNotOwnedNotLocallyOwned*numDof_);
  sharedPids_.reserve(sharedNotOwnedGids.capacity());

  // owned first:
  for(const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket & b = *bptr;
    for ( stk::mesh::Entity entity : b ) {
      int status = getDofStatus(entity);
      if (!(status & DS_SkippedDOF) && (status & DS_OwnedDOF))
        owned_nodes.push_back(entity);
    }
  }

  std::sort(owned_nodes.begin(), owned_nodes.end(), CompareEntityById(bulkData, realm_.naluGlobalId_) );

  // use the Contiguous Map constructor. 

   const Teuchos::RCP<LinSys::Comm> tpetraComm = Teuchos::rcp(new LinSys::Comm(bulkData.parallel()));
  ownedRowsMap_ = Teuchos::rcp(new LinSys::Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                               maxOwnedRowId_, // must have *numDof_ 
                                               1,
                                               tpetraComm));
  myLIDs_.clear();
  myLIDs_.reserve(maxOwnedRowId_ +  numSharedNotOwnedNotLocallyOwned*numDof_);

  LocalOrdinal localId = 0;
  int gstart = 0;

  GlobalOrdinal gomin   = ownedRowsMap_->getMinGlobalIndex();

  for(stk::mesh::Entity entity : owned_nodes) {
    const stk::mesh::EntityId entityId = *stk::mesh::field_data(*realm_.naluGlobalId_, entity);
    myLIDs_[entityId] = numDof_*localId++;
    auto *  thisgid = stk::mesh::field_data(*realm_.tpetGlobalId_, entity);
    auto basegid = gomin + numDof_ * gstart;
    (*thisgid) = basegid;
    for(unsigned idof=0; idof < numDof_; ++ idof) ownedGids.push_back(basegid + idof);
    gstart++;
  }
  ThrowRequire(localId == numOwnedNodes);
  // communicate the newly stored GID's.

  std::vector<const stk::mesh::FieldBase*> fVec{realm_.tpetGlobalId_};
  stk::mesh::copy_owned_to_shared(bulkData, fVec);
  stk::mesh::communicate_field_data(bulkData.aura_ghosting(), fVec);
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
    realm_.periodicManager_->parallel_communicate_field(realm_.tpetGlobalId_);
    realm_.periodicManager_->periodic_parallel_communicate_field(realm_.tpetGlobalId_);
  }

  // now sharedNotOwned:
  for(const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket & b = *bptr;
    for ( stk::mesh::Entity node : b) {
      int status = getDofStatus(node);
      if (!(status & DS_SkippedDOF) && (status & DS_SharedNotOwnedDOF))
        shared_not_owned_nodes.push_back(node);
    }
  }
  std::sort(shared_not_owned_nodes.begin(), shared_not_owned_nodes.end(), CompareEntityById(bulkData, realm_.naluGlobalId_) );
  std::vector<stk::mesh::Entity>::iterator iter = std::unique(shared_not_owned_nodes.begin(), shared_not_owned_nodes.end(), CompareEntityEqualById(bulkData, realm_.naluGlobalId_));
  shared_not_owned_nodes.erase(iter, shared_not_owned_nodes.end());

  for (unsigned inode=0; inode < shared_not_owned_nodes.size(); ++inode) {
    stk::mesh::Entity entity = shared_not_owned_nodes[inode];
    const stk::mesh::EntityId naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, entity);
    auto masterentity = get_entity_master(bulkData, entity, naluId);
    myLIDs_[naluId] = numDof_*localId++;
    int owner = bulkData.parallel_owner_rank(masterentity);
    auto basegid = *stk::mesh::field_data(*realm_.tpetGlobalId_, masterentity);

    if(entity != masterentity) 
      *stk::mesh::field_data(*realm_.tpetGlobalId_, entity) = basegid;
    
    for(unsigned idof=0; idof < numDof_; ++ idof) {
      GlobalOrdinal gid = basegid+idof;
      sharedNotOwnedGids.push_back(gid);
      sharedPids_.push_back(owner);
    }
  }

  sharedNotOwnedRowsMap_ = Teuchos::rcp(new LinSys::Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), sharedNotOwnedGids, 1, tpetraComm));

  exporter_ = Teuchos::rcp(new LinSys::Export(sharedNotOwnedRowsMap_, ownedRowsMap_));

  fill_entity_to_row_LID_mapping();
  ownedAndSharedNodes_.reserve(owned_nodes.size()+shared_not_owned_nodes.size());
  ownedAndSharedNodes_ = owned_nodes;
  ownedAndSharedNodes_.insert(ownedAndSharedNodes_.end(), shared_not_owned_nodes.begin(), shared_not_owned_nodes.end());
  connections_.resize(ownedAndSharedNodes_.size());
  for(std::vector<stk::mesh::Entity>& vec : connections_) { vec.reserve(8); }
} //beginConstruction

int CrsGraph::insert_connection(stk::mesh::Entity a, stk::mesh::Entity b)
{
    size_t idx = entityToLID_[a.local_offset()]/numDof_;

    ThrowRequireMsg(idx < ownedAndSharedNodes_.size(),"Error, insert_connection got index out of range.");

    bool correctEntity = ownedAndSharedNodes_[idx] == a;
    if (!correctEntity) {
      const stk::mesh::EntityId naluid_a = *stk::mesh::field_data(*realm_.naluGlobalId_, a);
      stk::mesh::Entity master = get_entity_master(realm_.bulk_data(), a, naluid_a);
      const stk::mesh::EntityId naluid_master = *stk::mesh::field_data(*realm_.naluGlobalId_, master);
      correctEntity = ownedAndSharedNodes_[idx] == master || naluid_a == naluid_master;
    }
    ThrowRequireMsg(correctEntity,"Error, indexing of rowEntities to connections isn't right.");

    std::vector<stk::mesh::Entity>& vec = connections_[idx];
    if (std::find(vec.begin(), vec.end(), b) == vec.end()) {
        vec.push_back(b);
    }
    return 0;
}

void CrsGraph::addConnections(const stk::mesh::Entity* entities, const size_t& num_entities)
{
  for(size_t a=0; a < num_entities; ++a) {
    const stk::mesh::Entity entity_a = entities[a];
    const stk::mesh::EntityId id_a = *stk::mesh::field_data(*realm_.naluGlobalId_, entity_a);
    insert_connection(entity_a, entity_a);

    for(size_t b=a+1; b < num_entities; ++b) {
      const stk::mesh::Entity entity_b = entities[b];
      const stk::mesh::EntityId id_b = *stk::mesh::field_data(*realm_.naluGlobalId_, entity_b);
      const bool a_then_b = id_a < id_b;
      const stk::mesh::Entity entity_min = a_then_b ? entity_a : entity_b;
      const stk::mesh::Entity entity_max = a_then_b ? entity_b : entity_a;
      insert_connection(entity_min, entity_max);
    }
  }
}

void CrsGraph::buildNodeGraph(const stk::mesh::PartVector & parts)
{
  beginConstruction();
  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::buildNodeGraph"));
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_owned );
  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity node = b[k];
      addConnections(&node, 1);
    }
  }
}

void CrsGraph::buildConnectedNodeGraph(stk::mesh::EntityRank rank,
                                                 const stk::mesh::PartVector& parts)
{
  beginConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets( rank, s_owned );

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const unsigned numNodes = b.num_nodes(k);
      stk::mesh::Entity const * nodes = b.begin_nodes(k);

      addConnections(nodes, numNodes);
    }
  }
}

void CrsGraph::buildEdgeToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginConstruction();
  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::buildEdgeToNodeGraph"));
  buildConnectedNodeGraph(stk::topology::EDGE_RANK, parts);
}

void CrsGraph::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginConstruction();
  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::buildFaceToNodeGraph"));
  stk::mesh::MetaData & metaData = realm_.meta_data();
  buildConnectedNodeGraph(metaData.side_rank(), parts);
}

void CrsGraph::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginConstruction();
  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::buildElemToNodeGraph"));
  buildConnectedNodeGraph(stk::topology::ELEM_RANK, parts);
}

void CrsGraph::buildReducedElemToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginConstruction();
  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::buildReducedElemToNodeGraph"));
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_owned );
  std::vector<stk::mesh::Entity> entities;
  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];

    // extract master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    // extract master element specifics
    const int numScsIp = meSCS->num_integration_points();
    const int *lrscv = meSCS->adjacentNodes();

    const stk::mesh::Bucket::size_type length   = b.size();
    //KOKKOS: intra BucketLoop noparallel addConnections insert (std::set)
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity const * elem_nodes = b.begin_nodes(k);

      const size_t numNodes = 2;
      entities.resize(numNodes);
      //KOKKOS: nested Loop noparallel addConnections insert (std::set)
      for (int j = 0; j < numScsIp; ++j){
        //KOKKOS: nested Loop parallel
        for(size_t n=0; n < numNodes; ++n) {
          entities[n] = elem_nodes[lrscv[2*j+n]];
        }
        addConnections(entities.data(), entities.size());
      }
    }
  }
}

void CrsGraph::buildFaceElemToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginConstruction();
  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::buildFaceElemToNodeGraph"));
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( metaData.side_rank(), s_owned );

  for(size_t ib=0; ib<face_buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *face_buckets[ib];
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const stk::mesh::Entity face = b[k];

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
      ThrowAssert( bulkData.num_elements(face) == 1 );

      // get connected element and nodal relations
      stk::mesh::Entity element = face_elem_rels[0];
      const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);

      // figure out the global dof ids for each dof on each node
      const size_t numNodes = bulkData.num_nodes(element);
      addConnections(elem_nodes, numNodes);
    }
  }
} //buildFaceElemToNodeGraph

void CrsGraph::buildNonConformalNodeGraph(const stk::mesh::PartVector & /* parts */)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  beginConstruction();
  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::buildNonConformalNodeGraph"));

  std::vector<stk::mesh::Entity> entities;

  // iterate nonConformalManager's dgInfoVecs
  for( NonConformalInfo * nonConfInfo : realm_.nonConformalManager_->nonConformalInfoVec_) {

    std::vector<std::vector<DgInfo*> >& dgInfoVec = nonConfInfo->dgInfoVec_;

    for( std::vector<DgInfo*>& faceDgInfoVec : dgInfoVec ) {

      // now loop over all the DgInfo objects on this particular exposed face
      for ( size_t k = 0; k < faceDgInfoVec.size(); ++k ) {

        DgInfo *dgInfo = faceDgInfoVec[k];

        // extract current/opposing element
        stk::mesh::Entity currentElement = dgInfo->currentElement_;
        stk::mesh::Entity opposingElement = dgInfo->opposingElement_;

        // node relations; current and opposing
        stk::mesh::Entity const* current_elem_node_rels = bulkData.begin_nodes(currentElement);
        const int current_num_elem_nodes = bulkData.num_nodes(currentElement);
        stk::mesh::Entity const* opposing_elem_node_rels = bulkData.begin_nodes(opposingElement);
        const int opposing_num_elem_nodes = bulkData.num_nodes(opposingElement);

        // resize based on both current and opposing face node size
        entities.resize(current_num_elem_nodes+opposing_num_elem_nodes);

        // fill in connected nodes; current
        //KOKKOS: nested Loop parallel
        for ( int ni = 0; ni < current_num_elem_nodes; ++ni ) {
          entities[ni] = current_elem_node_rels[ni];
        }

        // fill in connected nodes; opposing
        //KOKKOS: nested Loop parallel
        for ( int ni = 0; ni < opposing_num_elem_nodes; ++ni ) {
          entities[current_num_elem_nodes+ni] = opposing_elem_node_rels[ni];
        }

        // okay, now add the connections; will be symmetric
        // columns of current node row (opposing nodes) will add columns to opposing nodes row
        addConnections(entities.data(), entities.size());
      }
    }
  }
} //buildNonConformalNodeGraph

void CrsGraph::buildOversetNodeGraph(const stk::mesh::PartVector & /* parts */)
{
  // extract the rank
  const int theRank = NaluEnv::self().parallel_rank();

  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  beginConstruction();
  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::buildOversetNodeGraph"));

  std::vector<stk::mesh::Entity> entities;

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

    entities[0] = orphanNode;
    for(size_t n=0; n < numNodes; ++n) {
      entities[n+1] = elem_nodes[n];
    }
    addConnections(entities.data(), entities.size());
  }
} //buildOversetNodeGraph

void CrsGraph::compute_send_lengths(const std::vector<stk::mesh::Entity>& rowEntities,
                                              const std::vector<std::vector<stk::mesh::Entity> >& connections,
                                              const std::vector<int>& neighborProcs,
                                              stk::CommNeighbors& commNeighbors)
{
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  std::vector<int> sendLengths(neighborProcs.size(), 0);
  size_t maxColEntities = 128;
  std::vector<stk::mesh::EntityId> colEntityIds(maxColEntities);

  for(size_t i=0; i<rowEntities.size(); ++i)
  {
    const stk::mesh::Entity entity_a = rowEntities[i];
    const std::vector<stk::mesh::Entity>& colEntities = connections[i];
    unsigned numColEntities = colEntities.size();
    colEntityIds.resize(numColEntities);
    for(size_t j=0; j<colEntities.size(); ++j) {
      colEntityIds[j] = *stk::mesh::field_data(*realm_.naluGlobalId_, colEntities[j]);
    }

    const stk::mesh::EntityId entityId_a = *stk::mesh::field_data(*realm_.naluGlobalId_, entity_a);
    const int entity_a_status = getDofStatus(entity_a);
    const bool entity_a_shared = entity_a_status & DS_SharedNotOwnedDOF;

    if (entity_a_shared) {
        stk::mesh::Entity master = get_entity_master(bulk, entity_a, entityId_a);
        size_t idx = get_neighbor_index(neighborProcs, bulk.parallel_owner_rank(master));
        sendLengths[idx] += (1+numColEntities)*(sizeof(GlobalOrdinal)+sizeof(int));
    }

    for(size_t ii=0; ii<numColEntities; ++ii) {
        const stk::mesh::Entity entity_b = colEntities[ii];
        if (entity_b == entity_a) {
            continue;
        }
        const stk::mesh::EntityId entityId_b = colEntityIds[ii];
        const int entity_b_status = (entityId_a != entityId_b) ? getDofStatus(entity_b) : entity_a_status;
        const bool entity_b_shared = entity_b_status & DS_SharedNotOwnedDOF;
        if (entity_b_shared) {
            stk::mesh::Entity master = get_entity_master(bulk, entity_b, entityId_b);
            size_t idx = get_neighbor_index(neighborProcs, bulk.parallel_owner_rank(master));
            sendLengths[idx] += (1+numColEntities)*(sizeof(GlobalOrdinal)+sizeof(int));
        }
    }
  }

  for(size_t i=0; i<neighborProcs.size(); ++i) {
    stk::CommBufferV& sbuf = commNeighbors.send_buffer(neighborProcs[i]);
    sbuf.reserve(sendLengths[i]);
  }
} //compute_send_lengths

void CrsGraph::compute_graph_row_lengths(const std::vector<stk::mesh::Entity>& rowEntities,
                                                   const std::vector<std::vector<stk::mesh::Entity> >& connections,
                                                   LinSys::RowLengths& sharedNotOwnedRowLengths,
                                                   LinSys::RowLengths& locallyOwnedRowLengths,
                                                   stk::CommNeighbors& commNeighbors)
{
  LinSys::DeviceRowLengths deviceSharedNotOwnedRowLengths = sharedNotOwnedRowLengths.view<DeviceSpace>();
  LinSys::DeviceRowLengths deviceLocallyOwnedRowLengths = locallyOwnedRowLengths.view<DeviceSpace>();

  const stk::mesh::BulkData& bulk = realm_.bulk_data();

  size_t maxColEntities = 128;
  std::vector<stk::mesh::EntityId> colEntityIds(maxColEntities);
  std::vector<int> colOwners(maxColEntities);

  for(size_t i=0; i<rowEntities.size(); ++i)
  {
    const std::vector<stk::mesh::Entity>& colEntities = connections[i];
    unsigned numColEntities = colEntities.size();
    const stk::mesh::Entity entity_a = rowEntities[i];
    colEntityIds.resize(numColEntities);
    colOwners.resize(numColEntities);
    for(size_t j=0; j<numColEntities; ++j) {
        stk::mesh::Entity colEntity = colEntities[j];
        colEntityIds[j] = *stk::mesh::field_data(*realm_.naluGlobalId_, colEntity);
        colOwners[j] = bulk.parallel_owner_rank(get_entity_master(bulk, colEntity, colEntityIds[j]));
    }

    const stk::mesh::EntityId entityId_a = *stk::mesh::field_data(*realm_.naluGlobalId_, entity_a);

    const int entity_a_status = getDofStatus(entity_a);
    const bool entity_a_owned = entity_a_status & DS_OwnedDOF;
    LocalOrdinal lid_a = entityToLID_[entity_a.local_offset()];
    stk::mesh::Entity entity_a_master = get_entity_master(bulk, entity_a, entityId_a);
    int entity_a_owner = bulk.parallel_owner_rank(entity_a_master);

    add_to_length(deviceLocallyOwnedRowLengths, deviceSharedNotOwnedRowLengths, numDof_, lid_a, maxOwnedRowId_,
                  entity_a_owned, numColEntities);

    const bool entity_a_shared = entity_a_status & DS_SharedNotOwnedDOF;
    if (entity_a_shared) {
      add_lengths_to_comm_tpet(bulk, realm_.tpetGlobalId_,commNeighbors, entity_a_owner, entityId_a,
                               //                               numDof_, 
                               numColEntities, colEntityIds.data(), colOwners.data());
    }

    for(size_t ii=0; ii<numColEntities; ++ii) {
        const stk::mesh::Entity entity_b = colEntities[ii];
        if (entity_b == entity_a) {
            continue;
        }
        const stk::mesh::EntityId entityId_b = colEntityIds[ii];
        const int entity_b_status = getDofStatus(entity_b);
        const bool entity_b_owned = entity_b_status & DS_OwnedDOF;
        LocalOrdinal lid_b = entityToLID_[entity_b.local_offset()];
        add_to_length(deviceLocallyOwnedRowLengths, deviceSharedNotOwnedRowLengths, numDof_, lid_b, maxOwnedRowId_, entity_b_owned, 1);

        const bool entity_b_shared = entity_b_status & DS_SharedNotOwnedDOF;
        if (entity_b_shared) {
            add_lengths_to_comm_tpet(bulk, realm_.tpetGlobalId_, commNeighbors, colOwners[ii], entityId_b, 
                                     // numDof_, 
                                     1, &entityId_a, &entity_a_owner);
        }
    }
  }
} //compute_graph_row_lengths

void CrsGraph::insert_graph_connections(const std::vector<stk::mesh::Entity>& rowEntities,
                                                  const std::vector<std::vector<stk::mesh::Entity> >& connections,
                                                  LocalGraphArrays& locallyOwnedGraph,
                                                  LocalGraphArrays& sharedNotOwnedGraph)
{
  std::vector<LocalOrdinal> localDofs_a(1);
  unsigned max = 128;
  std::vector<int> dofStatus(max);
  std::vector<LocalOrdinal> localDofs_b(max);

  //KOKKOS: Loop noparallel Graph insert
  for(size_t i=0; i<rowEntities.size(); ++i) {
    const std::vector<stk::mesh::Entity>& entities_b = connections[i];
    unsigned numColEntities = entities_b.size();
    dofStatus.resize(numColEntities);
    localDofs_b.resize(numColEntities);

    const stk::mesh::Entity entity_a = rowEntities[i];
    int dofStatus_a = getDofStatus(entity_a);
    ThrowRequireMsg(entityToColLID_[entity_a.local_offset()] != -1 ,  "insert_graph_connections bad lid ");
    localDofs_a[0] = entityToColLID_[entity_a.local_offset()];
    for(size_t j=0; j<numColEntities; ++j) {
      const stk::mesh::Entity entity_b = entities_b[j];
      dofStatus[j] = getDofStatus(entity_b);

      ThrowRequireMsg(entityToColLID_[entity_b.local_offset()] != -1 , "insert_graph_connections bad lid #2 ");

      localDofs_b[j] = entityToColLID_[entity_b.local_offset()];
    }

    {
      LocalGraphArrays& crsGraph = (dofStatus_a & DS_OwnedDOF) ? locallyOwnedGraph : sharedNotOwnedGraph;
      insert_single_dof_row_into_graph(crsGraph, entityToLID_[entity_a.local_offset()], maxOwnedRowId_, numDof_, numColEntities, localDofs_b);
    }

    for(unsigned j=0; j<numColEntities; ++j) {
      if (entities_b[j] != entity_a) {
        LocalGraphArrays& crsGraph = (dofStatus[j] & DS_OwnedDOF) ? locallyOwnedGraph : sharedNotOwnedGraph;
        insert_single_dof_row_into_graph(crsGraph, entityToLID_[entities_b[j].local_offset()], maxOwnedRowId_, numDof_, 1, localDofs_a);
      }
    }
  }
} //insert_graph_connections

void CrsGraph::fill_entity_to_row_LID_mapping()
{
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  stk::mesh::Selector selector = bulk.mesh_meta_data().universal_part() & !(realm_.get_inactive_selector());
  entityToLID_ = LinSys::EntityToLIDView("entityToLID",bulk.get_size_of_entity_index_space());
  const stk::mesh::BucketVector& nodeBuckets = realm_.get_buckets(stk::topology::NODE_RANK, selector);
  for(const stk::mesh::Bucket* bptr : nodeBuckets) {
    const stk::mesh::Bucket& b = *bptr;
    const stk::mesh::EntityId* nodeIds = stk::mesh::field_data(*realm_.naluGlobalId_, b);
    for(size_t i=0; i<b.size(); ++i) {
      stk::mesh::Entity node = b[i];

      MyLIDMapType::const_iterator iter = myLIDs_.find(nodeIds[i]);
      if (iter != myLIDs_.end()) {
        entityToLID_[node.local_offset()] = iter->second;
        if (nodeIds[i] != bulk.identifier(node)) {
          stk::mesh::Entity master = get_entity_master(bulk, node, nodeIds[i]);
          if (master != node) {
            entityToLID_[master.local_offset()] = entityToLID_[node.local_offset()];
          }
        }
      }
    }
  }
} //fill_entity_to_row_LID_mapping

void CrsGraph::fill_entity_to_col_LID_mapping()
{
    const stk::mesh::BulkData& bulk = realm_.bulk_data();
    stk::mesh::Selector selector = bulk.mesh_meta_data().universal_part() & !(realm_.get_inactive_selector());
    entityToColLID_ = LinSys::EntityToLIDView("entityToLID",bulk.get_size_of_entity_index_space());
    const stk::mesh::BucketVector& nodeBuckets = realm_.get_buckets(stk::topology::NODE_RANK,selector);
    const bool throwIfMasterNotFound = false;
    for(const stk::mesh::Bucket* bptr : nodeBuckets) {
        const stk::mesh::Bucket& b = *bptr;
        const stk::mesh::EntityId* nodeIds = stk::mesh::field_data(*realm_.naluGlobalId_, b);
        for(size_t i=0; i<b.size(); ++i) {
            stk::mesh::Entity node = b[i];

            GlobalOrdinal gid =-1;
            // needed because of some shared and ghost nodes. 
            if (nodeIds[i] != bulk.identifier(node)) {
              stk::mesh::Entity master = get_entity_master(bulk, node, nodeIds[i],
                                                           throwIfMasterNotFound);
              if (bulk.is_valid(master) && master != node) {
                gid = * stk::mesh::field_data(*realm_.tpetGlobalId_, master);
                entityToColLID_[node.local_offset()] = totalColsMap_->getLocalElement(gid);
              }
              else if (!bulk.is_valid(master)) {
                gid = -1;
                entityToColLID_[node.local_offset()] = gid;
              }
              else {
                gid =  * stk::mesh::field_data(*realm_.tpetGlobalId_, node);
              }
            }
            else
              gid =  * stk::mesh::field_data(*realm_.tpetGlobalId_, node);

            if(gid == 0 || gid == -1 || gid == std::numeric_limits<LinSys::GlobalOrdinal>::max() ) {
	      // unit_test1 does produce ghost nodes that have a master that don't have a valid tpetGlobalId_
              entityToColLID_[node.local_offset()] = -1;
            }
            else
              entityToColLID_[node.local_offset()] = totalColsMap_->getLocalElement(gid);
        }
    }
} //fill_entity_to_col_LID_mapping

void CrsGraph::storeOwnersForShared()
{
  const stk::mesh::BulkData & bulkData = realm_.bulk_data();
  const stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector all = metaData.universal_part() & !(realm_.get_inactive_selector());
  const stk::mesh::BucketVector& buckets = realm_.get_buckets( stk::topology::NODE_RANK, all );

  for(const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket& bkt = *bptr;
    for(stk::mesh::Entity node : bkt) {
      int status = getDofStatus(node);
      if (status & DS_SharedNotOwnedDOF) {
        stk::mesh::EntityId naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
        stk::mesh::Entity master = get_entity_master(bulkData, node, naluId);
        GlobalOrdinal gidbase = * stk::mesh::field_data(*realm_.tpetGlobalId_, master);
        ThrowRequire(gidbase != 0);
        for(unsigned idof=0; idof < numDof_; ++ idof) {
          GlobalOrdinal gid = gidbase+idof;
          ownersAndGids_.insert(std::make_pair(bulkData.parallel_owner_rank(master), gid));
        }
      }
    }
  }
} //storeOwnersForShared

void CrsGraph::finalizeGraph()
{
  if (isFinalized_) return;
  isFinalized_ = true;

  const Teuchos::TimeMonitor timeMon(
    *Teuchos::TimeMonitor::getNewCounter("CrsGraph::finalizeGraph"));
  stk::mesh::BulkData & bulkData = realm_.bulk_data();

  sort_connections(connections_);

  size_t numSharedNotOwned = sharedNotOwnedRowsMap_->getMyGlobalIndices().extent(0);
  size_t numLocallyOwned = ownedRowsMap_->getMyGlobalIndices().extent(0);
  LinSys::RowLengths sharedNotOwnedRowLengths("rowLengths", numSharedNotOwned);
  LinSys::RowLengths locallyOwnedRowLengths("rowLengths", numLocallyOwned);
  LinSys::DeviceRowLengths ownedRowLengths = locallyOwnedRowLengths.view<DeviceSpace>();
  LinSys::DeviceRowLengths globalRowLengths = sharedNotOwnedRowLengths.view<DeviceSpace>();

  std::vector<int> neighborProcs;
  fill_neighbor_procs(neighborProcs, bulkData, realm_);

  stk::CommNeighbors commNeighbors(bulkData.parallel(), neighborProcs);

  compute_send_lengths(ownedAndSharedNodes_, connections_, neighborProcs, commNeighbors);
  compute_graph_row_lengths(ownedAndSharedNodes_, connections_, sharedNotOwnedRowLengths, locallyOwnedRowLengths, commNeighbors);

  ownersAndGids_.clear();
  storeOwnersForShared();

  communicate_remote_columns(bulkData, neighborProcs, commNeighbors, numDof_, ownedRowsMap_, ownedRowLengths, ownersAndGids_);

  LocalGraphArrays ownedGraph(ownedRowLengths);
  LocalGraphArrays sharedNotOwnedGraph(globalRowLengths);

  int localProc = bulkData.parallel_rank();

  std::vector<GlobalOrdinal> optColGids;
  std::vector<int> sourcePIDs;
  fill_owned_and_shared_then_nonowned_ordered_by_proc(optColGids, sourcePIDs, localProc, ownedRowsMap_, sharedNotOwnedRowsMap_, ownersAndGids_, sharedPids_);

  const Teuchos::RCP<LinSys::Comm> tpetraComm = Teuchos::rcp(new LinSys::Comm(bulkData.parallel()));
  totalColsMap_ = Teuchos::rcp(new LinSys::Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), optColGids, 1, tpetraComm));

  fill_entity_to_col_LID_mapping();

  insert_graph_connections(ownedAndSharedNodes_, connections_, ownedGraph, sharedNotOwnedGraph);

  insert_communicated_col_indices(neighborProcs, commNeighbors, numDof_, ownedGraph, *ownedRowsMap_, *totalColsMap_);

  fill_in_extra_dof_rows_per_node(ownedGraph, numDof_);
  fill_in_extra_dof_rows_per_node(sharedNotOwnedGraph, numDof_);

  remove_invalid_indices(ownedGraph, ownedRowLengths);

  sharedNotOwnedGraph_ = Teuchos::rcp(new LinSys::Graph(sharedNotOwnedRowsMap_, totalColsMap_, sharedNotOwnedRowLengths, Tpetra::StaticProfile));

  ownedGraph_ = Teuchos::rcp(new LinSys::Graph(ownedRowsMap_, totalColsMap_, locallyOwnedRowLengths, Tpetra::StaticProfile));

  ownedGraph_->setAllIndices(ownedGraph.rowPointers, ownedGraph.colIndices);
  sharedNotOwnedGraph_->setAllIndices(sharedNotOwnedGraph.rowPointers, sharedNotOwnedGraph.colIndices);

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList);
  params->set<bool>("No Nonlocal Changes", true);
  params->set<bool>("compute local triangular constants", false);

  bool allowedToReorderLocally = false;
  Teuchos::RCP<LinSys::Import> importer = Teuchos::rcp(new LinSys::Import(ownedRowsMap_, optColGids.data()+ownedRowLengths.size(), sourcePIDs.data(), sourcePIDs.size(), allowedToReorderLocally));

  ownedGraph_->expertStaticFillComplete(ownedRowsMap_, ownedRowsMap_, importer, Teuchos::null, params);
  sharedNotOwnedGraph_->expertStaticFillComplete(ownedRowsMap_, ownedRowsMap_, Teuchos::null, Teuchos::null, params);

} //finalizeGraph

int getDofStatus_impl(stk::mesh::Entity node, const Realm& realm)
{
  const stk::mesh::BulkData & bulkData = realm.bulk_data();

  const stk::mesh::Bucket & b = bulkData.bucket(node);
  const bool entityIsOwned = b.owned();
  const bool entityIsShared = b.shared();
  const bool entityIsGhosted = !entityIsOwned && !entityIsShared;

  bool has_non_matching_boundary_face_alg = realm.has_non_matching_boundary_face_alg();
  bool hasPeriodic = realm.hasPeriodic_;

  if (realm.hasPeriodic_ && realm.has_non_matching_boundary_face_alg()) {
    has_non_matching_boundary_face_alg = false;
    hasPeriodic = false;

    stk::mesh::Selector perSel = stk::mesh::selectUnion(realm.allPeriodicInteractingParts_);
    stk::mesh::Selector nonConfSel = stk::mesh::selectUnion(realm.allNonConformalInteractingParts_);
    //std::cout << "nonConfSel= " << nonConfSel << std::endl;

    for (auto part : b.supersets()) {
      if (perSel(*part)) {
        hasPeriodic = true;
      }
      if (nonConfSel(*part)) {
        has_non_matching_boundary_face_alg = true;
      }
    }
  }

  if (has_non_matching_boundary_face_alg && hasPeriodic) {
    std::ostringstream ostr;
    ostr << "node id= " << realm.bulkData_->identifier(node);
    throw std::logic_error("not ready for primetime to combine periodic and non-matching algorithm on same node: "+ostr.str());
  }

  // simple case
  if (!hasPeriodic && !has_non_matching_boundary_face_alg) {
    if (entityIsGhosted)
      return DS_GhostedDOF;
    if (entityIsOwned)
      return DS_OwnedDOF;
    if (entityIsShared && !entityIsOwned)
      return DS_SharedNotOwnedDOF;
  }

  if (has_non_matching_boundary_face_alg) {
    if (entityIsOwned)
      return DS_OwnedDOF;
    if (!entityIsOwned && (entityIsGhosted || entityIsShared)){
      return DS_SharedNotOwnedDOF;
    }
    // maybe return DS_GhostedDOF if entityIsGhosted
  }

  if (hasPeriodic) {
    const stk::mesh::EntityId stkId = bulkData.identifier(node);
    const stk::mesh::EntityId naluId = *stk::mesh::field_data(*realm.naluGlobalId_, node);

    // bool for type of ownership for this node
    const bool nodeOwned = bulkData.bucket(node).owned();
    const bool nodeShared = bulkData.bucket(node).shared();
    const bool nodeGhosted = !nodeOwned && !nodeShared;

    // really simple here.. ghosted nodes never part of the matrix
    if ( nodeGhosted ) {
      return DS_GhostedDOF;
    }

    // bool to see if this is possibly a periodic node
    const bool isSlaveNode = (stkId != naluId);

    if (!isSlaveNode) {
      if (nodeOwned)
        return DS_OwnedDOF;
      else if( nodeShared )
        return DS_SharedNotOwnedDOF;
      else
        return DS_GhostedDOF;
    }
    else {
      // I am a slave node.... get the master entity
      stk::mesh::Entity masterEntity = bulkData.get_entity(stk::topology::NODE_RANK, naluId);
      if ( bulkData.is_valid(masterEntity)) {
        const bool masterEntityOwned = bulkData.bucket(masterEntity).owned();
        const bool masterEntityShared = bulkData.bucket(masterEntity).shared();
        if (masterEntityOwned)
          return DS_SkippedDOF | DS_OwnedDOF;
        if (masterEntityShared)
          return DS_SkippedDOF | DS_SharedNotOwnedDOF;
        else
          return DS_SharedNotOwnedDOF;
      }
      else {
        return DS_SharedNotOwnedDOF;
      }
    }
  }

  // still got here? problem...
  throw std::logic_error("bad status2");

// Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
  return DS_SkippedDOF;
#endif
}

Teuchos::RCP<GraphTypes::Map>    CrsGraph::getOwnedRowsMap()          const {return ownedRowsMap_;}
Teuchos::RCP<GraphTypes::Graph>  CrsGraph::getOwnedGraph()            const {return ownedGraph_;}
Teuchos::RCP<GraphTypes::Map>    CrsGraph::getSharedNotOwnedRowsMap() const {return sharedNotOwnedRowsMap_;}
Teuchos::RCP<GraphTypes::Graph>  CrsGraph::getSharedNotOwnedGraph()   const {return sharedNotOwnedGraph_;}
Teuchos::RCP<GraphTypes::Export> CrsGraph::getExporter()              const {return exporter_;}

const LinSys::EntityToLIDView & CrsGraph::get_entity_to_row_LID_mapping() const {return entityToLID_;}
const LinSys::EntityToLIDView & CrsGraph::get_entity_to_col_LID_mapping() const {return entityToColLID_;}
const MyLIDMapType            & CrsGraph::get_my_LIDs() const {return myLIDs_;}
CrsGraph::LocalOrdinal          CrsGraph::getMaxOwnedRowID() const {return maxOwnedRowId_;}
CrsGraph::LocalOrdinal          CrsGraph::getMaxSharedNotOwnedRowID() const {return maxSharedNotOwnedRowId_;}

} // namespace nalu
} // namespace Sierra
