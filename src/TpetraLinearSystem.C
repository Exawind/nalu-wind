/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <TpetraLinearSystem.h>
#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <FieldTypeDef.h>
#include <DgInfo.h>
#include <Realm.h>
#include <PeriodicManager.h>
#include <Simulation.h>
#include <LinearSolver.h>
#include <master_element/MasterElement.h>
#include <EquationSystem.h>
#include <NaluEnv.h>
#include <utils/StkHelpers.h>

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
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Details_shortSort.hpp>
#include <Tpetra_Details_makeOptimizedColMap.hpp>

#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_FancyOStream.hpp>

#include <Tpetra_MatrixIO.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include <set>
#include <limits>
#include <type_traits>

#include <sstream>
#define KK_MAP
namespace sierra{
namespace nalu{

#define GID_(gid, ndof, idof)  ((ndof)*((gid)-1)+(idof)+1)
#define LID_(lid, ndof, idof)  ((ndof)*((lid))+(idof))

#define GLOBAL_ENTITY_ID(gid, ndof) ((gid-1)/ndof + 1)
#define GLOBAL_ENTITY_ID_IDOF(gid, ndof) ((gid-1) % ndof)

///====================================================================================================================================
///======== T P E T R A ===============================================================================================================
///====================================================================================================================================

//==========================================================================
// Class Definition
//==========================================================================
// TpetraLinearSystem - hook to Tpetra
//==========================================================================
TpetraLinearSystem::TpetraLinearSystem(
  Realm &realm,
  const unsigned numDof,
  EquationSystem *eqSys,
  LinearSolver * linearSolver)
  : LinearSystem(realm, numDof, eqSys, linearSolver)
{
  Teuchos::ParameterList junk;
  node_ = Teuchos::rcp(new LinSys::Node(junk));
}

TpetraLinearSystem::~TpetraLinearSystem()
{
  // dereference linear solver in safe manner
  TpetraLinearSolver *linearSolver = reinterpret_cast<TpetraLinearSolver *>(linearSolver_);
  linearSolver->destroyLinearSolver();
}

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
int TpetraLinearSystem::getDofStatus(stk::mesh::Entity node)
{
    return getDofStatus_impl(node, realm_);
}

stk::mesh::Entity get_entity_master(const stk::mesh::BulkData& bulk,
                             stk::mesh::Entity entity,
                             stk::mesh::EntityId naluId)
{ 
  bool thisEntityIsMaster = (bulk.identifier(entity) == naluId);
  if (thisEntityIsMaster) {
    return entity;
  }
  stk::mesh::Entity master = bulk.get_entity(stk::topology::NODE_RANK, naluId);
  if (!bulk.is_valid(master)) {
    std::ostringstream os;
    const stk::mesh::Entity* elems = bulk.begin_elements(entity);
    unsigned numElems = bulk.num_elements(entity);
    os<<" elems: ";
    for(unsigned i=0; i<numElems; ++i) {
       os<<"{"<<bulk.identifier(elems[i])<<","<<bulk.bucket(elems[i]).topology()
         <<",owned="<<bulk.bucket(elems[i]).owned()<<"}";
    }
    ThrowRequireMsg(bulk.is_valid(master),
                    "get_entity_master, P"<<bulk.parallel_rank()
                    <<" failed to get entity for naluId="<<naluId
                    <<", from entity with stkId="<<bulk.identifier(entity)
                    <<", owned="<<bulk.bucket(entity).owned()
                    <<", shared="<<bulk.bucket(entity).shared()
                    <<", "<<os.str());
  }
  return master;
}

void
TpetraLinearSystem::beginLinearSystemConstruction()
{
  if(inConstruction_) return;
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
  kokkos_parallel_for("Nalu::TpetraLinearSystem::beginLinearSystemConstructionA", buckets.size(), [&] (const int& ib) {
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
  LocalOrdinal localId = 0;

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
  std::vector<stk::mesh::Entity>::iterator iter = std::unique(owned_nodes.begin(), owned_nodes.end(), CompareEntityEqualById(bulkData, realm_.naluGlobalId_));
  owned_nodes.erase(iter, owned_nodes.end());

  myLIDs_.clear();
  //KOKKOS: Loop noparallel push_back totalGids_ (std::vector)
  for(stk::mesh::Entity entity : owned_nodes) {
    const stk::mesh::EntityId entityId = *stk::mesh::field_data(*realm_.naluGlobalId_, entity);
    myLIDs_[entityId] = numDof_*localId++;
    for(unsigned idof=0; idof < numDof_; ++ idof) {
      const GlobalOrdinal gid = GID_(entityId, numDof_, idof);
      ownedGids.push_back(gid);
    }
  }
  ThrowRequire(localId == numOwnedNodes);
  
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
  iter = std::unique(shared_not_owned_nodes.begin(), shared_not_owned_nodes.end(), CompareEntityEqualById(bulkData, realm_.naluGlobalId_));
  shared_not_owned_nodes.erase(iter, shared_not_owned_nodes.end());

  for (unsigned inode=0; inode < shared_not_owned_nodes.size(); ++inode) {
    stk::mesh::Entity entity = shared_not_owned_nodes[inode];
    const stk::mesh::EntityId naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, entity);
    entity = get_entity_master(bulkData, entity, naluId);
    myLIDs_[naluId] = numDof_*localId++;
    int owner = bulkData.parallel_owner_rank(entity);
    for(unsigned idof=0; idof < numDof_; ++ idof) {
      const GlobalOrdinal gid = GID_(naluId, numDof_, idof);
      sharedNotOwnedGids.push_back(gid);
      sharedPids_.push_back(owner);
    }
  }
  
  const Teuchos::RCP<LinSys::Comm> tpetraComm = Teuchos::rcp(new LinSys::Comm(bulkData.parallel()));
  ownedRowsMap_ = Teuchos::rcp(new LinSys::Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ownedGids, 1, tpetraComm, node_));
  sharedNotOwnedRowsMap_ = Teuchos::rcp(new LinSys::Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), sharedNotOwnedGids, 1, tpetraComm, node_));

  exporter_ = Teuchos::rcp(new LinSys::Export(sharedNotOwnedRowsMap_, ownedRowsMap_));

  fill_entity_to_row_LID_mapping();
  ownedAndSharedNodes_.reserve(owned_nodes.size()+shared_not_owned_nodes.size());
  ownedAndSharedNodes_ = owned_nodes;
  ownedAndSharedNodes_.insert(ownedAndSharedNodes_.end(), shared_not_owned_nodes.begin(), shared_not_owned_nodes.end());
  connections_.resize(ownedAndSharedNodes_.size());
  for(std::vector<stk::mesh::Entity>& vec : connections_) { vec.reserve(8); }
}

int TpetraLinearSystem::insert_connection(stk::mesh::Entity a, stk::mesh::Entity b)
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

void TpetraLinearSystem::addConnections(const stk::mesh::Entity* entities, const size_t& num_entities)
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

void
TpetraLinearSystem::buildNodeGraph(const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
//if (realm_.bulk_data().parallel_rank()==0) std::cerr<<"buildNodeGraph"<<std::endl;

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

void TpetraLinearSystem::buildConnectedNodeGraph(stk::mesh::EntityRank rank,
                                                 const stk::mesh::PartVector& parts)
{
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

void
TpetraLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
  buildConnectedNodeGraph(stk::topology::EDGE_RANK, parts);
}

void
TpetraLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  buildConnectedNodeGraph(metaData.side_rank(), parts);
}

void
TpetraLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
  buildConnectedNodeGraph(stk::topology::ELEM_RANK, parts);
}

void
TpetraLinearSystem::buildReducedElemToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
//if (realm_.bulk_data().parallel_rank()==0) std::cerr<<"buildReducedElemToNodeGraph"<<std::endl;

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
    const int numScsIp = meSCS->numIntPoints_;
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

void
TpetraLinearSystem::buildFaceElemToNodeGraph(const stk::mesh::PartVector & parts)
{
  beginLinearSystemConstruction();
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
}

void
TpetraLinearSystem::buildNonConformalNodeGraph(const stk::mesh::PartVector &parts)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  beginLinearSystemConstruction();
//if (realm_.bulk_data().parallel_rank()==0) std::cerr<<"buildNonConformalNodeGraph"<<std::endl;

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
}

void
TpetraLinearSystem::buildOversetNodeGraph(const stk::mesh::PartVector &parts)
{
  // extract the rank
  const int theRank = NaluEnv::self().parallel_rank();

  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  beginLinearSystemConstruction();

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
}

void
TpetraLinearSystem::copy_stk_to_tpetra(
  stk::mesh::FieldBase * stkField,
  const Teuchos::RCP<LinSys::MultiVector> tpetraField)
{
  ThrowAssert(!tpetraField.is_null());
  ThrowAssert(stkField);
  const int numVectors = tpetraField->getNumVectors();

  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const stk::mesh::Selector selector = stk::mesh::selectField(*stkField) 
    & metaData.locally_owned_part() 
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector())) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = bulkData.get_buckets(stk::topology::NODE_RANK, selector);

  for(const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket & b = *bptr;

    const int fieldSize = field_bytes_per_entity(*stkField, b) / (sizeof(double));

    ThrowRequire(numVectors == fieldSize);

    const stk::mesh::Bucket::size_type length = b.size();

    const double * stkFieldPtr = (double*)stk::mesh::field_data(*stkField, b);

    for (stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k )
    {
      const stk::mesh::Entity node = b[k];

      int status = getDofStatus(node);
      if ((status & DS_SkippedDOF) || (status & DS_SharedNotOwnedDOF))
        continue;

      const stk::mesh::EntityId nodeId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
      for(int d=0; d < fieldSize; ++d)
      {
        const size_t stkIndex = k*fieldSize + d;
        tpetraField->replaceGlobalValue(nodeId, d, stkFieldPtr[stkIndex]);
      }
    }
  }
}

void sort_connections(std::vector<std::vector<stk::mesh::Entity> >& connections)
{
  for(std::vector<stk::mesh::Entity>& vec : connections) {
    std::sort(vec.begin(), vec.end());
  }
}

template<typename ViewType, typename LocalOrdinal>
void add_to_length(ViewType& v_owned, ViewType& v_shared, unsigned numDof,
                   LocalOrdinal lid_a, LocalOrdinal maxOwnedRowId, bool a_owned, unsigned numColEntities)
{   
    ViewType& v_a = a_owned ? v_owned : v_shared;
    LocalOrdinal lid = a_owned ? lid_a : lid_a - maxOwnedRowId;
    
    for (unsigned d=0; d < numDof; ++d) {
      v_a(lid+d) += numDof*numColEntities;
    }
}

void add_lengths_to_comm(const stk::mesh::BulkData& bulk,
                         stk::CommNeighbors& commNeighbors,
                         int entity_a_owner,
                         stk::mesh::EntityId entityId_a,
                         unsigned numDof,
                         unsigned numColEntities,
                         const stk::mesh::EntityId* colEntityIds,
                         const int* colOwners)
{   
    int owner = entity_a_owner;
    stk::CommBufferV& sbuf = commNeighbors.send_buffer(owner);
    GlobalOrdinal rowGid = GID_(entityId_a, numDof , 0);
    
    sbuf.pack(rowGid);
    sbuf.pack(numColEntities*2);
    for(unsigned c=0; c<numColEntities; ++c) {
        GlobalOrdinal colGid0 = GID_(colEntityIds[c], numDof , 0);
        sbuf.pack(colGid0);
        sbuf.pack(colOwners[c]);
    }
}

template<typename ViewType>
void communicate_remote_columns(const stk::mesh::BulkData& bulk,
                                const std::vector<int>& neighborProcs,
                                stk::CommNeighbors& commNeighbors,
                                unsigned numDof,
                                const Teuchos::RCP<LinSys::Map>& ownedRowsMap,
                                ViewType& deviceLocallyOwnedRowLengths,
                                std::set<std::pair<int,GlobalOrdinal> >& communicatedColIndices)
{   
    commNeighbors.communicate();
    
    for(int p : neighborProcs) { 
        stk::CommBufferV& rbuf = commNeighbors.recv_buffer(p);
        size_t bufSize = rbuf.size_in_bytes();
        while(rbuf.size_in_bytes() > 0) {
            GlobalOrdinal rowGid = 0;
            rbuf.unpack(rowGid);
            unsigned len = 0;
            rbuf.unpack(len);
            unsigned numCols = len/2;
            LocalOrdinal lid = ownedRowsMap->getLocalElement(rowGid);
            if (lid < 0) {
                std::cerr<<"P"<<bulk.parallel_rank()<<" lid="<<lid<<" for rowGid="<<rowGid<<" sent from proc "<<p<<std::endl;
            }
            for(unsigned d=0; d<numDof; ++d) {
                deviceLocallyOwnedRowLengths(lid++) += numCols*numDof;
            }
            for(unsigned i=0; i<numCols; ++i) {
                GlobalOrdinal colGid = 0;
                rbuf.unpack(colGid);
                int owner = 0;
                rbuf.unpack(owner);
                for(unsigned dd=0; dd<numDof; ++dd) {
                    communicatedColIndices.insert(std::make_pair(owner,colGid++));
                }
            }
        }
        rbuf.resize(bufSize);
    }
}

size_t get_neighbor_index(const std::vector<int>& neighborProcs, int proc)
{   
    std::vector<int>::const_iterator neighbor = std::find(neighborProcs.begin(), neighborProcs.end(), proc);
    ThrowRequireMsg(neighbor != neighborProcs.end(),"Error, failed to find p="<<proc<<" in neighborProcs.");

    size_t neighborIndex = neighbor-neighborProcs.begin();
    return neighborIndex;
}

void
TpetraLinearSystem::compute_send_lengths(const std::vector<stk::mesh::Entity>& rowEntities,
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
}

void
TpetraLinearSystem::compute_graph_row_lengths(const std::vector<stk::mesh::Entity>& rowEntities,
        const std::vector<std::vector<stk::mesh::Entity> >& connections,
                                              LinSys::RowLengths& sharedNotOwnedRowLengths,
                                              LinSys::RowLengths& locallyOwnedRowLengths,
                                              stk::CommNeighbors& commNeighbors)
{
  Kokkos::View<size_t*,MemSpace> deviceSharedNotOwnedRowLengths = sharedNotOwnedRowLengths.view<MemSpace>();
  Kokkos::View<size_t*,MemSpace> deviceLocallyOwnedRowLengths = locallyOwnedRowLengths.view<MemSpace>();

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
        add_lengths_to_comm(bulk, commNeighbors, entity_a_owner, entityId_a,
                            numDof_, numColEntities, colEntityIds.data(), colOwners.data());
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
            add_lengths_to_comm(bulk, commNeighbors, colOwners[ii], entityId_b, numDof_, 1, &entityId_a, &entity_a_owner);
        }
    }
  }
}

void
insert_single_dof_row_into_graph(LocalGraphArrays& crsGraph, LocalOrdinal rowLid, LocalOrdinal maxOwnedRowId,
                unsigned numDof, unsigned numCols, const std::vector<LocalOrdinal>& colLids)
{
    if (rowLid >= maxOwnedRowId) {
      rowLid -= maxOwnedRowId;
    }
    crsGraph.insertIndices(rowLid++, numCols, colLids.data(), numDof);
}

void
TpetraLinearSystem::insert_graph_connections(const std::vector<stk::mesh::Entity>& rowEntities,
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
    localDofs_a[0] = entityToColLID_[entity_a.local_offset()];

    for(size_t j=0; j<numColEntities; ++j) {
      const stk::mesh::Entity entity_b = entities_b[j];
      dofStatus[j] = getDofStatus(entity_b);
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
}

void insert_communicated_col_indices(const std::vector<int>& neighborProcs,
                                     stk::CommNeighbors& commNeighbors,
                                     unsigned numDof,
                                     LocalGraphArrays& ownedGraph,
                                     const LinSys::Map& rowMap,
                                     const LinSys::Map& colMap)
{
    std::vector<LocalOrdinal> colLids;
    for(int p : neighborProcs) {
        stk::CommBufferV& rbuf = commNeighbors.recv_buffer(p);
        while(rbuf.size_in_bytes() > 0) {
            stk::mesh::EntityId rowGid = 0;
            rbuf.unpack(rowGid);
            unsigned len = 0;
            rbuf.unpack(len);
            unsigned numCols = len/2;
            colLids.resize(numCols);
            LocalOrdinal rowLid = rowMap.getLocalElement(rowGid);
            for(unsigned i=0; i<numCols; ++i) {
                GlobalOrdinal colGid = 0;
                rbuf.unpack(colGid);
                int owner = 0;
                rbuf.unpack(owner);
                colLids[i] = colMap.getLocalElement(colGid);
            }
            ownedGraph.insertIndices(rowLid++,numCols,colLids.data(), numDof);
        }
    }
}

void
TpetraLinearSystem::fill_entity_to_row_LID_mapping()
{
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  stk::mesh::Selector selector = bulk.mesh_meta_data().universal_part() & !(realm_.get_inactive_selector());
  entityToLID_.assign(bulk.get_size_of_entity_index_space(), 2000000000);
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
}

void
TpetraLinearSystem::fill_entity_to_col_LID_mapping()
{
    const stk::mesh::BulkData& bulk = realm_.bulk_data();
    entityToColLID_.assign(bulk.get_size_of_entity_index_space(), 2000000000);
    const stk::mesh::BucketVector& nodeBuckets = bulk.buckets(stk::topology::NODE_RANK);
    for(const stk::mesh::Bucket* bptr : nodeBuckets) {
        const stk::mesh::Bucket& b = *bptr;
        const stk::mesh::EntityId* nodeIds = stk::mesh::field_data(*realm_.naluGlobalId_, b);
        for(size_t i=0; i<b.size(); ++i) {
            stk::mesh::Entity node = b[i];
            GlobalOrdinal gid = GID_(nodeIds[i], numDof_, 0);
            entityToColLID_[node.local_offset()] = totalColsMap_->getLocalElement(gid);
        }
    }
}

void
TpetraLinearSystem::storeOwnersForShared()
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
        for(unsigned idof=0; idof < numDof_; ++ idof) {
          GlobalOrdinal gid = GID_(naluId, numDof_, idof);
          ownersAndGids_.insert(std::make_pair(bulkData.parallel_owner_rank(master), gid));
        }
      }
    }
  }
}

void add_procs_to_neighbors(const std::vector<int>& procs, std::vector<int>& neighbors)
{
  neighbors.insert(neighbors.end(), procs.begin(), procs.end());
  stk::util::sort_and_unique(neighbors);
}

void fill_neighbor_procs(std::vector<int>& neighborProcs,
                         const stk::mesh::BulkData& bulk,
                         const Realm& realm)
{   
  if (bulk.parallel_size() > 1) {
    neighborProcs = bulk.all_sharing_procs(stk::topology::NODE_RANK);
    if (bulk.is_automatic_aura_on()) {
      std::vector<int> ghostCommProcs;
      populate_ghost_comm_procs(bulk, bulk.aura_ghosting(), ghostCommProcs);
      add_procs_to_neighbors(ghostCommProcs, neighborProcs);
    }
    if (realm.hasPeriodic_) {
      add_procs_to_neighbors(realm.periodicManager_->ghostCommProcs_, neighborProcs);
    }
    if (realm.nonConformalManager_) {
      add_procs_to_neighbors(realm.nonConformalManager_->ghostCommProcs_, neighborProcs);
    }
    if (realm.oversetManager_) {
      add_procs_to_neighbors(realm.oversetManager_->ghostCommProcs_, neighborProcs);
    }
  }
}

void fill_owned_and_shared_then_nonowned_ordered_by_proc(std::vector<GlobalOrdinal>& totalGids,
                                    std::vector<int>& srcPids,
                                    int localProc,
                                    const Teuchos::RCP<LinSys::Map>& ownedRowsMap,
                                    const Teuchos::RCP<LinSys::Map>& sharedNotOwnedRowsMap,
                                    const std::set<std::pair<int,GlobalOrdinal> >& ownersAndGids,
                                    const std::vector<int>& sharedPids)
{ 
  auto ownedIndices = ownedRowsMap->getMyGlobalIndices();
  totalGids.clear();
  totalGids.reserve(ownedIndices.size() + ownersAndGids.size());
  
  srcPids.clear();
  srcPids.reserve(ownersAndGids.size());

  for(unsigned i=0; i<ownedIndices.size(); ++i) {
    totalGids.push_back(ownedIndices[i]);
  }
  
  auto sharedIndices = sharedNotOwnedRowsMap->getMyGlobalIndices();
  for(unsigned i=0; i<sharedIndices.size(); ++i) {
    totalGids.push_back(sharedIndices[i]);
    srcPids.push_back(sharedPids[i]);
    ThrowRequireMsg(sharedPids[i] != localProc && sharedPids[i] >= 0,
                    "Error, bad sharedPid = "<<sharedPids[i]<<
                    ", localProc = "<<localProc<<", gid = "<<sharedIndices[i]);
  }

  for(const std::pair<int,GlobalOrdinal>& procAndGid : ownersAndGids) {
    int proc = procAndGid.first;
    GlobalOrdinal gid = procAndGid.second;
    if (proc != localProc &&
        !ownedRowsMap->isNodeGlobalElement(gid) &&
        !sharedNotOwnedRowsMap->isNodeGlobalElement(gid)) {
      totalGids.push_back(gid);
      srcPids.push_back(procAndGid.first);
      ThrowRequireMsg(procAndGid.first != localProc && procAndGid.first >= 0,
                      "Error, bad remote proc = "<<procAndGid.first);
    }
  }

  ThrowRequireMsg(srcPids.size() == (totalGids.size() - ownedIndices.size()),
                  "Error, bad srcPids.size() = "<<srcPids.size());
}

void verify_same_except_sort_order(const std::vector<GlobalOrdinal>& vec1, const std::string& vec1name,
                                   const std::vector<GlobalOrdinal>& vec2, const std::string& vec2name,
                                   int localProc)
{ 
  std::vector<GlobalOrdinal> svec1(vec1);
  std::vector<GlobalOrdinal> svec2(vec2);
  
  std::sort(svec1.begin(), svec1.end());
  std::sort(svec2.begin(), svec2.end());
    
  std::vector<GlobalOrdinal> vec1NotInVec2;
  std::vector<GlobalOrdinal> vec2NotInVec1;
  
  for(GlobalOrdinal gid : vec1) {
    if (!std::binary_search(svec2.begin(), svec2.end(), gid)) {
      vec1NotInVec2.push_back(gid);
    }
  }
  for(GlobalOrdinal gid : vec2) {
    if (!std::binary_search(svec1.begin(), svec1.end(), gid)) {
      vec2NotInVec1.push_back(gid);
    }
  }
  std::vector<GlobalOrdinal>::iterator uniq1 = std::unique(svec1.begin(), svec1.end());
  std::vector<GlobalOrdinal>::iterator uniq2 = std::unique(svec2.begin(), svec2.end());
  unsigned idx1 = uniq1-svec1.begin();
  unsigned idx2 = uniq2-svec2.begin();
  bool foundDuplicates = ((svec1.size()-idx1)!=0) || ((svec2.size()-idx2)!=0);
  if (foundDuplicates) {
    std::ostringstream oss;
    oss<<"P"<<localProc<<" "<<(svec1.size()-idx1) << " duplicates in "<<vec1name<<std::endl;
    oss<<"P"<<localProc<<" "<<(svec2.size()-idx2) << " duplicates in "<<vec2name<<std::endl;
    oss<<"P"<<localProc<<" in "<<vec1name<<" but not in "<<vec2name<<":";
    for(GlobalOrdinal gid : vec1NotInVec2) { oss << gid << ","; }
    oss << ";; in "<<vec2name<<" but not in "<<vec1name<<":";
    for(GlobalOrdinal gid : vec2NotInVec1) { oss << gid << ","; }
    oss<<std::endl;
    std::cerr<<oss.str();
  }
  
  ThrowRequireMsg(vec1.size() == vec1.size() && vec1NotInVec2.empty() && vec2NotInVec1.empty() && !foundDuplicates,
                  "P"<<localProc<<", failed to verify "<<vec1name<<" against "<<vec2name);
}

void verify_row_lengths(const LinSys::Graph& graph,
                        const Kokkos::View<size_t*,MemSpace>& rowLengths, int localProc)
{
  ThrowRequireMsg(graph.getNodeNumRows() == rowLengths.size(),
                  "Error, graph.getNodeNumRows="<<graph.getNodeNumRows()<<" must equal "
                  <<"rowLengths.size="<<rowLengths.size());

  for(size_t i=0; i<rowLengths.size(); ++i) {
    if (rowLengths(i) < graph.getNumEntriesInLocalRow(i)) {
      std::ostringstream os;
      os<<"Error, P"<<localProc<<" expected global row "<<graph.getRowMap()->getGlobalElement(i)
         <<" to have "<<rowLengths(i)<<" entries, graph row has "<<graph.getNumEntriesInLocalRow(i)
         <<" entries: ";
      GlobalOrdinal rowGID = graph.getRowMap()->getGlobalElement(i);
      std::vector<GlobalOrdinal> vIndices(graph.getNumEntriesInGlobalRow(rowGID));
      Teuchos::ArrayView<GlobalOrdinal> colIndices(vIndices);
      size_t rowLen = 0;
      graph.getGlobalRowCopy(rowGID, colIndices, rowLen);

      for(unsigned j=0; j<graph.getNumEntriesInLocalRow(i); ++j) {
        os<<colIndices[j]<<",";
      }
      ThrowRequireMsg(rowLengths(i) >= graph.getNumEntriesInLocalRow(i),os.str());
    }
  }
}

void dump_graph(const std::string& name, int counter, int proc, LinSys::Graph& graph)
{
  std::string fullname(name+"."+std::to_string(counter)+"."+std::to_string(proc));
  std::ofstream ofs(fullname);
  Teuchos::RCP<const LinSys::Map> rowMap = graph.getRowMap();
  const auto myGlobalIndices = rowMap->getMyGlobalIndices();
  for(size_t i=0; i<myGlobalIndices.size(); ++i) {
    GlobalOrdinal rowGID = myGlobalIndices[i];
    std::vector<GlobalOrdinal> vIndices(graph.getNumEntriesInGlobalRow(rowGID));
    Teuchos::ArrayView<GlobalOrdinal> colIndices(vIndices);
    size_t rowLen = 0;
    graph.getGlobalRowCopy(rowGID, colIndices, rowLen);
    std::ostringstream os;
    os<<rowGID<<": ";
    for(size_t j=0; j<rowLen; ++j) {
        os<<colIndices[j]<<", ";
    }
    os<<std::endl;
    ofs<<os.str();
  }
}

template<typename ViewType>
void remove_invalid_indices(LocalGraphArrays& csg, ViewType& rowLengths)
{
  size_t nnz = csg.rowPointers(rowLengths.size());
  auto cols = csg.colIndices.data();
  auto rowPtrs = csg.rowPointers.data();
  size_t newNnz = 0;
  for(int i=0, ie=csg.rowPointers.size()-1; i<ie; ++i) {
    const LocalOrdinal* row = cols+rowPtrs[i];
    int rowLen = csg.get_row_length(i);
    for(int j=rowLen-1; j>=0; --j) {
      if (row[j] != INVALID) {
        rowLengths(i) = j+1;
        break;
      }
    }
    newNnz += rowLengths(i);
  }

  if (newNnz < nnz) {
    Kokkos::View<LocalOrdinal*,MemSpace> newColIndices(Kokkos::ViewAllocateWithoutInitializing("colInds"),newNnz);
    LocalOrdinal* newCols = newColIndices.data();
    auto rowLens = rowLengths.data();
    int index = 0;
    for(int i=0, ie=csg.rowPointers.size()-1; i<ie; ++i) {
      auto row = cols+rowPtrs[i];
      for(size_t j=0; j<rowLens[i]; ++j) {
        newCols[index++] = row[j];
      }
    }
    csg.colIndices = newColIndices;
    LocalGraphArrays::compute_row_pointers(csg.rowPointers, rowLengths);
  }
}

void fill_in_extra_dof_rows_per_node(LocalGraphArrays& csg, int numDof)
{
  if (numDof == 1) {
    return;
  }

  auto rowPtrs = csg.rowPointers.data();
  LocalOrdinal* cols = csg.colIndices.data();
  for(int i=0, ie=csg.rowPointers.size()-1; i<ie;) {
    const LocalOrdinal* row = cols+rowPtrs[i];
    int rowLen = csg.get_row_length(i);
    for(int d=1; d<numDof; ++d) {
      LocalOrdinal* row_d = cols + rowPtrs[i] + rowLen*d;
      for(int j=0; j<rowLen; ++j) {
        row_d[j] = row[j];
      }
    }
    i += numDof;
  }
}

void verify_no_empty_connections(const std::vector<stk::mesh::Entity>& rowEntities,
        const std::vector<std::vector<stk::mesh::Entity> >& connections)
{
  for(const std::vector<stk::mesh::Entity>& vec : connections) {
    ThrowRequireMsg(!vec.empty(), "Error, empty connections vec.");
  }
}

void
TpetraLinearSystem::finalizeLinearSystem()
{
  ThrowRequire(inConstruction_);
  inConstruction_ = false;

  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  sort_connections(connections_);

  size_t numSharedNotOwned = sharedNotOwnedRowsMap_->getMyGlobalIndices().extent(0);
  size_t numLocallyOwned = ownedRowsMap_->getMyGlobalIndices().extent(0);
  LinSys::RowLengths sharedNotOwnedRowLengths("rowLengths", numSharedNotOwned);
  LinSys::RowLengths locallyOwnedRowLengths("rowLengths", numLocallyOwned);
  auto ownedRowLengths = locallyOwnedRowLengths.view<MemSpace>();
  auto globalRowLengths = sharedNotOwnedRowLengths.view<MemSpace>();

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
  totalColsMap_ = Teuchos::rcp(new LinSys::Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), optColGids, 1, tpetraComm, node_));

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

  ownedMatrix_ = Teuchos::rcp(new LinSys::Matrix(ownedGraph_));
  sharedNotOwnedMatrix_ = Teuchos::rcp(new LinSys::Matrix(sharedNotOwnedGraph_));

  ownedLocalMatrix_ = ownedMatrix_->getLocalMatrix();
  sharedNotOwnedLocalMatrix_ = sharedNotOwnedMatrix_->getLocalMatrix();

  ownedRhs_ = Teuchos::rcp(new LinSys::Vector(ownedRowsMap_));
  sharedNotOwnedRhs_ = Teuchos::rcp(new LinSys::Vector(sharedNotOwnedRowsMap_));

  ownedLocalRhs_ = ownedRhs_->getLocalView<sierra::nalu::MemSpace>();
  sharedNotOwnedLocalRhs_ = sharedNotOwnedRhs_->getLocalView<sierra::nalu::MemSpace>();

  sln_ = Teuchos::rcp(new LinSys::Vector(ownedRowsMap_));

  const int nDim = metaData.spatial_dimension();

  Teuchos::RCP<LinSys::MultiVector> coords 
    = Teuchos::RCP<LinSys::MultiVector>(new LinSys::MultiVector(sln_->getMap(), nDim));

  TpetraLinearSolver *linearSolver = reinterpret_cast<TpetraLinearSolver *>(linearSolver_);

  VectorFieldType *coordinates = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  if (linearSolver->activeMueLu())
    copy_stk_to_tpetra(coordinates, coords);

  linearSolver->setupLinearSolver(sln_, ownedMatrix_, ownedRhs_, coords);
}

void
TpetraLinearSystem::zeroSystem()
{
  ThrowRequire(!ownedMatrix_.is_null());
  ThrowRequire(!sharedNotOwnedMatrix_.is_null());
  ThrowRequire(!sharedNotOwnedRhs_.is_null());
  ThrowRequire(!ownedRhs_.is_null());

  sharedNotOwnedMatrix_->resumeFill();
  ownedMatrix_->resumeFill();

  sharedNotOwnedMatrix_->setAllToScalar(0);
  ownedMatrix_->setAllToScalar(0);
  sharedNotOwnedRhs_->putScalar(0);
  ownedRhs_->putScalar(0);

  sln_->putScalar(0);
}

namespace
{
template<typename RowViewType>
void sum_into_row_vec_3(
  RowViewType row_view,
  const int num_entities,
  const int* localIds,
  const int* sort_permutation,
  const double* input_values)
{
  // assumes that the flattened column indices for block matrices are all stored sequentially
  // specialized for numDof == 3
  constexpr bool forceAtomic = !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;
  const LocalOrdinal length = row_view.length;

  LocalOrdinal offset = 0;
  for (int j = 0; j < num_entities; ++j) {
    // since the columns are sorted, we pass through the column idxs once,
    // updating the offset as we go
    const int id_index = 3 * j;
    const LocalOrdinal cur_local_column_idx = localIds[id_index];
    while (row_view.colidx(offset) != cur_local_column_idx) {
      offset += 3;
      if (offset >= length) return;
    }

    const int entry_offset = sort_permutation[id_index];
    if (forceAtomic) {
      Kokkos::atomic_add(&row_view.value(offset + 0), input_values[entry_offset + 0]);
      Kokkos::atomic_add(&row_view.value(offset + 1), input_values[entry_offset + 1]);
      Kokkos::atomic_add(&row_view.value(offset + 2), input_values[entry_offset + 2]);
    }
    else {
      row_view.value(offset + 0) += input_values[entry_offset + 0];
      row_view.value(offset + 1) += input_values[entry_offset + 1];
      row_view.value(offset + 2) += input_values[entry_offset + 2];
    }
    offset += 3;
  }
}

template <typename RowViewType>
void sum_into_row (
  RowViewType row_view,
  const int num_entities, const int numDof,
  const int* localIds,
  const int* sort_permutation,
  const double* input_values)
{
  if (numDof == 3) {
    sum_into_row_vec_3(row_view, num_entities, localIds, sort_permutation, input_values);
    return;
  }

  constexpr bool forceAtomic = !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;
  const LocalOrdinal length = row_view.length;

  const int numCols = num_entities * numDof;
  LocalOrdinal offset = 0;
  for (int j = 0; j < numCols; ++j) {
    const LocalOrdinal perm_index = sort_permutation[j];
    const LocalOrdinal cur_local_column_idx = localIds[j];

    // since the columns are sorted, we pass through the column idxs once,
    // updating the offset as we go
    while (row_view.colidx(offset) != cur_local_column_idx && offset < length) {
      ++offset;
    }

    if (offset < length) {
      ThrowAssertMsg(std::isfinite(input_values[perm_index]), "Inf or NAN lhs");
      if (forceAtomic) {
        Kokkos::atomic_add(&(row_view.value(offset)), input_values[perm_index]);
      }
      else {
        row_view.value(offset) += input_values[perm_index];
      }
    }
  }
}

}

void
TpetraLinearSystem::sumInto(
      unsigned numEntities,
      const stk::mesh::Entity* entities,
      const SharedMemView<const double*> & rhs,
      const SharedMemView<const double**> & lhs,
      const SharedMemView<int*> & localIds,
      const SharedMemView<int*> & sortPermutation,
      const char * trace_tag)
{
  constexpr bool forceAtomic = !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  ThrowAssertMsg(localIds.span_is_contiguous(), "localIds assumed contiguous");
  ThrowAssertMsg(sortPermutation.span_is_contiguous(), "sortPermutation assumed contiguous");

  const int n_obj = numEntities;
  const int numRows = n_obj * numDof_;

  for(int i = 0; i < n_obj; i++) {
    const stk::mesh::Entity entity = entities[i];
    const LocalOrdinal localOffset = entityToColLID_[entity.local_offset()];
    for(size_t d=0; d < numDof_; ++d) {
      size_t lid = i*numDof_ + d;
      localIds[lid] = localOffset + d;
    }
  }

  for (int i = 0; i < numRows; ++i) {
    sortPermutation[i] = i;
  }
  Tpetra::Details::shellSortKeysAndValues(localIds.data(), sortPermutation.data(), numRows);

  for (int r = 0; r < numRows; ++r) {
    int i = sortPermutation[r]/numDof_;
    LocalOrdinal rowLid = entityToLID_[entities[i].local_offset()];
    rowLid += sortPermutation[r]%numDof_;
    const LocalOrdinal cur_perm_index = sortPermutation[r];
    const double* const cur_lhs = &lhs(cur_perm_index, 0);
    const double cur_rhs = rhs[cur_perm_index];
    ThrowAssertMsg(std::isfinite(cur_rhs), "Inf or NAN rhs");

    if(rowLid < maxOwnedRowId_) {
      sum_into_row(ownedLocalMatrix_.row(rowLid), n_obj, numDof_, localIds.data(), sortPermutation.data(), cur_lhs);
      if (forceAtomic) {
        Kokkos::atomic_add(&ownedLocalRhs_(rowLid,0), cur_rhs);
      }
      else {
        ownedLocalRhs_(rowLid,0) += cur_rhs;
      }
    }
    else if (rowLid < maxSharedNotOwnedRowId_) {
      LocalOrdinal actualLocalId = rowLid - maxOwnedRowId_;
      sum_into_row(sharedNotOwnedLocalMatrix_.row(actualLocalId), n_obj, numDof_,
        localIds.data(), sortPermutation.data(), cur_lhs);

      if (forceAtomic) {
        Kokkos::atomic_add(&sharedNotOwnedLocalRhs_(actualLocalId,0), cur_rhs);
      }
      else {
        sharedNotOwnedLocalRhs_(actualLocalId,0) += cur_rhs;
      }
    }
  }
}

void
TpetraLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity> & entities,
  std::vector<int> &scratchIds,
  std::vector<double> &scratchVals,
  const std::vector<double> & rhs,
  const std::vector<double> & lhs,
  const char *trace_tag
  )
{
  const size_t n_obj = entities.size();
  const unsigned numRows = n_obj * numDof_;

  ThrowAssert(numRows == rhs.size());
  ThrowAssert(numRows*numRows == lhs.size());

  scratchIds.resize(numRows);
  sortPermutation_.resize(numRows);
  for(size_t i = 0; i < n_obj; i++) {
    const stk::mesh::Entity entity = entities[i];
    const LocalOrdinal localOffset = entityToColLID_[entity.local_offset()];
    for(size_t d=0; d < numDof_; ++d) {
      size_t lid = i*numDof_ + d;
      scratchIds[lid] = localOffset + d;
    }
  }

  for (unsigned i = 0; i < numRows; ++i) {
    sortPermutation_[i] = i;
  }
  Tpetra::Details::shellSortKeysAndValues(scratchIds.data(), sortPermutation_.data(), (int)numRows);

  for (unsigned r = 0; r < numRows; r++) {
    int i = sortPermutation_[r]/numDof_;
    LocalOrdinal rowLid = entityToLID_[entities[i].local_offset()];
    rowLid += sortPermutation_[r]%numDof_;
    const LocalOrdinal cur_perm_index = sortPermutation_[r];
    const double* const cur_lhs = &lhs[cur_perm_index*numRows];
    const double cur_rhs = rhs[cur_perm_index];
    ThrowAssertMsg(std::isfinite(cur_rhs), "Invalid rhs");

    if(rowLid < maxOwnedRowId_) {
      sum_into_row(ownedLocalMatrix_.row(rowLid),  n_obj, numDof_, scratchIds.data(), sortPermutation_.data(), cur_lhs);
      ownedLocalRhs_(rowLid,0) += cur_rhs;
    }
    else if (rowLid < maxSharedNotOwnedRowId_) {
      LocalOrdinal actualLocalId = rowLid - maxOwnedRowId_;
      sum_into_row(sharedNotOwnedLocalMatrix_.row(actualLocalId),  n_obj, numDof_,
        scratchIds.data(), sortPermutation_.data(), cur_lhs);

      sharedNotOwnedLocalRhs_(actualLocalId,0) += cur_rhs;
    }
  }
}

void
TpetraLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase * solutionField,
  stk::mesh::FieldBase * bcValuesField,
  const stk::mesh::PartVector & parts,
  const unsigned beginPos,
  const unsigned endPos)
{
  stk::mesh::MetaData & metaData = realm_.meta_data();

  double adbc_time = -NaluEnv::self().nalu_time();

  const stk::mesh::Selector selector 
    = (metaData.locally_owned_part() | metaData.globally_shared_part())
    & stk::mesh::selectUnion(parts)
    & stk::mesh::selectField(*solutionField) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, selector );

  const bool internalMatrixIsSorted = true;
  int nbc=0;
  for(const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket & b = *bptr;

    const unsigned fieldSize = field_bytes_per_entity(*solutionField, b) / sizeof(double);
    ThrowRequire(fieldSize == numDof_);

    const stk::mesh::Bucket::size_type length   = b.size();
    const double * solution = (double*)stk::mesh::field_data(*solutionField, *b.begin());
    const double * bcValues = (double*)stk::mesh::field_data(*bcValuesField, *b.begin());

    Teuchos::ArrayView<const LocalOrdinal> indices;
    Teuchos::ArrayView<const double> values;
    std::vector<double> new_values;

    for (stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const stk::mesh::Entity entity = b[k];
      const stk::mesh::EntityId naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, entity);
      const LocalOrdinal localIdOffset = lookup_myLID(myLIDs_, naluId, "applyDirichletBCs");

      for(unsigned d=beginPos; d < endPos; ++d) {
        const LocalOrdinal localId = localIdOffset + d;
        const bool useOwned = localId < maxOwnedRowId_;
        const LocalOrdinal actualLocalId = useOwned ? localId : localId - maxOwnedRowId_;
        Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
        const LinSys::Matrix::local_matrix_type& local_matrix = useOwned ? ownedLocalMatrix_ : sharedNotOwnedLocalMatrix_;

        if(localId > maxSharedNotOwnedRowId_) {
          std::cerr << "localId > maxSharedNotOwnedRowId_:: localId= " << localId << " maxSharedNotOwnedRowId_= " << maxSharedNotOwnedRowId_ << " useOwned = " << (localId < maxOwnedRowId_ ) << std::endl;
          throw std::runtime_error("logic error: localId > maxSharedNotOwnedRowId_");
        }

        // Adjust the LHS

        const double diagonal_value = useOwned ? 1.0 : 0.0;

        matrix->getLocalRowView(actualLocalId, indices, values);
        const size_t rowLength = values.size();
        if (rowLength > 0) {
          new_values.resize(rowLength);
          for(size_t i=0; i < rowLength; ++i) {
              new_values[i] = (indices[i] == localId) ? diagonal_value : 0;
          }
          local_matrix.replaceValues(actualLocalId, &indices[0], rowLength, new_values.data(), internalMatrixIsSorted);
        }

        // Replace the RHS residual with (desired - actual)
        Teuchos::RCP<LinSys::Vector> rhs = useOwned ? ownedRhs_: sharedNotOwnedRhs_;
        const double bc_residual = useOwned ? (bcValues[k*fieldSize + d] - solution[k*fieldSize + d]) : 0.0;
        rhs->replaceLocalValue(actualLocalId, bc_residual);
        ++nbc;
      }
    }
  }
  adbc_time += NaluEnv::self().nalu_time();
}

void
TpetraLinearSystem::prepareConstraints(
  const unsigned beginPos,
  const unsigned endPos)
{
  Teuchos::ArrayView<const LocalOrdinal> indices;
  Teuchos::ArrayView<const double> values;
  std::vector<double> new_values;

  const bool internalMatrixIsSorted = true;

  //KOKKOS: Loop noparallel RCP Vector Matrix replaceValues
  for( const OversetInfo* oversetInfo : realm_.oversetManager_->oversetInfoVec_) {

    // extract orphan node and global id; process both owned and shared
    stk::mesh::Entity orphanNode = oversetInfo->orphanNode_;
    const stk::mesh::EntityId naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, orphanNode);
    const LocalOrdinal localIdOffset = lookup_myLID(myLIDs_, naluId, "prepareConstraints");

    //KOKKOS: Nested Loop noparallel RCP Vector Matrix replaceValues
    for(unsigned d=beginPos; d < endPos; ++d) {
      const LocalOrdinal localId = localIdOffset + d;
      const bool useOwned = localId < maxOwnedRowId_;
      const LocalOrdinal actualLocalId = useOwned ? localId : localId - maxOwnedRowId_;
      Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
      const LinSys::Matrix::local_matrix_type& local_matrix = matrix->getLocalMatrix();
      
      if ( localId > maxSharedNotOwnedRowId_) {
        throw std::runtime_error("logic error: localId > maxSharedNotOwnedRowId_");
      }
      
      // Adjust the LHS; full row is perfectly zero
      matrix->getLocalRowView(actualLocalId, indices, values);
      const size_t rowLength = values.size();
      if (rowLength > 0) {
        new_values.resize(rowLength);
        for(size_t i=0; i < rowLength; ++i) {
          new_values[i] = 0.0;
        }
        local_matrix.replaceValues(actualLocalId, &indices[0], rowLength, new_values.data(), internalMatrixIsSorted);
      }
      
      // Replace the RHS residual with zero
      Teuchos::RCP<LinSys::Vector> rhs = useOwned ? ownedRhs_: sharedNotOwnedRhs_;
      const double bc_residual = 0.0;
      rhs->replaceLocalValue(actualLocalId, bc_residual);
    }
  }
}

void
TpetraLinearSystem::resetRows(
  const std::vector<stk::mesh::Entity> nodeList,
  const unsigned beginPos,
  const unsigned endPos)
{
  Teuchos::ArrayView<const LocalOrdinal> indices;
  Teuchos::ArrayView<const double> values;
  std::vector<double> new_values;
  constexpr double rhs_residual = 0.0;
  const bool internalMatrixIsSorted = true;

  for (auto node: nodeList) {
    const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
    const LocalOrdinal localIdOffset = lookup_myLID(myLIDs_, naluId, "resetRows");

    for (unsigned d=beginPos; d < endPos; ++d) {
      const LocalOrdinal localId = localIdOffset + d;
      const bool useOwned = (localId < maxOwnedRowId_);
      const LocalOrdinal actualLocalId =
        useOwned ? localId : (localId - maxOwnedRowId_);
      Teuchos::RCP<LinSys::Matrix> matrix =
        useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
      const LinSys::Matrix::local_matrix_type& local_matrix = matrix->getLocalMatrix();

      if (localId > maxSharedNotOwnedRowId_) {
        throw std::runtime_error("logic error: localId > maxSharedNotOwnedRowId");
      }

      // Adjust the LHS; zero out all entries (including diagonal)
      matrix->getLocalRowView(actualLocalId, indices, values);
      const size_t rowLength = values.size();
      if (rowLength > 0) {
        new_values.resize(rowLength);
        for (size_t i=0; i < rowLength; i++) {
          new_values[i] = 0.0;
        }
        local_matrix.replaceValues(actualLocalId, &indices[0], rowLength, new_values.data(), internalMatrixIsSorted);
      }

      // Replace RHS residual entry = 0.0
      Teuchos::RCP<LinSys::Vector> rhs =
        useOwned ? ownedRhs_ : sharedNotOwnedRhs_;
      rhs->replaceLocalValue(actualLocalId, rhs_residual);
    }
  }
}

void
TpetraLinearSystem::loadComplete()
{
  // LHS
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::parameterList ();
  params->set("No Nonlocal Changes", true);
  bool do_params=false;

  if (do_params)
    sharedNotOwnedMatrix_->fillComplete(params);
  else
    sharedNotOwnedMatrix_->fillComplete();

  ownedMatrix_->doExport(*sharedNotOwnedMatrix_, *exporter_, Tpetra::ADD);
  if (do_params)
    ownedMatrix_->fillComplete(params);
  else
    ownedMatrix_->fillComplete();

  // RHS
  ownedRhs_->doExport(*sharedNotOwnedRhs_, *exporter_, Tpetra::ADD);
}

int
TpetraLinearSystem::solve(
  stk::mesh::FieldBase * linearSolutionField)
{

  TpetraLinearSolver *linearSolver = reinterpret_cast<TpetraLinearSolver *>(linearSolver_);

  if ( realm_.debug() ) {
    checkForNaN(true);
    if (checkForZeroRow(true, false, true)) {
      throw std::runtime_error("ERROR checkForZeroRow in solve()");
    }
  }
   
  if (linearSolver->getConfig()->getWriteMatrixFiles()) {
    writeToFile(eqSysName_.c_str());
    writeToFile(eqSysName_.c_str(), false);
  }

  double solve_time = -NaluEnv::self().nalu_time();

  int iters;
  double finalResidNorm;
  
  // memory diagnostic
  if ( realm_.get_activate_memory_diagnostic() ) {
    NaluEnv::self().naluOutputP0() << "NaluMemory::TpetraLinearSystem::solve() PreSolve: " << eqSysName_ << std::endl;
    realm_.provide_memory_summary();
  }

  const int status = linearSolver->solve(
      sln_,
      iters,
      finalResidNorm,
      realm_.isFinalOuterIter_);

  solve_time += NaluEnv::self().nalu_time();

  if (linearSolver->getConfig()->getWriteMatrixFiles()) {
    writeSolutionToFile(eqSysName_.c_str());
    ++writeCounter_;
  }

  copy_tpetra_to_stk(sln_, linearSolutionField);
  sync_field(linearSolutionField);

  // computeL2 norm
  const double norm2 = ownedRhs_->norm2();

  // save off solver info
  linearSolveIterations_ = iters;
  nonLinearResidual_ = realm_.l2Scaling_*norm2;
  linearResidual_ = finalResidNorm;
   
  if ( eqSys_->firstTimeStepSolve_ )
    firstNonLinearResidual_ = nonLinearResidual_;
  scaledNonLinearResidual_ = nonLinearResidual_/std::max(std::numeric_limits<double>::epsilon(), firstNonLinearResidual_);

  if ( provideOutput_ ) {
    const int nameOffset = eqSysName_.length()+8;
    NaluEnv::self().naluOutputP0()
      << std::setw(nameOffset) << std::right << eqSysName_
      << std::setw(32-nameOffset)  << std::right << iters
      << std::setw(18) << std::right << finalResidNorm
      << std::setw(15) << std::right << nonLinearResidual_
      << std::setw(14) << std::right << scaledNonLinearResidual_ << std::endl;
  }

  eqSys_->firstTimeStepSolve_ = false;

  return status;
}

void
TpetraLinearSystem::checkForNaN(bool useOwned)
{
  Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::Vector> rhs = useOwned ? ownedRhs_ : sharedNotOwnedRhs_;

  Teuchos::ArrayView<const LocalOrdinal> indices;
  Teuchos::ArrayView<const double> values;

  size_t n = matrix->getRowMap()->getNodeNumElements();
  for(size_t i=0; i<n; ++i) {

    matrix->getLocalRowView(i, indices, values);
    const size_t rowLength = values.size();
    for(size_t k=0; k < rowLength; ++k) {
      if (values[k] != values[k])	{
        std::cerr << "LHS NaN: " << i << std::endl;
        throw std::runtime_error("bad LHS");
      }
    }
  }

  Teuchos::ArrayRCP<const Scalar> rhs_data = rhs->getData();
  n = rhs_data.size();
  for(size_t i=0; i<n; ++i) {
    if (rhs_data[i] != rhs_data[i]) {
      std::cerr << "rhs NaN: " << i << std::endl;
      throw std::runtime_error("bad rhs");
    }
  }
}

bool
TpetraLinearSystem::checkForZeroRow(bool useOwned, bool doThrow, bool doPrint)
{
  Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::Vector> rhs = useOwned ? ownedRhs_ : sharedNotOwnedRhs_;
  stk::mesh::BulkData & bulkData = realm_.bulk_data();

  Teuchos::ArrayView<const LocalOrdinal> indices;
  Teuchos::ArrayView<const double> values;

  size_t nrowG = matrix->getRangeMap()->getGlobalNumElements();
  size_t n = matrix->getRowMap()->getNodeNumElements();
  GlobalOrdinal max_gid = 0, g_max_gid=0;
  //KOKKOS: Loop parallel reduce
  kokkos_parallel_for("Nalu::TpetraLinearSystem::checkForZeroRowA", n, [&] (const size_t& i) {
    GlobalOrdinal gid = matrix->getGraph()->getRowMap()->getGlobalElement(i);
    max_gid = std::max(gid, max_gid);
  });
  stk::all_reduce_max(bulkData.parallel(), &max_gid, &g_max_gid, 1);

  nrowG = g_max_gid+1;
  std::vector<double> local_row_sums(nrowG, 0.0);
  std::vector<int> local_row_exists(nrowG, 0);
  std::vector<double> global_row_sums(nrowG, 0.0);
  std::vector<int> global_row_exists(nrowG, 0);

  for(size_t i=0; i<n; ++i) {
    GlobalOrdinal gid = matrix->getGraph()->getRowMap()->getGlobalElement(i);
    matrix->getLocalRowView(i, indices, values);
    const size_t rowLength = values.size();
    double row_sum = 0.0;
    for(size_t k=0; k < rowLength; ++k) {
      row_sum += std::abs(values[k]);
    }
    if (gid-1 >= (GlobalOrdinal)local_row_sums.size() || gid <= 0) {
      std::cerr << "gid= " << gid << " nrowG= " << nrowG << std::endl;
      throw std::runtime_error("bad gid");
    }
    local_row_sums[gid-1] = row_sum;
    local_row_exists[gid-1] = 1;
  }

  stk::all_reduce_sum(bulkData.parallel(), &local_row_sums[0], &global_row_sums[0], (unsigned)nrowG);
  stk::all_reduce_max(bulkData.parallel(), &local_row_exists[0], &global_row_exists[0], (unsigned)nrowG);

  bool found=false;
  //KOKKOS: Loop parallel
  kokkos_parallel_for("Nalu::TpetraLinearSystem::checkForZeroRowC", nrowG, [&] (const size_t& ii) {
    double row_sum = global_row_sums[ii];
    if (global_row_exists[ii] && bulkData.parallel_rank() == 0 && row_sum < 1.e-10) {
      found = true;
      GlobalOrdinal gid = ii+1;
      stk::mesh::EntityId nid = GLOBAL_ENTITY_ID(gid, numDof_);
      stk::mesh::Entity node = bulkData.get_entity(stk::topology::NODE_RANK, nid);
      stk::mesh::EntityId naluGlobalId;
      if (bulkData.is_valid(node)) naluGlobalId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);

      int idof = GLOBAL_ENTITY_ID_IDOF(gid, numDof_);
      GlobalOrdinal GID_check = GID_(nid, numDof_, idof);
      if (doPrint) {

        double dualVolume = -1.0;

        std::cout << "P[" << bulkData.parallel_rank() << "] LHS zero: " << ii
                  << " GID= " << gid << " GID_check= " << GID_check << " nid= " << nid
                  << " naluGlobalId " << naluGlobalId << " is_valid= " << bulkData.is_valid(node)
                  << " idof= " << idof << " numDof_= " << numDof_
                  << " row_sum= " << row_sum
                  << " dualVolume= " << dualVolume
                  << std::endl;
        NaluEnv::self().naluOutputP0() << "P[" << bulkData.parallel_rank() << "] LHS zero: " << ii
                        << " GID= " << gid << " GID_check= " << GID_check << " nid= " << nid
                        << " naluGlobalId " << naluGlobalId << " is_valid= " << bulkData.is_valid(node)
                        << " idof= " << idof << " numDof_= " << numDof_
                        << " row_sum= " << row_sum
                        << " dualVolume= " << dualVolume
                        << std::endl;
      }
    }
  });

  if (found && doThrow) {
    throw std::runtime_error("bad zero row LHS");
  }
  return found;
}

void
TpetraLinearSystem::writeToFile(const char * base_filename, bool useOwned)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  const unsigned p_rank = bulkData.parallel_rank();
  const unsigned p_size = bulkData.parallel_size();

  Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::Vector> rhs = useOwned ? ownedRhs_ : sharedNotOwnedRhs_;

  const int currentCount = writeCounter_;

  if (1)
    {
      std::ostringstream osLhs;
      std::ostringstream osRhs;
      osLhs << base_filename << "-" << (useOwned ? "O-":"G-") << currentCount << ".mm." << p_size; // A little hacky but whatever
      osRhs << base_filename << "-" << (useOwned ? "O-":"G-") << currentCount << ".rhs." << p_size; // A little hacky but whatever

      Tpetra::MatrixMarket::Writer<LinSys::Matrix>::writeSparseFile(osLhs.str().c_str(), matrix,
                                                                    eqSysName_, std::string("Tpetra matrix for: ")+eqSysName_, true);
      typedef Tpetra::MatrixMarket::Writer<LinSys::Matrix> writer_type;
      if (useOwned) writer_type::writeDenseFile (osRhs.str().c_str(), rhs);
    }

  if (1)
    {
      std::ostringstream osLhs;
      std::ostringstream osGra;
      std::ostringstream osRhs;

      osLhs << base_filename << "-" << (useOwned ? "O-":"G-") << currentCount << ".mm." << p_size << "." << p_rank; // A little hacky but whatever
      osGra << base_filename << "-" << (useOwned ? "O-":"G-") << currentCount << ".gra." << p_size << "." << p_rank; // A little hacky but whatever
      osRhs << base_filename << "-" << (useOwned ? "O-":"G-") << currentCount << ".rhs." << p_size << "." << p_rank; // A little hacky but whatever

      //Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
#define DUMP(A)  do {                                                   \
        out << "\n\n===============================================================================================\n"; \
        out << "===============================================================================================\n"; \
        out << "P[" << p_rank << "] writeToFile:: " #A "= " << "\n---------------------------\n" ; \
        out << Teuchos::describe(*A,Teuchos::VERB_EXTREME) << "\n";     \
        out << "===============================================================================================\n"; \
        out << "===============================================================================================\n\n\n"; \
      } while(0)

      {
        std::ostringstream out;
        DUMP(matrix);
        std::ofstream fout;
        fout.open (osLhs.str().c_str());
        fout << out.str() << std::endl;
      }

      {
        std::ostringstream out;
        DUMP(matrix->getGraph());
        std::ofstream fout;
        fout.open (osGra.str().c_str());
        fout << out.str() << std::endl;
      }

      {
        std::ostringstream out;
        DUMP(rhs);
        std::ofstream fout;
        fout.open (osRhs.str().c_str());
        fout << out.str() << std::endl;
      }


#undef DUMP

    }

}

void
TpetraLinearSystem::printInfo(bool useOwned)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  const unsigned p_rank = bulkData.parallel_rank();

  Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::Vector> rhs = useOwned ? ownedRhs_ : sharedNotOwnedRhs_;

  if (p_rank == 0) {
    std::cout << "\nMatrix for EqSystem: " << eqSysName_ << " :: N N NZ= " << matrix->getRangeMap()->getGlobalNumElements()
              << " "
              << matrix->getDomainMap()->getGlobalNumElements()
              << " "
              << matrix->getGlobalNumEntries()
              << std::endl;
    NaluEnv::self().naluOutputP0() << "\nMatrix for system: " << eqSysName_ << " :: N N NZ= " << matrix->getRangeMap()->getGlobalNumElements()
                                   << " "
                                   << matrix->getDomainMap()->getGlobalNumElements()
                                   << " "
                                   << matrix->getGlobalNumEntries()
                                   << std::endl;
  }
}

void
TpetraLinearSystem::writeSolutionToFile(const char * base_filename, bool useOwned)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  const unsigned p_rank = bulkData.parallel_rank();
  const unsigned p_size = bulkData.parallel_size();

  Teuchos::RCP<LinSys::Vector> sln = sln_;
  const int currentCount = writeCounter_;

  if (1)
    {
      std::ostringstream osSln;
      osSln << base_filename << "-" << (useOwned ? "O-":"G-") << currentCount << ".sln." << p_size; // A little hacky but whatever

      typedef Tpetra::MatrixMarket::Writer<LinSys::Matrix> writer_type;
      if (useOwned) writer_type::writeDenseFile (osSln.str().c_str(), sln);
    }

  if (1)
    {
      std::ostringstream osSln;

      osSln << base_filename << "-" << "O-" << currentCount << ".sln." << p_size << "." << p_rank; // A little hacky but whatever

#define DUMP(A)  do {                                                   \
        out << "\n\n===============================================================================================\n"; \
        out << "===============================================================================================\n"; \
        out << "P[" << p_rank << "] writeToFile:: " #A "= " << "\n---------------------------\n" ; \
        out << Teuchos::describe(*A,Teuchos::VERB_EXTREME) << "\n";     \
        out << "===============================================================================================\n"; \
        out << "===============================================================================================\n\n\n"; \
      } while(0)

      {
        std::ostringstream out;
        DUMP(sln);
        std::ofstream fout;
        fout.open (osSln.str().c_str());
        fout << out.str() << std::endl;
      }


#undef DUMP

    }

}

void
TpetraLinearSystem::copy_tpetra_to_stk(
  const Teuchos::RCP<LinSys::Vector> tpetraField,
  stk::mesh::FieldBase * stkField)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  ThrowAssert(!tpetraField.is_null());
  ThrowAssert(stkField);
  const LinSys::ConstOneDVector & tpetraVector = tpetraField->get1dView();

  const unsigned p_rank = bulkData.parallel_rank();

  const stk::mesh::Selector selector = stk::mesh::selectField(*stkField)
    & metaData.locally_owned_part() 
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());
  
  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, selector);

  for (size_t ib=0; ib < buckets.size(); ++ib) {
    stk::mesh::Bucket & b = *buckets[ib];

    const unsigned fieldSize = field_bytes_per_entity(*stkField, b) / sizeof(double);
    ThrowRequire(fieldSize == numDof_);

    const stk::mesh::Bucket::size_type length = b.size();
    double * stkFieldPtr = (double*)stk::mesh::field_data(*stkField, *b.begin());
    const stk::mesh::EntityId *naluGlobalId = stk::mesh::field_data(*realm_.naluGlobalId_, *b.begin());
    for (stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity node = b[k];
      const LocalOrdinal localIdOffset = entityToLID_[node.local_offset()];
      for(unsigned d=0; d < fieldSize; ++d) {
        const LocalOrdinal localId = localIdOffset + d;
        bool useOwned = true;
        LocalOrdinal actualLocalId = localId;
        if(localId >= maxOwnedRowId_) {
          actualLocalId = localId - maxOwnedRowId_;
          useOwned = false;
        }

        if (!useOwned) {
          stk::mesh::EntityId naluId = naluGlobalId[k];
          stk::mesh::EntityId stkId = bulkData.identifier(node);
          std::cout << "P[" << p_rank << "] useOwned = " << useOwned << " localId = " << localId << " maxOwnedRowId_= " << maxOwnedRowId_ << " actualLocalId= " << actualLocalId
                    << " naluGlobalId= " << naluGlobalId[k] << " stkId= " << stkId << " naluId= " << naluId << std::endl;
        }
        ThrowRequire(useOwned);

        const size_t stkIndex = k*numDof_ + d;
        if (useOwned){
          stkFieldPtr[stkIndex] = tpetraVector[localId];
        }
      }
    }
  }
}

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

  //std::cerr << "has_non_matching_boundary_face_alg= " << has_non_matching_boundary_face_alg << " hasPeriodic= " << hasPeriodic << std::endl;

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
    //if (entityIsShared && !entityIsOwned) {
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
  if (1)
    throw std::logic_error("bad status2");

  return DS_SkippedDOF;
}

} // namespace nalu
} // namespace Sierra
