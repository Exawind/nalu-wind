// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <TpetraLinearSystem.h>
#include <TpetraLinearSystemHelpers.h>
#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <FieldTypeDef.h>
#include <DgInfo.h>
#include <Realm.h>
#include <PeriodicManager.h>
#include <Simulation.h>
#include <LinearSolver.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementRepo.h>
#include <EquationSystem.h>
#include <NaluEnv.h>
#include <utils/StkHelpers.h>
#include <utils/CreateDeviceExpression.h>
#include <ngp_utils/NgpLoopUtils.h>
#include <ngp_utils/NgpFieldManager.h>

#ifdef NALU_HAS_MATRIXFREE
#include <matrix_free/NodeOrderMap.h>
#endif

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
#include <stk_mesh/base/NgpMesh.hpp>

// For Tpetra support
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
namespace sierra {
namespace nalu {

///====================================================================================================================================
///======== T P E T R A
///===============================================================================================================
///====================================================================================================================================

//==========================================================================
// Class Definition
//==========================================================================
// TpetraLinearSystem - hook to Tpetra
//==========================================================================
TpetraLinearSystem::TpetraLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver)
  : LinearSystem(realm, numDof, eqSys, linearSolver)
{
}

TpetraLinearSystem::~TpetraLinearSystem()
{
  // dereference linear solver in safe manner
  if (linearSolver_ != nullptr) {
    linearSolver_->destroyLinearSolver();
  }
}

struct CompareEntityEqualById
{
  const stk::mesh::BulkData& m_mesh;
  const GlobalIdFieldType* m_naluGlobalId;

  CompareEntityEqualById(
    const stk::mesh::BulkData& mesh, const GlobalIdFieldType* naluGlobalId)
    : m_mesh(mesh), m_naluGlobalId(naluGlobalId)
  {
  }

  bool operator()(const stk::mesh::Entity& e0, const stk::mesh::Entity& e1)
  {
    const stk::mesh::EntityId e0Id =
      *stk::mesh::field_data(*m_naluGlobalId, e0);
    const stk::mesh::EntityId e1Id =
      *stk::mesh::field_data(*m_naluGlobalId, e1);
    return e0Id == e1Id;
  }
};

struct CompareEntityById
{
  const stk::mesh::BulkData& m_mesh;
  const GlobalIdFieldType* m_naluGlobalId;

  CompareEntityById(
    const stk::mesh::BulkData& mesh, const GlobalIdFieldType* naluGlobalId)
    : m_mesh(mesh), m_naluGlobalId(naluGlobalId)
  {
  }

  bool operator()(const stk::mesh::Entity& e0, const stk::mesh::Entity& e1)
  {
    const stk::mesh::EntityId e0Id =
      *stk::mesh::field_data(*m_naluGlobalId, e0);
    const stk::mesh::EntityId e1Id =
      *stk::mesh::field_data(*m_naluGlobalId, e1);
    return e0Id < e1Id;
  }
  bool operator()(const Connection& c0, const Connection& c1)
  {
    const stk::mesh::EntityId c0firstId =
      *stk::mesh::field_data(*m_naluGlobalId, c0.first);
    const stk::mesh::EntityId c1firstId =
      *stk::mesh::field_data(*m_naluGlobalId, c1.first);
    if (c0firstId != c1firstId) {
      return c0firstId < c1firstId;
    }
    const stk::mesh::EntityId c0secondId =
      *stk::mesh::field_data(*m_naluGlobalId, c0.second);
    const stk::mesh::EntityId c1secondId =
      *stk::mesh::field_data(*m_naluGlobalId, c1.second);
    return c0secondId < c1secondId;
  }
};

// determines whether the node is to be put into which map/graph/matrix
// FIXME - note that the DOFStatus enum can be Or'd together if need be to
//   distinguish ever more complicated situations, for example, a DOF that
//   is both owned and ghosted: OwnedDOF | GhostedDOF
int
TpetraLinearSystem::getDofStatus(stk::mesh::Entity node)
{
  return getDofStatus_impl(node, realm_);
}

void
TpetraLinearSystem::beginLinearSystemConstruction()
{
  if (inConstruction_)
    return;
  inConstruction_ = true;
  STK_ThrowRequire(ownedGraph_.is_null());
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  stk::mesh::MetaData& metaData = realm_.meta_data();

  // create a localID for all active nodes in the mesh...
  const stk::mesh::Selector s_universal =
    metaData.universal_part() & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_universal);

  // we allow for ghosted nodes when nonconformal is active. When periodic is
  // active, we may also have ghosted nodes due to the periodicGhosting.
  // However, we want to exclude these nodes

  LocalOrdinal numGhostNodes = 0;
  LocalOrdinal numOwnedNodes = 0;
  LocalOrdinal numNodes = 0;
  LocalOrdinal numSharedNotOwnedNotLocallyOwned =
    0; // these are nodes on other procs
  // First, get the number of owned and sharedNotOwned (or
  // num_sharedNotOwned_nodes = num_nodes - num_owned_nodes)
  // KOKKOS: BucketLoop parallel "reduce" is accumulating 4 sums
  kokkos_parallel_for(
    "Nalu::TpetraLinearSystem::beginLinearSystemConstructionA", buckets.size(),
    [&](const int& ib) {
      stk::mesh::Bucket& b = *buckets[ib];
      const stk::mesh::Bucket::size_type length = b.size();
      // KOKKOS: intra BucketLoop parallel reduce
      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

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
  sharedNotOwnedGids.reserve(numSharedNotOwnedNotLocallyOwned * numDof_);
  sharedPids_.reserve(sharedNotOwnedGids.capacity());

  // owned first:
  for (const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket& b = *bptr;
    for (stk::mesh::Entity entity : b) {
      int status = getDofStatus(entity);
      if (!(status & DS_SkippedDOF) && (status & DS_OwnedDOF))
        owned_nodes.push_back(entity);
    }
  }

  std::sort(
    owned_nodes.begin(), owned_nodes.end(),
    CompareEntityById(bulkData, realm_.naluGlobalId_));

  // use the Contiguous Map constructor.

  const Teuchos::RCP<LinSys::Comm> tpetraComm =
    Teuchos::rcp(new LinSys::Comm(bulkData.parallel()));
  ownedRowsMap_ = Teuchos::rcp(new LinSys::Map(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
    maxOwnedRowId_, // must have *numDof_
    1, tpetraComm));
  myLIDs_.clear();
  myLIDs_.reserve(maxOwnedRowId_ + numSharedNotOwnedNotLocallyOwned * numDof_);

  LocalOrdinal localId = 0;
  int gstart = 0;

  GlobalOrdinal gomin = ownedRowsMap_->getMinGlobalIndex();

  for (stk::mesh::Entity entity : owned_nodes) {
    const stk::mesh::EntityId entityId =
      *stk::mesh::field_data(*realm_.naluGlobalId_, entity);
    myLIDs_[entityId] = numDof_ * localId++;
    auto* thisgid = stk::mesh::field_data(*realm_.tpetGlobalId_, entity);
    auto basegid = gomin + numDof_ * gstart;
    (*thisgid) = basegid;
    for (unsigned idof = 0; idof < numDof_; ++idof)
      ownedGids.push_back(basegid + idof);
    gstart++;
  }
  STK_ThrowRequire(localId == numOwnedNodes);
  // communicate the newly stored GID's.

  std::vector<const stk::mesh::FieldBase*> fVec{realm_.tpetGlobalId_};
  stk::mesh::copy_owned_to_shared(bulkData, fVec);
  stk::mesh::communicate_field_data(bulkData.aura_ghosting(), fVec);
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

  if (
    realm_.periodicManager_ != nullptr &&
    realm_.periodicManager_->periodicGhosting_ != nullptr) {
    realm_.periodicManager_->parallel_communicate_field(realm_.tpetGlobalId_);
    realm_.periodicManager_->periodic_parallel_communicate_field(
      realm_.tpetGlobalId_);
  }

  // now sharedNotOwned:
  for (const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket& b = *bptr;
    for (stk::mesh::Entity node : b) {
      int status = getDofStatus(node);
      if (!(status & DS_SkippedDOF) && (status & DS_SharedNotOwnedDOF))
        shared_not_owned_nodes.push_back(node);
    }
  }
  std::sort(
    shared_not_owned_nodes.begin(), shared_not_owned_nodes.end(),
    CompareEntityById(bulkData, realm_.naluGlobalId_));
  std::vector<stk::mesh::Entity>::iterator iter = std::unique(
    shared_not_owned_nodes.begin(), shared_not_owned_nodes.end(),
    CompareEntityEqualById(bulkData, realm_.naluGlobalId_));
  shared_not_owned_nodes.erase(iter, shared_not_owned_nodes.end());

  for (unsigned inode = 0; inode < shared_not_owned_nodes.size(); ++inode) {
    stk::mesh::Entity entity = shared_not_owned_nodes[inode];
    const stk::mesh::EntityId naluId =
      *stk::mesh::field_data(*realm_.naluGlobalId_, entity);
    auto masterentity = get_entity_master(bulkData, entity, naluId);
    myLIDs_[naluId] = numDof_ * localId++;
    int owner = bulkData.parallel_owner_rank(masterentity);
    auto basegid = *stk::mesh::field_data(*realm_.tpetGlobalId_, masterentity);

    if (entity != masterentity)
      *stk::mesh::field_data(*realm_.tpetGlobalId_, entity) = basegid;

    for (unsigned idof = 0; idof < numDof_; ++idof) {
      GlobalOrdinal gid = basegid + idof;
      sharedNotOwnedGids.push_back(gid);
      sharedPids_.push_back(owner);
    }
  }

  sharedNotOwnedRowsMap_ = Teuchos::rcp(new LinSys::Map(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
    sharedNotOwnedGids, 1, tpetraComm));

  exporter_ =
    Teuchos::rcp(new LinSys::Export(sharedNotOwnedRowsMap_, ownedRowsMap_));

  if (realm_.matrix_free()) {
    // assume owned and shared-not-owned are disjoint
    std::vector<GlobalOrdinal> ownedAndSharedGids = ownedGids;
    ownedAndSharedGids.insert(
      ownedAndSharedGids.end(), sharedNotOwnedGids.begin(),
      sharedNotOwnedGids.end());
    ownedAndSharedRowsMap_ = Teuchos::rcp(new LinSys::Map(
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
      ownedAndSharedGids, 1, tpetraComm));
  }

  fill_entity_to_row_LID_mapping();
  ownedAndSharedNodes_.reserve(
    owned_nodes.size() + shared_not_owned_nodes.size());
  ownedAndSharedNodes_ = owned_nodes;
  ownedAndSharedNodes_.insert(
    ownedAndSharedNodes_.end(), shared_not_owned_nodes.begin(),
    shared_not_owned_nodes.end());
  connections_.resize(ownedAndSharedNodes_.size());
  for (std::vector<stk::mesh::Entity>& vec : connections_) {
    vec.reserve(8);
  }
}

int
TpetraLinearSystem::insert_connection(stk::mesh::Entity a, stk::mesh::Entity b)
{
  size_t idx = entityToLIDHost_[a.local_offset()] / numDof_;

  STK_ThrowRequireMsg(
    idx < ownedAndSharedNodes_.size(),
    "Error, insert_connection got index out of range.");

  bool correctEntity = ownedAndSharedNodes_[idx] == a;
  if (!correctEntity) {
    const stk::mesh::EntityId naluid_a =
      *stk::mesh::field_data(*realm_.naluGlobalId_, a);
    stk::mesh::Entity master =
      get_entity_master(realm_.bulk_data(), a, naluid_a);
    const stk::mesh::EntityId naluid_master =
      *stk::mesh::field_data(*realm_.naluGlobalId_, master);
    correctEntity =
      ownedAndSharedNodes_[idx] == master || naluid_a == naluid_master;
  }
  STK_ThrowRequireMsg(
    correctEntity,
    "Error, indexing of rowEntities to connections isn't right.");

  std::vector<stk::mesh::Entity>& vec = connections_[idx];
  if (std::find(vec.begin(), vec.end(), b) == vec.end()) {
    vec.push_back(b);
  }
  return 0;
}

void
TpetraLinearSystem::addConnections(
  const stk::mesh::Entity* entities, const size_t& num_entities)
{
  for (size_t a = 0; a < num_entities; ++a) {
    const stk::mesh::Entity entity_a = entities[a];
    const stk::mesh::EntityId id_a =
      *stk::mesh::field_data(*realm_.naluGlobalId_, entity_a);
    insert_connection(entity_a, entity_a);

    for (size_t b = a + 1; b < num_entities; ++b) {
      const stk::mesh::Entity entity_b = entities[b];
      const stk::mesh::EntityId id_b =
        *stk::mesh::field_data(*realm_.naluGlobalId_, entity_b);
      const bool a_then_b = id_a < id_b;
      const stk::mesh::Entity entity_min = a_then_b ? entity_a : entity_b;
      const stk::mesh::Entity entity_max = a_then_b ? entity_b : entity_a;
      insert_connection(entity_min, entity_max);
    }
  }
}

void
TpetraLinearSystem::buildNodeGraph(const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned =
    metaData.locally_owned_part() & stk::mesh::selectUnion(parts) &
    !(stk::mesh::selectUnion(realm_.get_slave_part_vector())) &
    !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_owned);
  for (size_t ib = 0; ib < buckets.size(); ++ib) {
    const stk::mesh::Bucket& b = *buckets[ib];
    const stk::mesh::Bucket::size_type length = b.size();
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      stk::mesh::Entity node = b[k];
      addConnections(&node, 1);
    }
  }
}

void
TpetraLinearSystem::buildConnectedNodeGraph(
  stk::mesh::EntityRank rank, const stk::mesh::PartVector& parts)
{
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(rank, s_owned);

  for (size_t ib = 0; ib < buckets.size(); ++ib) {
    const stk::mesh::Bucket& b = *buckets[ib];
    const stk::mesh::Bucket::size_type length = b.size();
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const unsigned numNodes = b.num_nodes(k);
      stk::mesh::Entity const* nodes = b.begin_nodes(k);

      addConnections(nodes, numNodes);
    }
  }
}

void
TpetraLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();
  buildConnectedNodeGraph(stk::topology::EDGE_RANK, parts);
}

void
TpetraLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();
  buildConnectedNodeGraph(metaData.side_rank(), parts);
}

void
TpetraLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();
  buildConnectedNodeGraph(stk::topology::ELEM_RANK, parts);
}

void
TpetraLinearSystem::buildReducedElemToNodeGraph(
  const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::ELEMENT_RANK, s_owned);
  std::vector<stk::mesh::Entity> entities;
  for (size_t ib = 0; ib < buckets.size(); ++ib) {
    const stk::mesh::Bucket& b = *buckets[ib];

    // extract master element
    MasterElement* meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
        b.topology());
    // extract master element specifics
    const int numScsIp = meSCS->num_integration_points();
    const int* lrscv = meSCS->adjacentNodes();

    const stk::mesh::Bucket::size_type length = b.size();
    // KOKKOS: intra BucketLoop noparallel addConnections insert (std::set)
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      stk::mesh::Entity const* elem_nodes = b.begin_nodes(k);

      const size_t numNodes = 2;
      entities.resize(numNodes);
      // KOKKOS: nested Loop noparallel addConnections insert (std::set)
      for (int j = 0; j < numScsIp; ++j) {
        // KOKKOS: nested Loop parallel
        for (size_t n = 0; n < numNodes; ++n) {
          entities[n] = elem_nodes[lrscv[2 * j + n]];
        }
        addConnections(entities.data(), entities.size());
      }
    }
  }
}

void
TpetraLinearSystem::buildSparsifiedEdgeElemToNodeGraph(
  const stk::mesh::Selector& sel)
{
  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned =
    metaData.locally_owned_part() & sel & !(realm_.get_inactive_selector());

  constexpr int edge_conn[12][2][3] = {
    // bottom face
    {{0, 0, 0}, {0, 0, 1}},
    {{0, 0, 1}, {0, 1, 1}},
    {{0, 1, 1}, {0, 1, 0}},
    {{0, 1, 0}, {0, 0, 0}},

    // top face
    {{1, 0, 0}, {1, 0, 1}},
    {{1, 0, 1}, {1, 1, 1}},
    {{1, 1, 1}, {1, 1, 0}},
    {{1, 1, 0}, {1, 0, 0}},

    // edges from bottom to top
    {{0, 0, 0}, {1, 0, 0}},
    {{0, 0, 1}, {1, 0, 1}},
    {{0, 1, 1}, {1, 1, 1}},
    {{0, 1, 0}, {1, 1, 0}},
  };

  const int poly = realm_.polynomial_order();
  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::ELEMENT_RANK, s_owned);
  std::array<stk::mesh::Entity, 2> entities;
  for (const auto* ib : buckets) {
    const auto& b = *ib;
    if (poly == 1) {
      STK_ThrowRequire(b.topology() == stk::topology::HEX_8);
    } else if (poly == 2) {
      STK_ThrowRequire(b.topology() == stk::topology::HEX_27);
    } else {
      STK_ThrowRequire(b.topology().is_superelement());
    }
    for (size_t k = 0u; k < b.size(); ++k) {
      stk::mesh::Entity const* elem_nodes = b.begin_nodes(k);
      for (int n = 0; n < poly; ++n) {
        for (int m = 0; m < poly; ++m) {
          for (int l = 0; l < poly; ++l) {

            for (int iedge = 0; iedge < 12; ++iedge) {
              for (int lr = 0; lr < 2; ++lr) {
                const auto sub_n_index = n + edge_conn[iedge][lr][0];
                const auto sub_m_index = m + edge_conn[iedge][lr][1];
                const auto sub_l_index = l + edge_conn[iedge][lr][2];

#ifdef NALU_HAS_MATRIXFREE
                auto node_index = matrix_free::node_map(
                  poly, sub_n_index, sub_m_index, sub_l_index);
#else
                constexpr int nmap[2][2][2] = {
                  {{0, 1}, {3, 2}}, {{4, 5}, {7, 6}}};
                auto node_index = nmap[sub_n_index][sub_m_index][sub_l_index];
#endif
                entities[lr] = elem_nodes[node_index];
              }
              addConnections(entities.data(), 2u);
            }
          }
        }
      }
    }
  }
}

void
TpetraLinearSystem::buildFaceElemToNodeGraph(const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets(metaData.side_rank(), s_owned);

  for (size_t ib = 0; ib < face_buckets.size(); ++ib) {
    const stk::mesh::Bucket& b = *face_buckets[ib];
    const stk::mesh::Bucket::size_type length = b.size();
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const stk::mesh::Entity face = b[k];

      // extract the connected element to this exposed face; should be single in
      // size!
      const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
      STK_ThrowAssert(bulkData.num_elements(face) == 1);

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
TpetraLinearSystem::buildNonConformalNodeGraph(
  const stk::mesh::PartVector& /* parts */)
{
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  beginLinearSystemConstruction();

  std::vector<stk::mesh::Entity> entities;

  // iterate nonConformalManager's dgInfoVecs
  for (NonConformalInfo* nonConfInfo :
       realm_.nonConformalManager_->nonConformalInfoVec_) {

    std::vector<std::vector<DgInfo*>>& dgInfoVec = nonConfInfo->dgInfoVec_;

    for (std::vector<DgInfo*>& faceDgInfoVec : dgInfoVec) {

      // now loop over all the DgInfo objects on this particular exposed face
      for (size_t k = 0; k < faceDgInfoVec.size(); ++k) {

        DgInfo* dgInfo = faceDgInfoVec[k];

        // extract current/opposing element
        stk::mesh::Entity currentElement = dgInfo->currentElement_;
        stk::mesh::Entity opposingElement = dgInfo->opposingElement_;

        // node relations; current and opposing
        stk::mesh::Entity const* current_elem_node_rels =
          bulkData.begin_nodes(currentElement);
        const int current_num_elem_nodes = bulkData.num_nodes(currentElement);
        stk::mesh::Entity const* opposing_elem_node_rels =
          bulkData.begin_nodes(opposingElement);
        const int opposing_num_elem_nodes = bulkData.num_nodes(opposingElement);

        // resize based on both current and opposing face node size
        entities.resize(current_num_elem_nodes + opposing_num_elem_nodes);

        // fill in connected nodes; current
        // KOKKOS: nested Loop parallel
        for (int ni = 0; ni < current_num_elem_nodes; ++ni) {
          entities[ni] = current_elem_node_rels[ni];
        }

        // fill in connected nodes; opposing
        // KOKKOS: nested Loop parallel
        for (int ni = 0; ni < opposing_num_elem_nodes; ++ni) {
          entities[current_num_elem_nodes + ni] = opposing_elem_node_rels[ni];
        }

        // okay, now add the connections; will be symmetric
        // columns of current node row (opposing nodes) will add columns to
        // opposing nodes row
        addConnections(entities.data(), entities.size());
      }
    }
  }
}

void
TpetraLinearSystem::buildOversetNodeGraph(
  const stk::mesh::PartVector& /* parts */)
{
  // extract the rank
  const int theRank = NaluEnv::self().parallel_rank();

  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  beginLinearSystemConstruction();

  std::vector<stk::mesh::Entity> entities;

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

    entities[0] = orphanNode;
    for (size_t n = 0; n < numNodes; ++n) {
      entities[n + 1] = elem_nodes[n];
    }
    addConnections(entities.data(), entities.size());
  }
}

void
TpetraLinearSystem::copy_stk_to_tpetra(
  const stk::mesh::FieldBase* stkField,
  const Teuchos::RCP<LinSys::MultiVector> tpetraField)
{
  STK_ThrowAssert(!tpetraField.is_null());
  STK_ThrowAssert(stkField);
  const int numVectors = tpetraField->getNumVectors();

  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const stk::mesh::Selector selector =
    stk::mesh::selectField(*stkField) & metaData.locally_owned_part() &
    !(stk::mesh::selectUnion(realm_.get_slave_part_vector())) &
    !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    bulkData.get_buckets(stk::topology::NODE_RANK, selector);

  for (const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket& b = *bptr;

    const int fieldSize =
      field_bytes_per_entity(*stkField, b) / (sizeof(double));

    STK_ThrowRequireMsg(
      numVectors == fieldSize, "TpetraLinearSystem::copy_stk_to_tpetra");

    const stk::mesh::Bucket::size_type length = b.size();

    const double* stkFieldPtr = (double*)stk::mesh::field_data(*stkField, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const stk::mesh::Entity node = b[k];

      int status = getDofStatus(node);
      if ((status & DS_SkippedDOF) || (status & DS_SharedNotOwnedDOF))
        continue;

      const stk::mesh::EntityId nodeTpetGID =
        *stk::mesh::field_data(*realm_.tpetGlobalId_, node);
      STK_ThrowRequireMsg(
        nodeTpetGID != 0 &&
          nodeTpetGID != static_cast<stk::mesh::EntityId>(
                           std::numeric_limits<LinSys::GlobalOrdinal>::max()),
        " in copy_stk_to_tpetra ");
      for (int d = 0; d < fieldSize; ++d) {
        const size_t stkIndex = k * fieldSize + d;
        tpetraField->replaceGlobalValue(nodeTpetGID, d, stkFieldPtr[stkIndex]);
      }
    }
  }
}

void
TpetraLinearSystem::compute_send_lengths(
  const std::vector<stk::mesh::Entity>& rowEntities,
  const std::vector<std::vector<stk::mesh::Entity>>& connections,
  const std::vector<int>& neighborProcs,
  stk::CommNeighbors& commNeighbors)
{
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  std::vector<int> sendLengths(neighborProcs.size(), 0);
  size_t maxColEntities = 128;
  std::vector<stk::mesh::EntityId> colEntityIds(maxColEntities);

  for (size_t i = 0; i < rowEntities.size(); ++i) {
    const stk::mesh::Entity entity_a = rowEntities[i];
    const std::vector<stk::mesh::Entity>& colEntities = connections[i];
    unsigned numColEntities = colEntities.size();
    colEntityIds.resize(numColEntities);
    for (size_t j = 0; j < colEntities.size(); ++j) {
      colEntityIds[j] =
        *stk::mesh::field_data(*realm_.naluGlobalId_, colEntities[j]);
    }

    const stk::mesh::EntityId entityId_a =
      *stk::mesh::field_data(*realm_.naluGlobalId_, entity_a);
    const int entity_a_status = getDofStatus(entity_a);
    const bool entity_a_shared = entity_a_status & DS_SharedNotOwnedDOF;

    if (entity_a_shared) {
      stk::mesh::Entity master = get_entity_master(bulk, entity_a, entityId_a);
      size_t idx =
        get_neighbor_index(neighborProcs, bulk.parallel_owner_rank(master));
      sendLengths[idx] +=
        (1 + numColEntities) * (sizeof(GlobalOrdinal) + sizeof(int));
    }

    for (size_t ii = 0; ii < numColEntities; ++ii) {
      const stk::mesh::Entity entity_b = colEntities[ii];
      if (entity_b == entity_a) {
        continue;
      }
      const stk::mesh::EntityId entityId_b = colEntityIds[ii];
      const int entity_b_status =
        (entityId_a != entityId_b) ? getDofStatus(entity_b) : entity_a_status;
      const bool entity_b_shared = entity_b_status & DS_SharedNotOwnedDOF;
      if (entity_b_shared) {
        stk::mesh::Entity master =
          get_entity_master(bulk, entity_b, entityId_b);
        size_t idx =
          get_neighbor_index(neighborProcs, bulk.parallel_owner_rank(master));
        sendLengths[idx] +=
          (1 + numColEntities) * (sizeof(GlobalOrdinal) + sizeof(int));
      }
    }
  }

  for (size_t i = 0; i < neighborProcs.size(); ++i) {
    stk::CommBufferV& sbuf = commNeighbors.send_buffer(neighborProcs[i]);
    sbuf.reserve(sendLengths[i]);
  }
}

void
TpetraLinearSystem::compute_graph_row_lengths(
  const std::vector<stk::mesh::Entity>& rowEntities,
  const std::vector<std::vector<stk::mesh::Entity>>& connections,
  LinSys::RowLengths& sharedNotOwnedRowLengths,
  LinSys::RowLengths& locallyOwnedRowLengths,
  stk::CommNeighbors& commNeighbors)
{
  LinSys::HostRowLengths hostSharedNotOwnedRowLengths =
    sharedNotOwnedRowLengths.view<HostSpace>();
  LinSys::HostRowLengths hostLocallyOwnedRowLengths =
    locallyOwnedRowLengths.view<HostSpace>();

  const stk::mesh::BulkData& bulk = realm_.bulk_data();

  size_t maxColEntities = 128;
  std::vector<stk::mesh::EntityId> colEntityIds(maxColEntities);
  std::vector<int> colOwners(maxColEntities);

  for (size_t i = 0; i < rowEntities.size(); ++i) {
    const std::vector<stk::mesh::Entity>& colEntities = connections[i];
    unsigned numColEntities = colEntities.size();
    const stk::mesh::Entity entity_a = rowEntities[i];
    colEntityIds.resize(numColEntities);
    colOwners.resize(numColEntities);
    for (size_t j = 0; j < numColEntities; ++j) {
      stk::mesh::Entity colEntity = colEntities[j];
      colEntityIds[j] =
        *stk::mesh::field_data(*realm_.naluGlobalId_, colEntity);
      colOwners[j] = bulk.parallel_owner_rank(
        get_entity_master(bulk, colEntity, colEntityIds[j]));
    }

    const stk::mesh::EntityId entityId_a =
      *stk::mesh::field_data(*realm_.naluGlobalId_, entity_a);

    const int entity_a_status = getDofStatus(entity_a);
    const bool entity_a_owned = entity_a_status & DS_OwnedDOF;
    LocalOrdinal lid_a = entityToLIDHost_[entity_a.local_offset()];
    stk::mesh::Entity entity_a_master =
      get_entity_master(bulk, entity_a, entityId_a);
    int entity_a_owner = bulk.parallel_owner_rank(entity_a_master);

    add_to_length(
      hostLocallyOwnedRowLengths, hostSharedNotOwnedRowLengths, numDof_, lid_a,
      maxOwnedRowId_, entity_a_owned, numColEntities);

    const bool entity_a_shared = entity_a_status & DS_SharedNotOwnedDOF;
    if (entity_a_shared) {
      add_lengths_to_comm_tpet(
        bulk, realm_.tpetGlobalId_, commNeighbors, entity_a_owner, entityId_a,
        //                               numDof_,
        numColEntities, colEntityIds.data(), colOwners.data());
    }

    for (size_t ii = 0; ii < numColEntities; ++ii) {
      const stk::mesh::Entity entity_b = colEntities[ii];
      if (entity_b == entity_a) {
        continue;
      }
      const stk::mesh::EntityId entityId_b = colEntityIds[ii];
      const int entity_b_status = getDofStatus(entity_b);
      const bool entity_b_owned = entity_b_status & DS_OwnedDOF;
      LocalOrdinal lid_b = entityToLIDHost_[entity_b.local_offset()];
      add_to_length(
        hostLocallyOwnedRowLengths, hostSharedNotOwnedRowLengths, numDof_,
        lid_b, maxOwnedRowId_, entity_b_owned, 1);

      const bool entity_b_shared = entity_b_status & DS_SharedNotOwnedDOF;
      if (entity_b_shared) {
        add_lengths_to_comm_tpet(
          bulk, realm_.tpetGlobalId_, commNeighbors, colOwners[ii], entityId_b,
          // numDof_,
          1, &entityId_a, &entity_a_owner);
      }
    }
  }

  sync_dual_view_host_to_device(sharedNotOwnedRowLengths);
  sync_dual_view_host_to_device(locallyOwnedRowLengths);
}

void
TpetraLinearSystem::insert_graph_connections(
  const std::vector<stk::mesh::Entity>& rowEntities,
  const std::vector<std::vector<stk::mesh::Entity>>& connections,
  LocalGraphArrays& locallyOwnedGraph,
  LocalGraphArrays& sharedNotOwnedGraph)
{
  std::vector<LocalOrdinal> localDofs_a(1);
  unsigned max = 128;
  std::vector<int> dofStatus(max);
  std::vector<LocalOrdinal> localDofs_b(max);

  // KOKKOS: Loop noparallel Graph insert
  for (size_t i = 0; i < rowEntities.size(); ++i) {
    const std::vector<stk::mesh::Entity>& entities_b = connections[i];
    unsigned numColEntities = entities_b.size();
    dofStatus.resize(numColEntities);
    localDofs_b.resize(numColEntities);

    const stk::mesh::Entity entity_a = rowEntities[i];
    int dofStatus_a = getDofStatus(entity_a);
    STK_ThrowRequireMsg(
      entityToColLIDHost_[entity_a.local_offset()] != -1,
      "insert_graph_connections bad lid ");
    localDofs_a[0] = entityToColLIDHost_[entity_a.local_offset()];
    for (size_t j = 0; j < numColEntities; ++j) {
      const stk::mesh::Entity entity_b = entities_b[j];
      dofStatus[j] = getDofStatus(entity_b);

      STK_ThrowRequireMsg(
        entityToColLIDHost_[entity_b.local_offset()] != -1,
        "insert_graph_connections bad lid #2 ");

      localDofs_b[j] = entityToColLIDHost_[entity_b.local_offset()];
    }

    {
      LocalGraphArrays& crsGraph =
        (dofStatus_a & DS_OwnedDOF) ? locallyOwnedGraph : sharedNotOwnedGraph;
      insert_single_dof_row_into_graph(
        crsGraph, entityToLIDHost_[entity_a.local_offset()], maxOwnedRowId_,
        numDof_, numColEntities, localDofs_b);
    }

    for (unsigned j = 0; j < numColEntities; ++j) {
      if (entities_b[j] != entity_a) {
        LocalGraphArrays& crsGraph = (dofStatus[j] & DS_OwnedDOF)
                                       ? locallyOwnedGraph
                                       : sharedNotOwnedGraph;
        insert_single_dof_row_into_graph(
          crsGraph, entityToLIDHost_[entities_b[j].local_offset()],
          maxOwnedRowId_, numDof_, 1, localDofs_a);
      }
    }
  }
}

void
TpetraLinearSystem::fill_entity_to_row_LID_mapping()
{
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  stk::mesh::Selector selector =
    bulk.mesh_meta_data().universal_part() & !(realm_.get_inactive_selector());
  entityToLIDHost_ = LinSys::EntityToLIDHostView(
    "entityToLID", bulk.get_size_of_entity_index_space());
  entityToLID_ = Kokkos::create_mirror_view(LinSysMemSpace(), entityToLIDHost_);
  const stk::mesh::BucketVector& nodeBuckets =
    realm_.get_buckets(stk::topology::NODE_RANK, selector);
  for (const stk::mesh::Bucket* bptr : nodeBuckets) {
    const stk::mesh::Bucket& b = *bptr;
    const stk::mesh::EntityId* nodeIds =
      stk::mesh::field_data(*realm_.naluGlobalId_, b);
    for (size_t i = 0; i < b.size(); ++i) {
      stk::mesh::Entity node = b[i];

      MyLIDMapType::const_iterator iter = myLIDs_.find(nodeIds[i]);
      if (iter != myLIDs_.end()) {
        entityToLIDHost_[node.local_offset()] = iter->second;
        if (nodeIds[i] != bulk.identifier(node)) {
          stk::mesh::Entity master = get_entity_master(bulk, node, nodeIds[i]);
          if (master != node) {
            entityToLIDHost_[master.local_offset()] =
              entityToLIDHost_[node.local_offset()];
          }
        }
      }
    }
  }

  Kokkos::deep_copy(entityToLID_, entityToLIDHost_);
}

void
TpetraLinearSystem::fill_entity_to_col_LID_mapping()
{
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  stk::mesh::Selector selector =
    bulk.mesh_meta_data().universal_part() & !(realm_.get_inactive_selector());
  entityToColLIDHost_ = LinSys::EntityToLIDHostView(
    "entityToColLID", bulk.get_size_of_entity_index_space());
  entityToColLID_ =
    Kokkos::create_mirror_view(LinSysMemSpace(), entityToColLIDHost_);
  const stk::mesh::BucketVector& nodeBuckets =
    realm_.get_buckets(stk::topology::NODE_RANK, selector);
  const bool throwIfMasterNotFound = false;
  for (const stk::mesh::Bucket* bptr : nodeBuckets) {
    const stk::mesh::Bucket& b = *bptr;
    const stk::mesh::EntityId* nodeIds =
      stk::mesh::field_data(*realm_.naluGlobalId_, b);
    for (size_t i = 0; i < b.size(); ++i) {
      stk::mesh::Entity node = b[i];

      GlobalOrdinal gid = -1;
      // needed because of some shared and ghost nodes.
      if (nodeIds[i] != bulk.identifier(node)) {
        stk::mesh::Entity master =
          get_entity_master(bulk, node, nodeIds[i], throwIfMasterNotFound);
        if (bulk.is_valid(master) && master != node) {
          gid = *stk::mesh::field_data(*realm_.tpetGlobalId_, master);
          entityToColLIDHost_[node.local_offset()] =
            totalColsMap_->getLocalElement(gid);
        } else if (!bulk.is_valid(master)) {
          gid = -1;
          entityToColLIDHost_[node.local_offset()] = gid;
        } else {
          gid = *stk::mesh::field_data(*realm_.tpetGlobalId_, node);
        }
      } else
        gid = *stk::mesh::field_data(*realm_.tpetGlobalId_, node);

      if (
        gid == 0 || gid == -1 ||
        gid == std::numeric_limits<LinSys::GlobalOrdinal>::max()) {
        // unit_test1 does produce ghost nodes that have a master that don't
        // have a valid tpetGlobalId_
        entityToColLIDHost_[node.local_offset()] = -1;
      } else
        entityToColLIDHost_[node.local_offset()] =
          totalColsMap_->getLocalElement(gid);
    }
  }

  Kokkos::deep_copy(entityToColLID_, entityToColLIDHost_);
}

void
TpetraLinearSystem::storeOwnersForShared()
{
  const stk::mesh::BulkData& bulkData = realm_.bulk_data();
  const stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector all =
    metaData.universal_part() & !(realm_.get_inactive_selector());
  const stk::mesh::BucketVector& buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, all);

  for (const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket& bkt = *bptr;
    for (stk::mesh::Entity node : bkt) {
      int status = getDofStatus(node);
      if (status & DS_SharedNotOwnedDOF) {
        stk::mesh::EntityId naluId =
          *stk::mesh::field_data(*realm_.naluGlobalId_, node);
        stk::mesh::Entity master = get_entity_master(bulkData, node, naluId);
        GlobalOrdinal gidbase =
          *stk::mesh::field_data(*realm_.tpetGlobalId_, master);
        STK_ThrowRequire(gidbase != 0);
        for (unsigned idof = 0; idof < numDof_; ++idof) {
          GlobalOrdinal gid = gidbase + idof;
          ownersAndGids_.insert(
            std::make_pair(bulkData.parallel_owner_rank(master), gid));
        }
      }
    }
  }
}

void
TpetraLinearSystem::finalizeLinearSystem()
{
  STK_ThrowRequire(inConstruction_);
  inConstruction_ = false;

  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  stk::mesh::MetaData& metaData = realm_.meta_data();

  sort_connections(connections_);

  size_t numSharedNotOwned =
    sharedNotOwnedRowsMap_->getMyGlobalIndices().extent(0);
  size_t numLocallyOwned = ownedRowsMap_->getMyGlobalIndices().extent(0);
  LinSys::RowLengths sharedNotOwnedRowLengths("rowLengths", numSharedNotOwned);
  LinSys::RowLengths locallyOwnedRowLengths("rowLengths", numLocallyOwned);
  LinSys::HostRowLengths globalRowLengths =
    sharedNotOwnedRowLengths.view<HostSpace>();
  LinSys::HostRowLengths ownedRowLengths =
    locallyOwnedRowLengths.view<HostSpace>();

  std::vector<int> neighborProcs;
  fill_neighbor_procs(neighborProcs, bulkData, realm_);

  stk::CommNeighbors commNeighbors(bulkData.parallel(), neighborProcs);

  compute_send_lengths(
    ownedAndSharedNodes_, connections_, neighborProcs, commNeighbors);
  compute_graph_row_lengths(
    ownedAndSharedNodes_, connections_, sharedNotOwnedRowLengths,
    locallyOwnedRowLengths, commNeighbors);

  ownersAndGids_.clear();
  storeOwnersForShared();

  communicate_remote_columns(
    bulkData, neighborProcs, commNeighbors, numDof_, ownedRowsMap_,
    ownedRowLengths, ownersAndGids_);

  sync_dual_view_host_to_device(locallyOwnedRowLengths);

  LocalGraphArrays ownedGraph(ownedRowLengths);
  LocalGraphArrays sharedNotOwnedGraph(globalRowLengths);

  int localProc = bulkData.parallel_rank();

  std::vector<GlobalOrdinal> optColGids;
  std::vector<int> sourcePIDs;
  fill_owned_and_shared_then_nonowned_ordered_by_proc(
    optColGids, sourcePIDs, localProc, ownedRowsMap_, sharedNotOwnedRowsMap_,
    ownersAndGids_, sharedPids_);

  const Teuchos::RCP<LinSys::Comm> tpetraComm =
    Teuchos::rcp(new LinSys::Comm(bulkData.parallel()));
  totalColsMap_ = Teuchos::rcp(new LinSys::Map(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), optColGids, 1,
    tpetraComm));

  fill_entity_to_col_LID_mapping();

  insert_graph_connections(
    ownedAndSharedNodes_, connections_, ownedGraph, sharedNotOwnedGraph);

  insert_communicated_col_indices(
    neighborProcs, commNeighbors, numDof_, ownedGraph, *ownedRowsMap_,
    *totalColsMap_);

  fill_in_extra_dof_rows_per_node(ownedGraph, numDof_);
  fill_in_extra_dof_rows_per_node(sharedNotOwnedGraph, numDof_);

  remove_invalid_indices(ownedGraph, ownedRowLengths);

  sharedNotOwnedGraph_ = Teuchos::rcp(new LinSys::Graph(
    sharedNotOwnedRowsMap_, totalColsMap_, sharedNotOwnedRowLengths));

  ownedGraph_ = Teuchos::rcp(
    new LinSys::Graph(ownedRowsMap_, totalColsMap_, locallyOwnedRowLengths));

  auto deviceOwnedGraphRowPointers = Kokkos::create_mirror_view_and_copy(
    LinSysMemSpace(), ownedGraph.rowPointers);
  auto deviceOwnedGraphColIndices = Kokkos::create_mirror_view_and_copy(
    LinSysMemSpace(), ownedGraph.colIndices);
  ownedGraph_->setAllIndices(
    deviceOwnedGraphRowPointers, deviceOwnedGraphColIndices);

  auto deviceSharedNotOwnedGraphRowPointers =
    Kokkos::create_mirror_view_and_copy(
      LinSysMemSpace(), sharedNotOwnedGraph.rowPointers);
  auto deviceSharedNotOwnedGraphColIndices =
    Kokkos::create_mirror_view_and_copy(
      LinSysMemSpace(), sharedNotOwnedGraph.colIndices);
  sharedNotOwnedGraph_->setAllIndices(
    deviceSharedNotOwnedGraphRowPointers, deviceSharedNotOwnedGraphColIndices);

  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::rcp(new Teuchos::ParameterList);
  params->set<bool>("No Nonlocal Changes", true);
  params->set<bool>("compute local triangular constants", false);

  bool allowedToReorderLocally = false;
  Teuchos::RCP<LinSys::Import> importer = Teuchos::rcp(new LinSys::Import(
    ownedRowsMap_, optColGids.data() + ownedRowLengths.size(),
    sourcePIDs.data(), sourcePIDs.size(), allowedToReorderLocally));

  ownedGraph_->expertStaticFillComplete(
    ownedRowsMap_, ownedRowsMap_, importer, Teuchos::null, params);
  sharedNotOwnedGraph_->expertStaticFillComplete(
    ownedRowsMap_, ownedRowsMap_, Teuchos::null, Teuchos::null, params);

  ownedMatrix_ = Teuchos::rcp(new LinSys::Matrix(ownedGraph_));
  sharedNotOwnedMatrix_ =
    Teuchos::rcp(new LinSys::Matrix(sharedNotOwnedGraph_));

  ownedRhs_ = Teuchos::rcp(new LinSys::MultiVector(ownedRowsMap_, 1));
  sharedNotOwnedRhs_ =
    Teuchos::rcp(new LinSys::MultiVector(sharedNotOwnedRowsMap_, 1));

  sln_ = Teuchos::rcp(new LinSys::MultiVector(ownedRowsMap_, 1));

  const int nDim = metaData.spatial_dimension();

  Teuchos::RCP<LinSys::MultiVector> coords = Teuchos::RCP<LinSys::MultiVector>(
    new LinSys::MultiVector(sln_->getMap(), nDim));

  TpetraLinearSolver* linearSolver =
    reinterpret_cast<TpetraLinearSolver*>(linearSolver_);

  if (linearSolver != nullptr) {
    VectorFieldType* coordinates = metaData.get_field<double>(
      stk::topology::NODE_RANK, realm_.get_coordinates_name());
    if (linearSolver->activeMueLu())
      copy_stk_to_tpetra(coordinates, coords);

    linearSolver->setupLinearSolver(sln_, ownedMatrix_, ownedRhs_, coords);
  }
}

void
TpetraLinearSystem::zeroSystem()
{
  STK_ThrowRequire(!ownedMatrix_.is_null());
  STK_ThrowRequire(!sharedNotOwnedMatrix_.is_null());
  STK_ThrowRequire(!sharedNotOwnedRhs_.is_null());
  STK_ThrowRequire(!ownedRhs_.is_null());

  sharedNotOwnedMatrix_->resumeFill();
  ownedMatrix_->resumeFill();

  sharedNotOwnedMatrix_->setAllToScalar(0);
  ownedMatrix_->setAllToScalar(0);
  sharedNotOwnedRhs_->putScalar(0);
  ownedRhs_->putScalar(0);

  sln_->putScalar(0);
}

template <typename RowViewType>
KOKKOS_FUNCTION void
sum_into_row_vec_3(
  RowViewType row_view,
  const int num_entities,
  const int* localIds,
  const int* sort_permutation,
  const double* input_values)
{
  // assumes that the flattened column indices for block matrices are all stored
  // sequentially specialized for numDof == 3
  constexpr bool forceAtomic =
    !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;
  const LocalOrdinal length = row_view.length;

  LocalOrdinal offset = 0;
  for (int j = 0; j < num_entities; ++j) {
    // since the columns are sorted, we pass through the column idxs once,
    // updating the offset as we go
    const int id_index = 3 * j;
    const LocalOrdinal cur_local_column_idx = localIds[id_index];
    while (row_view.colidx(offset) != cur_local_column_idx) {
      offset += 3;
      if (offset >= length)
        return;
    }

    const int entry_offset = sort_permutation[id_index];
    if (forceAtomic) {
      Kokkos::atomic_add(
        &row_view.value(offset + 0), input_values[entry_offset + 0]);
      Kokkos::atomic_add(
        &row_view.value(offset + 1), input_values[entry_offset + 1]);
      Kokkos::atomic_add(
        &row_view.value(offset + 2), input_values[entry_offset + 2]);
    } else {
      row_view.value(offset + 0) += input_values[entry_offset + 0];
      row_view.value(offset + 1) += input_values[entry_offset + 1];
      row_view.value(offset + 2) += input_values[entry_offset + 2];
    }
    offset += 3;
  }
}

template <typename RowViewType>
KOKKOS_FUNCTION void
sum_into_row(
  RowViewType row_view,
  const int num_entities,
  const int numDof,
  const int* localIds,
  const int* sort_permutation,
  const double* input_values)
{
  if (numDof == 3) {
    sum_into_row_vec_3(
      row_view, num_entities, localIds, sort_permutation, input_values);
    return;
  }

  constexpr bool forceAtomic =
    !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;
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
      STK_ThrowAssertMsg(
        std::isfinite(input_values[perm_index]), "Inf or NAN lhs");
      if (forceAtomic) {
        Kokkos::atomic_add(&(row_view.value(offset)), input_values[perm_index]);
      } else {
        row_view.value(offset) += input_values[perm_index];
      }
    }
  }
}

template <
  typename MatrixType,
  typename RhsType,
  typename EntityArrayType,
  typename ShmemView1DType,
  typename ShmemView2DType,
  typename ShmemIntView1DType,
  typename EntityLIDType>
KOKKOS_FUNCTION void
sum_into(
  MatrixType ownedLocalMatrix,
  MatrixType sharedNotOwnedLocalMatrix,
  RhsType ownedLocalRhs,
  RhsType sharedNotOwnedLocalRhs,
  unsigned numEntities,
  const EntityArrayType& entities,
  const ShmemView1DType& rhs,
  const ShmemView2DType& lhs,
  const ShmemIntView1DType& localIds,
  const ShmemIntView1DType& sortPermutation,
  const EntityLIDType& entityToLID,
  const EntityLIDType& entityToColLID,
  int maxOwnedRowId,
  int maxSharedNotOwnedRowId,
  unsigned numDof)
{
  constexpr bool forceAtomic =
    !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;

  const int n_obj = numEntities;
  const int numRows = n_obj * numDof;

  for (int i = 0; i < n_obj; i++) {
    const stk::mesh::Entity entity = entities[i];
    const LocalOrdinal localOffset = entityToColLID[entity.local_offset()];
    for (size_t d = 0; d < numDof; ++d) {
      size_t lid = i * numDof + d;
      localIds[lid] = localOffset + d;
    }
  }

  for (int i = 0; i < numRows; ++i) {
    sortPermutation[i] = i;
  }
  Tpetra::Details::shellSortKeysAndValues(
    localIds.data(), sortPermutation.data(), numRows);

  for (int r = 0; r < numRows; ++r) {
    int i = sortPermutation[r] / numDof;
    LocalOrdinal rowLid = entityToLID[entities[i].local_offset()];
    rowLid += sortPermutation[r] % numDof;
    const LocalOrdinal cur_perm_index = sortPermutation[r];
    const double* const cur_lhs = &lhs(cur_perm_index, 0);
    const double cur_rhs = rhs[cur_perm_index];
    //    STK_ThrowAssertMsg(std::isfinite(cur_rhs), "Inf or NAN rhs");

    if (rowLid < maxOwnedRowId) {
      sum_into_row(
        ownedLocalMatrix.row(rowLid), n_obj, numDof, localIds.data(),
        sortPermutation.data(), cur_lhs);
      if (forceAtomic) {
        Kokkos::atomic_add(&ownedLocalRhs(rowLid, 0), cur_rhs);
      } else {
        ownedLocalRhs(rowLid, 0) += cur_rhs;
      }
    } else if (rowLid < maxSharedNotOwnedRowId) {
      LocalOrdinal actualLocalId = rowLid - maxOwnedRowId;
      sum_into_row(
        sharedNotOwnedLocalMatrix.row(actualLocalId), n_obj, numDof,
        localIds.data(), sortPermutation.data(), cur_lhs);

      if (forceAtomic) {
        Kokkos::atomic_add(&sharedNotOwnedLocalRhs(actualLocalId, 0), cur_rhs);
      } else {
        sharedNotOwnedLocalRhs(actualLocalId, 0) += cur_rhs;
      }
    }
  }
}

template <typename RowViewType>
KOKKOS_FUNCTION void
reset_row(RowViewType row_view, const int localRowId, const double diag_value)
{
  const LocalOrdinal length = row_view.length;

  for (LocalOrdinal i = 0; i < length; ++i) {
    if (row_view.colidx(i) == localRowId) {
      row_view.value(i) = diag_value;
    } else {
      row_view.value(i) = 0.0;
    }
  }
}

template <
  typename MatrixType,
  typename RhsType,
  typename EntityArrayType,
  typename EntityLIDType>
KOKKOS_FUNCTION void
reset_rows(
  MatrixType ownedLocalMatrix,
  MatrixType sharedNotOwnedLocalMatrix,
  RhsType ownedLocalRhs,
  RhsType sharedNotOwnedLocalRhs,
  unsigned numNodes,
  const EntityArrayType& nodeList,
  unsigned beginPos,
  unsigned endPos,
  double diag_value,
  double rhs_residual,
  const EntityLIDType& entityToLID,
  int maxOwnedRowId,
  int maxSharedNotOwnedRowId)
{
  for (unsigned nn = 0; nn < numNodes; ++nn) {
    stk::mesh::Entity node = nodeList[nn];
    const LocalOrdinal localIdOffset = entityToLID[node.local_offset()];
    const bool useOwned = (localIdOffset < maxOwnedRowId);
    const LinSys::LocalMatrix& localMatrix =
      useOwned ? ownedLocalMatrix : sharedNotOwnedLocalMatrix;
    const LinSys::LocalVector& localRhs =
      useOwned ? ownedLocalRhs : sharedNotOwnedLocalRhs;

    for (unsigned d = beginPos; d < endPos; ++d) {
      const LocalOrdinal localId = localIdOffset + d;
      const LocalOrdinal actualLocalId =
        useOwned ? localId : (localId - maxOwnedRowId);

      STK_NGP_ThrowRequireMsg(localId <= maxSharedNotOwnedRowId, "Error");

      // Adjust the LHS; zero out all entries (including diagonal)
      reset_row(localMatrix.row(actualLocalId), actualLocalId, diag_value);

      // Replace RHS residual entry
      localRhs(actualLocalId, 0) = rhs_residual;
    }
  }
}

sierra::nalu::CoeffApplier*
TpetraLinearSystem::get_coeff_applier()
{
  auto ownedLocalMatrix = getOwnedLocalMatrix();
  auto sharedNotOwnedLocalMatrix = getSharedNotOwnedLocalMatrix();
  auto ownedLocalRhs = getOwnedLocalRhs();
  auto sharedNotOwnedLocalRhs = getSharedNotOwnedLocalRhs();
  auto entityToLID = entityToLID_;
  auto entityToColLID = entityToColLID_;
  auto maxOwnedRowId = maxOwnedRowId_;
  auto maxSharedNotOwnedRowId = maxSharedNotOwnedRowId_;
  auto numDof = numDof_;
  auto newDeviceCoeffApplier =
    kokkos_malloc_on_device<TpetraLinSysCoeffApplier>("deviceCoeffApplier");
  Kokkos::parallel_for(
    DeviceRangePolicy(0, 1), KOKKOS_LAMBDA(const int&) {
      new (newDeviceCoeffApplier) TpetraLinSysCoeffApplier(
        ownedLocalMatrix, sharedNotOwnedLocalMatrix, ownedLocalRhs,
        sharedNotOwnedLocalRhs, entityToLID, entityToColLID, maxOwnedRowId,
        maxSharedNotOwnedRowId, numDof);
    });

  return newDeviceCoeffApplier;
}

void
TpetraLinearSystem::free_coeff_applier(CoeffApplier* coeffApplier)
{
  if (coeffApplier != nullptr) {
    sierra::nalu::kokkos_free_on_device(coeffApplier);
  }
}

KOKKOS_FUNCTION
void
TpetraLinearSystem::TpetraLinSysCoeffApplier::resetRows(
  unsigned numNodes,
  const stk::mesh::Entity* nodeList,
  const unsigned beginPos,
  const unsigned endPos,
  const double diag_value,
  const double rhs_residual)
{
  reset_rows(
    ownedLocalMatrix_, sharedNotOwnedLocalMatrix_, ownedLocalRhs_,
    sharedNotOwnedLocalRhs_, numNodes, nodeList, beginPos, endPos, diag_value,
    rhs_residual, entityToLID_, maxOwnedRowId_, maxSharedNotOwnedRowId_);
}

KOKKOS_FUNCTION
void
TpetraLinearSystem::TpetraLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  sum_into(
    ownedLocalMatrix_, sharedNotOwnedLocalMatrix_, ownedLocalRhs_,
    sharedNotOwnedLocalRhs_, numEntities, entities, rhs, lhs, localIds,
    sortPermutation, entityToLID_, entityToColLID_, maxOwnedRowId_,
    maxSharedNotOwnedRowId_, numDof_);
}

void
TpetraLinearSystem::sumInto(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  const char* /* trace_tag */)
{
  STK_ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  STK_ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  STK_ThrowAssertMsg(
    localIds.span_is_contiguous(), "localIds assumed contiguous");
  STK_ThrowAssertMsg(
    sortPermutation.span_is_contiguous(), "sortPermutation assumed contiguous");

  sum_into(
    getOwnedLocalMatrix(), getSharedNotOwnedLocalMatrix(), getOwnedLocalRhs(),
    getSharedNotOwnedLocalRhs(), numEntities, entities, rhs, lhs, localIds,
    sortPermutation, entityToLIDHost_, entityToColLIDHost_, maxOwnedRowId_,
    maxSharedNotOwnedRowId_, numDof_);
}

void
TpetraLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>& scratchIds,
  std::vector<double>& /* scratchVals */,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char* /* trace_tag */)
{
  const size_t n_obj = entities.size();
  const unsigned numRows = n_obj * numDof_;

  STK_ThrowAssert(numRows == rhs.size());
  STK_ThrowAssert(numRows * numRows == lhs.size());

  scratchIds.resize(numRows);
  sortPermutation_.resize(numRows);
  for (size_t i = 0; i < n_obj; i++) {
    const stk::mesh::Entity entity = entities[i];
    const LocalOrdinal localOffset = entityToColLIDHost_[entity.local_offset()];
    STK_ThrowRequireMsg(localOffset != -1, "sumInto bad lid #2 ");
    for (size_t d = 0; d < numDof_; ++d) {
      size_t lid = i * numDof_ + d;
      scratchIds[lid] = localOffset + d;
    }
  }

  for (unsigned i = 0; i < numRows; ++i) {
    sortPermutation_[i] = i;
  }
  Tpetra::Details::shellSortKeysAndValues(
    scratchIds.data(), sortPermutation_.data(), (int)numRows);

  for (unsigned r = 0; r < numRows; r++) {
    int i = sortPermutation_[r] / numDof_;
    LocalOrdinal rowLid = entityToLIDHost_[entities[i].local_offset()];
    rowLid += sortPermutation_[r] % numDof_;
    const LocalOrdinal cur_perm_index = sortPermutation_[r];
    const double* const cur_lhs = &lhs[cur_perm_index * numRows];
    const double cur_rhs = rhs[cur_perm_index];
    STK_ThrowAssertMsg(std::isfinite(cur_rhs), "Invalid rhs");

    if (rowLid < maxOwnedRowId_) {
      sum_into_row(
        getOwnedLocalMatrix().row(rowLid), n_obj, numDof_, scratchIds.data(),
        sortPermutation_.data(), cur_lhs);
      getOwnedLocalRhs()(rowLid, 0) += cur_rhs;
    } else if (rowLid < maxSharedNotOwnedRowId_) {
      LocalOrdinal actualLocalId = rowLid - maxOwnedRowId_;
      sum_into_row(
        getSharedNotOwnedLocalMatrix().row(actualLocalId), n_obj, numDof_,
        scratchIds.data(), sortPermutation_.data(), cur_lhs);

      getSharedNotOwnedLocalRhs()(actualLocalId, 0) += cur_rhs;
    }
  }
}

template <typename RowViewType>
KOKKOS_FUNCTION void
adjust_lhs_row(
  RowViewType row_view, const int localRowId, const double diagonalValue)
{
  const LocalOrdinal rowLength = row_view.length;
  for (LocalOrdinal i = 0; i < rowLength; ++i) {
    if (row_view.colidx(i) == localRowId) {
      row_view.value(i) = diagonalValue;
    } else {
      row_view.value(i) = 0.0;
    }
  }
}

void
TpetraLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned beginPos,
  const unsigned endPos)
{
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const stk::mesh::Selector selector =
    (metaData.locally_owned_part() | metaData.globally_shared_part()) &
    stk::mesh::selectUnion(parts) & stk::mesh::selectField(*solutionField) &
    !(realm_.get_inactive_selector());

  using Traits = nalu_ngp::NGPMeshTraits<>;
  using MeshIndex = typename Traits::MeshIndex;

  stk::mesh::NgpMesh ngpMesh = realm_.ngp_mesh();
  NGPDoubleFieldType ngpSolutionField =
    realm_.ngp_field_manager().get_field<double>(
      solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField =
    realm_.ngp_field_manager().get_field<double>(
      bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_device();
  ngpBCValuesField.sync_to_device();

  auto entityToLID = entityToLID_;
  const int maxOwnedRowId = maxOwnedRowId_;
  const int maxSharedNotOwnedRowId = maxSharedNotOwnedRowId_;
  auto ownedLocalMatrix = getOwnedLocalMatrix();
  auto sharedNotOwnedLocalMatrix = getSharedNotOwnedLocalMatrix();
  auto ownedLocalRhs = getOwnedLocalRhs();
  auto sharedNotOwnedLocalRhs = getSharedNotOwnedLocalRhs();

  // Suppress unused variable warning on non-debug builds
  (void)maxSharedNotOwnedRowId;

  nalu_ngp::run_entity_algorithm(
    "TpetraLinSys::applyDirichletBCs", ngpMesh, stk::topology::NODE_RANK,
    selector, KOKKOS_LAMBDA(const MeshIndex& meshIdx) {
      stk::mesh::Entity entity =
        ngpMesh.get_entity(stk::topology::NODE_RANK, meshIdx);
      const LocalOrdinal localIdOffset = entityToLID[entity.local_offset()];
      const bool useOwned = localIdOffset < maxOwnedRowId;
      const LinSys::LocalMatrix& local_matrix =
        useOwned ? ownedLocalMatrix : sharedNotOwnedLocalMatrix;
      const LinSys::LocalVector& localRhs =
        useOwned ? ownedLocalRhs : sharedNotOwnedLocalRhs;
      const double diagonalValue = useOwned ? 1.0 : 0.0;

      for (unsigned d = beginPos; d < endPos; ++d) {
        const LocalOrdinal localId = localIdOffset + d;
        const LocalOrdinal actualLocalId =
          useOwned ? localId : localId - maxOwnedRowId;

        STK_NGP_ThrowAssert(localId <= maxSharedNotOwnedRowId);

        adjust_lhs_row(
          local_matrix.row(actualLocalId), actualLocalId, diagonalValue);

        // Replace the RHS residual with (desired - actual)
        const double bc_residual = useOwned
                                     ? (ngpBCValuesField.get(meshIdx, d) -
                                        ngpSolutionField.get(meshIdx, d))
                                     : 0.0;
        localRhs(actualLocalId, 0) = bc_residual;
      }
    });
}

void
TpetraLinearSystem::resetRows(
  const std::vector<stk::mesh::Entity>& nodeList,
  const unsigned beginPos,
  const unsigned endPos,
  const double diag_value,
  const double rhs_residual)
{
  resetRows(
    nodeList.size(), nodeList.data(), beginPos, endPos, diag_value,
    rhs_residual);
}

void
TpetraLinearSystem::resetRows(
  unsigned numNodes,
  const stk::mesh::Entity* nodeList,
  const unsigned beginPos,
  const unsigned endPos,
  const double diag_value,
  const double rhs_residual)
{
  reset_rows(
    getOwnedLocalMatrix(), getSharedNotOwnedLocalMatrix(), getOwnedLocalRhs(),
    getSharedNotOwnedLocalRhs(), numNodes, nodeList, beginPos, endPos,
    diag_value, rhs_residual, entityToLIDHost_, maxOwnedRowId_,
    maxSharedNotOwnedRowId_);
}

void
TpetraLinearSystem::loadComplete()
{
  // LHS
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::parameterList();
  params->set("No Nonlocal Changes", true);
  bool do_params = false;

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
TpetraLinearSystem::solve(stk::mesh::FieldBase* linearSolutionField)
{

  TpetraLinearSolver* linearSolver =
    reinterpret_cast<TpetraLinearSolver*>(linearSolver_);

  if (NaluEnv::self().debug()) {
    checkForNaN(true);
    if (checkForZeroRow(true, false, true)) {
      throw std::runtime_error("ERROR checkForZeroRow in solve()");
    }
  }

  if (linearSolver->getConfig()->getWriteMatrixFiles()) {
    writeToFile(eqSysName_.c_str());
    writeToFile(eqSysName_.c_str(), false);
  }

  int iters;
  double finalResidNorm;

  // memory diagnostic
  if (realm_.get_activate_memory_diagnostic()) {
    NaluEnv::self().naluOutputP0()
      << "NaluMemory::TpetraLinearSystem::solve() PreSolve: " << eqSysName_
      << std::endl;
    realm_.provide_memory_summary();
  }

  const int status =
    linearSolver->solve(sln_, iters, finalResidNorm, realm_.isFinalOuterIter_);

  if (linearSolver->getConfig()->getWriteMatrixFiles()) {
    writeSolutionToFile(eqSysName_.c_str());
    ++eqSys_->linsysWriteCounter_;
  }

  copy_tpetra_to_stk(sln_, linearSolutionField);
  sync_field(linearSolutionField);

  // computeL2 norm
  Teuchos::Array<double> mv_norm(1);
  ownedRhs_->norm2(mv_norm());
  const double norm2 = mv_norm[0];

  // save off solver info
  linearSolveIterations_ = iters;
  nonLinearResidual_ = realm_.l2Scaling_ * norm2;
  linearResidual_ = finalResidNorm;

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
      << std::right << finalResidNorm << std::setw(15) << std::right
      << nonLinearResidual_ << std::setw(14) << std::right
      << scaledNonLinearResidual_ << std::endl;
  }

  eqSys_->firstTimeStepSolve_ = false;

  return status;
}

void
TpetraLinearSystem::checkForNaN(bool useOwned)
{
  Teuchos::RCP<LinSys::Matrix> matrix =
    useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> rhs =
    useOwned ? ownedRhs_ : sharedNotOwnedRhs_;

  LinSys::LocalIndicesHost indices;
  LinSys::LocalValuesHost values;

  size_t n = matrix->getRowMap()->getLocalNumElements();
  for (size_t i = 0; i < n; ++i) {

    matrix->getLocalRowView(i, indices, values);
    const size_t rowLength = values.size();
    for (size_t k = 0; k < rowLength; ++k) {
      if (values[k] != values[k]) {
        std::cerr << "LHS NaN: " << i << std::endl;
        throw std::runtime_error("bad LHS");
      }
    }
  }

  Teuchos::ArrayRCP<const Scalar> rhs_data = rhs->getData(0);
  n = rhs_data.size();
  for (size_t i = 0; i < n; ++i) {
    if (rhs_data[i] != rhs_data[i]) {
      std::cerr << "rhs NaN: " << i << std::endl;
      throw std::runtime_error("bad rhs");
    }
  }
}

bool
TpetraLinearSystem::checkForZeroRow(bool useOwned, bool doThrow, bool doPrint)
{
  Teuchos::RCP<LinSys::Matrix> matrix =
    useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> rhs =
    useOwned ? ownedRhs_ : sharedNotOwnedRhs_;
  stk::mesh::BulkData& bulkData = realm_.bulk_data();

  LinSys::LocalIndicesHost indices;
  LinSys::LocalValuesHost values;

  size_t nrowG = matrix->getRangeMap()->getGlobalNumElements();
  size_t n = matrix->getRowMap()->getLocalNumElements();
  GlobalOrdinal max_gid = 0, g_max_gid = 0;
  // KOKKOS: Loop parallel reduce
  kokkos_parallel_for(
    "Nalu::TpetraLinearSystem::checkForZeroRowA", n, [&](const size_t& i) {
      GlobalOrdinal gid = matrix->getGraph()->getRowMap()->getGlobalElement(i);
      max_gid = std::max(gid, max_gid);
    });
  stk::all_reduce_max(bulkData.parallel(), &max_gid, &g_max_gid, 1);

  nrowG = g_max_gid + 1;
  std::vector<double> local_row_sums(nrowG, 0.0);
  std::vector<int> local_row_exists(nrowG, 0);
  std::vector<double> global_row_sums(nrowG, 0.0);
  std::vector<int> global_row_exists(nrowG, 0);

  for (size_t i = 0; i < n; ++i) {
    GlobalOrdinal gid = matrix->getGraph()->getRowMap()->getGlobalElement(i);
    matrix->getLocalRowView(i, indices, values);
    const size_t rowLength = values.size();
    double row_sum = 0.0;
    for (size_t k = 0; k < rowLength; ++k) {
      row_sum += std::abs(values[k]);
    }
    if (gid - 1 >= (GlobalOrdinal)local_row_sums.size() || gid <= 0) {
      std::cerr << "gid= " << gid << " nrowG= " << nrowG << std::endl;
      throw std::runtime_error("bad gid");
    }
    local_row_sums[gid - 1] = row_sum;
    local_row_exists[gid - 1] = 1;
  }

  stk::all_reduce_sum(
    bulkData.parallel(), &local_row_sums[0], &global_row_sums[0],
    (unsigned)nrowG);
  stk::all_reduce_max(
    bulkData.parallel(), &local_row_exists[0], &global_row_exists[0],
    (unsigned)nrowG);

  bool found = false;
  // KOKKOS: Loop parallel
  kokkos_parallel_for(
    "Nalu::TpetraLinearSystem::checkForZeroRowC", nrowG, [&](const size_t& ii) {
      double row_sum = global_row_sums[ii];
      if (
        global_row_exists[ii] && bulkData.parallel_rank() == 0 &&
        row_sum < 1.e-10) {
        found = true;
        GlobalOrdinal gid = ii + 1;
        stk::mesh::EntityId nid = (gid - 1) / numDof_ + 1;
        stk::mesh::Entity node =
          bulkData.get_entity(stk::topology::NODE_RANK, nid);
        const stk::mesh::EntityId naluGlobalId =
          bulkData.is_valid(node)
            ? *stk::mesh::field_data(*realm_.naluGlobalId_, node)
            : -1;

        int idof = (gid - 1) % numDof_;
        GlobalOrdinal GID_check = GID_(nid, numDof_, idof);
        if (doPrint) {

          double dualVolume = -1.0;

          std::cout << "P[" << bulkData.parallel_rank() << "] LHS zero: " << ii
                    << " GID= " << gid << " GID_check= " << GID_check
                    << " nid= " << nid << " naluGlobalId " << naluGlobalId
                    << " is_valid= " << bulkData.is_valid(node)
                    << " idof= " << idof << " numDof_= " << numDof_
                    << " row_sum= " << row_sum << " dualVolume= " << dualVolume
                    << std::endl;
          NaluEnv::self().naluOutputP0()
            << "P[" << bulkData.parallel_rank() << "] LHS zero: " << ii
            << " GID= " << gid << " GID_check= " << GID_check << " nid= " << nid
            << " naluGlobalId " << naluGlobalId
            << " is_valid= " << bulkData.is_valid(node) << " idof= " << idof
            << " numDof_= " << numDof_ << " row_sum= " << row_sum
            << " dualVolume= " << dualVolume << std::endl;
        }
      }
    });

  if (found && doThrow) {
    throw std::runtime_error("bad zero row LHS");
  }
  return found;
}

void
TpetraLinearSystem::writeToFile(const char* base_filename, bool useOwned)
{
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  const unsigned p_rank = bulkData.parallel_rank();
  const unsigned p_size = bulkData.parallel_size();

  Teuchos::RCP<LinSys::Matrix> matrix =
    useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> rhs =
    useOwned ? ownedRhs_ : sharedNotOwnedRhs_;

  const int currentCount = eqSys_->linsysWriteCounter_;

  if (1) {
    std::ostringstream osLhs;
    std::ostringstream osRhs;
    osLhs << base_filename << "-" << (useOwned ? "O-" : "G-") << currentCount
          << ".mm." << p_size; // A little hacky but whatever
    osRhs << base_filename << "-" << (useOwned ? "O-" : "G-") << currentCount
          << ".rhs." << p_size; // A little hacky but whatever

    Tpetra::MatrixMarket::Writer<LinSys::Matrix>::writeSparseFile(
      osLhs.str().c_str(), matrix, eqSysName_,
      std::string("Tpetra matrix for: ") + eqSysName_, true);
    typedef Tpetra::MatrixMarket::Writer<LinSys::Matrix> writer_type;
    if (useOwned)
      writer_type::writeDenseFile(osRhs.str().c_str(), rhs);
  }

  if (1) {
    std::ostringstream osLhs;
    std::ostringstream osGra;
    std::ostringstream osRhs;

    osLhs << base_filename << "-" << (useOwned ? "O-" : "G-") << currentCount
          << ".mm." << p_size << "." << p_rank; // A little hacky but whatever
    osGra << base_filename << "-" << (useOwned ? "O-" : "G-") << currentCount
          << ".gra." << p_size << "." << p_rank; // A little hacky but whatever
    osRhs << base_filename << "-" << (useOwned ? "O-" : "G-") << currentCount
          << ".rhs." << p_size << "." << p_rank; // A little hacky but whatever

    // Teuchos::RCP<Teuchos::FancyOStream> out =
    // Teuchos::VerboseObjectBase::getDefaultOStream();
#define DUMP(A)                                                                \
  do {                                                                         \
    out << "\n\n=============================================================" \
           "==================================\n";                             \
    out << "=================================================================" \
           "==============================\n";                                 \
    out << "P[" << p_rank << "] writeToFile:: " #A "= "                        \
        << "\n---------------------------\n";                                  \
    out << Teuchos::describe(*A, Teuchos::VERB_EXTREME) << "\n";               \
    out << "=================================================================" \
           "==============================\n";                                 \
    out << "=================================================================" \
           "==============================\n\n\n";                             \
  } while (0)

    {
      std::ostringstream out;
      DUMP(matrix);
      std::ofstream fout;
      fout.open(osLhs.str().c_str());
      fout << out.str() << std::endl;
    }

    {
      std::ostringstream out;
      DUMP(matrix->getGraph());
      std::ofstream fout;
      fout.open(osGra.str().c_str());
      fout << out.str() << std::endl;
    }

    {
      std::ostringstream out;
      DUMP(rhs);
      std::ofstream fout;
      fout.open(osRhs.str().c_str());
      fout << out.str() << std::endl;
    }

#undef DUMP
  }
}

void
TpetraLinearSystem::printInfo(bool useOwned)
{
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  const unsigned p_rank = bulkData.parallel_rank();

  Teuchos::RCP<LinSys::Matrix> matrix =
    useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> rhs =
    useOwned ? ownedRhs_ : sharedNotOwnedRhs_;

  if (p_rank == 0) {
    std::cout << "\nMatrix for EqSystem: " << eqSysName_
              << " :: N N NZ= " << matrix->getRangeMap()->getGlobalNumElements()
              << " " << matrix->getDomainMap()->getGlobalNumElements() << " "
              << matrix->getGlobalNumEntries() << std::endl;
    NaluEnv::self().naluOutputP0()
      << "\nMatrix for system: " << eqSysName_
      << " :: N N NZ= " << matrix->getRangeMap()->getGlobalNumElements() << " "
      << matrix->getDomainMap()->getGlobalNumElements() << " "
      << matrix->getGlobalNumEntries() << std::endl;
  }
}

void
TpetraLinearSystem::writeSolutionToFile(
  const char* base_filename, bool useOwned)
{
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  const unsigned p_rank = bulkData.parallel_rank();
  const unsigned p_size = bulkData.parallel_size();

  Teuchos::RCP<LinSys::MultiVector> sln = sln_;
  const int currentCount = eqSys_->linsysWriteCounter_;

  if (1) {
    std::ostringstream osSln;
    osSln << base_filename << "-" << (useOwned ? "O-" : "G-") << currentCount
          << ".sln." << p_size; // A little hacky but whatever

    typedef Tpetra::MatrixMarket::Writer<LinSys::Matrix> writer_type;
    if (useOwned)
      writer_type::writeDenseFile(osSln.str().c_str(), sln);
  }

  if (1) {
    std::ostringstream osSln;

    osSln << base_filename << "-"
          << "O-" << currentCount << ".sln." << p_size << "."
          << p_rank; // A little hacky but whatever

#define DUMP(A)                                                                \
  do {                                                                         \
    out << "\n\n=============================================================" \
           "==================================\n";                             \
    out << "=================================================================" \
           "==============================\n";                                 \
    out << "P[" << p_rank << "] writeToFile:: " #A "= "                        \
        << "\n---------------------------\n";                                  \
    out << Teuchos::describe(*A, Teuchos::VERB_EXTREME) << "\n";               \
    out << "=================================================================" \
           "==============================\n";                                 \
    out << "=================================================================" \
           "==============================\n\n\n";                             \
  } while (0)

    {
      std::ostringstream out;
      DUMP(sln);
      std::ofstream fout;
      fout.open(osSln.str().c_str());
      fout << out.str() << std::endl;
    }

#undef DUMP
  }
}

void
TpetraLinearSystem::copy_tpetra_to_stk(
  const Teuchos::RCP<LinSys::MultiVector> tpetraField,
  stk::mesh::FieldBase* stkField)
{
  using Traits = nalu_ngp::NGPMeshTraits<>;
  using MeshIndex = typename Traits::MeshIndex;

  const stk::mesh::MetaData& metaData = realm_.meta_data();

  STK_ThrowAssert(!tpetraField.is_null());
  STK_ThrowAssert(stkField);
  const auto deviceVector =
    tpetraField->getLocalViewDevice(Tpetra::Access::ReadWrite);

  const int maxOwnedRowId = maxOwnedRowId_;
  const unsigned numDof = numDof_;
  auto entityToLID = entityToLID_;

  const stk::mesh::Selector selector =
    stk::mesh::selectField(*stkField) & metaData.locally_owned_part() &
    !(stk::mesh::selectUnion(realm_.get_slave_part_vector())) &
    !(realm_.get_inactive_selector());

  NGPDoubleFieldType ngpField = realm_.ngp_field_manager().get_field<double>(
    stkField->mesh_meta_data_ordinal());

  stk::mesh::NgpMesh ngpMesh = realm_.ngp_mesh();

  nalu_ngp::run_entity_algorithm(
    "TpetraLinSys::copy_tpetra_to_stk", ngpMesh, stk::topology::NODE_RANK,
    selector, KOKKOS_LAMBDA(const MeshIndex& meshIdx) {
      stk::mesh::Entity node =
        ngpMesh.get_entity(stk::topology::NODE_RANK, meshIdx);
      const LocalOrdinal localIdOffset = entityToLID[node.local_offset()];
      for (unsigned d = 0; d < numDof; ++d) {
        const LocalOrdinal localId = localIdOffset + d;
        STK_NGP_ThrowRequireMsg(localId < maxOwnedRowId, "Error");

        ngpField.get(meshIdx, d) = deviceVector(localId, 0);
      }
    });

  ngpField.modify_on_device();
}

int
getDofStatus_impl(stk::mesh::Entity node, const Realm& realm)
{
  const stk::mesh::BulkData& bulkData = realm.bulk_data();

  const stk::mesh::Bucket& b = bulkData.bucket(node);
  const bool entityIsOwned = b.owned();
  const bool entityIsShared = b.shared();
  const bool entityIsGhosted = !entityIsOwned && !entityIsShared;

  bool has_non_matching_boundary_face_alg =
    realm.has_non_matching_boundary_face_alg();
  bool hasPeriodic = realm.hasPeriodic_;

  if (realm.hasPeriodic_ && realm.has_non_matching_boundary_face_alg()) {
    has_non_matching_boundary_face_alg = false;
    hasPeriodic = false;

    stk::mesh::Selector perSel =
      stk::mesh::selectUnion(realm.allPeriodicInteractingParts_);
    stk::mesh::Selector nonConfSel =
      stk::mesh::selectUnion(realm.allNonConformalInteractingParts_);
    // std::cout << "nonConfSel= " << nonConfSel << std::endl;

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
    throw std::logic_error(
      "not ready for primetime to combine periodic and non-matching algorithm "
      "on same node: " +
      ostr.str());
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
    // if (entityIsShared && !entityIsOwned) {
    if (!entityIsOwned && (entityIsGhosted || entityIsShared)) {
      return DS_SharedNotOwnedDOF;
    }
    // maybe return DS_GhostedDOF if entityIsGhosted
  }

  if (hasPeriodic) {
    const stk::mesh::EntityId stkId = bulkData.identifier(node);
    const stk::mesh::EntityId naluId =
      *stk::mesh::field_data(*realm.naluGlobalId_, node);

    // bool for type of ownership for this node
    const bool nodeOwned = bulkData.bucket(node).owned();
    const bool nodeShared = bulkData.bucket(node).shared();
    const bool nodeGhosted = !nodeOwned && !nodeShared;

    // really simple here.. ghosted nodes never part of the matrix
    if (nodeGhosted) {
      return DS_GhostedDOF;
    }

    // bool to see if this is possibly a periodic node
    const bool isSlaveNode = (stkId != naluId);

    if (!isSlaveNode) {
      if (nodeOwned)
        return DS_OwnedDOF;
      else if (nodeShared)
        return DS_SharedNotOwnedDOF;
      else
        return DS_GhostedDOF;
    } else {
      // I am a slave node.... get the master entity
      stk::mesh::Entity masterEntity =
        bulkData.get_entity(stk::topology::NODE_RANK, naluId);
      if (bulkData.is_valid(masterEntity)) {
        const bool masterEntityOwned = bulkData.bucket(masterEntity).owned();
        const bool masterEntityShared = bulkData.bucket(masterEntity).shared();
        if (masterEntityOwned)
          return DS_SkippedDOF | DS_OwnedDOF;
        if (masterEntityShared)
          return DS_SkippedDOF | DS_SharedNotOwnedDOF;
        else
          return DS_SharedNotOwnedDOF;
      } else {
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

} // namespace nalu
} // namespace sierra
