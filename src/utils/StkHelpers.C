// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "utils/StkHelpers.h"

#include <element_promotion/PromotedPartHelper.h>
#include <Realm.h>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Ghosting.hpp>
#include <stk_topology/topology.hpp>

// stk_util
#include "stk_util/parallel/ParallelReduce.hpp"
#include "stk_util/parallel/CommSparse.hpp"
#include "stk_util/util/SortAndUnique.hpp"

namespace sierra {
namespace nalu {

void
populate_ghost_comm_procs(
  const stk::mesh::BulkData& bulk_data,
  stk::mesh::Ghosting& ghosting,
  std::vector<int>& ghostCommProcs)
{
  ghostCommProcs.clear();

  std::vector<stk::mesh::EntityProc> sendList;
  ghosting.send_list(sendList);

  for (const stk::mesh::EntityProc& entProc : sendList) {
    stk::util::insert_keep_sorted_and_unique(entProc.second, ghostCommProcs);
  }

  std::vector<stk::mesh::EntityKey> recvList;
  ghosting.receive_list(recvList);

  for (const stk::mesh::EntityKey& key : recvList) {
    stk::mesh::Entity entity = bulk_data.get_entity(key);
    stk::util::insert_keep_sorted_and_unique(
      bulk_data.parallel_owner_rank(entity), ghostCommProcs);
  }
}

stk::topology
get_elem_topo(const Realm& realm, const stk::mesh::Part& surfacePart)
{
  if (realm.doPromotion_) {
    return get_promoted_elem_topo(
      realm.spatialDimension_, realm.promotionOrder_);
  }

  std::vector<const stk::mesh::Part*> blockParts =
    realm.meta_data().get_blocks_touching_surface(&surfacePart);

  ThrowRequireMsg(
    blockParts.size() >= 1,
    "Error, expected at least 1 block for surface " << surfacePart.name());

  stk::topology elemTopo = blockParts[0]->topology();
  if (blockParts.size() > 1) {
    for (size_t i = 1; i < blockParts.size(); ++i) {
      ThrowRequireMsg(
        blockParts[i]->topology() == elemTopo,
        "Error, found blocks of different topology connected to surface '"
          << surfacePart.name() << "', " << elemTopo << " and "
          << blockParts[i]->topology());
    }
  }

  ThrowRequireMsg(
    elemTopo != stk::topology::INVALID_TOPOLOGY,
    "Error, didn't find valid topology block for surface "
      << surfacePart.name());
  return elemTopo;
}

void
add_downward_relations(
  const stk::mesh::BulkData& bulk,
  std::vector<stk::mesh::EntityKey>& entityKeys)
{
  size_t numEntities = entityKeys.size();
  for (size_t i = 0; i < numEntities; ++i) {
    stk::mesh::Entity ent = bulk.get_entity(entityKeys[i]);
    if (bulk.is_valid(ent)) {
      stk::mesh::EntityRank thisRank = bulk.entity_rank(ent);

      for (stk::mesh::EntityRank irank = stk::topology::NODE_RANK;
           irank < thisRank; ++irank) {
        unsigned num = bulk.num_connectivity(ent, irank);
        const stk::mesh::Entity* downwardEntities = bulk.begin(ent, irank);

        for (unsigned j = 0; j < num; ++j) {
          stk::mesh::EntityKey key = bulk.entity_key(downwardEntities[j]);
          const stk::mesh::Bucket& bkt = bulk.bucket(downwardEntities[j]);
          if (!bkt.shared()) {
            entityKeys.push_back(key);
          }
        }
      }
    }
  }
}

void
keep_elems_not_already_ghosted(
  const stk::mesh::BulkData& /* bulk */,
  const stk::mesh::EntityProcVec& alreadyGhosted,
  stk::mesh::EntityProcVec& elemsToGhost)
{
  if (!alreadyGhosted.empty()) {
    size_t numKept = 0;
    size_t num = elemsToGhost.size();
    for (size_t i = 0; i < num; ++i) {
      if (!std::binary_search(
            alreadyGhosted.begin(), alreadyGhosted.end(), elemsToGhost[i])) {
        elemsToGhost[numKept++] = elemsToGhost[i];
      }
    }
    elemsToGhost.resize(numKept);
  }
}

void
fill_send_ghosts_to_remove_from_ghosting(
  const stk::mesh::EntityProcVec& curSendGhosts,
  const stk::mesh::EntityProcVec& intersection,
  stk::mesh::EntityProcVec& sendGhostsToRemove)
{
  sendGhostsToRemove.reserve(curSendGhosts.size() - intersection.size());
  for (size_t i = 0; i < curSendGhosts.size(); ++i) {
    if (!std::binary_search(
          intersection.begin(), intersection.end(), curSendGhosts[i])) {
      sendGhostsToRemove.push_back(curSendGhosts[i]);
    }
  }
}

void
communicate_to_fill_recv_ghosts_to_remove(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::EntityProcVec& sendGhostsToRemove,
  std::vector<stk::mesh::EntityKey>& recvGhostsToRemove)
{
  stk::CommSparse commSparse(bulk.parallel());
  stk::pack_and_communicate(commSparse, [&]() {
    for (const stk::mesh::EntityProc& entityProc : sendGhostsToRemove) {
      stk::mesh::EntityKey key = bulk.entity_key(entityProc.first);
      stk::CommBuffer& buf = commSparse.send_buffer(entityProc.second);
      buf.pack<stk::mesh::EntityKey>(key);
    }
  });

  int numProcs = bulk.parallel_size();
  for (int p = 0; p < numProcs; ++p) {
    if (p == bulk.parallel_rank()) {
      continue;
    }
    stk::CommBuffer& buf = commSparse.recv_buffer(p);
    while (buf.remaining()) {
      stk::mesh::EntityKey key;
      buf.unpack<stk::mesh::EntityKey>(key);
      recvGhostsToRemove.push_back(key);
    }
  }

  add_downward_relations(bulk, recvGhostsToRemove);
}

void
keep_only_elems(
  const stk::mesh::BulkData& bulk, stk::mesh::EntityProcVec& entityProcs)
{
  size_t elemCounter = 0;
  for (size_t i = 0; i < entityProcs.size(); ++i) {
    if (bulk.entity_rank(entityProcs[i].first) == stk::topology::ELEM_RANK) {
      entityProcs[elemCounter++] = entityProcs[i];
    }
  }
  entityProcs.resize(elemCounter);
}

void
compute_precise_ghosting_lists(
  const stk::mesh::BulkData& bulk,
  stk::mesh::EntityProcVec& elemsToGhost,
  stk::mesh::EntityProcVec& curSendGhosts,
  std::vector<stk::mesh::EntityKey>& recvGhostsToRemove)
{
  keep_only_elems(bulk, curSendGhosts);
  stk::util::sort_and_unique(curSendGhosts);
  stk::util::sort_and_unique(elemsToGhost);

  stk::mesh::EntityProcVec intersection;
  std::set_intersection(
    curSendGhosts.begin(), curSendGhosts.end(), elemsToGhost.begin(),
    elemsToGhost.end(), std::back_inserter(intersection));

  keep_elems_not_already_ghosted(bulk, intersection, elemsToGhost);

  stk::mesh::EntityProcVec sendGhostsToRemove;
  fill_send_ghosts_to_remove_from_ghosting(
    curSendGhosts, intersection, sendGhostsToRemove);

  communicate_to_fill_recv_ghosts_to_remove(
    bulk, sendGhostsToRemove, recvGhostsToRemove);
}

} // namespace nalu
} // namespace sierra
