// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <TpetraLinearSystemHelpers.h>

#include <Realm.h>
#include <PeriodicManager.h>
#include <NonConformalManager.h>
#include <utils/StkHelpers.h>
#include <LinearSolver.h>
#include <overset/OversetManager.h>

#include <stk_util/parallel/CommNeighbors.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_topology/topology.hpp>

#include <stdio.h>

namespace sierra {
namespace nalu {

void
add_procs_to_neighbors(
  const std::vector<int>& procs, std::vector<int>& neighbors)
{
  neighbors.insert(neighbors.end(), procs.begin(), procs.end());
  stk::util::sort_and_unique(neighbors);
}

void
fill_neighbor_procs(
  std::vector<int>& neighborProcs,
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
      add_procs_to_neighbors(
        realm.periodicManager_->ghostCommProcs_, neighborProcs);
    }
    if (realm.nonConformalManager_) {
      add_procs_to_neighbors(
        realm.nonConformalManager_->ghostCommProcs_, neighborProcs);
    }
    if (realm.oversetManager_) {
      add_procs_to_neighbors(
        realm.oversetManager_->ghostCommProcs_, neighborProcs);
    }
  }
}

void
fill_owned_and_shared_then_nonowned_ordered_by_proc(
  std::vector<LinSys::GlobalOrdinal>& totalGids,
  std::vector<int>& srcPids,
  int localProc,
  const Teuchos::RCP<LinSys::Map>& ownedRowsMap,
  const Teuchos::RCP<LinSys::Map>& sharedNotOwnedRowsMap,
  const std::set<std::pair<int, LinSys::GlobalOrdinal>>& ownersAndGids,
  const std::vector<int>& sharedPids)
{
  auto ownedIndices = ownedRowsMap->getMyGlobalIndices();
  totalGids.clear();
  totalGids.reserve(ownedIndices.size() + ownersAndGids.size());

  srcPids.clear();
  srcPids.reserve(ownersAndGids.size());

  for (unsigned i = 0; i < ownedIndices.size(); ++i) {
    totalGids.push_back(ownedIndices[i]);
  }

  auto sharedIndices = sharedNotOwnedRowsMap->getMyGlobalIndices();
  for (unsigned i = 0; i < sharedIndices.size(); ++i) {
    totalGids.push_back(sharedIndices[i]);
    srcPids.push_back(sharedPids[i]);
    STK_ThrowRequireMsg(
      sharedPids[i] != localProc && sharedPids[i] >= 0,
      "Error, bad sharedPid = " << sharedPids[i] << ", localProc = "
                                << localProc << ", gid = " << sharedIndices[i]);
  }

  for (const std::pair<int, LinSys::GlobalOrdinal>& procAndGid :
       ownersAndGids) {
    int proc = procAndGid.first;
    LinSys::GlobalOrdinal gid = procAndGid.second;
    if (
      proc != localProc && !ownedRowsMap->isNodeGlobalElement(gid) &&
      !sharedNotOwnedRowsMap->isNodeGlobalElement(gid)) {
      totalGids.push_back(gid);
      srcPids.push_back(procAndGid.first);
      STK_ThrowRequireMsg(
        procAndGid.first != localProc && procAndGid.first >= 0,
        "Error, bad remote proc = " << procAndGid.first);
    }
  }

  STK_ThrowRequireMsg(
    srcPids.size() == (totalGids.size() - ownedIndices.size()),
    "Error, bad srcPids.size() = " << srcPids.size());
}

stk::mesh::Entity
get_entity_master(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Entity entity,
  stk::mesh::EntityId naluId,
  bool throwIfMasterNotFound)
{
  bool thisEntityIsMaster = (bulk.identifier(entity) == naluId);
  if (thisEntityIsMaster) {
    return entity;
  }
  stk::mesh::Entity master = bulk.get_entity(stk::topology::NODE_RANK, naluId);
  if (throwIfMasterNotFound && !bulk.is_valid(master)) {
    std::ostringstream os;
    const stk::mesh::Entity* elems = bulk.begin_elements(entity);
    unsigned numElems = bulk.num_elements(entity);
    os << " elems: ";
    for (unsigned i = 0; i < numElems; ++i) {
      os << "{" << bulk.identifier(elems[i]) << ","
         << bulk.bucket(elems[i]).topology()
         << ",owned=" << bulk.bucket(elems[i]).owned() << "}";
    }
    STK_ThrowRequireMsg(
      bulk.is_valid(master),
      "get_entity_master, P"
        << bulk.parallel_rank() << " failed to get entity for naluId=" << naluId
        << ", from entity with stkId=" << bulk.identifier(entity)
        << ", owned=" << bulk.bucket(entity).owned()
        << ", shared=" << bulk.bucket(entity).shared() << ", " << os.str());
  }
  return master;
}

size_t
get_neighbor_index(const std::vector<int>& neighborProcs, int proc)
{
  std::vector<int>::const_iterator neighbor =
    std::find(neighborProcs.begin(), neighborProcs.end(), proc);
  STK_ThrowRequireMsg(
    neighbor != neighborProcs.end(),
    "Error, failed to find p=" << proc << " in neighborProcs.");

  size_t neighborIndex = neighbor - neighborProcs.begin();
  return neighborIndex;
}

void
sort_connections(std::vector<std::vector<stk::mesh::Entity>>& connections)
{
  for (std::vector<stk::mesh::Entity>& vec : connections) {
    std::sort(vec.begin(), vec.end());
  }
}

void
add_to_length(
  LinSys::HostRowLengths& v_owned,
  LinSys::HostRowLengths& v_shared,
  unsigned numDof,
  LinSys::LocalOrdinal lid_a,
  LinSys::LocalOrdinal maxOwnedRowId,
  bool a_owned,
  unsigned numColEntities)
{
  LinSys::HostRowLengths& v_a = a_owned ? v_owned : v_shared;
  LinSys::LocalOrdinal lid = a_owned ? lid_a : lid_a - maxOwnedRowId;

  for (unsigned d = 0; d < numDof; ++d) {
    v_a(lid + d) += numDof * numColEntities;
  }
}

void
add_lengths_to_comm(
  const stk::mesh::BulkData& /* bulk */,
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
  LinSys::GlobalOrdinal rowGid = GID_(entityId_a, numDof, 0);

  sbuf.pack(rowGid);
  sbuf.pack(numColEntities * 2);
  for (unsigned c = 0; c < numColEntities; ++c) {
    LinSys::GlobalOrdinal colGid0 = GID_(colEntityIds[c], numDof, 0);
    sbuf.pack(colGid0);
    sbuf.pack(colOwners[c]);
  }
}

void
add_lengths_to_comm_tpet(
  const stk::mesh::BulkData& bulk /* bulk */,
  TpetIDFieldType* tpetGID_label,
  stk::CommNeighbors& commNeighbors,
  int entity_a_owner,
  stk::mesh::EntityId entityId_a,
  //                         unsigned numDof,
  unsigned numColEntities,
  const stk::mesh::EntityId* colEntityIds,
  const int* colOwners)
{
  int owner = entity_a_owner;
  stk::CommBufferV& sbuf = commNeighbors.send_buffer(owner);
  const auto node = bulk.get_entity(stk::topology::NODE_RANK, entityId_a);
  LinSys::GlobalOrdinal rowGid = *stk::mesh::field_data(*tpetGID_label, node);
  STK_ThrowRequireMsg(
    rowGid != 0 && rowGid != std::numeric_limits<LinSys::GlobalOrdinal>::max(),
    "add_lengths_to_comm_tpet");
  sbuf.pack(rowGid);
  sbuf.pack(numColEntities * 2);
  for (unsigned c = 0; c < numColEntities; ++c) {
    const auto centity =
      bulk.get_entity(stk::topology::NODE_RANK, colEntityIds[c]);
    LinSys::GlobalOrdinal colGid0 =
      *stk::mesh::field_data(*tpetGID_label, centity);
    STK_ThrowRequireMsg(
      colGid0 != 0 &&
        colGid0 != std::numeric_limits<LinSys::GlobalOrdinal>::max(),
      "add_lengths_to_comm_tpet");
    sbuf.pack(colGid0);
    sbuf.pack(colOwners[c]);
  }
}
void
communicate_remote_columns(
  const stk::mesh::BulkData& bulk,
  const std::vector<int>& neighborProcs,
  stk::CommNeighbors& commNeighbors,
  unsigned numDof,
  const Teuchos::RCP<LinSys::Map>& ownedRowsMap,
  LinSys::HostRowLengths& hostLocallyOwnedRowLengths,
  std::set<std::pair<int, LinSys::GlobalOrdinal>>& communicatedColIndices)
{
  commNeighbors.communicate();

  for (int p : neighborProcs) {
    stk::CommBufferV& rbuf = commNeighbors.recv_buffer(p);
    size_t bufSize = rbuf.size_in_bytes();
    while (rbuf.size_in_bytes() > 0) {
      LinSys::GlobalOrdinal rowGid = 0;
      rbuf.unpack(rowGid);

      STK_ThrowRequireMsg(
        rowGid != 0 &&
          rowGid != std::numeric_limits<LinSys::GlobalOrdinal>::max(),
        "communicate_remote_columns");

      unsigned len = 0;
      rbuf.unpack(len);
      unsigned numCols = len / 2;
      LinSys::LocalOrdinal lid = ownedRowsMap->getLocalElement(rowGid);
      if (lid < 0) {
        std::cerr << "P" << bulk.parallel_rank() << " lid=" << lid
                  << " for rowGid=" << rowGid << " sent from proc " << p
                  << std::endl;
      }
      for (unsigned d = 0; d < numDof; ++d) {
        hostLocallyOwnedRowLengths(lid++) += numCols * numDof;
      }
      for (unsigned i = 0; i < numCols; ++i) {
        LinSys::GlobalOrdinal colGid = 0;
        rbuf.unpack(colGid);

        STK_ThrowRequireMsg(
          colGid != 0 &&
            colGid != std::numeric_limits<LinSys::GlobalOrdinal>::max(),
          "communicate_remote_columns");

        int owner = 0;
        rbuf.unpack(owner);
        for (unsigned dd = 0; dd < numDof; ++dd) {
          communicatedColIndices.insert(std::make_pair(owner, colGid++));
        }
      }
    }
    rbuf.resize(bufSize);
  }
}

void
insert_single_dof_row_into_graph(
  LocalGraphArrays& crsGraph,
  LinSys::LocalOrdinal rowLid,
  LinSys::LocalOrdinal maxOwnedRowId,
  unsigned numDof,
  unsigned numCols,
  const std::vector<LinSys::LocalOrdinal>& colLids)
{
  if (rowLid >= maxOwnedRowId) {
    rowLid -= maxOwnedRowId;
  }
  crsGraph.insertIndices(rowLid++, numCols, colLids.data(), numDof);
}

void
insert_communicated_col_indices(
  const std::vector<int>& neighborProcs,
  stk::CommNeighbors& commNeighbors,
  unsigned numDof,
  LocalGraphArrays& ownedGraph,
  const LinSys::Map& rowMap,
  const LinSys::Map& colMap)
{
  std::vector<LocalOrdinal> colLids;
  for (int p : neighborProcs) {
    stk::CommBufferV& rbuf = commNeighbors.recv_buffer(p);
    while (rbuf.size_in_bytes() > 0) {
      stk::mesh::EntityId rowGid = 0;
      rbuf.unpack(rowGid);

      STK_ThrowRequireMsg(
        rowGid != 0 &&
          rowGid != static_cast<stk::mesh::EntityId>(
                      std::numeric_limits<LinSys::GlobalOrdinal>::max()),
        " insert_communicated_col_indices");

      unsigned len = 0;
      rbuf.unpack(len);
      unsigned numCols = len / 2;
      colLids.resize(numCols);
      LocalOrdinal rowLid = rowMap.getLocalElement(rowGid);
      for (unsigned i = 0; i < numCols; ++i) {
        GlobalOrdinal colGid = 0;
        rbuf.unpack(colGid);

        STK_ThrowRequireMsg(
          colGid != 0 &&
            colGid != std::numeric_limits<LinSys::GlobalOrdinal>::max(),
          " insert_communicated_col_indices");

        int owner = 0;
        rbuf.unpack(owner);
        colLids[i] = colMap.getLocalElement(colGid);
      }
      ownedGraph.insertIndices(rowLid++, numCols, colLids.data(), numDof);
    }
  }
}

void
fill_in_extra_dof_rows_per_node(LocalGraphArrays& csg, int numDof)
{
  if (numDof == 1) {
    return;
  }

  auto rowPtrs = csg.rowPointers.data();
  LocalOrdinal* cols = csg.colIndices.data();
  for (int i = 0, ie = csg.rowPointers.size() - 1; i < ie;) {
    const LocalOrdinal* row = cols + rowPtrs[i];
    int rowLen = csg.get_row_length(i);
    for (int d = 1; d < numDof; ++d) {
      LocalOrdinal* row_d = cols + rowPtrs[i] + rowLen * d;
      for (int j = 0; j < rowLen; ++j) {
        row_d[j] = row[j];
      }
    }
    i += numDof;
  }
}

void
remove_invalid_indices(
  LocalGraphArrays& csg, LinSys::HostRowLengths& rowLengths)
{
  size_t nnz = csg.rowPointers(rowLengths.size());
  auto cols = csg.colIndices.data();
  auto rowPtrs = csg.rowPointers.data();
  size_t newNnz = 0;
  for (int i = 0, ie = csg.rowPointers.size() - 1; i < ie; ++i) {
    const LocalOrdinal* row = cols + rowPtrs[i];
    int rowLen = csg.get_row_length(i);
    for (int j = rowLen - 1; j >= 0; --j) {
      if (row[j] != INVALID) {
        rowLengths(i) = j + 1;
        break;
      }
    }
    newNnz += rowLengths(i);
  }

  if (newNnz < nnz) {
    Kokkos::View<LocalOrdinal*, typename LinSys::HostRowLengths::memory_space>
      newColIndices(Kokkos::ViewAllocateWithoutInitializing("colInds"), newNnz);
    LocalOrdinal* newCols = newColIndices.data();
    auto rowLens = rowLengths.data();
    int index = 0;
    for (int i = 0, ie = csg.rowPointers.size() - 1; i < ie; ++i) {
      auto row = cols + rowPtrs[i];
      for (size_t j = 0; j < rowLens[i]; ++j) {
        newCols[index++] = row[j];
      }
    }
    csg.colIndices = newColIndices;
    LocalGraphArrays::compute_row_pointers(csg.rowPointers, rowLengths);
  }
}

} // namespace nalu
} // namespace sierra
