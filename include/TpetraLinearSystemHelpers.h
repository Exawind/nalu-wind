/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TpetraLinearSystemHelpers_h
#define TpetraLinearSystemHelpers_h

#include <stdio.h>

#include <Realm.h>
#include <PeriodicManager.h>
#include <NonConformalManager.h>
#include <utils/StkHelpers.h>
#include <LinearSolverTypes.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_topology/topology.hpp>

namespace stk {
class CommNeighbors;
}

namespace sierra {
namespace nalu {

class LocalGraphArrays;

#define GID_(gid, ndof, idof)  ((ndof)*((gid)-1)+(idof)+1)

enum DOFStatus {
  DS_NotSet            = 0,
  DS_SkippedDOF        = 1 << 1,
  DS_OwnedDOF          = 1 << 2,
  DS_SharedNotOwnedDOF = 1 << 3,
  DS_GhostedDOF        = 1 << 4
};

void add_procs_to_neighbors(const std::vector<int>& procs, std::vector<int>& neighbors);

void fill_neighbor_procs(std::vector<int>& neighborProcs,
                         const stk::mesh::BulkData& bulk,
                         const Realm& realm);

void fill_owned_and_shared_then_nonowned_ordered_by_proc(std::vector<LinSys::GlobalOrdinal>& totalGids,
                                                         std::vector<int>& srcPids,
                                                         int localProc,
                                                         const Teuchos::RCP<LinSys::Map>& ownedRowsMap,
                                                         const Teuchos::RCP<LinSys::Map>& sharedNotOwnedRowsMap,
                                                         const std::set<std::pair<int, LinSys::GlobalOrdinal> >& ownersAndGids,
                                                         const std::vector<int>& sharedPids);

stk::mesh::Entity get_entity_master(const stk::mesh::BulkData& bulk,
                                    stk::mesh::Entity entity,
                                    stk::mesh::EntityId naluId);

size_t get_neighbor_index(const std::vector<int>& neighborProcs, int proc);

void sort_connections(std::vector<std::vector<stk::mesh::Entity> >& connections);

void add_to_length(LinSys::DeviceRowLengths& v_owned, LinSys::DeviceRowLengths& v_shared,
                   unsigned numDof, LinSys::LocalOrdinal lid_a, LinSys::LocalOrdinal maxOwnedRowId,
                   bool a_owned, unsigned numColEntities);

void add_lengths_to_comm(const stk::mesh::BulkData&  /* bulk */,
                         stk::CommNeighbors& commNeighbors,
                         int entity_a_owner,
                         stk::mesh::EntityId entityId_a,
                         unsigned numDof,
                         unsigned numColEntities,
                         const stk::mesh::EntityId* colEntityIds,
                         const int* colOwners);

void communicate_remote_columns(const stk::mesh::BulkData& bulk,
                                const std::vector<int>& neighborProcs,
                                stk::CommNeighbors& commNeighbors,
                                unsigned numDof,
                                const Teuchos::RCP<LinSys::Map>& ownedRowsMap,
                                LinSys::DeviceRowLengths& deviceLocallyOwnedRowLengths,
                                std::set<std::pair<int, LinSys::GlobalOrdinal> >& communicatedColIndices);

void insert_single_dof_row_into_graph(LocalGraphArrays& crsGraph, LinSys::LocalOrdinal rowLid,
                                      LinSys::LocalOrdinal maxOwnedRowId, unsigned numDof,
                                      unsigned numCols, const std::vector<LinSys::LocalOrdinal>& colLids);

void insert_communicated_col_indices(const std::vector<int>& neighborProcs,
                                     stk::CommNeighbors& commNeighbors,
                                     unsigned numDof,
                                     LocalGraphArrays& ownedGraph,
                                     const LinSys::Map& rowMap,
                                     const LinSys::Map& colMap);

void fill_in_extra_dof_rows_per_node(LocalGraphArrays& csg, int numDof);

void remove_invalid_indices(LocalGraphArrays& csg, LinSys::DeviceRowLengths& rowLengths);

} // nalu
} // sierra

#endif
