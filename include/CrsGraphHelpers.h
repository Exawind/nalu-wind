// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef CrsGraphHelpers_h
#define CrsGraphHelpers_h

#include <stdio.h>

#include <Realm.h>
#include <PeriodicManager.h>
#include <NonConformalManager.h>
#include <utils/StkHelpers.h>
#include <CrsGraphTypes.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_topology/topology.hpp>

namespace stk {
class CommNeighbors;
}

namespace sierra {
namespace nalu {

const GraphTypes::LocalOrdinal INVALID = std::numeric_limits<GraphTypes::LocalOrdinal>::max();
typedef typename GraphTypes::LocalGraph::row_map_type::non_const_type RowPointers;
typedef typename GraphTypes::LocalGraph::entries_type::non_const_type ColumnIndices;

/** LocalGraphArrays is a helper class for building the arrays describing
 * the local csr graph, rowPointers and colIndices. These arrays are passed
 * to the TpetraCrsGraph::setAllIndices method. This helper class is used
 * within nalu's TpetraLinearSystem class.
 * See unit-tests in UnitTestLocalGraphArrays.C.
 */
class LocalGraphArrays {
public:

  template<typename ViewType>
  LocalGraphArrays(const ViewType& rowLengths)
  : rowPointers(),
    rowPointersData(nullptr),
    colIndices()
  {
    RowPointers rowPtrs("rowPtrs", rowLengths.size()+1);
    rowPointers = rowPtrs;
    rowPointersData = rowPointers.data();

    size_t nnz = compute_row_pointers(rowPointers, rowLengths);
    colIndices = Kokkos::View<GraphTypes::LocalOrdinal*,LinSysMemSpace>(Kokkos::ViewAllocateWithoutInitializing("colIndices"), nnz);
    Kokkos::deep_copy(colIndices, INVALID);
  }

  size_t get_row_length(size_t localRow) const { return rowPointersData[localRow+1]-rowPointersData[localRow]; }

  void insertIndices(size_t localRow, size_t numInds, const GraphTypes::LocalOrdinal* inds, int numDof)
  {
    GraphTypes::LocalOrdinal* row = &colIndices((int)rowPointersData[localRow]);
    size_t rowLen = get_row_length(localRow);
    GraphTypes::LocalOrdinal* rowEnd = std::find(row, row+rowLen, INVALID);
    for(size_t i=0; i<numInds; ++i) {
      GraphTypes::LocalOrdinal* insertPoint = std::lower_bound(row, rowEnd, inds[i]);
      if (insertPoint <= rowEnd && *insertPoint != inds[i]) {
        insert(inds[i], numDof, insertPoint, rowEnd+numDof);
        rowEnd += numDof;
      }
    }
  }

  template<typename ViewType1, typename ViewType2>
  static size_t compute_row_pointers(ViewType1& rowPtrs,
                                   const ViewType2& rowLengths)
  {
    size_t nnz = 0;
    auto rowPtrData = rowPtrs.data();
    auto rowLens = rowLengths.data();
    for(unsigned i=0, iend=rowLengths.size(); i<iend; ++i) {
      rowPtrData[i] = nnz;
      nnz += rowLens[i];
    }
    rowPtrData[rowLengths.size()] = nnz;
    return nnz;
  }

  RowPointers rowPointers;
  typename RowPointers::traits::data_type rowPointersData;
  ColumnIndices colIndices;

private:

  void insert(GraphTypes::LocalOrdinal ind, int numDof, GraphTypes::LocalOrdinal* insertPoint, GraphTypes::LocalOrdinal* rowEnd)
  {
    for(GraphTypes::LocalOrdinal* ptr = rowEnd-1; ptr!= insertPoint; --ptr) {
        *ptr = *(ptr-numDof);
    }
    for(int i=0; i<numDof; ++i) {
      *insertPoint++ = ind+i;
    }
  }
};

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

void fill_owned_and_shared_then_nonowned_ordered_by_proc(std::vector<GraphTypes::GlobalOrdinal>& totalGids,
                                                         std::vector<int>& srcPids,
                                                         int localProc,
                                                         const Teuchos::RCP<GraphTypes::Map>& ownedRowsMap,
                                                         const Teuchos::RCP<GraphTypes::Map>& sharedNotOwnedRowsMap,
                                                         const std::set<std::pair<int, GraphTypes::GlobalOrdinal> >& ownersAndGids,
                                                         const std::vector<int>& sharedPids);

stk::mesh::Entity get_entity_master(const stk::mesh::BulkData& bulk,
                                    stk::mesh::Entity entity,
                                    stk::mesh::EntityId naluId,
                                    bool throwIfMasterNotFound = true);

size_t get_neighbor_index(const std::vector<int>& neighborProcs, int proc);

void sort_connections(std::vector<std::vector<stk::mesh::Entity> >& connections);

void add_to_length(GraphTypes::DeviceRowLengths& v_owned, GraphTypes::DeviceRowLengths& v_shared,
                   unsigned numDof, GraphTypes::LocalOrdinal lid_a, GraphTypes::LocalOrdinal maxOwnedRowId,
                   bool a_owned, unsigned numColEntities);

void add_lengths_to_comm(const stk::mesh::BulkData&  /* bulk */,
                         stk::CommNeighbors& commNeighbors,
                         int entity_a_owner,
                         stk::mesh::EntityId entityId_a,
                         unsigned numDof,
                         unsigned numColEntities,
                         const stk::mesh::EntityId* colEntityIds,
                         const int* colOwners);

void add_lengths_to_comm_tpet(const stk::mesh::BulkData&  /* bulk */,
                              TpetIDFieldType * tpetGID_label,
                         stk::CommNeighbors& commNeighbors,
                         int entity_a_owner,
                         stk::mesh::EntityId entityId_a,
                              //                         unsigned numDof,
                         unsigned numColEntities,
                         const stk::mesh::EntityId* colEntityIds,
                         const int* colOwners);

void communicate_remote_columns(const stk::mesh::BulkData& bulk,
                                const std::vector<int>& neighborProcs,
                                stk::CommNeighbors& commNeighbors,
                                unsigned numDof,
                                const Teuchos::RCP<GraphTypes::Map>& ownedRowsMap,
                                GraphTypes::DeviceRowLengths& deviceLocallyOwnedRowLengths,
                                std::set<std::pair<int, GraphTypes::GlobalOrdinal> >& communicatedColIndices);

void insert_single_dof_row_into_graph(LocalGraphArrays& crsGraph, GraphTypes::LocalOrdinal rowLid,
                                      GraphTypes::LocalOrdinal maxOwnedRowId, unsigned numDof,
                                      unsigned numCols, const std::vector<GraphTypes::LocalOrdinal>& colLids);

void insert_communicated_col_indices(const std::vector<int>& neighborProcs,
                                     stk::CommNeighbors& commNeighbors,
                                     unsigned numDof,
                                     LocalGraphArrays& ownedGraph,
                                     const GraphTypes::Map& rowMap,
                                     const GraphTypes::Map& colMap);

void fill_in_extra_dof_rows_per_node(LocalGraphArrays& csg, int numDof);

void remove_invalid_indices(LocalGraphArrays& csg, GraphTypes::DeviceRowLengths& rowLengths);

} // nalu
} // sierra

#endif
