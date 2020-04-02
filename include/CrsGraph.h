/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef CrsGraph_
#define CrsGraph_

#include <KokkosInterface.h>
#include <FieldTypeDef.h>

#include <Kokkos_DefaultNode.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <stk_ngp/Ngp.hpp>

#include <vector>
#include <string>
#include <unordered_map>

#include "LinearSolverTypes.h"

namespace stk {
class CommNeighbors;
}

namespace sierra {
namespace nalu {

class Realm;

typedef std::unordered_map<stk::mesh::EntityId, size_t>  MyLIDMapType;

typedef std::pair<stk::mesh::Entity, stk::mesh::Entity> Connection;


class CrsGraph
{
public:
  typedef GraphTypes::GlobalOrdinal   GlobalOrdinal;
  typedef GraphTypes::LocalOrdinal    LocalOrdinal;

  CrsGraph(
    Realm &realm,
    const unsigned numDof);
  ~CrsGraph();

   // Graph/Matrix Construction
  void buildNodeGraph(const stk::mesh::PartVector & parts); // for nodal assembly (e.g., lumped mass and source)
  void buildFaceToNodeGraph(const stk::mesh::PartVector & parts); // face->node assembly
  void buildEdgeToNodeGraph(const stk::mesh::PartVector & parts); // edge->node assembly
  void buildElemToNodeGraph(const stk::mesh::PartVector & parts); // elem->node assembly
  void buildReducedElemToNodeGraph(const stk::mesh::PartVector & parts); // elem (nearest nodes only)->node assembly
  void buildFaceElemToNodeGraph(const stk::mesh::PartVector & parts); // elem:face->node assembly
  void buildNonConformalNodeGraph(const stk::mesh::PartVector & parts); // nonConformal->node assembly
  void buildOversetNodeGraph(const stk::mesh::PartVector & parts); // overset->elem_node assembly
  void storeOwnersForShared();
  void finalizeGraph();

  int getDofStatus(stk::mesh::Entity node);

  int getRowLID(stk::mesh::Entity node) { return entityToLID_[node.local_offset()]; }
  int getColLID(stk::mesh::Entity node) { return entityToColLID_[node.local_offset()]; }

  Teuchos::RCP<GraphTypes::Map>    getOwnedRowsMap() const;
  Teuchos::RCP<GraphTypes::Graph>  getOwnedGraph() const;
  Teuchos::RCP<GraphTypes::Map>    getSharedNotOwnedRowsMap() const;
  Teuchos::RCP<GraphTypes::Graph>  getSharedNotOwnedGraph() const;
  Teuchos::RCP<GraphTypes::Export> getExporter() const;

  void buildConnectedNodeGraph(stk::mesh::EntityRank rank,
                               const stk::mesh::PartVector& parts);

  const LinSys::EntityToLIDView & get_entity_to_row_LID_mapping() const;
  const LinSys::EntityToLIDView & get_entity_to_col_LID_mapping() const;
  const MyLIDMapType &            get_my_LIDs() const;
  LocalOrdinal                    getMaxOwnedRowID() const;
  LocalOrdinal                    getMaxSharedNotOwnedRowID() const;
private:

  void beginConstruction();

  void checkError( const int /* err_code */, const char * /* msg */) {}

  void compute_send_lengths(const std::vector<stk::mesh::Entity>& rowEntities,
         const std::vector<std::vector<stk::mesh::Entity> >& connections,
                            const std::vector<int>& neighborProcs,
                            stk::CommNeighbors& commNeighbors);

  void compute_graph_row_lengths(const std::vector<stk::mesh::Entity>& rowEntities,
         const std::vector<std::vector<stk::mesh::Entity> >& connections,
                                 LinSys::RowLengths& sharedNotOwnedRowLengths,
                                 LinSys::RowLengths& locallyOwnedRowLengths,
                                 stk::CommNeighbors& commNeighbors);

  void insert_graph_connections(const std::vector<stk::mesh::Entity>& rowEntities,
         const std::vector<std::vector<stk::mesh::Entity> >& connections,
                                LocalGraphArrays& locallyOwnedGraph,
                                LocalGraphArrays& sharedNotOwnedGraph);

  void fill_entity_to_row_LID_mapping();
  void fill_entity_to_col_LID_mapping();

  int insert_connection(stk::mesh::Entity a, stk::mesh::Entity b);
  void addConnections(const stk::mesh::Entity* entities,const size_t&);
  void expand_unordered_map(unsigned newCapacityNeeded);

  std::vector<stk::mesh::Entity> ownedAndSharedNodes_;
  std::vector<std::vector<stk::mesh::Entity> > connections_;
  std::vector<GlobalOrdinal> totalGids_;
  std::set<std::pair<int,GlobalOrdinal> > ownersAndGids_;
  std::vector<int> sharedPids_;

  Teuchos::RCP<LinSys::Node>   node_;

  // all rows, otherwise known as col map
  Teuchos::RCP<LinSys::Map>    totalColsMap_;
  Teuchos::RCP<LinSys::Map>    optColsMap_;

  // Map of rows my proc owns (locally owned)
  Teuchos::RCP<LinSys::Map>    ownedRowsMap_;

  // Only nodes that share with other procs that I don't own
  Teuchos::RCP<LinSys::Map>    sharedNotOwnedRowsMap_;

  Teuchos::RCP<LinSys::Graph>  ownedGraph_;
  Teuchos::RCP<LinSys::Graph>  sharedNotOwnedGraph_;

  Teuchos::RCP<LinSys::Export>      exporter_;

  MyLIDMapType myLIDs_;
  LinSys::EntityToLIDView entityToColLID_;
  LinSys::EntityToLIDView entityToLID_;
  LocalOrdinal maxOwnedRowId_; // = num_owned_nodes * numDof_
  LocalOrdinal maxSharedNotOwnedRowId_; // = (num_owned_nodes + num_sharedNotOwned_nodes) * numDof_

  std::vector<int> sortPermutation_;

  Realm &realm_;
  const unsigned numDof_;
  bool inConstruction_;
  bool isFinalized_;

}; //CrsGraph class

template<typename T1, typename T2>
void copy_kokkos_unordered_map(const Kokkos::UnorderedMap<T1,T2>& src,
                               Kokkos::UnorderedMap<T1,T2>& dest)
{
  if (src.capacity() > dest.capacity()) {
    dest = Kokkos::UnorderedMap<T1,T2>(src.capacity());
  }

  unsigned capacity = src.capacity();
  unsigned fail_count = 0;
  for(unsigned i=0; i<capacity; ++i) {
    if (src.valid_at(i)) {
      auto insert_result = dest.insert(src.key_at(i));
      fail_count += insert_result.failed() ? 1 : 0;
    }
  }
  ThrowRequire(fail_count == 0);
}

int getDofStatus_impl(stk::mesh::Entity node, const Realm& realm);

} // namespace nalu
} // namespace Sierra

#endif
