// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TpetraSegregatedLinearSystem_h
#define TpetraSegregatedLinearSystem_h

#include <LinearSystem.h>

#include <KokkosInterface.h>
#include <FieldTypeDef.h>

#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpMesh.hpp>

#include <vector>
#include <string>
#include <unordered_map>

namespace stk {
class CommNeighbors;
}

namespace sierra {
namespace nalu {

class Realm;
class EquationSystem;
class LinearSolver;
class LocalGraphArrays;

typedef std::unordered_map<stk::mesh::EntityId, size_t> MyLIDMapType;

typedef std::pair<stk::mesh::Entity, stk::mesh::Entity> Connection;

class TpetraSegregatedLinearSystem : public LinearSystem
{
public:
  typedef LinSys::GlobalOrdinal GlobalOrdinal;
  typedef LinSys::LocalOrdinal LocalOrdinal;

  TpetraSegregatedLinearSystem(
    Realm& realm,
    const unsigned numDof,
    EquationSystem* eqSys,
    LinearSolver* linearSolver);
  ~TpetraSegregatedLinearSystem();

  // Graph/Matrix Construction
  void
  buildNodeGraph(const stk::mesh::PartVector&
                   parts); // for nodal assembly (e.g., lumped mass and source)
  void buildFaceToNodeGraph(
    const stk::mesh::PartVector& parts); // face->node assembly
  void buildEdgeToNodeGraph(
    const stk::mesh::PartVector& parts); // edge->node assembly
  void buildElemToNodeGraph(
    const stk::mesh::PartVector& parts); // elem->node assembly
  void buildReducedElemToNodeGraph(
    const stk::mesh::PartVector&
      parts); // elem (nearest nodes only)->node assembly
  void buildFaceElemToNodeGraph(
    const stk::mesh::PartVector& parts); // elem:face->node assembly
  void buildNonConformalNodeGraph(
    const stk::mesh::PartVector& parts); // nonConformal->node assembly
  void buildOversetNodeGraph(
    const stk::mesh::PartVector& parts); // overset->elem_node assembly
  void storeOwnersForShared();
  void finalizeLinearSystem();

  CoeffApplier* get_coeff_applier();
  void free_coeff_applier(CoeffApplier* coeffApplier);

  // Matrix Assembly
  void zeroSystem();

  void sumInto(
    unsigned numEntities,
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    const SharedMemView<const double*, DeviceShmem>& rhs,
    const SharedMemView<const double**, DeviceShmem>& lhs,
    const SharedMemView<int*, DeviceShmem>& localIds,
    const SharedMemView<int*, DeviceShmem>& sortPermutation,
    const char* trace_tag);

  void sumInto(
    const std::vector<stk::mesh::Entity>& entities,
    std::vector<int>& scratchIds,
    std::vector<double>& scratchVals,
    const std::vector<double>& rhs,
    const std::vector<double>& lhs,
    const char* trace_tag = 0);

  void applyDirichletBCs(
    stk::mesh::FieldBase* solutionField,
    stk::mesh::FieldBase* bcValuesField,
    const stk::mesh::PartVector& parts,
    const unsigned beginPos,
    const unsigned endPos);

  /** Reset LHS and RHS for the given set of nodes to 0
   *
   *  @param nodeList A list of STK node entities whose rows are zeroed out
   *  @param beginPos Starting index (usually 0)
   *  @param endPos Terminating index (1 for scalar quantities; nDim for
   * vectors)
   */
  virtual void resetRows(
    const std::vector<stk::mesh::Entity>& nodeList,
    const unsigned beginPos,
    const unsigned endPos,
    const double diag_value = 0.0,
    const double rhs_residual = 0.0);

  virtual void resetRows(
    unsigned numNodes,
    const stk::mesh::Entity* nodeList,
    const unsigned beginPos,
    const unsigned endPos,
    const double diag_value = 0.0,
    const double rhs_residual = 0.0);

  // Solve
  int solve(stk::mesh::FieldBase* linearSolutionField);
  void loadComplete();
  void writeToFile(const char* filename, bool useOwned = true);
  void printInfo(bool useOwned = true);
  void writeSolutionToFile(const char* filename, bool useOwned = true);
  size_t lookup_myLID(
    MyLIDMapType& myLIDs,
    stk::mesh::EntityId entityId,
    const char* /* msg */ = nullptr,
    stk::mesh::Entity /* entity */ = stk::mesh::Entity())
  {
    return myLIDs[entityId];
  }

  int getDofStatus(stk::mesh::Entity node);

  int getRowLID(stk::mesh::Entity node)
  {
    return entityToLID_[node.local_offset()];
  }
  int getColLID(stk::mesh::Entity node)
  {
    return entityToColLID_[node.local_offset()];
  }

  Teuchos::RCP<LinSys::Graph> getOwnedGraph() { return ownedGraph_; }
  Teuchos::RCP<LinSys::Matrix> getOwnedMatrix() { return ownedMatrix_; }
  Teuchos::RCP<LinSys::MultiVector> getOwnedRhs() { return ownedRhs_; }

  LinSys::LocalMatrix getOwnedLocalMatrix()
  {
    return ownedMatrix_->getLocalMatrixDevice();
  }
  LinSys::LocalMatrix getSharedNotOwnedLocalMatrix()
  {
    return sharedNotOwnedMatrix_->getLocalMatrixDevice();
  }

  LinSys::LocalVector getOwnedLocalRhs()
  {
    return ownedRhs_->getLocalViewDevice(Tpetra::Access::ReadWrite);
  }
  LinSys::LocalVector getSharedNotOwnedLocalRhs()
  {
    return sharedNotOwnedRhs_->getLocalViewDevice(Tpetra::Access::ReadWrite);
  }

  class TpetraLinSysCoeffApplier : public CoeffApplier
  {
  public:
    KOKKOS_FUNCTION
    TpetraLinSysCoeffApplier(
      LinSys::LocalMatrix ownedLclMatrix,
      LinSys::LocalMatrix sharedNotOwnedLclMatrix,
      LinSys::LocalVector ownedLclRhs,
      LinSys::LocalVector sharedNotOwnedLclRhs,
      LinSys::EntityToLIDView entityLIDs,
      LinSys::EntityToLIDView entityColLIDs,
      int maxOwnedRowId,
      int maxSharedNotOwnedRowId,
      unsigned numDof)
      : ownedLocalMatrix_(ownedLclMatrix),
        sharedNotOwnedLocalMatrix_(sharedNotOwnedLclMatrix),
        ownedLocalRhs_(ownedLclRhs),
        sharedNotOwnedLocalRhs_(sharedNotOwnedLclRhs),
        entityToLID_(entityLIDs),
        entityToColLID_(entityColLIDs),
        maxOwnedRowId_(maxOwnedRowId),
        maxSharedNotOwnedRowId_(maxSharedNotOwnedRowId),
        numDof_(numDof)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION
    ~TpetraLinSysCoeffApplier() = default;

    KOKKOS_FUNCTION
    virtual void resetRows(
      unsigned numNodes,
      const stk::mesh::Entity* nodeList,
      const unsigned beginPos,
      const unsigned endPos,
      const double diag_value = 0.0,
      const double rhs_residual = 0.0);

    KOKKOS_FUNCTION
    virtual void operator()(
      unsigned numEntities,
      const stk::mesh::NgpMesh::ConnectedNodes& entities,
      const SharedMemView<int*, DeviceShmem>& localIds,
      const SharedMemView<int*, DeviceShmem>& sortPermutation,
      const SharedMemView<const double*, DeviceShmem>& rhs,
      const SharedMemView<const double**, DeviceShmem>& lhs,
      const char* trace_tag);

    void free_device_pointer() {}

    sierra::nalu::CoeffApplier* device_pointer() { return nullptr; }

  private:
    LinSys::LocalMatrix ownedLocalMatrix_, sharedNotOwnedLocalMatrix_;
    LinSys::LocalVector ownedLocalRhs_, sharedNotOwnedLocalRhs_;
    LinSys::EntityToLIDView entityToLID_;
    LinSys::EntityToLIDView entityToColLID_;
    int maxOwnedRowId_, maxSharedNotOwnedRowId_;
    unsigned numDof_;
  };

private:
  void buildConnectedNodeGraph(
    stk::mesh::EntityRank rank, const stk::mesh::PartVector& parts);

  void beginLinearSystemConstruction();

  void checkError(const int /* err_code */, const char* /* msg */) {}

  void compute_send_lengths(
    const std::vector<stk::mesh::Entity>& rowEntities,
    const std::vector<std::vector<stk::mesh::Entity>>& connections,
    const std::vector<int>& neighborProcs,
    stk::CommNeighbors& commNeighbors);

  void compute_graph_row_lengths(
    const std::vector<stk::mesh::Entity>& rowEntities,
    const std::vector<std::vector<stk::mesh::Entity>>& connections,
    LinSys::RowLengths& sharedNotOwnedRowLengths,
    LinSys::RowLengths& locallyOwnedRowLengths,
    stk::CommNeighbors& commNeighbors);

  void insert_graph_connections(
    const std::vector<stk::mesh::Entity>& rowEntities,
    const std::vector<std::vector<stk::mesh::Entity>>& connections,
    LocalGraphArrays& locallyOwnedGraph,
    LocalGraphArrays& sharedNotOwnedGraph);

  void fill_entity_to_row_LID_mapping();
  void fill_entity_to_col_LID_mapping();

  void copy_tpetra_to_stk(
    const Teuchos::RCP<LinSys::MultiVector> tpetraVector,
    stk::mesh::FieldBase* stkField);

  // This method copies a stk::mesh::field to a tpetra multivector. Each
  // dof/node is written into a different vector in the multivector.
  void copy_stk_to_tpetra(
    stk::mesh::FieldBase* stkField,
    const Teuchos::RCP<LinSys::MultiVector> tpetraVector);

  int insert_connection(stk::mesh::Entity a, stk::mesh::Entity b);
  void addConnections(const stk::mesh::Entity* entities, const size_t&);
  void expand_unordered_map(unsigned newCapacityNeeded);
  void checkForNaN(bool useOwned);
  bool checkForZeroRow(bool useOwned, bool doThrow, bool doPrint = false);

  std::vector<stk::mesh::Entity> ownedAndSharedNodes_;
  std::vector<std::vector<stk::mesh::Entity>> connections_;
  std::vector<GlobalOrdinal> totalGids_;
  std::set<std::pair<int, GlobalOrdinal>> ownersAndGids_;
  std::vector<int> sharedPids_;

  // all rows, otherwise known as col map
  Teuchos::RCP<LinSys::Map> totalColsMap_;
  Teuchos::RCP<LinSys::Map> optColsMap_;

  // Map of rows my proc owns (locally owned)
  Teuchos::RCP<LinSys::Map> ownedRowsMap_;

  // Only nodes that share with other procs that I don't own
  Teuchos::RCP<LinSys::Map> sharedNotOwnedRowsMap_;

  Teuchos::RCP<LinSys::Graph> ownedGraph_;
  Teuchos::RCP<LinSys::Graph> sharedNotOwnedGraph_;

  Teuchos::RCP<LinSys::Matrix> ownedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> ownedRhs_;

  Teuchos::RCP<LinSys::Matrix> sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> sharedNotOwnedRhs_;

  Teuchos::RCP<LinSys::MultiVector> sln_;
  Teuchos::RCP<LinSys::MultiVector> globalSln_;
  Teuchos::RCP<LinSys::Export> exporter_;

  MyLIDMapType myLIDs_;
  LinSys::EntityToLIDView entityToColLID_;
  LinSys::EntityToLIDHostView entityToColLIDHost_;
  LinSys::EntityToLIDView entityToLID_;
  LinSys::EntityToLIDHostView entityToLIDHost_;
  LocalOrdinal maxOwnedRowId_;          // = num_owned_nodes * numDof_
  LocalOrdinal maxSharedNotOwnedRowId_; // = (num_owned_nodes +
                                        // num_sharedNotOwned_nodes) * numDof_

  std::vector<int> sortPermutation_;
};

int getDofStatus_impl(stk::mesh::Entity node, const Realm& realm);

} // namespace nalu
} // namespace sierra

#endif
