// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef TpetraLinearSystem_h
#define TpetraLinearSystem_h

#include <LinearSystem.h>

#include <KokkosInterface.h>
#include <FieldTypeDef.h>

#include <Kokkos_DefaultNode.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <stk_ngp/Ngp.hpp>

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
class CrsGraph;

typedef std::unordered_map<stk::mesh::EntityId, size_t>  MyLIDMapType;

typedef std::pair<stk::mesh::Entity, stk::mesh::Entity> Connection;


class TpetraLinearSystem : public LinearSystem
{
public:
  typedef LinSys::GlobalOrdinal GlobalOrdinal;
  typedef LinSys::LocalOrdinal  LocalOrdinal;

  TpetraLinearSystem(
    Realm &realm,
    const unsigned numDof,
    EquationSystem *eqSys,
    LinearSolver * linearSolver);
  ~TpetraLinearSystem();

  // Graph/Matrix Construction
  // These all call through to the CrsGraph methods of the same name
  void buildNodeGraph(const stk::mesh::PartVector & parts); // for nodal assembly (e.g., lumped mass and source)
  void buildFaceToNodeGraph(const stk::mesh::PartVector & parts); // face->node assembly
  void buildEdgeToNodeGraph(const stk::mesh::PartVector & parts); // edge->node assembly
  void buildElemToNodeGraph(const stk::mesh::PartVector & parts); // elem->node assembly
  void buildReducedElemToNodeGraph(const stk::mesh::PartVector & parts); // elem (nearest nodes only)->node assembly
  void buildFaceElemToNodeGraph(const stk::mesh::PartVector & parts); // elem:face->node assembly
  void buildNonConformalNodeGraph(const stk::mesh::PartVector & parts); // nonConformal->node assembly
  void buildOversetNodeGraph(const stk::mesh::PartVector & parts); // overset->elem_node assembly
  void finalizeLinearSystem();

  sierra::nalu::CoeffApplier* get_coeff_applier();

  // Matrix Assembly
  void zeroSystem();

  void sumInto(
    unsigned numEntities,
    const ngp::Mesh::ConnectedNodes& entities,
    const SharedMemView<const double*,DeviceShmem> & rhs,
    const SharedMemView<const double**,DeviceShmem> & lhs,
    const SharedMemView<int*,DeviceShmem> & localIds,
    const SharedMemView<int*,DeviceShmem> & sortPermutation,
    const char * trace_tag);

  void sumInto(
    const std::vector<stk::mesh::Entity> & entities,
    std::vector<int> &scratchIds,
    std::vector<double> &scratchVals,
    const std::vector<double> & rhs,
    const std::vector<double> & lhs,
    const char *trace_tag=0
    );

  void applyDirichletBCs(
    stk::mesh::FieldBase * solutionField,
    stk::mesh::FieldBase * bcValuesField,
    const stk::mesh::PartVector & parts,
    const unsigned beginPos,
    const unsigned endPos);

  /** Reset LHS and RHS for the given set of nodes to 0
   *
   *  @param nodeList A list of STK node entities whose rows are zeroed out
   *  @param beginPos Starting index (usually 0)
   *  @param endPos Terminating index (1 for scalar quantities; nDim for vectors)
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
  int solve(stk::mesh::FieldBase * linearSolutionField);
  void loadComplete();
  void writeToFile(const char * filename, bool useOwned=true);
  void printInfo(bool useOwned=true);
  void writeSolutionToFile(const char * filename, bool useOwned=true);
  size_t lookup_myLID(MyLIDMapType& myLIDs, stk::mesh::EntityId entityId, const char* /* msg */ =nullptr, stk::mesh::Entity /* entity */ = stk::mesh::Entity())
  {
    return myLIDs[entityId];
  }

  void copy_tpetra_to_stk(const Teuchos::RCP<LinSys::MultiVector> tpetraVector,
                          stk::mesh::FieldBase * stkField);

  // This method copies a stk::mesh::field to a tpetra multivector. Each dof/node is written
  // into a different vector in the multivector.
  void copy_stk_to_tpetra(const stk::mesh::FieldBase * stkField,
                          const Teuchos::RCP<LinSys::MultiVector> tpetraVector);


  int getDofStatus(stk::mesh::Entity node);

  //for unit testing
  int getRowLID(stk::mesh::Entity node) { return entityToLID_[node.local_offset()]; }
  //for unit testing
  int getColLID(stk::mesh::Entity node) { return entityToColLID_[node.local_offset()]; }

  Teuchos::RCP<LinSys::Graph>  getOwnedGraph();
  Teuchos::RCP<LinSys::Matrix> getOwnedMatrix();
  Teuchos::RCP<LinSys::MultiVector> getOwnedRhs();

  class TpetraLinSysCoeffApplier : public CoeffApplier
  {
  public:
    KOKKOS_FUNCTION
    TpetraLinSysCoeffApplier(LinSys::LocalMatrix ownedLclMatrix,
                             LinSys::LocalMatrix sharedNotOwnedLclMatrix,
                             LinSys::LocalVector ownedLclRhs,
                             LinSys::LocalVector sharedNotOwnedLclRhs,
                             LinSys::EntityToLIDView entityLIDs,
                             LinSys::EntityToLIDView entityColLIDs,
                             int maxOwnedRowId, int maxSharedNotOwnedRowId, unsigned numDof)
    : ownedLocalMatrix_(ownedLclMatrix),
      sharedNotOwnedLocalMatrix_(sharedNotOwnedLclMatrix),
      ownedLocalRhs_(ownedLclRhs),
      sharedNotOwnedLocalRhs_(sharedNotOwnedLclRhs),
      entityToLID_(entityLIDs),
      entityToColLID_(entityColLIDs),
      maxOwnedRowId_(maxOwnedRowId), maxSharedNotOwnedRowId_(maxSharedNotOwnedRowId), numDof_(numDof),
      devicePointer_(nullptr)
    {}

    KOKKOS_FUNCTION
    ~TpetraLinSysCoeffApplier() {}

    KOKKOS_FUNCTION
    virtual void resetRows(unsigned numNodes,
                           const stk::mesh::Entity* nodeList,
                           const unsigned beginPos,
                           const unsigned endPos,
                           const double diag_value = 0.0,
                           const double rhs_residual = 0.0);

    KOKKOS_FUNCTION
    virtual void operator()(unsigned numEntities,
                            const ngp::Mesh::ConnectedNodes& entities,
                            const SharedMemView<int*,DeviceShmem> & localIds,
                            const SharedMemView<int*,DeviceShmem> & sortPermutation,
                            const SharedMemView<const double*,DeviceShmem> & rhs,
                            const SharedMemView<const double**,DeviceShmem> & lhs,
                            const char * trace_tag);

    void free_device_pointer();

    sierra::nalu::CoeffApplier* device_pointer();

  private:
    LinSys::LocalMatrix ownedLocalMatrix_, sharedNotOwnedLocalMatrix_;
    LinSys::LocalVector ownedLocalRhs_, sharedNotOwnedLocalRhs_;
    LinSys::EntityToLIDView entityToLID_;
    LinSys::EntityToLIDView entityToColLID_;
    int maxOwnedRowId_, maxSharedNotOwnedRowId_;
    unsigned numDof_;
    TpetraLinSysCoeffApplier* devicePointer_;
  };

private:

  Teuchos::RCP<CrsGraph>   crsGraph_;

  //calls through to CrsGraph::buildConnectedNodeGraph()
  void buildConnectedNodeGraph(stk::mesh::EntityRank rank,
                               const stk::mesh::PartVector& parts);

  void beginLinearSystemConstruction();

  void checkError( const int /* err_code */, const char * /* msg */) {}

  void expand_unordered_map(unsigned newCapacityNeeded);

  void checkForNaN(bool useOwned);
  bool checkForZeroRow(bool useOwned, bool doThrow, bool doPrint=false);

  Teuchos::RCP<LinSys::Matrix> ownedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> ownedRhs_;
  LinSys::LocalMatrix ownedLocalMatrix_;
  LinSys::LocalMatrix sharedNotOwnedLocalMatrix_;
  LinSys::LocalVector ownedLocalRhs_;
  LinSys::LocalVector sharedNotOwnedLocalRhs_;

  Teuchos::RCP<LinSys::Matrix>      sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> sharedNotOwnedRhs_;

  Teuchos::RCP<LinSys::MultiVector> sln_;
  Teuchos::RCP<LinSys::MultiVector> globalSln_;
  Teuchos::RCP<LinSys::Export>      exporter_;

  MyLIDMapType myLIDs_;
  LinSys::EntityToLIDView entityToColLID_;
  LinSys::EntityToLIDView entityToLID_;
  LocalOrdinal maxOwnedRowId_; // = num_owned_nodes * numDof_
  LocalOrdinal maxSharedNotOwnedRowId_; // = (num_owned_nodes + num_sharedNotOwned_nodes) * numDof_

  std::vector<int> sortPermutation_;
};

int getDofStatus_impl(stk::mesh::Entity node, const Realm& realm);

} // namespace nalu
} // namespace Sierra

#endif
