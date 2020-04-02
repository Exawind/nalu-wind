// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <TpetraSegregatedLinearSystem.h>
#include <CrsGraphHelpers.h>
#include <CrsGraph.h>
#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <FieldTypeDef.h>
#include <DgInfo.h>
#include <Realm.h>
#include <PeriodicManager.h>
#include <Simulation.h>
#include <LinearSolver.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <EquationSystem.h>
#include <NaluEnv.h>
#include <utils/StkHelpers.h>
#include <utils/CreateDeviceExpression.h>

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

///====================================================================================================================================
///======== T P E T R A ===============================================================================================================
///====================================================================================================================================

//==========================================================================
// Class Definition
//==========================================================================
// TpetraLinearSystem - hook to Tpetra
//==========================================================================
TpetraSegregatedLinearSystem::TpetraSegregatedLinearSystem(
  Realm &realm,
  const unsigned numDof,
  EquationSystem *eqSys,
  LinearSolver * linearSolver)
  : LinearSystem(realm, numDof, eqSys, linearSolver)
{
  if (realm.scalarGraph_ == Teuchos::null)
    realm.scalarGraph_ = Teuchos::rcp(new CrsGraph(realm,1));
  crsGraph_ = realm.scalarGraph_;
}

TpetraSegregatedLinearSystem::~TpetraSegregatedLinearSystem()
{
  // dereference linear solver in safe manner
  if (linearSolver_ != nullptr) {
    TpetraLinearSolver *linearSolver = reinterpret_cast<TpetraLinearSolver *>(linearSolver_);
    linearSolver->destroyLinearSolver();
  }
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
int TpetraSegregatedLinearSystem::getDofStatus(stk::mesh::Entity node)
{
    return getDofStatus_impl(node, realm_);
}

void TpetraSegregatedLinearSystem::beginLinearSystemConstruction()
{
  if(inConstruction_) return;
  inConstruction_ = true;
}

void TpetraSegregatedLinearSystem::buildNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildNodeGraph(parts);
}

void TpetraSegregatedLinearSystem::buildConnectedNodeGraph(stk::mesh::EntityRank rank,
                                                           const stk::mesh::PartVector& parts)
{
  crsGraph_->buildConnectedNodeGraph(rank, parts);
}

void TpetraSegregatedLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildConnectedNodeGraph(stk::topology::EDGE_RANK, parts);
}

void TpetraSegregatedLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // stk::mesh::MetaData & metaData = realm_.meta_data();
  // crsGraph_->buildConnectedNodeGraph(metaData.side_rank(), parts);
}

void TpetraSegregatedLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildConnectedNodeGraph(stk::topology::ELEM_RANK, parts);
}

void TpetraSegregatedLinearSystem::buildReducedElemToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildReducedElemToNodeGraph(parts);
}

void TpetraSegregatedLinearSystem::buildFaceElemToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildFaceElemToNodeGraph(parts);
}

void TpetraSegregatedLinearSystem::buildNonConformalNodeGraph(const stk::mesh::PartVector &)
{
  beginLinearSystemConstruction();
  // crsGraph_->buildNonConformalNodeGraph(parts);
}

void TpetraSegregatedLinearSystem::buildOversetNodeGraph(const stk::mesh::PartVector &)
{
  beginLinearSystemConstruction();
  // crsGraph_->buildOversetNodeGraph(parts);
}

void TpetraSegregatedLinearSystem::copy_stk_to_tpetra(stk::mesh::FieldBase * stkField,
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

      const stk::mesh::EntityId nodeId = *stk::mesh::field_data(*realm_.tpetGlobalId_, node);
      ThrowRequireMsg(nodeId != 0 && nodeId != std::numeric_limits<LinSys::GlobalOrdinal>::max()
                            , " in copy_stk_to_tpetra ");
      for(int dofIdx = 0; dofIdx < fieldSize; ++dofIdx)
      {
        const size_t stkIndex = k*fieldSize + dofIdx;
        tpetraField->replaceGlobalValue(nodeId, dofIdx, stkFieldPtr[stkIndex]);
      }
    }
  }
}

void TpetraSegregatedLinearSystem::finalizeLinearSystem()
{
  ThrowRequire(inConstruction_);
  inConstruction_ = false;
  crsGraph_->finalizeGraph();
  entityToColLID_ = crsGraph_->get_entity_to_col_LID_mapping();
  entityToLID_ = crsGraph_->get_entity_to_row_LID_mapping();
  exporter_ = crsGraph_->getExporter();
  myLIDs_ = crsGraph_->get_my_LIDs();
  maxOwnedRowId_ = crsGraph_->getMaxOwnedRowID();
  maxSharedNotOwnedRowId_ = crsGraph_->getMaxSharedNotOwnedRowID();

  ownedMatrix_ = Teuchos::rcp(new LinSys::Matrix(crsGraph_->getOwnedGraph()));
  sharedNotOwnedMatrix_ = Teuchos::rcp(new LinSys::Matrix(crsGraph_->getSharedNotOwnedGraph()));

  ownedLocalMatrix_ = ownedMatrix_->getLocalMatrix();
  sharedNotOwnedLocalMatrix_ = sharedNotOwnedMatrix_->getLocalMatrix();

  ownedRhs_ = Teuchos::rcp(new LinSys::MultiVector(crsGraph_->getOwnedRowsMap(), numDof_));
  sharedNotOwnedRhs_ = Teuchos::rcp(new LinSys::MultiVector(crsGraph_->getSharedNotOwnedRowsMap(), numDof_));

  ownedLocalRhs_ = ownedRhs_->getLocalView<sierra::nalu::DeviceSpace>();
  sharedNotOwnedLocalRhs_ = sharedNotOwnedRhs_->getLocalView<sierra::nalu::DeviceSpace>();

  sln_ = Teuchos::rcp(new LinSys::MultiVector(crsGraph_->getOwnedRowsMap(), numDof_));

  stk::mesh::MetaData & metaData = realm_.meta_data();
  const int nDim = metaData.spatial_dimension();

  Teuchos::RCP<LinSys::MultiVector> coords
    = Teuchos::RCP<LinSys::MultiVector>(new LinSys::MultiVector(sln_->getMap(), nDim));

  TpetraLinearSolver *linearSolver = reinterpret_cast<TpetraLinearSolver *>(linearSolver_);

  if (linearSolver != nullptr) {
    VectorFieldType *coordinates = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
    if (linearSolver->activeMueLu())
      copy_stk_to_tpetra(coordinates, coords);

    linearSolver->setupLinearSolver(sln_, ownedMatrix_, ownedRhs_, coords);
  }
}

void TpetraSegregatedLinearSystem::zeroSystem()
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

template <typename RowViewType>
KOKKOS_FUNCTION
void segregated_sum_into_row (RowViewType row_view,
                              const int num_entities,
                              const int numDof,
                              const int* localIds,
                              const int* sort_permutation,
                              const double* input_values)
{

  constexpr bool forceAtomic = !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;
  const LocalOrdinal length = row_view.length;

  const int numCols = num_entities;
  LocalOrdinal offset = 0;
  for (int colIdx = 0; colIdx < numCols; ++colIdx) {
    const LocalOrdinal perm_index = sort_permutation[colIdx];
    const LocalOrdinal cur_local_column_idx = localIds[colIdx];

    // since the columns are sorted, we pass through the column idxs once,
    // updating the offset as we go
    while (row_view.colidx(offset) != cur_local_column_idx && offset < length) {
      ++offset;
    }

    if (offset < length) {
      ThrowAssertMsg(std::isfinite(input_values[perm_index*numDof]), "Inf or NAN lhs");
      if (forceAtomic) {
        Kokkos::atomic_add(&(row_view.value(offset)), input_values[perm_index*numDof]);
      }
      else {
        row_view.value(offset) += input_values[perm_index*numDof];
      }
    }
  }
}

template<typename MatrixType,
         typename RhsType,
         typename EntityArrayType,
         typename ShmemView1DType,
         typename ShmemView2DType,
         typename ShmemIntView1DType,
         typename EntityLIDType>
KOKKOS_FUNCTION
void segregated_sum_into(MatrixType ownedLocalMatrix,
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
  constexpr bool forceAtomic = !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;

  const int n_obj = numEntities;
  const int numRows = n_obj;

  for(int i = 0; i < n_obj; i++) {
    localIds[i] = entityToColLID[entities[i].local_offset()];
    sortPermutation[i] = i;
  }
  Tpetra::Details::shellSortKeysAndValues(localIds.data(), sortPermutation.data(), numRows);


  for (int r = 0; r < numRows; ++r) {
    int i = sortPermutation[r];
    LocalOrdinal rowLid = entityToLID[entities[i].local_offset()];
    const LocalOrdinal cur_perm_index = sortPermutation[r];
    const double* const cur_lhs = &lhs(cur_perm_index*numDof, 0);

    if(rowLid < maxOwnedRowId) {
      segregated_sum_into_row(ownedLocalMatrix.row(rowLid), n_obj, numDof,
                              localIds.data(), sortPermutation.data(), cur_lhs);

      for(unsigned dofIdx = 0; dofIdx < numDof; ++dofIdx) {
        const double cur_rhs = rhs[cur_perm_index*numDof + dofIdx];
        if (forceAtomic) {
          Kokkos::atomic_add(&ownedLocalRhs(rowLid, dofIdx), cur_rhs);
        } else {
          ownedLocalRhs(rowLid, dofIdx) += cur_rhs;
        }
      }
    } else if (rowLid < maxSharedNotOwnedRowId) {
      LocalOrdinal actualLocalId = rowLid - maxOwnedRowId;
      segregated_sum_into_row(sharedNotOwnedLocalMatrix.row(actualLocalId), n_obj, numDof,
                              localIds.data(), sortPermutation.data(), cur_lhs);

      for(unsigned dofIdx = 0; dofIdx < numDof; ++dofIdx) {
        const double cur_rhs = rhs[cur_perm_index*numDof + dofIdx];
        if (forceAtomic) {
          Kokkos::atomic_add(&sharedNotOwnedLocalRhs(actualLocalId, dofIdx), cur_rhs);
        } else {
          sharedNotOwnedLocalRhs(actualLocalId, dofIdx) += cur_rhs;
        }
      }
    }
  }
}

template <typename RowViewType>
KOKKOS_FUNCTION
void reset_row(
  RowViewType row_view,
  const int localRowId,
  const double diag_value)
{
  const LocalOrdinal length = row_view.length;

  for(LocalOrdinal i=0; i<length; ++i) {
    if (row_view.colidx(i) == localRowId) {
      row_view.value(i) = diag_value;
    }
    else {
      row_view.value(i) = 0.0;
    }
  }
}

template<typename MatrixType,
         typename RhsType,
         typename EntityArrayType,
         typename EntityLIDType>
KOKKOS_FUNCTION
void reset_rows(
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
  for (unsigned nn=0; nn<numNodes; ++nn) {
    stk::mesh::Entity node = nodeList[nn];
    const LocalOrdinal localIdOffset = entityToLID[node.local_offset()];
    const bool useOwned = (localIdOffset < maxOwnedRowId);
    const LinSys::LocalMatrix& localMatrix = useOwned ?  ownedLocalMatrix : sharedNotOwnedLocalMatrix;
    const LinSys::LocalVector& localRhs = useOwned ? ownedLocalRhs : sharedNotOwnedLocalRhs;

    for (unsigned d=beginPos; d < endPos; ++d) {
      const LocalOrdinal localId = localIdOffset + d;
      const LocalOrdinal actualLocalId =
        useOwned ? localId : (localId - maxOwnedRowId);

      NGP_ThrowRequire(localId <= maxSharedNotOwnedRowId);

      // Adjust the LHS; zero out all entries (including diagonal)
      reset_row(localMatrix.row(actualLocalId), actualLocalId, diag_value);

      // Replace RHS residual entry
      localRhs(actualLocalId,0) = rhs_residual;
    }
  }
}

sierra::nalu::CoeffApplier* TpetraSegregatedLinearSystem::get_coeff_applier()
{
  if (!hostCoeffApplier) {
    hostCoeffApplier.reset(new TpetraLinSysCoeffApplier(
      ownedLocalMatrix_, sharedNotOwnedLocalMatrix_, ownedLocalRhs_,
      sharedNotOwnedLocalRhs_, entityToLID_, entityToColLID_, maxOwnedRowId_,
      maxSharedNotOwnedRowId_, numDof_));
    deviceCoeffApplier = hostCoeffApplier->device_pointer();
  }

  return deviceCoeffApplier;
}

KOKKOS_FUNCTION
void TpetraSegregatedLinearSystem::TpetraLinSysCoeffApplier::resetRows(unsigned numNodes,
                           const stk::mesh::Entity* nodeList,
                           const unsigned beginPos,
                           const unsigned endPos,
                           const double diag_value,
                           const double rhs_residual)
{
  reset_rows(ownedLocalMatrix_, sharedNotOwnedLocalMatrix_,
             ownedLocalRhs_, sharedNotOwnedLocalRhs_,
             numNodes, nodeList, beginPos, endPos, diag_value, rhs_residual,
             entityToLID_, maxOwnedRowId_, maxSharedNotOwnedRowId_);
}

KOKKOS_FUNCTION
void TpetraSegregatedLinearSystem::TpetraLinSysCoeffApplier::operator() (unsigned numEntities,
                                                                         const ngp::Mesh::ConnectedNodes& entities,
                                                                         const SharedMemView<int*,DeviceShmem> & localIds,
                                                                         const SharedMemView<int*,DeviceShmem> & sortPermutation,
                                                                         const SharedMemView<const double*,DeviceShmem> & rhs,
                                                                         const SharedMemView<const double**,DeviceShmem> & lhs,
                                                                         const char * /*trace_tag*/)
{
  segregated_sum_into(ownedLocalMatrix_, sharedNotOwnedLocalMatrix_,
                      ownedLocalRhs_, sharedNotOwnedLocalRhs_,
                      numEntities, entities,
                      rhs, lhs,
                      localIds, sortPermutation,
                      entityToLID_, entityToColLID_,
                      maxOwnedRowId_, maxSharedNotOwnedRowId_,
                      numDof_);
}

void TpetraSegregatedLinearSystem::TpetraLinSysCoeffApplier::free_device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier* TpetraSegregatedLinearSystem::TpetraLinSysCoeffApplier::device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (devicePointer_ != nullptr) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
  devicePointer_ = sierra::nalu::create_device_expression(*this);
#else
  devicePointer_ = this;
#endif
  return devicePointer_;
}

void TpetraSegregatedLinearSystem::sumInto(unsigned numEntities,
                                           const ngp::Mesh::ConnectedNodes& entities,
                                           const SharedMemView<const double*,DeviceShmem> & rhs,
                                           const SharedMemView<const double**,DeviceShmem> & lhs,
                                           const SharedMemView<int*,DeviceShmem> & localIds,
                                           const SharedMemView<int*,DeviceShmem> & sortPermutation,
                                           const char *  /* trace_tag */)
{
  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  ThrowAssertMsg(localIds.span_is_contiguous(), "localIds assumed contiguous");
  ThrowAssertMsg(sortPermutation.span_is_contiguous(), "sortPermutation assumed contiguous");

  segregated_sum_into(ownedLocalMatrix_, sharedNotOwnedLocalMatrix_,
                      ownedLocalRhs_, sharedNotOwnedLocalRhs_,
                      numEntities, entities,
                      rhs, lhs,
                      localIds, sortPermutation,
                      entityToLID_, entityToColLID_,
                      maxOwnedRowId_, maxSharedNotOwnedRowId_,
                      numDof_);
}

void TpetraSegregatedLinearSystem::sumInto(const std::vector<stk::mesh::Entity> & entities,
                                           std::vector<int> &scratchIds,
                                           std::vector<double> & /* scratchVals */,
                                           const std::vector<double> & rhs,
                                           const std::vector<double> & lhs,
                                           const char * /* trace_tag */)
{
  const size_t n_obj = entities.size();
  const unsigned numRows = n_obj;

  ThrowAssert(numRows*numDof_ == rhs.size());
  ThrowAssert(numRows*numDof_*numRows*numDof_ == lhs.size());

  scratchIds.resize(numRows);
  sortPermutation_.resize(numRows);
  for(size_t i = 0; i < n_obj; i++) {
    scratchIds[i] = entityToColLID_[entities[i].local_offset()];
    sortPermutation_[i] = i;
  }
  Tpetra::Details::shellSortKeysAndValues(scratchIds.data(), sortPermutation_.data(), (int)numRows);

  for (unsigned r = 0; r < numRows; r++) {
    int i = sortPermutation_[r];
    LocalOrdinal rowLid = entityToLID_[entities[i].local_offset()];
    const LocalOrdinal cur_perm_index = sortPermutation_[r];
    const double* const cur_lhs = &lhs[(cur_perm_index*numDof_)*(numRows*numDof_)];

    if(rowLid < maxOwnedRowId_) {
      segregated_sum_into_row(ownedLocalMatrix_.row(rowLid),  n_obj, numDof_,
                              scratchIds.data(), sortPermutation_.data(), cur_lhs);

      for(unsigned dofIdx = 0; dofIdx < numDof_; ++dofIdx) {
        const double cur_rhs = rhs[cur_perm_index*numDof_ + dofIdx];
        ThrowAssertMsg(std::isfinite(cur_rhs), "Invalid rhs");
        ownedLocalRhs_(rowLid, dofIdx) += cur_rhs;
      }
    }
    else if (rowLid < maxSharedNotOwnedRowId_) {
      LocalOrdinal actualLocalId = rowLid - maxOwnedRowId_;
      segregated_sum_into_row(sharedNotOwnedLocalMatrix_.row(actualLocalId),  n_obj, numDof_,
                              scratchIds.data(), sortPermutation_.data(), cur_lhs);

      for(unsigned dofIdx = 0; dofIdx < numDof_; ++dofIdx) {
        const double cur_rhs = rhs[cur_perm_index*numDof_ + dofIdx];
        ThrowAssertMsg(std::isfinite(cur_rhs), "Invalid rhs");
        sharedNotOwnedLocalRhs_(actualLocalId, dofIdx) += cur_rhs;
      }
    }
  }
}

void TpetraSegregatedLinearSystem::applyDirichletBCs(stk::mesh::FieldBase * solutionField,
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

      const bool useOwned = localIdOffset < maxOwnedRowId_;
      const LocalOrdinal actualLocalId = useOwned ? localIdOffset : localIdOffset - maxOwnedRowId_;
      Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
      const LinSys::LocalMatrix& local_matrix = useOwned ? ownedLocalMatrix_ : sharedNotOwnedLocalMatrix_;

      if(localIdOffset > maxSharedNotOwnedRowId_) {
        std::cerr << "localId > maxSharedNotOwnedRowId_:: localId= " << localIdOffset
                  << " maxSharedNotOwnedRowId_= " << maxSharedNotOwnedRowId_
                  << " useOwned = " << (localIdOffset < maxOwnedRowId_ ) << std::endl;
        throw std::runtime_error("logic error: localId > maxSharedNotOwnedRowId_");
      }

      // Adjust the LHS

      const double diagonal_value = useOwned ? 1.0 : 0.0;

      matrix->getLocalRowView(actualLocalId, indices, values);
      const size_t rowLength = values.size();
      if (rowLength > 0) {
        new_values.resize(rowLength);
        for(size_t i=0; i < rowLength; ++i) {
          new_values[i] = (indices[i] == localIdOffset) ? diagonal_value : 0;
        }
        local_matrix.replaceValues(actualLocalId, &indices[0], rowLength, new_values.data(), internalMatrixIsSorted);
      }

      // Replace the RHS residual with (desired - actual)
      Teuchos::RCP<LinSys::MultiVector> rhs = useOwned ? ownedRhs_: sharedNotOwnedRhs_;
      for(unsigned d = beginPos; d < endPos; ++d) {
        const double bc_residual = useOwned ? (bcValues[k*fieldSize + d] - solution[k*fieldSize + d]) : 0.0;
        rhs->replaceLocalValue(actualLocalId, d, bc_residual);
        ++nbc;
      }
    }
  }
  adbc_time += NaluEnv::self().nalu_time();
}

void TpetraSegregatedLinearSystem::resetRows(const std::vector<stk::mesh::Entity>& nodeList,
                                             const unsigned beginPos,
                                             const unsigned endPos,
                                             const double diag_value,
                                             const double rhs_residual)
{
  resetRows(nodeList.size(), nodeList.data(), beginPos, endPos, diag_value, rhs_residual);
}

void TpetraSegregatedLinearSystem::resetRows(unsigned numNodes,
                                             const stk::mesh::Entity* nodeList,
                                             const unsigned beginPos,
                                             const unsigned endPos,
                                             const double diag_value,
                                             const double rhs_residual)
{
  reset_rows(ownedLocalMatrix_, sharedNotOwnedLocalMatrix_,
             ownedLocalRhs_, sharedNotOwnedLocalRhs_,
             numNodes, nodeList, beginPos, endPos, diag_value, rhs_residual,
             entityToLID_, maxOwnedRowId_, maxSharedNotOwnedRowId_);
}

void TpetraSegregatedLinearSystem::loadComplete()
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

int TpetraSegregatedLinearSystem::solve(stk::mesh::FieldBase * linearSolutionField)
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
    NaluEnv::self().naluOutputP0() << "NaluMemory::TpetraSegregatedLinearSystem::solve() PreSolve: " << eqSysName_ << std::endl;
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
    ++eqSys_->linsysWriteCounter_;
  }

  copy_tpetra_to_stk(sln_, linearSolutionField);
  sync_field(linearSolutionField);

  // computeL2 norm
  Teuchos::Array<double> mv_norm(ownedRhs_->getNumVectors());
  ownedRhs_->norm2(mv_norm());
  double norm2 = 0.0;
  for(size_t vecIdx = 0; vecIdx < ownedRhs_->getNumVectors(); ++vecIdx) {
    norm2 += mv_norm[vecIdx]*mv_norm[vecIdx];
  }
  norm2 = std::sqrt(norm2);

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

void TpetraSegregatedLinearSystem::checkForNaN(bool useOwned)
{
  Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> rhs = useOwned ? ownedRhs_ : sharedNotOwnedRhs_;

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

  for(unsigned dofIdx = 0; dofIdx < numDof_; ++dofIdx) {
    Teuchos::ArrayRCP<const Scalar> rhs_data = rhs->getData(dofIdx);
    n = rhs_data.size();
    for(size_t i = 0; i < n; ++i) {
      if (rhs_data[i] != rhs_data[i]) {
        std::cerr << "rhs NaN: (" << i << ", " << dofIdx << ")" << std::endl;
        throw std::runtime_error("bad rhs");
      }
    }
  }
}

bool TpetraSegregatedLinearSystem::checkForZeroRow(bool useOwned, bool doThrow, bool doPrint)
{
  Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> rhs = useOwned ? ownedRhs_ : sharedNotOwnedRhs_;
  stk::mesh::BulkData & bulkData = realm_.bulk_data();

  Teuchos::ArrayView<const LocalOrdinal> indices;
  Teuchos::ArrayView<const double> values;

  size_t nrowG = matrix->getRangeMap()->getGlobalNumElements();
  size_t n = matrix->getRowMap()->getNodeNumElements();
  GlobalOrdinal max_gid = 0, g_max_gid=0;
  //KOKKOS: Loop parallel reduce
  kokkos_parallel_for("Nalu::TpetraSegregatedLinearSystem::checkForZeroRowA", n, [&] (const size_t& i) {
    GlobalOrdinal gid = matrix->getGraph()->getRowMap()->getGlobalElement(i);
    max_gid = std::max(gid, max_gid);
  });
  stk::all_reduce_max(bulkData.parallel(), &max_gid, &g_max_gid, 1);

  nrowG = g_max_gid+1;
  std::vector<double> local_row_sums   (nrowG, 0.0);
  std::vector<int>    local_row_exists (nrowG, 0);
  std::vector<double> global_row_sums  (nrowG, 0.0);
  std::vector<int>    global_row_exists(nrowG, 0);

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
  kokkos_parallel_for("Nalu::TpetraSegregatedLinearSystem::checkForZeroRowC", nrowG, [&] (const size_t& ii) {
    double row_sum = global_row_sums[ii];
    if (global_row_exists[ii] && bulkData.parallel_rank() == 0 && row_sum < 1.e-10) {
      found = true;
      GlobalOrdinal gid = ii+1;
      stk::mesh::EntityId nid = static_cast<stk::mesh::EntityId>(gid);
      stk::mesh::Entity node = bulkData.get_entity(stk::topology::NODE_RANK, nid);
      stk::mesh::EntityId naluGlobalId;
      if (bulkData.is_valid(node)) naluGlobalId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);

      if (doPrint) {

        double dualVolume = -1.0;

        std::cout << "P[" << bulkData.parallel_rank() << "] LHS zero: " << ii
                  << " GID= " << gid
                  << " nid= " << nid
                  << " naluGlobalId " << naluGlobalId
                  << " is_valid= " << bulkData.is_valid(node)
                  << " numDof_= " << numDof_
                  << " row_sum= " << row_sum
                  << " dualVolume= " << dualVolume
                  << std::endl;
        NaluEnv::self().naluOutputP0() << "P[" << bulkData.parallel_rank() << "] LHS zero: " << ii
                                       << " GID= " << gid << " nid= " << nid
                                       << " naluGlobalId " << naluGlobalId
                                       << " is_valid= " << bulkData.is_valid(node)
                                       << " numDof_= " << numDof_
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

void TpetraSegregatedLinearSystem::writeToFile(const char * base_filename, bool useOwned)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  const unsigned p_rank = bulkData.parallel_rank();
  const unsigned p_size = bulkData.parallel_size();

  Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> rhs = useOwned ? ownedRhs_ : sharedNotOwnedRhs_;

  const int currentCount = eqSys_->linsysWriteCounter_;

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

void TpetraSegregatedLinearSystem::printInfo(bool useOwned)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  const unsigned p_rank = bulkData.parallel_rank();

  Teuchos::RCP<LinSys::Matrix> matrix = useOwned ? ownedMatrix_ : sharedNotOwnedMatrix_;
  Teuchos::RCP<LinSys::MultiVector> rhs = useOwned ? ownedRhs_ : sharedNotOwnedRhs_;

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

void TpetraSegregatedLinearSystem::writeSolutionToFile(const char * base_filename, bool useOwned)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  const unsigned p_rank = bulkData.parallel_rank();
  const unsigned p_size = bulkData.parallel_size();

  Teuchos::RCP<LinSys::MultiVector> sln = sln_;
  const int currentCount = eqSys_->linsysWriteCounter_;

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

void TpetraSegregatedLinearSystem::copy_tpetra_to_stk(const Teuchos::RCP<LinSys::MultiVector> tpetraField,
                                                      stk::mesh::FieldBase * stkField)
{
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  ThrowAssert(!tpetraField.is_null());
  ThrowAssert(stkField);
  const LinSys::ConstOneDVector & tpetraVector = tpetraField->get1dView();
  const size_t numNodes = tpetraField->getLocalLength();

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
      for(unsigned dofIdx = 0; dofIdx < fieldSize; ++dofIdx) {
        const LocalOrdinal localId = localIdOffset;
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

        const size_t stkIndex = k*numDof_ + dofIdx;
        if (useOwned){
          stkFieldPtr[stkIndex] = tpetraVector[localId + dofIdx*numNodes];
        }
      }
    }
  }
}

  Teuchos::RCP<LinSys::Graph>  TpetraSegregatedLinearSystem::getOwnedGraph() { return crsGraph_->getOwnedGraph(); }
  Teuchos::RCP<LinSys::Matrix> TpetraSegregatedLinearSystem::getOwnedMatrix() { return ownedMatrix_; }
  Teuchos::RCP<LinSys::MultiVector> TpetraSegregatedLinearSystem::getOwnedRhs() { return ownedRhs_; }

} // namespace nalu
} // namespace Sierra
