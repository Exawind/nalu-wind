// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <TpetraLinearSystem.h>
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
#include <ngp_utils/NgpLoopUtils.h>

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
TpetraLinearSystem::TpetraLinearSystem(
  Realm &realm,
  const unsigned numDof,
  EquationSystem *eqSys,
  LinearSolver * linearSolver)
  : LinearSystem(realm, numDof, eqSys, linearSolver)
{
  if (numDof == 1) {
    if (realm.scalarGraph_ == Teuchos::null)
      realm.scalarGraph_ = Teuchos::rcp(new CrsGraph(realm,numDof));
    crsGraph_ = realm.scalarGraph_;
  } else {
    if (realm.systemGraph_ == Teuchos::null)
      realm.systemGraph_ = Teuchos::rcp(new CrsGraph(realm,numDof));
    crsGraph_ = realm.systemGraph_;
  }
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

void TpetraLinearSystem::beginLinearSystemConstruction()
{
  inConstruction_ = true;
  return;
}

void TpetraLinearSystem::buildNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildNodeGraph(parts);
}

void TpetraLinearSystem::buildConnectedNodeGraph(stk::mesh::EntityRank rank,
                                                 const stk::mesh::PartVector& parts)
{
  crsGraph_->buildConnectedNodeGraph(rank, parts);
}

void TpetraLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildEdgeToNodeGraph(parts);
}

void TpetraLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildFaceToNodeGraph(parts);
}

void TpetraLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildElemToNodeGraph(parts);
}

void TpetraLinearSystem::buildReducedElemToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildReducedElemToNodeGraph(parts);
}

void TpetraLinearSystem::buildFaceElemToNodeGraph(const stk::mesh::PartVector & )
{
  beginLinearSystemConstruction();
  // crsGraph_->buildFaceElemToNodeGraph(parts);
}

void TpetraLinearSystem::buildNonConformalNodeGraph(const stk::mesh::PartVector &)
{
  beginLinearSystemConstruction();
  // crsGraph_->buildNonConformalNodeGraph(parts);
}

void TpetraLinearSystem::buildOversetNodeGraph(const stk::mesh::PartVector &)
{
  beginLinearSystemConstruction();
  // crsGraph_->buildOversetNodeGraph(parts);
}

void TpetraLinearSystem::copy_stk_to_tpetra(const stk::mesh::FieldBase * stkField,
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

    ThrowRequireMsg(numVectors == fieldSize, "TpetraLinearSystem::copy_stk_to_tpetra");

    const stk::mesh::Bucket::size_type length = b.size();

    const double * stkFieldPtr = (double*)stk::mesh::field_data(*stkField, b);

    for (stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k )
    {
      const stk::mesh::Entity node = b[k];

      int status = getDofStatus(node);
      if ((status & DS_SkippedDOF) || (status & DS_SharedNotOwnedDOF))
        continue;

      const stk::mesh::EntityId nodeTpetGID = *stk::mesh::field_data(*realm_.tpetGlobalId_, node);
      ThrowRequireMsg(nodeTpetGID != 0 && nodeTpetGID != std::numeric_limits<LinSys::GlobalOrdinal>::max()
                      , " in copy_stk_to_tpetra ");
      for(int d=0; d < fieldSize; ++d)
      {
        const size_t stkIndex = k*fieldSize + d;
        tpetraField->replaceGlobalValue(nodeTpetGID, d, stkFieldPtr[stkIndex]);
      }
    }
  }
}

void TpetraLinearSystem::finalizeLinearSystem()
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

  ownedRhs_ = Teuchos::rcp(new LinSys::MultiVector(crsGraph_->getOwnedRowsMap(), 1));
  sharedNotOwnedRhs_ = Teuchos::rcp(new LinSys::MultiVector(crsGraph_->getSharedNotOwnedRowsMap(), 1));

  ownedLocalRhs_ = ownedRhs_->getLocalView<sierra::nalu::DeviceSpace>();
  sharedNotOwnedLocalRhs_ = sharedNotOwnedRhs_->getLocalView<sierra::nalu::DeviceSpace>();

  sln_ = Teuchos::rcp(new LinSys::MultiVector(crsGraph_->getOwnedRowsMap(), 1));

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

void TpetraLinearSystem::zeroSystem()
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

template<typename RowViewType>
KOKKOS_FUNCTION
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
KOKKOS_FUNCTION
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

template<typename MatrixType,
         typename RhsType,
         typename EntityArrayType,
         typename ShmemView1DType,
         typename ShmemView2DType,
         typename ShmemIntView1DType,
         typename EntityLIDType>
KOKKOS_FUNCTION
void sum_into(
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
  constexpr bool forceAtomic = !std::is_same<sierra::nalu::DeviceSpace, Kokkos::Serial>::value;

  const int n_obj = numEntities;
  const int numRows = n_obj * numDof;

  for(int i = 0; i < n_obj; i++) {
    const stk::mesh::Entity entity = entities[i];
    const LocalOrdinal localOffset = entityToColLID[entity.local_offset()];
    for(size_t d=0; d < numDof; ++d) {
      size_t lid = i*numDof + d;
      localIds[lid] = localOffset + d;
    }
  }

  for (int i = 0; i < numRows; ++i) {
    sortPermutation[i] = i;
  }
  Tpetra::Details::shellSortKeysAndValues(localIds.data(), sortPermutation.data(), numRows);

  for (int r = 0; r < numRows; ++r) {
    int i = sortPermutation[r]/numDof;
    LocalOrdinal rowLid = entityToLID[entities[i].local_offset()];
    rowLid += sortPermutation[r]%numDof;
    const LocalOrdinal cur_perm_index = sortPermutation[r];
    const double* const cur_lhs = &lhs(cur_perm_index, 0);
    const double cur_rhs = rhs[cur_perm_index];
//    ThrowAssertMsg(std::isfinite(cur_rhs), "Inf or NAN rhs");

    if(rowLid < maxOwnedRowId) {
      sum_into_row(ownedLocalMatrix.row(rowLid), n_obj, numDof, localIds.data(), sortPermutation.data(), cur_lhs);
      if (forceAtomic) {
        Kokkos::atomic_add(&ownedLocalRhs(rowLid,0), cur_rhs);
      }
      else {
        ownedLocalRhs(rowLid,0) += cur_rhs;
      }
    }
    else if (rowLid < maxSharedNotOwnedRowId) {
      LocalOrdinal actualLocalId = rowLid - maxOwnedRowId;
      sum_into_row(sharedNotOwnedLocalMatrix.row(actualLocalId), n_obj, numDof,
        localIds.data(), sortPermutation.data(), cur_lhs);

      if (forceAtomic) {
        Kokkos::atomic_add(&sharedNotOwnedLocalRhs(actualLocalId,0), cur_rhs);
      }
      else {
        sharedNotOwnedLocalRhs(actualLocalId,0) += cur_rhs;
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

sierra::nalu::CoeffApplier* TpetraLinearSystem::get_coeff_applier()
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
void TpetraLinearSystem::TpetraLinSysCoeffApplier::resetRows(unsigned numNodes,
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
void
TpetraLinearSystem::TpetraLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const ngp::Mesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  sum_into(
      ownedLocalMatrix_, sharedNotOwnedLocalMatrix_,
      ownedLocalRhs_, sharedNotOwnedLocalRhs_,
      numEntities, entities,
      rhs, lhs,
      localIds, sortPermutation,
      entityToLID_, entityToColLID_,
      maxOwnedRowId_, maxSharedNotOwnedRowId_,
      numDof_);
}

void TpetraLinearSystem::TpetraLinSysCoeffApplier::free_device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier* TpetraLinearSystem::TpetraLinSysCoeffApplier::device_pointer()
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

void TpetraLinearSystem::sumInto(unsigned numEntities,
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

  sum_into(
      ownedLocalMatrix_, sharedNotOwnedLocalMatrix_,
      ownedLocalRhs_, sharedNotOwnedLocalRhs_,
      numEntities, entities,
      rhs, lhs,
      localIds, sortPermutation,
      entityToLID_, entityToColLID_,
      maxOwnedRowId_, maxSharedNotOwnedRowId_,
      numDof_);
}

void TpetraLinearSystem::sumInto(const std::vector<stk::mesh::Entity> & entities,
                                 std::vector<int> &scratchIds,
                                 std::vector<double> & /* scratchVals */,
                                 const std::vector<double> & rhs,
                                 const std::vector<double> & lhs,
                                 const char * /* trace_tag */)
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
    ThrowRequireMsg(localOffset != -1 , "sumInto bad lid #2 ");
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

template<typename RowViewType>
KOKKOS_FUNCTION
void adjust_lhs_row(
  RowViewType row_view,
  const int localRowId,
  const double diagonalValue)
{
  const LocalOrdinal rowLength = row_view.length;
  for(LocalOrdinal i=0; i<rowLength; ++i) {
    if (row_view.colidx(i) == localRowId) {
      row_view.value(i) = diagonalValue;
    }
    else {
      row_view.value(i) = 0.0;
    }
  }
}

void TpetraLinearSystem::applyDirichletBCs(stk::mesh::FieldBase * solutionField,
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

  using Traits = nalu_ngp::NGPMeshTraits<>;
  using MeshIndex = typename Traits::MeshIndex;

  ngp::Mesh ngpMesh = realm_.ngp_mesh();
  NGPDoubleFieldType ngpSolutionField = realm_.ngp_field_manager().get_field<double>(solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField = realm_.ngp_field_manager().get_field<double>(bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_device();
  ngpBCValuesField.sync_to_device();

  auto entityToLID = entityToLID_;
  const int maxOwnedRowId = maxOwnedRowId_;
  const int maxSharedNotOwnedRowId = maxSharedNotOwnedRowId_;
  auto ownedLocalMatrix = ownedLocalMatrix_;
  auto sharedNotOwnedLocalMatrix = sharedNotOwnedLocalMatrix_;
  auto ownedLocalRhs = ownedLocalRhs_;
  auto sharedNotOwnedLocalRhs = sharedNotOwnedLocalRhs_;

  // Suppress unused variable warning on non-debug builds
  (void) maxSharedNotOwnedRowId;

  nalu_ngp::run_entity_algorithm(
    "TpetraLinSys::applyDirichletBCs", ngpMesh, stk::topology::NODE_RANK, selector,
    KOKKOS_LAMBDA(const MeshIndex& meshIdx)
    {
      stk::mesh::Entity entity = (*meshIdx.bucket)[meshIdx.bucketOrd];
      const LocalOrdinal localIdOffset = entityToLID[entity.local_offset()];
      const bool useOwned = localIdOffset < maxOwnedRowId;
      const LinSys::LocalMatrix& local_matrix = useOwned ? ownedLocalMatrix : sharedNotOwnedLocalMatrix;
      const LinSys::LocalVector& localRhs = useOwned ? ownedLocalRhs : sharedNotOwnedLocalRhs;
      const double diagonalValue = useOwned ? 1.0 : 0.0;

      for(unsigned d=beginPos; d < endPos; ++d) {
        const LocalOrdinal localId = localIdOffset + d;
        const LocalOrdinal actualLocalId = useOwned ? localId : localId - maxOwnedRowId;

        NGP_ThrowAssert(localId <= maxSharedNotOwnedRowId);

        adjust_lhs_row(local_matrix.row(actualLocalId), actualLocalId, diagonalValue);

        // Replace the RHS residual with (desired - actual)
        const double bc_residual = useOwned ? (ngpBCValuesField.get(meshIdx, d) - ngpSolutionField.get(meshIdx, d)) : 0.0;
        localRhs(actualLocalId,0) = bc_residual;
      }
    }
  );

  adbc_time += NaluEnv::self().nalu_time();
}

void TpetraLinearSystem::resetRows(const std::vector<stk::mesh::Entity>& nodeList,
                                   const unsigned beginPos,
                                   const unsigned endPos,
                                   const double diag_value,
                                   const double rhs_residual)
{
  resetRows(nodeList.size(), nodeList.data(), beginPos, endPos, diag_value, rhs_residual);
}

void TpetraLinearSystem::resetRows(
    unsigned numNodes,
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

void TpetraLinearSystem::loadComplete()
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

int TpetraLinearSystem::solve(stk::mesh::FieldBase * linearSolutionField)
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

void TpetraLinearSystem::checkForNaN(bool useOwned)
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

  Teuchos::ArrayRCP<const Scalar> rhs_data = rhs->getData(0);
  n = rhs_data.size();
  for(size_t i=0; i<n; ++i) {
    if (rhs_data[i] != rhs_data[i]) {
      std::cerr << "rhs NaN: " << i << std::endl;
      throw std::runtime_error("bad rhs");
    }
  }
}

bool TpetraLinearSystem::checkForZeroRow(bool useOwned, bool doThrow, bool doPrint)
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
      stk::mesh::EntityId nid = (gid - 1) / numDof_ + 1;
      stk::mesh::Entity node = bulkData.get_entity(stk::topology::NODE_RANK, nid);
      stk::mesh::EntityId naluGlobalId;
      if (bulkData.is_valid(node)) naluGlobalId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);

      int idof = (gid - 1) % numDof_;
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

void TpetraLinearSystem::writeToFile(const char * base_filename, bool useOwned)
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

void TpetraLinearSystem::printInfo(bool useOwned)
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

void TpetraLinearSystem::writeSolutionToFile(const char * base_filename, bool useOwned)
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

void TpetraLinearSystem::copy_tpetra_to_stk(
  const Teuchos::RCP<LinSys::MultiVector> tpetraField,
  stk::mesh::FieldBase * stkField)
{
  using Traits    = nalu_ngp::NGPMeshTraits<>;
  using MeshIndex = typename Traits::MeshIndex;

  const stk::mesh::MetaData & metaData = realm_.meta_data();

  ThrowAssert(!tpetraField.is_null());
  ThrowAssert(stkField);
  const auto deviceVector = tpetraField->getLocalView<sierra::nalu::DeviceSpace>();

  const int maxOwnedRowId = maxOwnedRowId_;
  const unsigned numDof = numDof_;
  auto entityToLID = entityToLID_;

  const stk::mesh::Selector selector = stk::mesh::selectField(*stkField)
    & metaData.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  NGPDoubleFieldType ngpField = realm_.ngp_field_manager().get_field<double>(stkField->mesh_meta_data_ordinal());

  ngp::Mesh ngpMesh = realm_.ngp_mesh();

  nalu_ngp::run_entity_algorithm(
    "TpetraLinSys::copy_tpetra_to_stk",
    ngpMesh, stk::topology::NODE_RANK, selector,
  KOKKOS_LAMBDA(const MeshIndex& meshIdx)
  {
      stk::mesh::Entity node = (*meshIdx.bucket)[meshIdx.bucketOrd];
      const LocalOrdinal localIdOffset = entityToLID[node.local_offset()];
      for(unsigned d=0; d < numDof; ++d) {
        const LocalOrdinal localId = localIdOffset + d;
        NGP_ThrowRequire(localId < maxOwnedRowId);
  
        ngpField.get(meshIdx, d) = deviceVector(localId,0);
      }
  });

  ngpField.modify_on_device();
}

  Teuchos::RCP<LinSys::Graph>  TpetraLinearSystem::getOwnedGraph() { return crsGraph_->getOwnedGraph(); }
  Teuchos::RCP<LinSys::Matrix> TpetraLinearSystem::getOwnedMatrix() { return ownedMatrix_; }
  Teuchos::RCP<LinSys::MultiVector> TpetraLinearSystem::getOwnedRhs() { return ownedRhs_; }

} // namespace nalu
} // namespace Sierra
