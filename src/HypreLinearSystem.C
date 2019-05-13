/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "HypreLinearSystem.h"
#include "HypreDirectSolver.h"
#include "Realm.h"
#include "EquationSystem.h"
#include "LinearSolver.h"
#include "PeriodicManager.h"
#include "NonConformalManager.h"
#include "overset/OversetManager.h"
#include "overset/OversetInfo.h"


#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE.h"
#include "HYPRE_config.h"

#include <cmath>
#include <cstdint>

namespace sierra {
namespace nalu {

HypreLinearSystem::HypreLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver)
  : LinearSystem(realm, numDof, eqSys, linearSolver),
    rowFilled_(0),
    rowStatus_(0),
    idBuffer_(0)
{}

HypreLinearSystem::~HypreLinearSystem()
{
  if (systemInitialized_) {
    HYPRE_IJMatrixDestroy(mat_);
    HYPRE_IJVectorDestroy(rhs_);
    HYPRE_IJVectorDestroy(sln_);
    systemInitialized_ = false;
  }
}

void
HypreLinearSystem::beginLinearSystemConstruction()
{
  if (inConstruction_) return;
  inConstruction_ = true;

#ifndef HYPRE_BIGINT
  // Make sure that HYPRE is compiled with 64-bit integer support when running
  // O(~1B) linear systems.
  uint64_t totalRows = (static_cast<uint64_t>(realm_.hypreNumNodes_) *
                        static_cast<uint64_t>(numDof_));
  uint64_t maxHypreSize = static_cast<uint64_t>(std::numeric_limits<HypreIntType>::max());

  if (totalRows > maxHypreSize)
    throw std::runtime_error(
      "The linear system size is greater than what HYPRE is compiled for. "
      "Please recompile with bigint support and link to Nalu");
#endif

  const int rank = realm_.bulk_data().parallel_rank();

  if (rank == 0) {
    iLower_ = realm_.hypreILower_;
  } else {
    iLower_ = realm_.hypreILower_ * numDof_ ;
  }

  iUpper_ = realm_.hypreIUpper_  * numDof_ - 1;
  // For now set column indices the same as row indices
  jLower_ = iLower_;
  jUpper_ = iUpper_;

  // The total number of rows handled by this MPI rank for Hypre
  numRows_ = (iUpper_ - iLower_ + 1);
  // Total number of global rows in the system
  maxRowID_ = realm_.hypreNumNodes_ * numDof_ - 1;

#if 0
  if (numDof_ > 0)
    std::cerr << rank << "\t" << numDof_ << "\t"
              << realm_.hypreILower_ << "\t" << realm_.hypreIUpper_ << "\t"
                << iLower_ << "\t" << iUpper_ << "\t"
                << numRows_ << "\t" << maxRowID_ << std::endl;
#endif
  // Allocate memory for the arrays used to track row types and row filled status.
  rowFilled_.resize(numRows_);
  rowStatus_.resize(numRows_);
  skippedRows_.clear();
  // All nodes start out as NORMAL; "build*NodeGraph" methods might alter the
  // row status to modify behavior of sumInto method.
  for (HypreIntType i=0; i < numRows_; i++)
    rowStatus_[i] = RT_NORMAL;

  auto& bulk = realm_.bulk_data();
  std::vector<const stk::mesh::FieldBase*> fVec{realm_.hypreGlobalId_};

  stk::mesh::copy_owned_to_shared(bulk, fVec);
  stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);

  if (realm_.oversetManager_ != nullptr &&
      realm_.oversetManager_->oversetGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.oversetManager_->oversetGhosting_, fVec);

  if (realm_.nonConformalManager_ != nullptr &&
      realm_.nonConformalManager_->nonConformalGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, fVec);

  if (realm_.periodicManager_ != nullptr &&
      realm_.periodicManager_->periodicGhosting_ != nullptr) {
    realm_.periodicManager_->parallel_communicate_field(realm_.hypreGlobalId_);
    realm_.periodicManager_->periodic_parallel_communicate_field(
      realm_.hypreGlobalId_);
  }
}

void
HypreLinearSystem::buildNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildFaceToNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildEdgeToNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildElemToNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildFaceElemToNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildReducedElemToNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildNonConformalNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreLinearSystem::buildOversetNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  // Mark all the fringe nodes as skipped so that sumInto doesn't add into these
  // rows during assembly process
  for(auto* oinfo: realm_.oversetManager_->oversetInfoVec_) {
    auto node = oinfo->orphanNode_;
    HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
    skippedRows_.insert(hid * numDof_);
  }
}

void
HypreLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::PartVector& parts)
{
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  // Grab nodes regardless of whether they are owned or shared
  const stk::mesh::Selector sel = stk::mesh::selectUnion(parts);
  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
      skippedRows_.insert(hid * numDof_);
    }
  }
}

void
HypreLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  for (const auto& node: nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);
    skippedRows_.insert(hid * numDof_);
  }
}

void
HypreLinearSystem::finalizeLinearSystem()
{
  ThrowRequire(inConstruction_);
  inConstruction_ = false;

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  finalizeSolver();

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;

  // At this stage the LHS and RHS data structures are ready for
  // sumInto/assembly.
  systemInitialized_ = true;
}

void
HypreLinearSystem::finalizeSolver()
{
  MPI_Comm comm = realm_.bulk_data().parallel();
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_);
  HYPRE_IJVectorSetObjectType(rhs_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_);
  HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));
}

void
HypreLinearSystem::loadComplete()
{
  // All algorithms have called sumInto and populated LHS/RHS. Now we are ready
  // to finalize the matrix at the HYPRE end. However, before we do that we need
  // to process unfilled rows and process them appropriately. Any row acted on
  // by sumInto method will have toggled the rowFilled_ array to RS_FILLED
  // status. Before finalizing assembly, we process rows that still have an
  // RS_UNFILLED status and set their diagonal entries to 1.0 (dummy row)
  //
  // TODO: Alternate design to eliminate dummy rows. This will require
  // load-balancing on HYPRE end.

  HypreIntType hnrows = 1;
  HypreIntType hncols = 1;
  double getval;
  double setval = 1.0;
  for (HypreIntType i=0; i < numRows_; i++) {
    if (rowFilled_[i] == RS_FILLED) continue;
    HypreIntType lid = iLower_ + i;
    HYPRE_IJMatrixGetValues(mat_, hnrows, &hncols, &lid, &lid, &getval);
    if (std::fabs(getval) < 1.0e-12)
      HYPRE_IJMatrixSetValues(mat_, hnrows, &hncols, &lid, &lid, &setval);
  }

  loadCompleteSolver();
}

void
HypreLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorAssemble(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorAssemble(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));

  solver->comm_ = realm_.bulk_data().parallel();

  // Set flag to indicate zeroSystem that the matrix must be reinitialized
  // during the next invocation.
  matrixAssembled_ = true;
}

void
HypreLinearSystem::zeroSystem()
{
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  // It is unsafe to call IJMatrixInitialize multiple times without intervening
  // call to IJMatrixAssemble. This occurs during the first outer iteration (of
  // first timestep in static application and every timestep in moving mesh
  // applications) when the data structures have been created but never used and
  // zeroSystem is called for a reset. Include a check to ensure we only
  // initialize if it was previously assembled.
  if (matrixAssembled_) {
    HYPRE_IJMatrixInitialize(mat_);
    HYPRE_IJVectorInitialize(rhs_);
    HYPRE_IJVectorInitialize(sln_);

    // Set flag to false until next invocation of IJMatrixAssemble in loadComplete
    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parRhs_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parSln_, 0.0);

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;
}

void
HypreLinearSystem::sumInto(
  unsigned numEntities,
  const stk::mesh::Entity* entities,
  const SharedMemView<const double*>& rhs,
  const SharedMemView<const double**>& lhs,
  const SharedMemView<int*>&,
  const SharedMemView<int*>&,
  const char*  /* trace_tag */)
{
  const size_t n_obj = numEntities;
  HypreIntType numRows = n_obj * numDof_;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    HypreIntType hid = get_entity_hypre_id(entities[in]);
    HypreIntType localOffset = hid * numDof_;
    for (size_t d=0; d < numDof_; d++) {
      size_t lid = in * numDof_ + d;
      idBuffer_[lid] = localOffset + d;
    }
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * numDof_;
    HypreIntType hid = idBuffer_[ix];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    for (size_t d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = idBuffer_[ir];

      const double* cur_lhs = &lhs(ir, 0);
      HYPRE_IJMatrixAddToValues(mat_, 1, &numRows, &lid,
                                &idBuffer_[0], cur_lhs);
      HYPRE_IJVectorAddToValues(rhs_, 1, &lid, &rhs[ir]);

      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
}

void
HypreLinearSystem::sumInto(
  unsigned numEntities,
  const ngp::Mesh::ConnectedNodes& entities,
  const SharedMemView<const double*>& rhs,
  const SharedMemView<const double**>& lhs,
  const SharedMemView<int*>&,
  const SharedMemView<int*>&,
  const char*  /* trace_tag */)
{
#ifndef KOKKOS_ENABLE_CUDA
  const size_t n_obj = numEntities;
  HypreIntType numRows = n_obj * numDof_;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    HypreIntType hid = get_entity_hypre_id(entities[in]);
    HypreIntType localOffset = hid * numDof_;
    for (size_t d=0; d < numDof_; d++) {
      size_t lid = in * numDof_ + d;
      idBuffer_[lid] = localOffset + d;
    }
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * numDof_;
    HypreIntType hid = idBuffer_[ix];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    for (size_t d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = idBuffer_[ir];

      const double* cur_lhs = &lhs(ir, 0);
      HYPRE_IJMatrixAddToValues(mat_, 1, &numRows, &lid,
                                &idBuffer_[0], cur_lhs);
      HYPRE_IJVectorAddToValues(rhs_, 1, &lid, &rhs[ir]);

      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
#endif
}


void
HypreLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& scratchVals,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char*  /* trace_tag */)
{
  const size_t n_obj = entities.size();
  HypreIntType numRows = n_obj * numDof_;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssert(numRows == static_cast<HypreIntType>(rhs.size()));
  ThrowAssert(numRows*numRows == static_cast<HypreIntType>(lhs.size()));

  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    HypreIntType hid = get_entity_hypre_id(entities[in]);
    HypreIntType localOffset = hid * numDof_;
    for (size_t d=0; d < numDof_; d++) {
      size_t lid = in * numDof_ + d;
      idBuffer_[lid] = localOffset + d;
    }
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * numDof_;
    HypreIntType hid = idBuffer_[ix];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    for (size_t d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = idBuffer_[ir];

      for (int c=0; c < numRows; c++)
        scratchVals[c] = lhs[ir * numRows + c];

      HYPRE_IJMatrixAddToValues(mat_, 1, &numRows, &lid,
                                &idBuffer_[0], &scratchVals[0]);
      HYPRE_IJVectorAddToValues(rhs_, 1, &lid, &rhs[ir]);
      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
}

void
HypreLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned,
  const unsigned)
{
  auto& meta = realm_.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm_.get_inactive_selector()));

  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  HypreIntType ncols = 1;
  double diag_value = 1.0;
  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);

      for (size_t d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        double bcval = bcValues[in*numDof_ + d] - solution[in*numDof_ + d];

        HYPRE_IJMatrixSetValues(mat_, 1, &ncols, &lid, &lid, &diag_value);
        HYPRE_IJVectorSetValues(rhs_, 1, &lid, &bcval);
        rowFilled_[lid - iLower_] = RS_FILLED;
      }
    }
  }
}

HypreIntType
HypreLinearSystem::get_entity_hypre_id(const stk::mesh::Entity& node)
{
  auto& bulk = realm_.bulk_data();
  const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
  const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluId);
#ifndef NDEBUG
  if (!bulk.is_valid(node))
    throw std::runtime_error("BAD STK NODE");
#endif
  HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);

#ifndef NDEBUG
  HypreIntType chk = ((hid+1) * numDof_ - 1);
  if ((hid < 0) || (chk > maxRowID_)) {
    std::cerr << bulk.parallel_rank() << "\t"
              << hid << "\t" << iLower_ << "\t" << iUpper_ << std::endl;
    throw std::runtime_error("BAD STK to hypre conversion");
  }
#endif

  return hid;
}

int
HypreLinearSystem::solve(stk::mesh::FieldBase* linearSolutionField)
{
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(
    linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    const std::string matFile = eqSysName_ + ".IJM.mat";
    const std::string rhsFile = eqSysName_ + ".IJV.rhs";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());
    HYPRE_IJVectorPrint(rhs_, rhsFile.c_str());
  }

  int iters = 0;
  double finalResidNorm = 0.0;

  // Call solve
  int status = 0;

  status = solver->solve(iters, finalResidNorm, realm_.isFinalOuterIter_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    const std::string slnFile = eqSysName_ + ".IJV.sln";
    HYPRE_IJVectorPrint(sln_, slnFile.c_str());
  }

  double norm2 = copy_hypre_to_stk(linearSolutionField);
  sync_field(linearSolutionField);

  linearSolveIterations_ = iters;
  // Hypre provides relative residuals not the final residual, so multiply by
  // the non-linear residual to obtain a final residual that is comparable to
  // what is reported by TpetraLinearSystem. Note that this assumes the initial
  // solution vector is set to 0 at the start of linear iterations.
  linearResidual_ = finalResidNorm * norm2;
  nonLinearResidual_ = realm_.l2Scaling_ * norm2;

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
      << std::right << linearResidual_ << std::setw(15) << std::right
      << nonLinearResidual_ << std::setw(14) << std::right
      << scaledNonLinearResidual_ << std::endl;
  }

  eqSys_->firstTimeStepSolve_ = false;

  return status;
}

double
HypreLinearSystem::copy_hypre_to_stk(
  stk::mesh::FieldBase* stkField)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const auto sel = stk::mesh::selectField(*stkField)
    & meta.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, sel);

  double lclnorm2 = 0.0;
  double rhsVal = 0.0;
  for (auto b: bkts) {
    double* field = (double*) stk::mesh::field_data(*stkField, *b);
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = get_entity_hypre_id(node);

      for (size_t d=0; d < numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        int sid = in * numDof_ + d;
        HYPRE_IJVectorGetValues(sln_, 1, &lid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_, 1, &lid, &rhsVal);

        lclnorm2 += rhsVal * rhsVal;
      }
    }
  }

  double gblnorm2 = 0.0;
  stk::all_reduce_sum(bulk.parallel(), &lclnorm2, &gblnorm2, 1);

  return std::sqrt(gblnorm2);
}

}  // nalu
}  // sierra
