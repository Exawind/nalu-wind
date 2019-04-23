/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "HypreUVWLinearSystem.h"
#include "HypreUVWSolver.h"
#include "Realm.h"
#include "EquationSystem.h"

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

#include <limits>
#include <vector>
#include <string>
#include <cmath>

namespace sierra {
namespace nalu {

HypreUVWLinearSystem::HypreUVWLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver
) : HypreLinearSystem(realm, 1, eqSys, linearSolver),
    rhs_(numDof, nullptr),
    sln_(numDof, nullptr),
    nDim_(numDof)
{}

HypreUVWLinearSystem::~HypreUVWLinearSystem()
{
  if (systemInitialized_) {
    HYPRE_IJMatrixDestroy(mat_);

    for (int i=0; i < nDim_; i++) {
      HYPRE_IJVectorDestroy(rhs_[i]);
      HYPRE_IJVectorDestroy(sln_[i]);
    }
  }
  systemInitialized_ = false;
}

void
HypreUVWLinearSystem::finalizeSolver()
{

  MPI_Comm comm = realm_.bulk_data().parallel();
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  for (int i=0; i < nDim_; i++) {
    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_[i]);
    HYPRE_IJVectorSetObjectType(rhs_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));

    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_[i]);
    HYPRE_IJVectorSetObjectType(sln_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));
  }
}

void
HypreUVWLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  for (int i=0; i < nDim_; i++) {
    HYPRE_IJVectorAssemble(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));

    HYPRE_IJVectorAssemble(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));
  }

  solver->comm_ = realm_.bulk_data().parallel();

  matrixAssembled_ = true;
}

void
HypreUVWLinearSystem::zeroSystem()
{
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  if (matrixAssembled_) {
    HYPRE_IJMatrixInitialize(mat_);
    for (int i=0; i < nDim_; i++) {
      HYPRE_IJVectorInitialize(rhs_[i]);
      HYPRE_IJVectorInitialize(sln_[i]);
    }

    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  for (int i=0; i < nDim_; i++) {
    HYPRE_ParVectorSetConstantValues((solver->parRhsU_[i]), 0.0);
    HYPRE_ParVectorSetConstantValues((solver->parSlnU_[i]), 0.0);
  }

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
HypreUVWLinearSystem::sumInto(
  unsigned numEntities,
  const stk::mesh::Entity* entities,
  const SharedMemView<const double*>& rhs,
  const SharedMemView<const double**>& lhs,
  const SharedMemView<int*>&,
  const SharedMemView<int*>&,
  const char*  /* trace_tag */)
{
  HypreIntType numRows = numEntities;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) {
    idBuffer_.resize(numRows);
    scratchRowVals_.resize(numRows);
  }

  for (size_t in=0; in < numEntities; in++) {
    idBuffer_[in] = get_entity_hypre_id(entities[in]);
  }

  for (size_t in=0; in < numEntities; in++) {
    int ix = in * nDim_;
    HypreIntType hid = idBuffer_[in];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    int offset = 0;
    for (int c=0; c < numRows; c++) {
      scratchRowVals_[c] = lhs(ix, offset);
      offset += nDim_;
    }
    HYPRE_IJMatrixAddToValues(
      mat_, 1, &numRows, &hid, &idBuffer_[0], &scratchRowVals_[0]);

    for (int d=0; d < nDim_; d++) {
      int ir = ix + d;
      HYPRE_IJVectorAddToValues(rhs_[d], 1, &hid, &rhs[ir]);
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
}

void
HypreUVWLinearSystem::sumInto(
  unsigned numEntities,
  const ngp::Mesh::ConnectedNodes& entities,
  const SharedMemView<const double*>& rhs,
  const SharedMemView<const double**>& lhs,
  const SharedMemView<int*>&,
  const SharedMemView<int*>&,
  const char*  /* trace_tag */)
{
#ifndef KOKKOS_ENABLE_CUDA
  HypreIntType numRows = numEntities;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) {
    idBuffer_.resize(numRows);
    scratchRowVals_.resize(numRows);
  }

  for (size_t in=0; in < numEntities; in++) {
    idBuffer_[in] = get_entity_hypre_id(entities[in]);
  }

  for (size_t in=0; in < numEntities; in++) {
    int ix = in * nDim_;
    HypreIntType hid = idBuffer_[in];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    int offset = 0;
    for (int c=0; c < numRows; c++) {
      scratchRowVals_[c] = lhs(ix, offset);
      offset += nDim_;
    }
    HYPRE_IJMatrixAddToValues(
      mat_, 1, &numRows, &hid, &idBuffer_[0], &scratchRowVals_[0]);

    for (int d=0; d < nDim_; d++) {
      int ir = ix + d;
      HYPRE_IJVectorAddToValues(rhs_[d], 1, &hid, &rhs[ir]);
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
#endif
}

void
HypreUVWLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& scratchVals,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char*  /* trace_tag */)
{
  const size_t n_obj = entities.size();
  HypreIntType numRows = n_obj;
  const HypreIntType bufSize = idBuffer_.size();

#ifndef NDEBUG
  size_t vecSize = numRows * nDim_;
  ThrowAssert(vecSize == rhs.size());
  ThrowAssert(vecSize*vecSize == lhs.size());
#endif
  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    idBuffer_[in] = get_entity_hypre_id(entities[in]);
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * nDim_;
    HypreIntType hid = get_entity_hypre_id(entities[in]);

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    int offset = 0;
    int ic = ix * numRows * nDim_;
    for (int c=0; c < numRows; c++) {
      scratchVals[c] = lhs[ic + offset];
      offset += nDim_;
    }
    HYPRE_IJMatrixAddToValues(
      mat_, 1, &numRows, &hid, &idBuffer_[0], &scratchVals[0]);

    for (int d = 0; d < nDim_; d++) {
      int ir = ix + d;
      HYPRE_IJVectorAddToValues(rhs_[d], 1, &hid, &rhs[ir]);
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
}

void
HypreUVWLinearSystem::applyDirichletBCs(
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

      HYPRE_IJMatrixSetValues(mat_, 1, &ncols, &hid, &hid, &diag_value);
      for (int d=0; d<nDim_; d++) {
        double bcval = bcValues[in*nDim_ + d] - solution[in*nDim_ + d];

        HYPRE_IJVectorSetValues(rhs_[d], 1, &hid, &bcval);
      }
      rowFilled_[hid - iLower_] = RS_FILLED;
    }
  }
}

int
HypreUVWLinearSystem::solve(stk::mesh::FieldBase* slnField)
{
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    const std::string matFile = eqSysName_ + ".IJM.mat";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());

    for (int d=0; d < nDim_; d++) {
      const std::string rhsFile = eqSysName_ + std::to_string(d) + ".IJV.rhs";
      HYPRE_IJVectorPrint(rhs_[d], rhsFile.c_str());
    }
  }

  int status = 0;
  std::vector<int> iters(nDim_, 0);
  std::vector<double> finalNorm(nDim_, 1.0);
  std::vector<double> rhsNorm(nDim_, std::numeric_limits<double>::max());

  for (int d=0; d < nDim_; d++) {
    status = solver->solve(d, iters[d], finalNorm[d], realm_.isFinalOuterIter_);
  }
  copy_hypre_to_stk(slnField, rhsNorm);
  sync_field(slnField);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    for (int d=0; d < nDim_; d++) {
      const std::string slnFile = eqSysName_ + std::to_string(d) + ".IJV.sln";
      HYPRE_IJVectorPrint(sln_[d], slnFile.c_str());
    }
  }

  {
    linearSolveIterations_ = 0;
    linearResidual_ = 0.0;
    nonLinearResidual_ = 0.0;
    double linres, nonlinres, scaledres, tmp, scaleFac = 0.0;

    for (int d=0; d < nDim_; d++) {
      linres = finalNorm[d] * rhsNorm[d];
      nonlinres = realm_.l2Scaling_ * rhsNorm[d];

      if (eqSys_->firstTimeStepSolve_)
        firstNLR_[d] = nonlinres;

      tmp = std::max(std::numeric_limits<double>::epsilon(), firstNLR_[d]);
      scaledres = nonlinres / tmp;
      scaleFac += tmp * tmp;

      linearResidual_ += linres * linres;
      nonLinearResidual_ += nonlinres * nonlinres;
      scaledNonLinearResidual_ += scaledres * scaledres;
      linearSolveIterations_ += iters[d];

      if (provideOutput_) {
        const int nameOffset = eqSysName_.length() + 10;

        NaluEnv::self().naluOutputP0()
          << std::setw(nameOffset) << std::right << eqSysName_+"_"+vecNames_[d]
          << std::setw(32 - nameOffset) << std::right << iters[d] << std::setw(18)
          << std::right << linres << std::setw(15) << std::right
          << nonlinres << std::setw(14) << std::right
          << scaledres << std::endl;
      }
    }
    linearResidual_ = std::sqrt(linearResidual_);
    nonLinearResidual_ = std::sqrt(nonLinearResidual_);
    scaledNonLinearResidual_ = nonLinearResidual_ / std::sqrt(scaleFac);

    if (provideOutput_) {
      const int nameOffset = eqSysName_.length() + 8;
      NaluEnv::self().naluOutputP0()
        << std::setw(nameOffset) << std::right << eqSysName_
        << std::setw(32 - nameOffset) << std::right << linearSolveIterations_ << std::setw(18)
        << std::right << linearResidual_ << std::setw(15) << std::right
        << nonLinearResidual_ << std::setw(14) << std::right
        << scaledNonLinearResidual_ << std::endl;
    }
  }

  eqSys_->firstTimeStepSolve_ = false;

  return status;
}


void
HypreUVWLinearSystem::copy_hypre_to_stk(
  stk::mesh::FieldBase* stkField, std::vector<double>& rhsNorm)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const auto sel = stk::mesh::selectField(*stkField)
    & meta.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, sel);

  std::vector<double> lclnorm(nDim_, 0.0);
  std::vector<double> gblnorm(nDim_, 0.0);
  double rhsVal = 0.0;

  for (auto b: bkts) {
    double* field = (double*) stk::mesh::field_data(*stkField, *b);
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = get_entity_hypre_id(node);

      for (int d=0; d < nDim_; d++) {
        int sid = in * nDim_ + d;
        HYPRE_IJVectorGetValues(sln_[d], 1, &hid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_[d], 1, &hid, &rhsVal);

        lclnorm[d] += rhsVal * rhsVal;
      }
    }
  }

  stk::all_reduce_sum(bulk.parallel(), lclnorm.data(), gblnorm.data(), nDim_);

  for (int d=0; d < nDim_; d++)
    rhsNorm[d] = std::sqrt(gblnorm[d]);
}

}  // nalu
}  // sierra
