// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "HypreUVWLinearSystem.h"

namespace sierra {
namespace nalu {

HypreUVWLinearSystem::HypreUVWLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver)
  : HypreLinearSystem(realm, 1, eqSys, linearSolver),
    rhs_(numDof, nullptr),
    sln_(numDof, nullptr),
    nDim_(numDof)
{
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  output_ = fopen(oname_, "at");
  fprintf(
    output_, "rank_=%d EqnName=%s : %s %s %d\n", rank_, name_.c_str(), __FILE__,
    __FUNCTION__, __LINE__);
  fclose(output_);
#endif
}

HypreUVWLinearSystem::~HypreUVWLinearSystem()
{
  if (hypreMatrixVectorsCreated_) {
    HYPRE_IJMatrixDestroy(mat_);

    for (unsigned i = 0; i < nDim_; ++i) {
      HYPRE_IJVectorDestroy(rhs_[i]);
      HYPRE_IJVectorDestroy(sln_[i]);
    }
  }
  hypreMatrixVectorsCreated_ = false;
}

void
HypreUVWLinearSystem::finalizeLinearSystem()
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  ThrowRequire(inConstruction_);
  inConstruction_ = false;

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  size_t used1 = 0, free1 = 0;
  stk::get_gpu_memory_info(used1, free1);
#endif

  /**********************************************************************************/
  /* Build the coeff applier ... host data structure for building the linear
   * system */
  if (!hostCoeffApplier)
    hostCoeffApplier.reset(
      new HypreUVWLinSysCoeffApplier(1, nDim_, iLower_, iUpper_));

  /* make the periodic node maps */
  HypreUVWLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreUVWLinSysCoeffApplier*>(hostCoeffApplier.get());

  hcApplier->ngpMesh_ = realm_.ngp_mesh();
  hcApplier->ngpHypreGlobalId_ =
    realm_.ngp_field_manager().get_field<HypreIntType>(
      realm_.hypreGlobalId_->mesh_meta_data_ordinal());

  /* create these mappings */
  buildCoeffApplierPeriodicNodeToHIDMapping();

  /* fill the various device data structures need in device coeff applier */
  buildCoeffApplierDeviceDataStructures();

  /* compute the exact row sizes by reducing row counts at row indices across
   * all ranks */
  computeRowSizes();

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  size_t used2 = 0, free2 = 0;
  stk::get_gpu_memory_info(used2, free2);
  size_t total = used2 + free2;
  output_ = fopen(oname_, "at");
  fprintf(
    output_,
    "rank_=%d EqnName=%s : %s %s %d : usedMem before=%1.5g, usedMem "
    "after=%1.5g, total=%1.5g\n",
    rank_, name_.c_str(), __FILE__, __FUNCTION__, __LINE__, used1 / 1.e9,
    used2 / 1.e9, total / 1.e9);
  fclose(output_);
#endif

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  finalizeLinearSystemTimer_.push_back(msec);
#endif
}

void
HypreUVWLinearSystem::hypreIJVectorSetAddToValues()
{
  HypreUVWLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreUVWLinSysCoeffApplier*>(hostCoeffApplier.get());

  auto num_rows_owned = hcApplier->num_rows_owned_;
  auto num_rows_shared = hcApplier->num_rows_shared_;

  HypreDirectSolver* solver =
    reinterpret_cast<HypreDirectSolver*>(linearSolver_);
  HypreLinearSolverConfig* config =
    reinterpret_cast<HypreLinearSolverConfig*>(solver->getConfig());

  for (unsigned i = 0; i < nDim_; ++i) {
    if (config->simpleHypreMatrixAssemble()) {
#if 0
      /* set the key hypre parameters */
      HYPRE_IJVectorSetMaxOnProcElmts(rhs_[i], num_rows_owned);
      HYPRE_IJVectorSetOffProcSendElmts(rhs_[i], offProcRhsToSend_);
      HYPRE_IJVectorSetOffProcRecvElmts(rhs_[i], offProcRhsToRecv_);
#endif
    }

    if (config->getWritePreassemblyMatrixFiles()) {
      MPI_Barrier(realm_.bulk_data().parallel());

      char rank_str[8];
      sprintf(rank_str, "%05d", rank_);
      std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
      const std::string rhsFileRows = eqSysName_ + std::to_string(i) + ".IJV." +
                                      writeCounter + ".rhs." +
                                      std::string(rank_str) + ".preassem.i";
      const std::string rhsFileVals = eqSysName_ + std::to_string(i) + ".IJV." +
                                      writeCounter + ".rhs." +
                                      std::string(rank_str) + ".preassem.v";
      const std::string rhsFileMeta = eqSysName_ + std::to_string(i) + ".IJV." +
                                      writeCounter + ".rhs." +
                                      std::string(rank_str) + ".preassem.meta";

      FILE* fid = fopen(rhsFileRows.c_str(), "wb");
      fwrite(
        rhs_rows_host_.data(), sizeof(HypreIntType),
        num_rows_owned + num_rows_shared, fid);
      fclose(fid);

      DoubleView2DHost temp(
        "temp", hcApplier->rhs_dev_.extent(0), hcApplier->nDim_);
      Kokkos::deep_copy(temp, hcApplier->rhs_dev_);
      fid = fopen(rhsFileVals.c_str(), "wb");
      fwrite(
        temp.data() + i * rhs_rows_dev_.extent(0), sizeof(double),
        num_rows_owned + num_rows_shared, fid);
      fclose(fid);

      fid = fopen(rhsFileMeta.c_str(), "wb");
      HypreIntType meta[3] = {
        num_rows_owned, num_rows_shared, (HypreIntType)rhs_rows_dev_.extent(0)};
      fwrite(meta, sizeof(HypreIntType), 3, fid);
      fclose(fid);

      MPI_Barrier(realm_.bulk_data().parallel());
    }

    if (num_rows_owned) {
      /* Set the owned part */
      HYPRE_IJVectorSetValues(
        rhs_[i], num_rows_owned,
        rhs_rows_dev_.data() + i * rhs_rows_dev_.extent(0),
        hcApplier->rhs_dev_.data() + i * rhs_rows_dev_.extent(0));
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
      double* ptr = hcApplier->rhs_dev_.data() + i * rhs_rows_dev_.extent(0);
      scanBufferForBadValues(
        ptr, num_rows_owned, __FILE__, __FUNCTION__, __LINE__, "Owned RHS");
#endif
    }

    if (num_rows_shared) {
      /* Add the shared part */
      HYPRE_IJVectorAddToValues(
        rhs_[i], num_rows_shared,
        rhs_rows_dev_.data() + i * rhs_rows_dev_.extent(0) + num_rows_owned,
        hcApplier->rhs_dev_.data() + i * rhs_rows_dev_.extent(0) +
          num_rows_owned);
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
      double* ptr = hcApplier->rhs_dev_.data() + i * rhs_rows_dev_.extent(0) +
                    num_rows_owned;
      scanBufferForBadValues(
        ptr, num_rows_shared, __FILE__, __FUNCTION__, __LINE__, "Shared RHS");
#endif
    }
  }
}

void
HypreUVWLinearSystem::loadComplete()
{
  HypreUVWLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreUVWLinSysCoeffApplier*>(hostCoeffApplier.get());

  /* finish assembly for the coupled overset case */
  finishCoupledOversetAssembly();

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  /* Matrix */
  hypreIJMatrixSetAddToValues();

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the stop time */
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  hypreMatAssemblyTimer_.push_back(msec);
  gettimeofday(&_start, NULL);
#endif

  /* Rhs */
  hypreIJVectorSetAddToValues();

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the stop time */
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
         1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  hypreRhsAssemblyTimer_.push_back(msec);
#endif

  /* Reset after assembly */
  hcApplier->reinitialize_ = true;

  /* call IJMatrix/IJVectorAssemble */
  loadCompleteSolver();
}

void
HypreUVWLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  hypre_CSRMatrix* diag =
    hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix*)hypre_IJMatrixObject(mat_));
  hypre_CSRMatrix* offd =
    hypre_ParCSRMatrixOffd((hypre_ParCSRMatrix*)hypre_IJMatrixObject(mat_));
  HYPRE_Int nnz_diag = hypre_CSRMatrixNumNonzeros(diag);
  HYPRE_Int nnz_offd = hypre_CSRMatrixNumNonzeros(offd);
  double* ptr_diag = hypre_CSRMatrixData(diag);
  double* ptr_offd = hypre_CSRMatrixData(offd);
  scanBufferForBadValues(
    ptr_diag, nnz_diag, __FILE__, __FUNCTION__, __LINE__, "Diag Matrix");
  scanBufferForBadValues(
    ptr_offd, nnz_offd, __FILE__, __FUNCTION__, __LINE__, "Offd Matrix");
  output_ = fopen(oname_, "at");
  fprintf(
    output_,
    "rank=%d : diag num_rows=%d, num_cols=%d, offd num_rows=%d, num_cols=%d\n",
    rank_, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag),
    hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd));
  fclose(output_);
#endif

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  hypreMatAssemblyTimer_.back() += msec;
  gettimeofday(&_start, NULL);
#endif

  for (unsigned i = 0; i < nDim_; ++i) {
    HYPRE_IJVectorAssemble(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));

    HYPRE_IJVectorAssemble(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
    double* rhs_data = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(rhs_[i])));
    double* sln_data = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(sln_[i])));
    scanBufferForBadValues(
      rhs_data, numRows_, __FILE__, __FUNCTION__, __LINE__, "RHS");
    scanBufferForBadValues(
      sln_data, numRows_, __FILE__, __FUNCTION__, __LINE__, "SLN");
#endif
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
         1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  hypreRhsAssemblyTimer_.back() += msec;
#endif

  solver->comm_ = realm_.bulk_data().parallel();

  HypreLinearSolverConfig* config =
    reinterpret_cast<HypreLinearSolverConfig*>(solver->getConfig());
  if (config->dumpHypreMatrixStats() && !matrixStatsDumped_) {
    dumpMatrixStats();
    matrixStatsDumped_ = true;
  }
}

void
HypreUVWLinearSystem::zeroSystem()
{
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  MPI_Comm comm = realm_.bulk_data().parallel();

  if (hypreMatrixVectorsCreated_) {
#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
    sprintf(oname_, "debug_out_%d.txt", rank_);
    output_ = fopen(oname_, "wt");
    fprintf(
      output_, "rank_=%d EqnName=%s : %s %s %d\n", rank_, name_.c_str(),
      __FILE__, __FUNCTION__, __LINE__);
    fclose(output_);
#endif
    HYPRE_IJMatrixDestroy(mat_);
    for (unsigned i = 0; i < nDim_; ++i) {
      HYPRE_IJVectorDestroy(rhs_[i]);
      HYPRE_IJVectorDestroy(sln_[i]);
    }
    hypreMatrixVectorsCreated_ = false;
  }

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));
  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);

  for (unsigned i = 0; i < nDim_; ++i) {
    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_[i]);
    HYPRE_IJVectorSetObjectType(rhs_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));
    HYPRE_ParVectorSetConstantValues((solver->parRhsU_[i]), 0.0);

    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_[i]);
    HYPRE_IJVectorSetObjectType(sln_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));
    HYPRE_ParVectorSetConstantValues((solver->parSlnU_[i]), 0.0);
  }
  hypreMatrixVectorsCreated_ = true;
}

void
HypreUVWLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned,
  const unsigned)
{
  HypreUVWLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreUVWLinSysCoeffApplier*>(hostCoeffApplier.get());

  /* Step 1: execute the old CPU code */
  auto& meta = realm_.meta_data();

  const stk::mesh::Selector selector =
    (meta.locally_owned_part() & stk::mesh::selectUnion(parts) &
     stk::mesh::selectField(*solutionField) &
     !(realm_.get_inactive_selector()));

  NGPDoubleFieldType ngpSolutionField =
    realm_.ngp_field_manager().get_field<double>(
      solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField =
    realm_.ngp_field_manager().get_field<double>(
      bcValuesField->mesh_meta_data_ordinal());

  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  /* data from hcApplier */
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto hypreGID = hcApplier->ngpHypreGlobalId_;
  auto mat_row_start_owned = hcApplier->mat_row_start_owned_;
  auto vals = hcApplier->values_dev_;
  auto rhs_vals = hcApplier->rhs_dev_;

  auto nDim = nDim_;
  auto iLower = iLower_;

  nalu_ngp::run_entity_algorithm(
    "HypreUVWLinearSystem::applyDirichletBCs", ngpMesh,
    stk::topology::NODE_RANK, selector,
    KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
      const auto node = (*mi.bucket)[mi.bucketOrd];
      HypreIntType hid = hypreGID.get(ngpMesh, node, 0);
      unsigned matIndex = mat_row_start_owned(hid - iLower);
      vals(matIndex) = 1.0;
      for (unsigned d = 0; d < nDim; ++d) {
        rhs_vals(hid - iLower, d) = ngpBCValuesField.get(ngpMesh, node, d) -
                                    ngpSolutionField.get(ngpMesh, node, d);
      }
    });
}

int
HypreUVWLinearSystem::solve(stk::mesh::FieldBase* slnField)
{
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFile = eqSysName_ + ".IJM." + writeCounter + ".mat";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());

    for (unsigned d = 0; d < nDim_; ++d) {
      const std::string rhsFile =
        eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".rhs";
      HYPRE_IJVectorPrint(rhs_[d], rhsFile.c_str());
    }
  }

  int status = 0;
  std::vector<int> iters(nDim_, 0);
  std::vector<double> finalNorm(nDim_, 1.0);
  std::vector<double> rhsNorm(nDim_, std::numeric_limits<double>::max());

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  output_ = fopen(oname_, "at");
  fprintf(
    output_, "%s %s %d %s : rank=%d\n", __FILE__, __FUNCTION__, __LINE__,
    eqSysName_.c_str(), rank_);
#endif

  for (unsigned d = 0; d < nDim_; ++d) {
    status = solver->solve(d, iters[d], finalNorm[d], realm_.isFinalOuterIter_);
  }

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  output_ = fopen(oname_, "at");
  fprintf(
    output_, "%s %s %d %s : rank=%d\n", __FILE__, __FUNCTION__, __LINE__,
    eqSysName_.c_str(), rank_);
#endif

  copy_hypre_to_stk(slnField, rhsNorm);
  sync_field(slnField);

  /* set this after the solve calls */
  solver->set_initialize_solver_flag();

  if (solver->getConfig()->getWriteMatrixFiles()) {
    for (unsigned d = 0; d < nDim_; ++d) {
      std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
      const std::string slnFile =
        eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".sln";
      HYPRE_IJVectorPrint(sln_[d], slnFile.c_str());
    }
  }

  HypreLinearSolverConfig* config =
    reinterpret_cast<HypreLinearSolverConfig*>(solver->getConfig());
  if (
    solver->getConfig()->getWriteMatrixFiles() ||
    config->getWritePreassemblyMatrixFiles()) {
    ++eqSys_->linsysWriteCounter_;
  }

  {
    linearSolveIterations_ = 0;
    linearResidual_ = 0.0;
    nonLinearResidual_ = 0.0;
    double linres, nonlinres, scaledres, tmp, scaleFac = 0.0;

    for (unsigned d = 0; d < nDim_; ++d) {
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
          << std::setw(nameOffset) << std::right
          << eqSysName_ + "_" + vecNames_[d] << std::setw(32 - nameOffset)
          << std::right << iters[d] << std::setw(18) << std::right << linres
          << std::setw(15) << std::right << nonlinres << std::setw(14)
          << std::right << scaledres << std::endl;
      }
    }
    linearResidual_ = std::sqrt(linearResidual_);
    nonLinearResidual_ = std::sqrt(nonLinearResidual_);
    scaledNonLinearResidual_ = nonLinearResidual_ / std::sqrt(scaleFac);

    if (provideOutput_) {
      const int nameOffset = eqSysName_.length() + 8;
      NaluEnv::self().naluOutputP0()
        << std::setw(nameOffset) << std::right << eqSysName_
        << std::setw(32 - nameOffset) << std::right << linearSolveIterations_
        << std::setw(18) << std::right << linearResidual_ << std::setw(15)
        << std::right << nonLinearResidual_ << std::setw(14) << std::right
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
  const auto selector =
    stk::mesh::selectField(*stkField) & meta.locally_owned_part() &
    !(stk::mesh::selectUnion(realm_.get_slave_part_vector())) &
    !(realm_.get_inactive_selector());

  HypreUVWLinSysCoeffApplier* hcApplier =
    dynamic_cast<HypreUVWLinSysCoeffApplier*>(hostCoeffApplier.get());

  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  auto ngpField = realm_.ngp_field_manager().get_field<double>(
    stkField->mesh_meta_data_ordinal());
  auto ngpHypreGlobalId = hcApplier->ngpHypreGlobalId_;
  const auto& ngpMesh = hcApplier->ngpMesh_;
  const auto periodic_node_to_hypre_id = hcApplier->periodic_node_to_hypre_id_;

  auto iLower = iLower_;
  auto iUpper = iUpper_;
  auto nDim = nDim_;

  /******************************/
  /* Move solution to stk field */

  if (nDim == 2) {
    /* use internal hypre APIs to get directly at the pointer to the owned SLN
     * vector */
    double* sln_data0 = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(sln_[0])));
    double* sln_data1 = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(sln_[1])));

    nalu_ngp::run_entity_algorithm(
      "HypreUVWLinearSystem::copy_hypre_to_stk_3D", ngpMesh,
      stk::topology::NODE_RANK, selector,
      KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
        const auto node = (*mi.bucket)[mi.bucketOrd];
        HypreIntType hid;
        if (periodic_node_to_hypre_id.exists(node.local_offset()))
          hid = periodic_node_to_hypre_id.value_at(
            periodic_node_to_hypre_id.find(node.local_offset()));
        else
          hid = ngpHypreGlobalId.get(ngpMesh, node, 0);

        if (hid >= iLower && hid <= iUpper) {
          ngpField.get(mi, 0) = sln_data0[hid - iLower];
          ngpField.get(mi, 1) = sln_data1[hid - iLower];
        }
      });
  } else {
    /* use internal hypre APIs to get directly at the pointer to the owned SLN
     * vector */
    double* sln_data0 = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(sln_[0])));
    double* sln_data1 = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(sln_[1])));
    double* sln_data2 = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(sln_[2])));

    nalu_ngp::run_entity_algorithm(
      "HypreUVWLinearSystem::copy_hypre_to_stk_3D", ngpMesh,
      stk::topology::NODE_RANK, selector,
      KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
        const auto node = (*mi.bucket)[mi.bucketOrd];
        HypreIntType hid;
        if (periodic_node_to_hypre_id.exists(node.local_offset()))
          hid = periodic_node_to_hypre_id.value_at(
            periodic_node_to_hypre_id.find(node.local_offset()));
        else
          hid = ngpHypreGlobalId.get(ngpMesh, node, 0);

        if (hid >= iLower && hid <= iUpper) {
          ngpField.get(mi, 0) = sln_data0[hid - iLower];
          ngpField.get(mi, 1) = sln_data1[hid - iLower];
          ngpField.get(mi, 2) = sln_data2[hid - iLower];
        }
      });
  }
  ngpField.modify_on_device();

  /********************/
  /* Compute RHS norm */
  std::vector<double> rhsnorm(nDim);
  std::fill(rhsnorm.begin(), rhsnorm.end(), 0);
#if 1
  /* Use Hypre internal APIs */
  for (unsigned d = 0; d < nDim; ++d)
    rhsNorm[d] = hypre_ParVectorInnerProd(
      (hypre_ParVector*)hypre_IJVectorObject(rhs_[d]),
      (hypre_ParVector*)hypre_IJVectorObject(rhs_[d]));
#else

  for (unsigned d = 0; d < nDim; ++d) {
    double* rhs_data = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(rhs_[d])));
    Kokkos::parallel_reduce(
      "HypreUVWLinearSystem::Reduction", N,
      KOKKOS_LAMBDA(const int i, double& update) {
        double t = rhs_data[i];
        update += t * t;
      },
      rhsnorm[d]);
  }

  stk::all_reduce_sum(bulk.parallel(), rhsnorm.data(), rhsNorm.data(), nDim);
#endif

  for (unsigned i = 0; i < nDim; ++i)
    rhsNorm[i] = std::sqrt(rhsNorm[i]);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  for (unsigned d = 0; d < nDim; ++d) {
    double* rhs_data = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(rhs_[d])));
    double* sln_data = hypre_VectorData(hypre_ParVectorLocalVector(
      (hypre_ParVector*)hypre_IJVectorObject(sln_[d])));

    scanBufferForBadValues(
      rhs_data, numRows_, __FILE__, __FUNCTION__, __LINE__, "RHS");
    scanBufferForBadValues(
      sln_data, numRows_, __FILE__, __FUNCTION__, __LINE__, "SLN");
  }
#endif
}

sierra::nalu::CoeffApplier*
HypreUVWLinearSystem::get_coeff_applier()
{
  /* call this before getting the device coeff applier
     Do NOT move this!
   */
  resetCoeffApplierData();
  return hostCoeffApplier->device_pointer();
}

/********************************************************************************************************/
/*                     Beginning of HypreUVWLinSysCoeffApplier implementations
 */
/********************************************************************************************************/

HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::HypreUVWLinSysCoeffApplier(
  unsigned numDof, unsigned nDim, HypreIntType iLower, HypreIntType iUpper)
  : HypreLinSysCoeffApplier(numDof, nDim, iLower, iUpper)
{
}

KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::sum_into(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const HypreIntType& iLower,
  const HypreIntType& iUpper,
  unsigned nDim,
  HypreIntType memShift)
{

  for (unsigned i = 0; i < numEntities; ++i) {
    auto node = entities[i];
    if (periodic_node_to_hypre_id_.exists(node.local_offset()))
      localIds[i] = periodic_node_to_hypre_id_.value_at(
        periodic_node_to_hypre_id_.find(node.local_offset()));
    else
      localIds[i] = ngpHypreGlobalId_.get(ngpMesh_, node, 0);
    sortPermutation[i] = i * nDim;
  }

  // sort the local ids
  sort(localIds, sortPermutation, numEntities);

  for (unsigned i = 0; i < numEntities; ++i) {
    int ix = sortPermutation[i];
    HypreIntType hid = localIds[i];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid))
        continue;
    }

    if (hid >= iLower && hid <= iUpper) {
      HypreIntType index = hid - iLower;
      unsigned matIndex = mat_row_start_owned_ra_(index);
      for (unsigned k = 0; k < numEntities; ++k) {
        /* search sorted list from where we left off */
        HypreIntType col = localIds[k];
        while (cols_dev_ra_(matIndex) < col)
          matIndex++;
        /* write the matrix element */
        Kokkos::atomic_add(&values_dev_(matIndex), lhs(ix, sortPermutation[k]));
        matIndex++;
      }
      for (unsigned d = 0; d < nDim; ++d) {
        int ir = ix + d;
        Kokkos::atomic_add(&rhs_dev_(index, d), rhs[ir]);
      }
    } else {
      if (!map_shared_.exists(hid))
        continue;
      unsigned index = map_shared_.value_at(map_shared_.find(hid));
      unsigned matIndex = mat_row_start_shared_ra_(index) + memShift;
      for (unsigned k = 0; k < numEntities; ++k) {
        /* search sorted list from where we left off */
        HypreIntType col = localIds[k];
        while (cols_dev_ra_(matIndex) < col)
          matIndex++;
        /* write the matrix element */
        Kokkos::atomic_add(&values_dev_(matIndex), lhs(ix, sortPermutation[k]));
        matIndex++;
      }

      unsigned rhsIndex = rhs_row_start_shared_(index) + (iUpper - iLower + 1);
      for (unsigned d = 0; d < nDim; ++d) {
        int ir = ix + d;
        Kokkos::atomic_add(&rhs_dev_(rhsIndex, d), rhs[ir]);
      }
    }
  }
}

KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  sum_into(
    numEntities, entities, localIds, sortPermutation, rhs, lhs, iLower_,
    iUpper_, nDim_, num_nonzeros_owned_);
}

KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::reset_rows(
  unsigned numNodes,
  const stk::mesh::Entity* nodeList,
  const double diag_value,
  const double rhs_residual,
  const HypreIntType iLower,
  const HypreIntType iUpper,
  const unsigned nDim,
  HypreIntType memShift)
{
  for (unsigned i = 0; i < numNodes; ++i) {
    auto node = nodeList[i];
    HypreIntType hid;
    if (periodic_node_to_hypre_id_.exists(node.local_offset()))
      hid = periodic_node_to_hypre_id_.value_at(
        periodic_node_to_hypre_id_.find(node.local_offset()));
    else
      hid = ngpHypreGlobalId_.get(ngpMesh_, node, 0);

    if (hid >= iLower && hid <= iUpper) {
      HypreIntType index = hid - iLower;
      unsigned lower = mat_row_start_owned_ra_(index);
      unsigned upper = mat_row_start_owned_ra_(index + 1);
      for (unsigned k = lower; k < upper; ++k) {
        values_dev_(k) = 0.0;
        if (cols_dev_ra_(k) == hid)
          values_dev_(k) = diag_value;
      }
      for (unsigned d = 0; d < nDim; ++d)
        rhs_dev_(index, d) = rhs_residual;

    } else {
      if (!map_shared_.exists(hid))
        continue;

      unsigned index = map_shared_.value_at(map_shared_.find(hid));
      unsigned lower = mat_row_start_shared_ra_(index) + memShift;
      unsigned upper = mat_row_start_shared_ra_(index + 1) + memShift;
      for (unsigned k = lower; k < upper; ++k) {
        values_dev_(k) = 0.0;
        if (cols_dev_ra_(k) == hid)
          values_dev_(k) = diag_value;
      }
      unsigned rhsIndex = rhs_row_start_shared_(index) + (iUpper - iLower + 1);
      for (unsigned d = 0; d < nDim; ++d)
        rhs_dev_(rhsIndex, d) = rhs_residual;
    }
  }
}

KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::resetRows(
  unsigned numNodes,
  const stk::mesh::Entity* nodeList,
  const unsigned,
  const unsigned,
  const double diag_value,
  const double rhs_residual)
{
  checkSkippedRows_() = 0;
  reset_rows(
    numNodes, nodeList, diag_value, rhs_residual, iLower_, iUpper_, nDim_,
    num_nonzeros_owned_);
}

void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::free_device_pointer()
{
#if defined(KOKKOS_ENABLE_GPU)
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier*
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::device_pointer()
{
#if defined(KOKKOS_ENABLE_GPU)
  if (devicePointer_ != nullptr) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
  devicePointer_ = sierra::nalu::create_device_expression(*this);
  return devicePointer_;
#else
  return this;
#endif
}

/*********************************************************************************************************/
/*                           End of HypreUVWLinSysCoeffApplier implementations
 */
/*********************************************************************************************************/

void
HypreUVWLinearSystem::buildNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned =
    metaData.locally_owned_part() & stk::mesh::selectUnion(parts) &
    !(stk::mesh::selectUnion(realm_.get_slave_part_vector())) &
    !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_owned);

  std::vector<HypreIntType> hids(1);

  for (size_t ib = 0; ib < buckets.size(); ++ib) {
    const stk::mesh::Bucket& b = *buckets[ib];
    for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

      stk::mesh::Entity node = b[k];
      hids[0] = get_entity_hypre_id(node);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(1, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreUVWLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(realm_.meta_data().side_rank(), s_owned);

  std::vector<HypreIntType> hids(0);

  for (size_t ib = 0; ib < buckets.size(); ++ib) {
    const stk::mesh::Bucket& b = *buckets[ib];

    auto numNodes = b.topology().num_nodes();
    hids.resize(numNodes);

    for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

      stk::mesh::Entity const* nodes = b.begin_nodes(k);

      /* save the hypre ids */
      for (unsigned i = 0; i < numNodes; ++i)
        hids[i] = get_entity_hypre_id(nodes[i]);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(numNodes, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildFaceToNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreUVWLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::EDGE_RANK, s_owned);

  std::vector<HypreIntType> hids(0);

  for (size_t ib = 0; ib < buckets.size(); ++ib) {
    const stk::mesh::Bucket& b = *buckets[ib];

    auto numNodes = b.topology().num_nodes();
    hids.resize(numNodes);

    for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

      stk::mesh::Entity const* nodes = b.begin_nodes(k);

      /* save the hypre ids */
      for (unsigned i = 0; i < numNodes; ++i)
        hids[i] = get_entity_hypre_id(nodes[i]);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(numNodes, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildEdgeToNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreUVWLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets(stk::topology::ELEM_RANK, s_owned);

  std::vector<HypreIntType> hids(0);

  for (size_t ib = 0; ib < buckets.size(); ++ib) {
    const stk::mesh::Bucket& b = *buckets[ib];

    auto numNodes = b.topology().num_nodes();
    hids.resize(numNodes);

    for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {

      stk::mesh::Entity const* nodes = b.begin_nodes(k);

      /* save the hypre ids */
      for (unsigned i = 0; i < numNodes; ++i)
        hids[i] = get_entity_hypre_id(nodes[i]);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(numNodes, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildElemToNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreUVWLinearSystem::buildFaceElemToNodeGraph(
  const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  stk::mesh::MetaData& metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part() &
                                      stk::mesh::selectUnion(parts) &
                                      !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets(metaData.side_rank(), s_owned);

  std::vector<HypreIntType> hids(0);

  for (size_t ib = 0; ib < face_buckets.size(); ++ib) {
    const stk::mesh::Bucket& b = *face_buckets[ib];
    for (stk::mesh::Bucket::size_type k = 0; k < b.size(); ++k) {
      const stk::mesh::Entity face = b[k];

      // extract the connected element to this exposed face; should be single in
      // size!
      const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
      ThrowAssert(bulkData.num_elements(face) == 1);

      // get connected element and nodal relations
      stk::mesh::Entity element = face_elem_rels[0];
      const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);

      // figure out the global dof ids for each dof on each node
      const unsigned numNodes = (unsigned)bulkData.num_nodes(element);
      hids.resize(numNodes);

      /* save the hypre ids */
      for (unsigned i = 0; i < numNodes; ++i)
        hids[i] = get_entity_hypre_id(elem_nodes[i]);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(numNodes, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildFaceElemToNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreUVWLinearSystem::buildReducedElemToNodeGraph(const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreUVWLinearSystem::buildNonConformalNodeGraph(const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  // Grab nodes regardless of whether they are owned or shared
  const stk::mesh::Selector sel = stk::mesh::selectUnion(parts);
  const auto& bkts = realm_.get_buckets(stk::topology::NODE_RANK, sel);

  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
      skippedRows_.insert(hid);
      if (hid >= iLower_ && hid <= iUpper_) {
        HypreIntType lid = hid - iLower_;
        rowCountOwned_[lid]++;
        columnsOwned_[lid].push_back(hid);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildDirichletNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  for (const auto& node : nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);
    skippedRows_.insert(hid);
    if (hid >= iLower_ && hid <= iUpper_) {
      HypreIntType lid = hid - iLower_;
      rowCountOwned_[lid]++;
      columnsOwned_[lid].push_back(hid);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildDirichletNodeGraphTimer_.push_back(msec);
#endif
}

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::NgpMesh::ConnectedNodes nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  for (unsigned i = 0; i < nodeList.size(); ++i) {
    HypreIntType hid = get_entity_hypre_id(nodeList[i]);
    skippedRows_.insert(hid);
    if (hid >= iLower_ && hid <= iUpper_) {
      HypreIntType lid = hid - iLower_;
      rowCountOwned_[lid]++;
      columnsOwned_[lid].push_back(hid);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 +
                1.e3 * ((double)(_stop.tv_sec - _start.tv_sec));
  buildDirichletNodeGraphTimer_.push_back(msec);
#endif
}

} // namespace nalu
} // namespace sierra
