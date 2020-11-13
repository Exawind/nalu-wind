// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "HypreUVWLinearSystem.h"
#include "HypreUVWSolver.h"
#include "NaluEnv.h"
#include "Realm.h"
#include "EquationSystem.h"

#include <utils/CreateDeviceExpression.h>

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
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

    for (unsigned i=0; i<nDim_; ++i) {
      HYPRE_IJVectorDestroy(rhs_[i]);
      HYPRE_IJVectorDestroy(sln_[i]);
    }
  }
  systemInitialized_ = false;
}



void
HypreUVWLinearSystem::finalizeLinearSystem()
{
  ThrowRequire(inConstruction_);
  inConstruction_ = false;
  
  finalizeSolver();

  /* get this field */
  auto ngpHypreGlobalId_ = realm_.ngp_field_manager().get_field<HypreIntType>(realm_.hypreGlobalId_->mesh_meta_data_ordinal());

  /* create these mappings */
  fill_periodic_node_to_hid_mapping();

  /* fill the various device data structures need in device coeff applier */
  fill_device_data_structures();

  /**********************************************************************************/
  /* Build the coeff applier ... host data structure for building the linear system */
  if (!hostCoeffApplier) {
    hostCoeffApplier.reset(new HypreUVWLinSysCoeffApplier(realm_.ngp_mesh(), ngpHypreGlobalId_, 1, nDim_,
							  globalNumRows_, rank_, iLower_, iUpper_, jLower_, jUpper_,
							  map_shared_, mat_elem_cols_owned_uvm_, mat_elem_cols_shared_uvm_,
							  mat_row_start_owned_, mat_row_start_shared_, rhs_row_start_shared_,
							  row_indices_owned_uvm_, row_indices_shared_uvm_, row_counts_owned_uvm_,
							  row_counts_shared_uvm_, periodic_bc_rows_owned_, periodic_node_to_hypre_id_,
							  skippedRowsMap_, skippedRowsMapHost_, oversetRowsMap_, oversetRowsMapHost_,
							  num_mat_overset_pts_owned_, num_rhs_overset_pts_owned_));
    deviceCoeffApplier = hostCoeffApplier->device_pointer();
  }

  // At this stage the LHS and RHS data structures are ready for
  // sumInto/assembly.
  systemInitialized_ = true;
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

  for (unsigned i=0; i<nDim_; ++i) {
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
HypreUVWLinearSystem::loadComplete()
{
  HypreUVWLinSysCoeffApplier* hcApplier = dynamic_cast<HypreUVWLinSysCoeffApplier*>(hostCoeffApplier.get());
  std::vector<HYPRE_IJVector> rhs(nDim_);
  for (unsigned i=0; i<nDim_; ++i) rhs[i] = rhs_[i];
  hcApplier->finishAssembly(mat_, rhs);
  loadCompleteSolver();
}


void
HypreUVWLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  for (unsigned i=0; i<nDim_; ++i) {
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
    for (unsigned i=0; i<nDim_; ++i) {
      HYPRE_IJVectorInitialize(rhs_[i]);
      HYPRE_IJVectorInitialize(sln_[i]);
    }

    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  for (unsigned i=0; i<nDim_; ++i) {
    HYPRE_ParVectorSetConstantValues((solver->parRhsU_[i]), 0.0);
    HYPRE_ParVectorSetConstantValues((solver->parSlnU_[i]), 0.0);
  }
}

void
HypreUVWLinearSystem::sumInto(
  unsigned,
  const stk::mesh::NgpMesh::ConnectedNodes&,
  const SharedMemView<const double*, DeviceShmem>&,
  const SharedMemView<const double**, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const char*  /* trace_tag */)
{
}

void
HypreUVWLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& /*entities */,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& /* scratchVals */,
  const std::vector<double>& /* rhs */,
  const std::vector<double>& /* lhs */,
  const char*  /* trace_tag */)
{
}

void
HypreUVWLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned,
  const unsigned)
{
  HypreUVWLinSysCoeffApplier* hcApplier = dynamic_cast<HypreUVWLinSysCoeffApplier*>(hostCoeffApplier.get());
  hcApplier->applyDirichletBCs(realm_, solutionField, bcValuesField, parts);
}

int
HypreUVWLinearSystem::solve(stk::mesh::FieldBase* slnField)
{
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFile = eqSysName_ + ".IJM." + writeCounter + ".mat";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());

    for (unsigned d=0; d<nDim_; ++d) {
      const std::string rhsFile =
        eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".rhs";
      HYPRE_IJVectorPrint(rhs_[d], rhsFile.c_str());
    }
  }

  int status = 0;
  std::vector<int> iters(nDim_, 0);
  std::vector<double> finalNorm(nDim_, 1.0);
  std::vector<double> rhsNorm(nDim_, std::numeric_limits<double>::max());

  for (unsigned d=0; d<nDim_; ++d) {
    status = solver->solve(d, iters[d], finalNorm[d], realm_.isFinalOuterIter_);
  }
  copy_hypre_to_stk(slnField, rhsNorm);
  sync_field(slnField);

  /* set this after the solve calls */
  solver->set_initialize_solver_flag();

  if (solver->getConfig()->getWriteMatrixFiles()) {
    for (unsigned d=0; d < nDim_; ++d) {
      std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
      const std::string slnFile = eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".sln";
      HYPRE_IJVectorPrint(sln_[d], slnFile.c_str());
    }
    ++eqSys_->linsysWriteCounter_;
  }

  {
    linearSolveIterations_ = 0;
    linearResidual_ = 0.0;
    nonLinearResidual_ = 0.0;
    double linres, nonlinres, scaledres, tmp, scaleFac = 0.0;

    for (unsigned d=0; d<nDim_; ++d) {
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

      for (unsigned d=0; d<nDim_; ++d) {
        int sid = in * nDim_ + d;
        HYPRE_IJVectorGetValues(sln_[d], 1, &hid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_[d], 1, &hid, &rhsVal);
        lclnorm[d] += rhsVal * rhsVal;
      }
    }
  }

  NGPDoubleFieldType ngpField = realm_.ngp_field_manager().get_field<double>(stkField->mesh_meta_data_ordinal());
  ngpField.modify_on_host();
  ngpField.sync_to_device();

  stk::all_reduce_sum(bulk.parallel(), lclnorm.data(), gblnorm.data(), nDim_);

  for (unsigned d=0; d<nDim_; ++d)
    rhsNorm[d] = std::sqrt(gblnorm[d]);
}



sierra::nalu::CoeffApplier* HypreUVWLinearSystem::get_coeff_applier()
{
  /* reset the internal counters */
  HypreUVWLinSysCoeffApplier* hcApplier = dynamic_cast<HypreUVWLinSysCoeffApplier*>(hostCoeffApplier.get());
  hcApplier->resetInternalData();
  return deviceCoeffApplier;
}

/********************************************************************************************************/
/*                     Beginning of HypreUVWLinSysCoeffApplier implementations                          */
/********************************************************************************************************/

HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::HypreUVWLinSysCoeffApplier(const stk::mesh::NgpMesh ngpMesh, NGPHypreIDFieldType ngpHypreGlobalId,
									     unsigned numDof, unsigned nDim, HypreIntType globalNumRows, int rank, 
									     HypreIntType iLower, HypreIntType iUpper, HypreIntType jLower, HypreIntType jUpper, MemoryMap map_shared,
									     HypreIntTypeViewUVM mat_elem_cols_owned_uvm, HypreIntTypeViewUVM mat_elem_cols_shared_uvm,
									     UnsignedView mat_row_start_owned, UnsignedView mat_row_start_shared, UnsignedView rhs_row_start_shared,
									     HypreIntTypeViewUVM row_indices_owned_uvm, HypreIntTypeViewUVM row_indices_shared_uvm, 
									     HypreIntTypeViewUVM row_counts_owned_uvm, HypreIntTypeViewUVM row_counts_shared_uvm,
									     HypreIntTypeView periodic_bc_rows_owned, PeriodicNodeMap periodic_node_to_hypre_id,
									     HypreIntTypeUnorderedMap skippedRowsMap, HypreIntTypeUnorderedMapHost skippedRowsMapHost,
									     HypreIntTypeUnorderedMap oversetRowsMap, HypreIntTypeUnorderedMapHost oversetRowsMapHost,
									     HypreIntType num_mat_overset_pts_owned, HypreIntType num_rhs_overset_pts_owned)
  : HypreLinSysCoeffApplier(ngpMesh, ngpHypreGlobalId, numDof, nDim, globalNumRows, rank,
			    iLower, iUpper, jLower, jUpper, map_shared, mat_elem_cols_owned_uvm, mat_elem_cols_shared_uvm,
			    mat_row_start_owned, mat_row_start_shared, rhs_row_start_shared,
			    row_indices_owned_uvm, row_indices_shared_uvm, row_counts_owned_uvm, row_counts_shared_uvm,
			    periodic_bc_rows_owned, periodic_node_to_hypre_id, skippedRowsMap, skippedRowsMapHost,
			    oversetRowsMap, oversetRowsMapHost, num_mat_overset_pts_owned, num_rhs_overset_pts_owned) { }


KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::sum_into(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const HypreIntType& iLower, const HypreIntType& iUpper,
  unsigned nDim) {

  for(unsigned i=0; i<numEntities; ++i) {
    auto node = entities[i];
    if (periodic_node_to_hypre_id_.exists(node.local_offset()))
      localIds[i] = periodic_node_to_hypre_id_.value_at(periodic_node_to_hypre_id_.find(node.local_offset()));
    else
      localIds[i] = ngpHypreGlobalId_.get(ngpMesh_, node, 0);
  }

  for (unsigned i=0; i<numEntities; ++i) {
    int ix = i * nDim;
    HypreIntType hid = localIds[i];
    if (checkSkippedRows_()) {
      if (skippedRowsMap_.exists(hid)) continue;
    }
    
    if (hid>=iLower && hid<=iUpper) {
      int offset = 0;
      unsigned lower = mat_row_start_owned_(hid-iLower);
      unsigned upper = mat_row_start_owned_(hid-iLower+1)-1;

      for (unsigned k=0; k<numEntities; ++k) {
	/* binary search subrange rather than a map.find */
	HypreIntType col = localIds[k];
	unsigned matIndex;
	binarySearch(mat_elem_cols_owned_uvm_,lower,upper,col,matIndex);	  
	/* write the matrix element */
	Kokkos::atomic_add(&values_owned_uvm_(matIndex),lhs(ix, offset));
	offset += nDim;
      }
      for (unsigned d=0; d<nDim; ++d) {
	int ir = ix + d;
	Kokkos::atomic_add(&rhs_owned_uvm_(hid-iLower,d),rhs[ir]);
      }
    } else {
      if (!map_shared_.exists(hid)) continue;
      int offset = 0;
      unsigned index = map_shared_.value_at(map_shared_.find(hid));
      unsigned lower = mat_row_start_shared_(index);
      unsigned upper = mat_row_start_shared_(index+1)-1;
      for (unsigned k=0; k<numEntities; ++k) {
	/* binary search subrange rather than a map.find */
	HypreIntType col = localIds[k];
	unsigned matIndex;
	binarySearch(mat_elem_cols_shared_uvm_,lower,upper,col,matIndex);	  
	/* write the matrix element */
	Kokkos::atomic_add(&values_shared_uvm_(matIndex),lhs(ix, offset));
	offset += nDim;
      }

      unsigned rhsIndex = rhs_row_start_shared_(index);
      for (unsigned d=0; d<nDim; ++d) {
	int ir = ix + d;
	Kokkos::atomic_add(&rhs_shared_uvm_(rhsIndex,d),rhs[ir]);
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
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  sum_into(numEntities,entities,localIds,rhs,lhs,iLower_,iUpper_,nDim_);
}


KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::reset_rows(unsigned numNodes,
							     const stk::mesh::Entity* nodeList,
							     const double diag_value,
							     const double rhs_residual,
							     const HypreIntType iLower, const HypreIntType iUpper,
							     const unsigned nDim) {

  for (unsigned i=0; i<numNodes; ++i) {
    auto node = nodeList[i];
    HypreIntType hid;
    if (periodic_node_to_hypre_id_.exists(node.local_offset()))
      hid = periodic_node_to_hypre_id_.value_at(periodic_node_to_hypre_id_.find(node.local_offset()));
    else
      hid = ngpHypreGlobalId_.get(ngpMesh_, node, 0);

    if (hid>=iLower && hid<=iUpper) {
      unsigned lower = mat_row_start_owned_(hid-iLower);
      unsigned upper = mat_row_start_owned_(hid-iLower+1)-1;
      for (unsigned k=lower; k<=upper; ++k)
        values_owned_uvm_(k) = 0.0;

      unsigned matIndex;
      binarySearch(mat_elem_cols_owned_uvm_,lower,upper,hid,matIndex);
      values_owned_uvm_(matIndex) = diag_value;

      for (unsigned d=0; d<nDim; ++d)
	rhs_owned_uvm_(hid-iLower,d) = rhs_residual;

    } else {
      if (!map_shared_.exists(hid)) continue;

      unsigned index = map_shared_.value_at(map_shared_.find(hid));
      unsigned lower = mat_row_start_shared_(index);
      unsigned upper = mat_row_start_shared_(index+1)-1;
      for (unsigned k=lower; k<=upper; ++k)
	values_shared_uvm_(k) = 0.0;

      unsigned matIndex;
      binarySearch(mat_elem_cols_shared_uvm_,lower,upper,hid,matIndex);
      values_shared_uvm_(matIndex) = diag_value;

      unsigned rhsIndex = rhs_row_start_shared_(index);
      for (unsigned d=0; d<nDim; ++d)
	rhs_shared_uvm_(rhsIndex,d) = rhs_residual;
    }
  }
}


KOKKOS_FUNCTION
void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::resetRows(unsigned numNodes,
                           const stk::mesh::Entity* nodeList,
                           const unsigned,
                           const unsigned,
                           const double diag_value,
                           const double rhs_residual) {
  checkSkippedRows_() = 0;
  reset_rows(numNodes, nodeList, diag_value, rhs_residual, iLower_, iUpper_, nDim_);
}


void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::applyDirichletBCs(Realm & realm, 
								    stk::mesh::FieldBase * solutionField,
								    stk::mesh::FieldBase * bcValuesField,
								    const stk::mesh::PartVector& parts) {
  resetInternalData();

  /************************************************************/
  /* this is a hack to get dirichlet bcs working consistently */

  /* Step 1: execute the old CPU code */
  auto& meta = realm.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm.get_inactive_selector()));

  const auto& bkts = realm.get_buckets(
    stk::topology::NODE_RANK, sel);

  double diag_value = 1.0;
  std::vector<HypreIntType> tCols(0);
  std::vector<double> tVals(0);
  std::vector<std::vector<double> >trhsVals(nDim_);
  for (unsigned i=0;i<nDim_;++i) {
    trhsVals[i].resize(0);
  }

  NGPDoubleFieldType ngpSolutionField = realm.ngp_field_manager().get_field<double>(solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField = realm.ngp_field_manager().get_field<double>(bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_host();
  ngpBCValuesField.sync_to_host();

  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (unsigned in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm.hypreGlobalId_, node);

      /* fill these temp values */
      tCols.push_back(hid);
      tVals.push_back(diag_value);
      
      for (unsigned d=0; d<nDim_; d++) {
        double bcval = bcValues[in*nDim_ + d] - solution[in*nDim_ + d];
	trhsVals[d].push_back(bcval);
      }
    }
  }

  /* Step 2 : allocate space in which to push the temporaries */
  HypreIntTypeView c = HypreIntTypeView("c",tCols.size());
  HypreIntTypeViewHost ch  = Kokkos::create_mirror_view(c);

  DoubleView v = DoubleView("v",tVals.size());
  DoubleViewHost vh  = Kokkos::create_mirror_view(v);

  DoubleView2D rv = DoubleView2D("rv",trhsVals[0].size(),nDim_);
  DoubleView2DHost rvh  = Kokkos::create_mirror_view(rv);

  /* Step 3 : next copy the std::vectors into the host mirrors */
  for (unsigned int i=0; i<tCols.size(); ++i) {
    ch(i) = tCols[i];
    vh(i) = tVals[i];
    for (unsigned j=0; j<nDim_;++j) {
      rvh(i,j) = trhsVals[j][i];
    }
  }

  /* Step 4 : deep copy this to device */
  Kokkos::deep_copy(c,ch);
  Kokkos::deep_copy(v,vh);
  Kokkos::deep_copy(rv,rvh);

  /* For device capture */
  auto mat_row_start_owned = mat_row_start_owned_;
  auto nDim = nDim_;
  auto iLower = iLower_;
  int N = (int) tCols.size();
  auto vals = values_owned_uvm_;
  auto rhs_vals = rhs_owned_uvm_;
  Kokkos::parallel_for("dirichlet_bcs_UVW", N, KOKKOS_LAMBDA(const unsigned& i) {
      HypreIntType hid = c(i);
      unsigned matIndex = mat_row_start_owned(hid-iLower);
      vals(matIndex)=v(i);
      for (unsigned d=0; d<nDim; ++d) {
	rhs_vals(hid-iLower,d) = rv(i,d);
	}
    });
}

void HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::free_device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier* HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
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
/*                           End of HypreUVWLinSysCoeffApplier implementations                           */
/*********************************************************************************************************/

void
HypreUVWLinearSystem::buildNodeGraph(const stk::mesh::PartVector & parts)
{
 #ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

 beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_owned );

  std::vector<HypreIntType> hids(1);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {

      stk::mesh::Entity node = b[k];
      hids[0] = get_entity_hypre_id(node);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(1, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  buildNodeGraphTimer_.push_back(msec);
  //printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreUVWLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(realm_.meta_data().side_rank(), s_owned);

  std::vector<HypreIntType> hids(0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];

    auto numNodes = b.topology().num_nodes();
    hids.resize(numNodes);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {

      stk::mesh::Entity const * nodes = b.begin_nodes(k);

      /* save the hypre ids */
      for (unsigned i=0; i<numNodes; ++i) hids[i] = get_entity_hypre_id(nodes[i]);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(numNodes, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  buildFaceToNodeGraphTimer_.push_back(msec);
  //printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}

void
HypreUVWLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::EDGE_RANK, s_owned);

  std::vector<HypreIntType> hids(0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];

    auto numNodes = b.topology().num_nodes();
    hids.resize(numNodes);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {

      stk::mesh::Entity const * nodes = b.begin_nodes(k);

      /* save the hypre ids */
      for (unsigned i=0; i<numNodes; ++i) hids[i] = get_entity_hypre_id(nodes[i]);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(numNodes, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  buildEdgeToNodeGraphTimer_.push_back(msec);
  //printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreUVWLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::ELEM_RANK, s_owned);

  std::vector<HypreIntType> hids(0);

  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];

    auto numNodes = b.topology().num_nodes();
    hids.resize(numNodes);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {

      stk::mesh::Entity const * nodes = b.begin_nodes(k);

      /* save the hypre ids */
      for (unsigned i=0; i<numNodes; ++i) hids[i] = get_entity_hypre_id(nodes[i]);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(numNodes, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  buildElemToNodeGraphTimer_.push_back(msec);
  //printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreUVWLinearSystem::buildFaceElemToNodeGraph(
  const stk::mesh::PartVector & parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();
  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( metaData.side_rank(), s_owned );

  std::vector<HypreIntType> hids(0);

  for(size_t ib=0; ib<face_buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *face_buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      const stk::mesh::Entity face = b[k];

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
      ThrowAssert( bulkData.num_elements(face) == 1 );

      // get connected element and nodal relations
      stk::mesh::Entity element = face_elem_rels[0];
      const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);

      // figure out the global dof ids for each dof on each node
      const unsigned numNodes = (unsigned)bulkData.num_nodes(element);
      hids.resize(numNodes);

      /* save the hypre ids */
      for (unsigned i=0; i<numNodes; ++i) hids[i] = get_entity_hypre_id(elem_nodes[i]);

      /* fill owned/shared 1 Dof */
      fill_owned_shared_data_structures_1DoF(numNodes, hids);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  buildFaceElemToNodeGraphTimer_.push_back(msec);
  //printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}


void
HypreUVWLinearSystem::buildReducedElemToNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}


void
HypreUVWLinearSystem::buildNonConformalNodeGraph(
  const stk::mesh::PartVector&)
{
  beginLinearSystemConstruction();
}


void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::PartVector& parts)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  // Grab nodes regardless of whether they are owned or shared
  const stk::mesh::Selector sel = stk::mesh::selectUnion(parts);
  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
      skippedRows_.insert(hid);
      if (hid>=iLower_ && hid<=iUpper_) {
	HypreIntType lid = hid-iLower_;
	rowCountOwned_[lid]++;
	columnsOwned_[lid].push_back(hid);
      }
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  buildDirichletNodeGraphTimer_.push_back(msec);
  //printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  for (const auto& node: nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);
    skippedRows_.insert(hid);
    if (hid>=iLower_ && hid<=iUpper_) {
      HypreIntType lid = hid-iLower_;
      rowCountOwned_[lid]++;
      columnsOwned_[lid].push_back(hid);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  buildDirichletNodeGraphTimer_.push_back(msec);
  //printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}

void 
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::NgpMesh::ConnectedNodes nodeList)
{
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  /* record the start time */
  struct timeval _start, _stop;
  gettimeofday(&_start, NULL);
#endif

  beginLinearSystemConstruction();

  for (unsigned i=0; i<nodeList.size();++i) {
    HypreIntType hid = get_entity_hypre_id(nodeList[i]);
    skippedRows_.insert(hid);
    if (hid>=iLower_ && hid<=iUpper_) {
      HypreIntType lid = hid-iLower_;
      rowCountOwned_[lid]++;
      columnsOwned_[lid].push_back(hid);
    }
  }

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  gettimeofday(&_stop, NULL);
  double msec = (double)(_stop.tv_usec - _start.tv_usec) / 1.e3 + 1.e3*((double)(_stop.tv_sec - _start.tv_sec));
  buildDirichletNodeGraphTimer_.push_back(msec);
  //printf("rank_=%d EqnName=%s : %s %s %d : dt=%1.5lf\n",rank_,name_.c_str(),__FILE__,__FUNCTION__,__LINE__,msec);
#endif
}



}  // nalu
}  // sierra
