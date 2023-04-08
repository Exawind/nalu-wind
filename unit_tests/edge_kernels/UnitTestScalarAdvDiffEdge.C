// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"
#include "UnitTestTpetraHelperObjects.h"
#include "FixPressureAtNodeInfo.h"
#include "FixPressureAtNodeAlgorithm.h"
#include "stk_mesh/base/NgpField.hpp"

#include "edge_kernels/ScalarEdgeSolverAlg.h"

namespace {
namespace hex8_golds {
namespace adv_diff {

static const std::vector<int> rowOffsets_serial = {0,  4,  8,  12, 16,
                                                   20, 24, 28, 32};

static const std::vector<int> cols_serial = {0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3,
                                             6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4,
                                             5, 7, 2, 4, 6, 7, 3, 5, 6, 7};

static const std::vector<double> vals_serial = {
  1.3875e-05, -4.625e-06,         -4.625e-06,         -4.625e-06,
  -4.625e-06, 1.3875e-05,         -4.625e-06,         -4.625e-06,
  -4.625e-06, -0.001357507586766, -0.001376007586766, -4.625e-06,
  -4.625e-06, 0.001366757586766,  0.001385257586766,  -4.625e-06,
  -4.625e-06, 1.3875e-05,         -4.625e-06,         -4.625e-06,
  -4.625e-06, -4.625e-06,         1.3875e-05,         -4.625e-06,
  -4.625e-06, -4.625e-06,         1.3875e-05,         -4.625e-06,
  -4.625e-06, -4.625e-06,         -4.625e-06,         1.3875e-05};

static const std::vector<double> fixed_vals_serial = {
  1.0,
  0.0,
  0.0,
  0.0,
  -4.625e-06,
  1.3875e-05,
  -4.625e-06,
  -4.625e-06,
  -4.625e-06,
  -0.001357507586766,
  -0.001376007586766,
  -4.625e-06,
  -4.625e-06,
  0.001366757586766,
  0.001385257586766,
  -4.625e-06,
  -4.625e-06,
  1.3875e-05,
  -4.625e-06,
  -4.625e-06,
  -4.625e-06,
  -4.625e-06,
  1.3875e-05,
  -4.625e-06,
  -4.625e-06,
  -4.625e-06,
  1.3875e-05,
  -4.625e-06,
  -4.625e-06,
  -4.625e-06,
  -4.625e-06,
  1.3875e-05};

static const std::vector<double> dirichlet_vals_serial = {
  1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
  0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1};

static const std::vector<double> rhs_serial = {-5.55e-05, 5.55e-05, 5.55e-05,
                                               -5.55e-05, 5.55e-05, -5.55e-05,
                                               -5.55e-05, 5.55e-05};

static const std::vector<double> fixed_rhs_serial = {
  0.91245334547109691, 5.55e-05,  5.55e-05, -5.55e-05, 5.55e-05,
  -5.55e-05,           -5.55e-05, 5.55e-05};

static const std::vector<double> dirichlet_rhs_serial = {-1, -2, -2, -2,
                                                         -2, -2, -2, -2};

//-------- P0 ------------------

static const std::vector<int> rowOffsets_P0 = {0, 4, 8, 12, 16, 21, 26, 31, 36};

static const std::vector<int> cols_P0 = {0, 1, 2, 4, 0, 1, 3,  5, 0, 2, 3, 6,
                                         1, 2, 3, 7, 0, 4, 5,  6, 8, 1, 4, 5,
                                         7, 9, 2, 4, 6, 7, 10, 3, 5, 6, 7, 11};

static const std::vector<double> vals_P0 = {
  1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05,
  -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06,
  -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, 1.85e-05,
  -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.85e-05,
  -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.85e-05,   -4.625e-06,
  -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.85e-05,   -4.625e-06};

static const std::vector<double> fixed_vals_P0 = {
  1.0,        0.0,        0.0,        0.0,        -4.625e-06, 1.3875e-05,
  -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06,
  -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, 1.85e-05,
  -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.85e-05,
  -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.85e-05,   -4.625e-06,
  -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.85e-05,   -4.625e-06};

static const std::vector<double> dirichlet_vals_P0 = {
  1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0};

static const std::vector<double> rhs_P0 = {-5.55e-05, 5.55e-05, 5.55e-05,
                                           -5.55e-05, 7.4e-05,  -7.4e-05,
                                           -7.4e-05,  7.4e-05};

static const std::vector<double> fixed_rhs_P0 = {
  0.91245334547109691,
  5.55e-05,
  5.55e-05,
  -5.55e-05,
  7.4e-05,
  -7.4e-05,
  -7.4e-05,
  7.4e-05};

static const std::vector<double> dirichlet_rhs_P0 = {-1, -2, -2, -2,
                                                     -2, -2, -2, -2};

//-------- P1 ------------------

static const std::vector<int> rowOffsets_P1 = {0, 4, 8, 12, 16};

static const std::vector<int> cols_P1 = {0, 1, 2, 4, 0, 1, 3, 5,
                                         0, 2, 3, 6, 1, 2, 3, 7};

static const std::vector<double> vals_P1 = {
  1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05,
  -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06,
  -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06};

static const std::vector<double> dirichlet_vals_P1 = {1, 0, 0, 0, 0, 1, 0, 0,
                                                      0, 1, 0, 0, 0, 0, 1, 0};

static const std::vector<double> rhs_P1 = {
  -5.55e-05, 5.55e-05, 5.55e-05, -5.55e-05};

static const std::vector<double> dirichlet_rhs_P1 = {-2, -2, -2, -2};

} // namespace adv_diff
} // namespace hex8_golds
} // namespace

TEST_F(MixtureFractionKernelHex8Mesh, NGP_adv_diff_edge_tpetra)
{
  int numProcs = bulk_->parallel_size();
  if (numProcs > 2)
    return;

  int myProc = bulk_->parallel_rank();

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.alphaMap_["mixture_fraction"] = 0.0;
  solnOpts_.alphaUpwMap_["mixture_fraction"] = 0.0;
  solnOpts_.upwMap_["mixture_fraction"] = 0.0;

  const int numDof = 1;
  unit_test_utils::TpetraHelperObjectsEdge helperObjs(bulk_, numDof);

  helperObjs.realm.naluGlobalId_ = naluGlobalId_;
  helperObjs.realm.tpetGlobalId_ = tpetGlobalId_;

  helperObjs.realm.set_global_id();

  bool useAvgMdot_ = false;

  helperObjs.create<sierra::nalu::ScalarEdgeSolverAlg>(
    partVec_[0], mixFraction_, dzdx_, viscosity_, useAvgMdot_);

  helperObjs.execute();

  namespace golds = ::hex8_golds::adv_diff;

  if (numProcs == 1) {
    helperObjs.check_against_sparse_gold_values(
      golds::rowOffsets_serial, golds::cols_serial, golds::vals_serial,
      golds::rhs_serial);
  } else {
    if (myProc == 0) {
      helperObjs.check_against_sparse_gold_values(
        golds::rowOffsets_P0, golds::cols_P0, golds::vals_P0, golds::rhs_P0);
    } else {
      helperObjs.check_against_sparse_gold_values(
        golds::rowOffsets_P1, golds::cols_P1, golds::vals_P1, golds::rhs_P1);
    }
  }

  // copy_stk_to_tpetra is not converted to stk::mesh::NgpField yet, but this
  // test still works due to tpetra using UVM space.
  helperObjs.linsys->copy_stk_to_tpetra(
    viscosity_, helperObjs.linsys->getOwnedRhs());
  helperObjs.linsys->copy_tpetra_to_stk(
    helperObjs.linsys->getOwnedRhs(), mixFraction_);

  auto ngpField = helperObjs.realm.ngp_field_manager().get_field<double>(
    mixFraction_->mesh_meta_data_ordinal());
  ngpField.sync_to_host();

  const stk::mesh::BucketVector& buckets = bulk_->get_buckets(
    stk::topology::NODE_RANK, bulk_->mesh_meta_data().locally_owned_part());
  for (const stk::mesh::Bucket* bptr : buckets) {
    for (stk::mesh::Entity node : *bptr) {
      const double* data1 =
        static_cast<double*>(stk::mesh::field_data(*viscosity_, node));
      const double* data2 =
        static_cast<double*>(stk::mesh::field_data(*mixFraction_, node));
      EXPECT_NEAR(*data1, *data2, 1.e-12);
    }
  }
}

TEST_F(
  MixtureFractionKernelHex8Mesh, NGP_adv_diff_edge_tpetra_fix_pressure_at_node)
{
  int numProcs = bulk_->parallel_size();
  if (numProcs > 2)
    return;

  int myProc = bulk_->parallel_rank();

  fill_mesh_and_init_fields();

  const int numDof = 1;
  unit_test_utils::TpetraHelperObjectsEdge helperObjs(bulk_, numDof);

  sierra::nalu::SolutionOptions* solnOpts = helperObjs.realm.solutionOptions_;

  // Setup solution options for default advection kernel
  solnOpts->meshMotion_ = false;
  solnOpts->meshDeformation_ = false;
  solnOpts->alphaMap_["mixture_fraction"] = 0.0;
  solnOpts->alphaUpwMap_["mixture_fraction"] = 0.0;
  solnOpts->upwMap_["mixture_fraction"] = 0.0;

  solnOpts->fixPressureInfo_.reset(new sierra::nalu::FixPressureAtNodeInfo);
  solnOpts->fixPressureInfo_->refPressure_ = 1.0;
  solnOpts->fixPressureInfo_->lookupType_ =
    sierra::nalu::FixPressureAtNodeInfo::STK_NODE_ID;
  solnOpts->fixPressureInfo_->stkNodeId_ = 1;

  helperObjs.realm.naluGlobalId_ = naluGlobalId_;
  helperObjs.realm.tpetGlobalId_ = tpetGlobalId_;

  helperObjs.realm.set_global_id();

  helperObjs.create<sierra::nalu::ScalarEdgeSolverAlg>(
    partVec_[0], mixFraction_, dzdx_, viscosity_);

  helperObjs.execute();

  sierra::nalu::FixPressureAtNodeAlgorithm fixPressure(
    helperObjs.realm, partVec_[0], &helperObjs.eqSystem);

  fixPressure.pressure_ =
    density_; // any scalar field should work for this unit-test...

  fixPressure.initialize();
  fixPressure.execute();

  namespace golds = ::hex8_golds::adv_diff;

  if (numProcs == 1) {
    helperObjs.check_against_sparse_gold_values(
      golds::rowOffsets_serial, golds::cols_serial, golds::fixed_vals_serial,
      golds::fixed_rhs_serial);
  } else {
    if (myProc == 0) {
      helperObjs.check_against_sparse_gold_values(
        golds::rowOffsets_P0, golds::cols_P0, golds::fixed_vals_P0,
        golds::fixed_rhs_P0);
    } else {
      helperObjs.check_against_sparse_gold_values(
        golds::rowOffsets_P1, golds::cols_P1, golds::vals_P1, golds::rhs_P1);
    }
  }
}

TEST_F(MixtureFractionKernelHex8Mesh, NGP_adv_diff_edge_tpetra_dirichlet)
{
  int numProcs = bulk_->parallel_size();
  if (numProcs > 2)
    return;

  fill_mesh_and_init_fields();

  const int numDof = 1;
  unit_test_utils::TpetraHelperObjectsEdge helperObjs(bulk_, numDof);

  sierra::nalu::SolutionOptions* solnOpts = helperObjs.realm.solutionOptions_;

  // Setup solution options for default advection kernel
  solnOpts->meshMotion_ = false;
  solnOpts->meshDeformation_ = false;
  solnOpts->alphaMap_["mixture_fraction"] = 0.0;
  solnOpts->alphaUpwMap_["mixture_fraction"] = 0.0;
  solnOpts->upwMap_["mixture_fraction"] = 0.0;

  solnOpts->fixPressureInfo_.reset(new sierra::nalu::FixPressureAtNodeInfo);
  solnOpts->fixPressureInfo_->refPressure_ = 1.0;
  solnOpts->fixPressureInfo_->lookupType_ =
    sierra::nalu::FixPressureAtNodeInfo::STK_NODE_ID;
  solnOpts->fixPressureInfo_->stkNodeId_ = 1;

  helperObjs.realm.naluGlobalId_ = naluGlobalId_;
  helperObjs.realm.tpetGlobalId_ = tpetGlobalId_;

  helperObjs.realm.set_global_id();

  helperObjs.create<sierra::nalu::ScalarEdgeSolverAlg>(
    partVec_[0], mixFraction_, dzdx_, viscosity_);

  helperObjs.execute();

  // next, test the applyDirichletBCs method.
  // any scalar nodal fields should work for this unit-test...
  stk::mesh::FieldBase* solutionField = mixFraction_;
  stk::mesh::FieldBase* bcValuesField = viscosity_;

  auto ngpSolutionField =
    helperObjs.realm.ngp_field_manager().get_field<double>(
      solutionField->mesh_meta_data_ordinal());
  auto ngpBCValuesField =
    helperObjs.realm.ngp_field_manager().get_field<double>(
      bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_host();
  ngpBCValuesField.sync_to_host();

  stk::mesh::field_fill(2.0, *solutionField);
  stk::mesh::field_fill(0.0, *bcValuesField);

  ngpSolutionField.modify_on_host();
  ngpBCValuesField.modify_on_host();

  stk::mesh::Entity node1 = bulk_->get_entity(stk::topology::NODE_RANK, 1);
  if (bulk_->is_valid(node1)) {
    double* node1value =
      static_cast<double*>(stk::mesh::field_data(*bcValuesField, node1));
    *node1value = 1.0;
  }

  helperObjs.linsys->applyDirichletBCs(
    solutionField, bcValuesField, partVec_, 0, 1);

  namespace golds = ::hex8_golds::adv_diff;

  int myProc = bulk_->parallel_rank();

  if (numProcs == 1) {
    helperObjs.check_against_sparse_gold_values(
      golds::rowOffsets_serial, golds::cols_serial,
      golds::dirichlet_vals_serial, golds::dirichlet_rhs_serial);
  } else {
    if (myProc == 0) {
      helperObjs.check_against_sparse_gold_values(
        golds::rowOffsets_P0, golds::cols_P0, golds::dirichlet_vals_P0,
        golds::dirichlet_rhs_P0);
    } else {
      helperObjs.check_against_sparse_gold_values(
        golds::rowOffsets_P1, golds::cols_P1, golds::dirichlet_vals_P1,
        golds::dirichlet_rhs_P1);
    }
  }
}
