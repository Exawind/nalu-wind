// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/VOFAdvectionEdgeAlg.h"
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
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  -83.128243716220496,
  0.0,
  0.0,
  83.128243716220496,
  0.0,
  0.0,
  0.0,
  -83.128243716220496,
  83.128243716220496,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  83.449380029351119,
  0.0,
  0.0,
  0.0,
  0.0,
  -83.449380029351119,
  0.0,
  -83.449380029351119,
  0.0,
  83.449380029351119};

static const std::vector<double> rhs_serial = {
  0.0, 166.256487432440991,  166.256487432440991,  -332.512974864881983,
  0.0, -166.898760058702237, -166.898760058702237, 333.797520117404474};

//-------- P0 ------------------

static const std::vector<int> rowOffsets_P0 = {0, 4, 8, 12, 16, 21, 26, 31, 36};

static const std::vector<int> cols_P0 = {0, 1, 2, 4, 0, 1, 3,  5, 0, 2, 3, 6,
                                         1, 2, 3, 7, 0, 4, 5,  6, 8, 1, 4, 5,
                                         7, 9, 2, 4, 6, 7, 10, 3, 5, 6, 7, 11};

static const std::vector<double> vals_P0 = {
  0.000000000000000,   0.000000000000000,   0.000000000000000,
  0.000000000000000,   0.000000000000000,   0.000000000000000,
  -83.128243716220496, 0.000000000000000,   0.000000000000000,
  83.128243716220496,  0.000000000000000,   0.000000000000000,
  0.000000000000000,   -83.128243716220496, 83.128243716220496,
  0.000000000000000,   0.000000000000000,   0.000000000000000,
  0.000000000000000,   0.000000000000000,   0.000000000000000,
  0.000000000000000,   0.000000000000000,   83.449380029351119,
  0.000000000000000,   0.000000000000000,   0.000000000000000,
  0.000000000000000,   0.000000000000000,   -83.449380029351119,
  0.000000000000000,   0.000000000000000,   -83.449380029351119,
  0.000000000000000,   83.449380029351119,  0.000000000000000};

static const std::vector<double> rhs_P0 = {
  0.0, 166.256487432440991,  166.256487432440991,  -332.512974864881983,
  0.0, -166.898760058702237, -166.898760058702237, 333.797520117404474};

//-------- P1 ------------------

static const std::vector<int> rowOffsets_P1 = {0, 4, 8, 12, 16};

static const std::vector<int> cols_P1 = {0, 1, 2, 4, 0, 1, 3, 5,
                                         0, 2, 3, 6, 1, 2, 3, 7};

static const std::vector<double> vals_P1 = {
  0.000000000000000,   0.000000000000000,   0.000000000000000,
  0.000000000000000,   0.000000000000000,   0.000000000000000,
  -83.128243716220496, 0.000000000000000,   0.000000000000000,
  83.128243716220496,  0.000000000000000,   0.000000000000000,
  0.000000000000000,   -83.128243716220496, 83.128243716220496,
  0.000000000000000};

static const std::vector<double> rhs_P1 = {
  0.000000000000000, 166.256487432440991, 166.256487432440991,
  -332.512974864881983};

} // namespace adv_diff
} // namespace hex8_golds
} // namespace

TEST_F(VOFKernelHex8Mesh, NGP_adv_diff_edge_tpetra)
{
  int numProcs = bulk_->parallel_size();

  if (numProcs > 2)
    return;

  int myProc = bulk_->parallel_rank();

  fill_mesh_and_init_fields();

  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.alphaMap_["volume_of_fluid"] = 0.0;
  solnOpts_.alphaUpwMap_["volume_of_fluid"] = 0.0;
  solnOpts_.upwMap_["volume_of_fluid"] = 0.0;

  const int numDof = 1;
  unit_test_utils::TpetraHelperObjectsEdge helperObjs(bulk_, numDof);

  helperObjs.realm.naluGlobalId_ = naluGlobalId_;
  helperObjs.realm.tpetGlobalId_ = tpetGlobalId_;

  helperObjs.realm.set_global_id();

  helperObjs.create<sierra::nalu::VOFAdvectionEdgeAlg>(
    partVec_[0], volumeOfFluid_, dvolumeOfFluidDx_);
  helperObjs.execute();

  namespace golds = ::hex8_golds::adv_diff;

  const double tol = 1e-13;
  if (numProcs == 1) {
    helperObjs.check_against_sparse_gold_values(
      golds::rowOffsets_serial, golds::cols_serial, golds::vals_serial,
      golds::rhs_serial, tol);
  } else {
    if (myProc == 0) {
      helperObjs.check_against_sparse_gold_values(
        golds::rowOffsets_P0, golds::cols_P0, golds::vals_P0, golds::rhs_P0,
        tol);
    } else {
      helperObjs.check_against_sparse_gold_values(
        golds::rowOffsets_P1, golds::cols_P1, golds::vals_P1, golds::rhs_P1,
        tol);
    }
  }
}
