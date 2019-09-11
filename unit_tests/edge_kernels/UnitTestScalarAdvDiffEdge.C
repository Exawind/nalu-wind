/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"
#include "UnitTestTpetraHelperObjects.h"

#include "edge_kernels/ScalarEdgeSolverAlg.h"

namespace {
namespace hex8_golds {
namespace adv_diff {

static const std::vector<int> rowOffsets_serial = {0, 4, 8, 12, 16, 20, 24, 28, 32};

static const std::vector<int> cols_serial = {0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4, 5, 7, 2, 4, 6, 7, 3, 5, 6, 7};

static const std::vector<double> vals_serial = {1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -0.001357507586766, -0.001376007586766, -4.625e-06, -4.625e-06, 0.001366757586766, 0.001385257586766, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05};

static const std::vector<double> rhs_serial = {-5.55e-05, 5.55e-05, 5.55e-05, -5.55e-05, 5.55e-05, -5.55e-05, -5.55e-05, 5.55e-05};

//-------- P0 ------------------

static const std::vector<int> rowOffsets_P0 = {0, 4, 8, 12, 16, 21, 26, 31, 36};

static const std::vector<int> cols_P0 = {0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 8, 1, 4, 5, 7, 9, 2, 4, 6, 7, 10, 3, 5, 6, 7, 11};

static const std::vector<double> vals_P0 = {1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, 1.85e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.85e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.85e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.85e-05, -4.625e-06};

static const std::vector<double> rhs_P0 = {-5.55e-05, 5.55e-05, 5.55e-05, -5.55e-05, 7.4e-05, -7.4e-05, -7.4e-05, 7.4e-05};

//-------- P1 ------------------

static const std::vector<int> rowOffsets_P1 = {0, 4, 8, 12, 16};

static const std::vector<int> cols_P1 = {0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7};

static const std::vector<double> vals_P1 = {1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06, -4.625e-06, -4.625e-06, -4.625e-06, 1.3875e-05, -4.625e-06};

static const std::vector<double> rhs_P1 = {-5.55e-05, 5.55e-05, 5.55e-05, -5.55e-05};

}
}
}

TEST_F(MixtureFractionKernelHex8Mesh, NGP_adv_diff_edge_tpetra)
{
  int numProcs = bulk_.parallel_size();
  if (numProcs > 2) return;

  int myProc = bulk_.parallel_rank();

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.alphaMap_["mixture_fraction"] = 0.0;
  solnOpts_.alphaUpwMap_["mixture_fraction"] = 0.0;
  solnOpts_.upwMap_["mixture_fraction"] = 0.0;

  const int numDof = 1;
  unit_test_utils::TpetraHelperObjectsEdge helperObjs(bulk_, numDof);

  helperObjs.realm.naluGlobalId_ = naluGlobalId_;
  helperObjs.realm.linSysGlobalId_ = linSysGlobalId_;

  helperObjs.realm.set_global_id();

  helperObjs.create<sierra::nalu::ScalarEdgeSolverAlg>(
    partVec_[0], mixFraction_, dzdx_, viscosity_);

  helperObjs.execute();

  namespace golds = ::hex8_golds::adv_diff;

  if (numProcs == 1) {
    helperObjs.check_against_sparse_gold_values(golds::rowOffsets_serial, golds::cols_serial,
                                                golds::vals_serial, golds::rhs_serial);
  }
  else {
    if (myProc == 0) {
      helperObjs.check_against_sparse_gold_values(golds::rowOffsets_P0, golds::cols_P0,
                                                  golds::vals_P0, golds::rhs_P0);
    }
    else {
      helperObjs.check_against_sparse_gold_values(golds::rowOffsets_P1, golds::cols_P1,
                                                  golds::vals_P1, golds::rhs_P1);
    }
  }
}
