/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "edge_kernels/ScalarEdgeSolverAlg.h"

#ifndef KOKKOS_ENABLE_CUDA
namespace {
namespace hex8_golds {
namespace adv_diff {
static constexpr double rhs[8] = {
  -5.55e-05, 5.55e-05, 5.55e-05, -5.55e-05,
  5.55e-05, -5.55e-05, -5.55e-05, 5.55e-05,
};

static constexpr double lhs[8][8] = {
  {1.3875e-05, -4.625e-06, -4.625e-06, 0, -4.625e-06, 0, 0, 0, },
  {-4.625e-06, 1.3875e-05, 0, -4.625e-06, 0, -4.625e-06, 0, 0, },
  {-4.625e-06, 0, -0.001357507586766, -0.001376007586766, 0, 0, -4.625e-06, 0, },
  {0, -4.625e-06, 0.001366757586766, 0.001385257586766, 0, 0, 0, -4.625e-06, },
  {-4.625e-06, 0, 0, 0, 1.3875e-05, -4.625e-06, -4.625e-06, 0, },
  {0, -4.625e-06, 0, 0, -4.625e-06, 1.3875e-05, 0, -4.625e-06, },
  {0, 0, -4.625e-06, 0, -4.625e-06, 0, 1.3875e-05, -4.625e-06, },
  {0, 0, 0, -4.625e-06, 0, -4.625e-06, -4.625e-06, 1.3875e-05, },
};
}
}
}

#endif

TEST_F(MixtureFractionKernelHex8Mesh, NGP_adv_diff_edge)
{
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.alphaMap_["mixture_fraction"] = 0.0;
  solnOpts_.alphaUpwMap_["mixture_fraction"] = 0.0;
  solnOpts_.upwMap_["mixture_fraction"] = 0.0;

  unit_test_utils::EdgeHelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1);

  helperObjs.create<sierra::nalu::ScalarEdgeSolverAlg>(
    partVec_[0], mixFraction_, dzdx_, viscosity_);

  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_, 12);

  namespace gold_values = ::hex8_golds::adv_diff;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, gold_values::lhs, 1.0e-12);
#endif
}
