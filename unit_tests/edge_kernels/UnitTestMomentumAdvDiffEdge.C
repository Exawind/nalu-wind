/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "edge_kernels/MomentumEdgeSolverAlg.h"

#ifndef KOKKOS_ENABLE_CUDA
namespace {
namespace hex8_golds {
namespace adv_diff {
static constexpr double rhs[24] = {
  -0.0015197705440727, 0.0015197705440727, 0,
  -0.00089329871267447, -0.018194207355441, 0,
  -0.084934324255693, 0.03907064661541, 0,
  0.087347393512441, -0.022396209804042, 0,
  -0.0015197705440727, 0.0015197705440727, 0,
  -0.00089329871267447, -0.018194207355441, 0,
  0.018194207355441, 0.00089329871267447, 0,
  -0.015781138098694, 0.015781138098694, 0,
};

static constexpr double lhs[24][24] = {
{0.1, 0, 0, -0.05, 0, 0, -0.025, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{0, 0.1, 0, 0, -0.025, 0, 0, -0.05, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{0, 0, 0.1, 0, 0, -0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
{-0.05, 0, 0, 0.1, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, },
{0, -0.025, 0, 0, 0.1, 0, 0, 0, 0, 0, -0.05, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, },
{0, 0, -0.025, 0, 0, 0.1, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, -0.05, 0, 0, 0, 0, 0, 0, },
{-0.025, 0, 0, 0, 0, 0, 0.019715921717342, 0, 0, -0.13028407828266,
 0, 0, 0, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, },
{0, -0.05, 0, 0, 0, 0, 0, 0.019715921717342, 0, 0, -0.10528407828266,
 0, 0, 0, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, },
{0, 0, -0.025, 0, 0, 0, 0, 0, 0.019715921717342, 0, 0, -0.10528407828266,
 0, 0, 0, 0, 0, 0, 0, 0, -0.05, 0, 0, 0, },
{0, 0, 0, -0.025, 0, 0, 0.030284078282658, 0, 0, 0.18028407828266,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.025, 0, 0, },
{0, 0, 0, 0, -0.05, 0, 0, 0.055284078282658, 0, 0, 0.18028407828266,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.025, 0, },
{0, 0, 0, 0, 0, -0.025, 0, 0, 0.055284078282658, 0, 0, 0.18028407828266,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, },
{-0.025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, -0.05, 0, 0, -0.025, 0, 0, 0, 0, 0, },
{0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, -0.025, 0, 0, -0.05, 0, 0, 0, 0, },
{0, 0, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, -0.025, 0, 0, -0.025, 0, 0, 0, },
{0, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, 0, 0, 0.1, 0, 0, 0, 0, 0, -0.025, 0, 0, },
{0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, -0.025, 0, 0, 0.1, 0, 0, 0, 0, 0, -0.05, 0, },
{0, 0, 0, 0, 0, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, -0.025, 0, 0, 0.1, 0, 0, 0, 0, 0, -0.025, },
{0, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, 0.1, 0, 0, -0.05, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, -0.05, 0, 0, 0, 0, 0, 0.1, 0, 0, -0.025, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, -0.05, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, 0.1, 0, 0, -0.025, },
{0, 0, 0, 0, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, -0.025, 0, 0, -0.05, 0, 0, 0.1, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.025, 0, 0, 0, 0, 0, -0.05, 0, 0, -0.025, 0, 0, 0.1, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, 0, 0, 0, 0, 0, -0.025, 0, 0, -0.025, 0, 0, 0.1, },
};
} // adv_diff
} // hex8_golds
} // anonymous namespace
#endif

TEST_F(MomentumEdgeHex8Mesh, NGP_advection_diffusion)
{
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.alphaMap_["velocity"] = 0.0;
  solnOpts_.alphaUpwMap_["velocity"] = 0.0;
  solnOpts_.upwMap_["velocity"] = 0.0;

  unit_test_utils::EdgeHelperObjects helperObjs(bulk_, stk::topology::HEX_8, 3);

  helperObjs.create<sierra::nalu::MomentumEdgeSolverAlg>(partVec_[0]);

  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_, 12);

  namespace gold_values = ::hex8_golds::adv_diff;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(
    helperObjs.linsys->lhs_, gold_values::lhs, 1.0e-12);
#endif
}
