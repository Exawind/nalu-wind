/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "edge_kernels/WallDistEdgeSolverAlg.h"

#ifndef KOKKOS_ENABLE_CUDA
namespace {
namespace hex8_golds {
static constexpr double lhs[8][8] = {
  { 0.421875, -0.046875, -0.078125, -0.046875, -0.046875, -0.078125, -0.046875, -0.078125, },
  {-0.046875,  0.421875, -0.046875, -0.078125, -0.078125, -0.046875, -0.078125, -0.046875, },
  {-0.078125, -0.046875,  0.421875, -0.046875, -0.046875, -0.078125, -0.046875, -0.078125, },
  {-0.046875, -0.078125, -0.046875,  0.421875, -0.078125, -0.046875, -0.078125, -0.046875, },
  {-0.046875, -0.078125, -0.046875, -0.078125,  0.421875, -0.046875, -0.078125, -0.046875, },
  {-0.078125, -0.046875, -0.078125, -0.046875, -0.046875,  0.421875, -0.046875, -0.078125, },
  {-0.046875, -0.078125, -0.046875, -0.078125, -0.078125, -0.046875,  0.421875, -0.046875, },
  {-0.078125, -0.046875, -0.078125, -0.046875, -0.046875, -0.078125, -0.046875,  0.421875, },
};
} // hex8_golds
} // anonymous namespace
#endif

TEST_F(WallDistKernelHex8Mesh, NGP_wall_dist_edge)
{
  if (bulk_.parallel_size() > 1) return;

  fill_mesh();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  unit_test_utils::EdgeHelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1);
  helperObjs.create<sierra::nalu::WallDistEdgeSolverAlg>(partVec_[0]);

  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, 0.125);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, hex8_golds::lhs);
#endif
}
