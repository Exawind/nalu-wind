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

#include "kernel/WallDistElemKernel.h"

namespace {
namespace hex8_golds {
namespace wall_dist_default {
static constexpr double lhs[8][8] = {
  {
    0.421875,
    -0.046875,
    -0.078125,
    -0.046875,
    -0.046875,
    -0.078125,
    -0.046875,
    -0.078125,
  },
  {
    -0.046875,
    0.421875,
    -0.046875,
    -0.078125,
    -0.078125,
    -0.046875,
    -0.078125,
    -0.046875,
  },
  {
    -0.078125,
    -0.046875,
    0.421875,
    -0.046875,
    -0.046875,
    -0.078125,
    -0.046875,
    -0.078125,
  },
  {
    -0.046875,
    -0.078125,
    -0.046875,
    0.421875,
    -0.078125,
    -0.046875,
    -0.078125,
    -0.046875,
  },
  {
    -0.046875,
    -0.078125,
    -0.046875,
    -0.078125,
    0.421875,
    -0.046875,
    -0.078125,
    -0.046875,
  },
  {
    -0.078125,
    -0.046875,
    -0.078125,
    -0.046875,
    -0.046875,
    0.421875,
    -0.046875,
    -0.078125,
  },
  {
    -0.046875,
    -0.078125,
    -0.046875,
    -0.078125,
    -0.078125,
    -0.046875,
    0.421875,
    -0.046875,
  },
  {
    -0.078125,
    -0.046875,
    -0.078125,
    -0.046875,
    -0.046875,
    -0.078125,
    -0.046875,
    0.421875,
  },
};
} // namespace wall_dist_default

namespace wall_dist_lumped {
static constexpr double lhs[8][8] = {
  {
    0.75,
    -0.25,
    0,
    -0.25,
    -0.25,
    0,
    0,
    0,
  },
  {
    -0.25,
    0.75,
    -0.25,
    0,
    0,
    -0.25,
    0,
    0,
  },
  {
    0,
    -0.25,
    0.75,
    -0.25,
    0,
    0,
    -0.25,
    0,
  },
  {
    -0.25,
    0,
    -0.25,
    0.75,
    0,
    0,
    0,
    -0.25,
  },
  {
    -0.25,
    0,
    0,
    0,
    0.75,
    -0.25,
    0,
    -0.25,
  },
  {
    0,
    -0.25,
    0,
    0,
    -0.25,
    0.75,
    -0.25,
    0,
  },
  {
    0,
    0,
    -0.25,
    0,
    0,
    -0.25,
    0.75,
    -0.25,
  },
  {
    0,
    0,
    0,
    -0.25,
    -0.25,
    0,
    -0.25,
    0.75,
  },
};
}
} // namespace hex8_golds
} // namespace

TEST_F(WallDistKernelHex8Mesh, NGP_wall_dist)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> wallKernel(
    new sierra::nalu::WallDistElemKernel<sierra::nalu::AlgTraitsHex8>(
      *bulk_, solnOpts_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(wallKernel.get());

  // Populate LHS and RHS
  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::wall_dist_default;
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, 0.125);
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->lhs_, gold_values::lhs);
}

TEST_F(WallDistKernelHex8Mesh, NGP_wall_dist_shifted)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.shiftedGradOpMap_["ndtw"] = true;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> wallKernel(
    new sierra::nalu::WallDistElemKernel<sierra::nalu::AlgTraitsHex8>(
      *bulk_, solnOpts_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(wallKernel.get());

  // Populate LHS and RHS
  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::wall_dist_lumped;
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, 0.125);
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->lhs_, gold_values::lhs);
}
