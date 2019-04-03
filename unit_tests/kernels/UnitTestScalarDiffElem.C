/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/ScalarDiffElemKernel.h"

namespace {
namespace hex8_golds {
namespace heatcond {
namespace cvfem_diff {
static constexpr double lhs[8][8] = {
  {0.421875, -0.046875, -0.078125, -0.046875, -0.046875, -0.078125, -0.046875, -0.078125},
  {-0.046875, 0.421875, -0.046875, -0.078125, -0.078125, -0.046875, -0.078125, -0.046875},
  {-0.078125, -0.046875, 0.421875, -0.046875, -0.046875, -0.078125, -0.046875, -0.078125},
  {-0.046875, -0.078125, -0.046875, 0.421875, -0.078125, -0.046875, -0.078125, -0.046875},
  {-0.046875, -0.078125, -0.046875, -0.078125, 0.421875, -0.046875, -0.078125, -0.046875},
  {-0.078125, -0.046875, -0.078125, -0.046875, -0.046875, 0.421875, -0.046875, -0.078125},
  {-0.046875, -0.078125, -0.046875, -0.078125, -0.078125, -0.046875, 0.421875, -0.046875},
  {-0.078125, -0.046875, -0.078125, -0.046875, -0.046875, -0.078125, -0.046875, 0.421875},
};
} // cvfem_diff
} // heatcond
} // hex8_golds
} // anonymous namespace

#ifndef KOKKOS_ENABLE_CUDA

/// Scalar diffusion kernel applied to heat conduction equation
TEST_F(HeatCondKernelHex8Mesh, cvfem_diff)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  int numDof = 1;
  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, numDof, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::ScalarDiffElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, temperature_, thermalCond_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::heatcond::cvfem_diff;
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, 0.0);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, gold_values::lhs);
}

#endif

