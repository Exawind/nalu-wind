/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "user_functions/SteadyThermal3dContactSrcElemKernel.h"

namespace {
namespace hex8_golds {
namespace steady_3d_thermal {

static constexpr double rhs[8] = {
  2.26627114475e-16, -7.55423714915e-17,
  -3.77711857458e-16, -7.55423714915e-17,
  -7.55423714915e-17, -3.77711857458e-16,
  -6.79881343424e-16, -3.77711857458e-16,
};
} // steady_3d_thermal
} // hex8_golds
} // anonymous namespace

#ifndef KOKKOS_ENABLE_CUDA

/// Steady 3D MMS source term
TEST_F(HeatCondKernelHex8Mesh, steady_3d_thermal)
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
    new sierra::nalu::SteadyThermal3dContactSrcElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = ::hex8_golds::steady_3d_thermal;

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, 0.0);
}

#endif

