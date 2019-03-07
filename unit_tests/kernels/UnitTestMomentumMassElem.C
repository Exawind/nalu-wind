/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/MomentumMassElemKernel.h"

namespace {
namespace hex8_golds {
namespace momentum_time_derivative {
static constexpr double rhs[24] = {
  0.2127585399521817, -0.2407694664966337, 0.0000000000000000,
  0.1326399983722029, -0.6942974729454492, 0.0000000000000000,
  0.4819527747499651, -0.5659855543833213, 0.0000000000000000,
  0.6662865464009970, -0.2166727780055592, 0.0000000000000000,
  0.2127585399521817, -0.2407694664966337, 0.0000000000000000,
  0.1326399983722029, -0.6942974729454492, 0.0000000000000000,
  0.4819527747499651, -0.5659855543833212, 0.0000000000000000,
  0.6662865464009970, -0.2166727780055592, 0.0000000000000000,
};

static constexpr double lhs[24][24] = {
  {
    0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0,
    0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0,
  },
  {
    0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0,
    0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0,
  },
  {
    0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125,
    0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375,
  },
  {
    0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0,
    0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0,
  },
  {
    0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0,
    0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0,
  },
  {
    0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375,
    0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125,
  },
  {
    0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0,
    0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0,
  },
  {
    0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0,
    0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0,
  },
  {
    0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125,
    0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375,
  },
  {
    0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0,
    0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0,
  },
  {
    0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0,
    0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0,
  },
  {
    0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375,
    0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125,
  },
  {
    0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0,
    0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0,
  },
  {
    0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0,
    0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0,
  },
  {
    0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375,
    0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125,
  },
  {
    0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0,
    0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0,
  },
  {
    0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125, 0,
    0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0,
  },
  {
    0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.01953125,
    0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0, 0.05859375,
  },
  {
    0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0,
    0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0, 0,
  },
  {
    0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375, 0,
    0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125, 0,
  },
  {
    0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.05859375,
    0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0, 0.17578125,
  },
  {
    0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0,
    0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0, 0,
  },
  {
    0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0,
    0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375, 0,
  },
  {
    0, 0, 0.05859375, 0, 0, 0.01953125, 0, 0, 0.05859375, 0, 0, 0.17578125,
    0, 0, 0.17578125, 0, 0, 0.05859375, 0, 0, 0.17578125, 0, 0, 0.52734375,
  },
};

} // momentum_time_derivative

namespace momentum_time_derivative_lumped {
static constexpr double rhs[24] = {
  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
  -0.0560218530889042, -1.0112712429686841, 0.0000000000000000,
  0.5383884695955669,  -0.6504321757733752, 0.0000000000000000,
  1.0112712429686841,  -0.0560218530889042, 0.0000000000000000,
  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
  -0.0560218530889042, -1.0112712429686841, 0.0000000000000000,
  0.5383884695955669,  -0.6504321757733752, 0.0000000000000000,
  1.0112712429686841,  -0.0560218530889042, 0.0000000000000000,
};
} // momentum_time_derivative_lumped
} // hex8_golds
} // anonymous namespace

#ifndef KOKKOS_ENABLE_CUDA

TEST_F(MomentumKernelHex8Mesh, momentum_time_derivative)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> massKernel(
    new sierra::nalu::MomentumMassElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_, false));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(massKernel.get());

  // Mass terms need time integration information
  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;

  // Call Kernel setup with the time integrator to setup Kernel values
  massKernel->setup(timeIntegrator);
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  // Populate LHS and RHS
  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace gold_values = ::hex8_golds::momentum_time_derivative;
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<24>(helperObjs.linsys->lhs_, gold_values::lhs);
}

TEST_F(MomentumKernelHex8Mesh, momentum_time_derivative_lumped)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> massKernel(
    new sierra::nalu::MomentumMassElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_, true));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(massKernel.get());

  // Mass terms need time integration information
  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;

  // Call Kernel setup with the time integrator to setup Kernel values
  massKernel->setup(timeIntegrator);
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  // Populate LHS and RHS
  helperObjs.assembleElemSolverAlg->execute();

  namespace gold_values = ::hex8_golds::momentum_time_derivative_lumped;
  // Exact LHS expected
  std::vector<double> lhsExact(576, 0.0);
  for (int i=0; i<24; i++)
    lhsExact[i*24+i] = 1.25;

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->lhs_, lhsExact.data());
}

#endif

