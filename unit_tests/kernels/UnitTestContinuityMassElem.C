/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/ContinuityMassElemKernel.h"

#ifndef KOKKOS_ENABLE_CUDA

TEST_F(ContinuityKernelHex8Mesh, density_time_derivative)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> massKernel(
    new sierra::nalu::ContinuityMassElemKernel<sierra::nalu::AlgTraitsHex8>(
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

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_,-12.5);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_,0.0);
}

TEST_F(ContinuityKernelHex8Mesh, density_time_derivative_lumped)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> massKernel(
    new sierra::nalu::ContinuityMassElemKernel<sierra::nalu::AlgTraitsHex8>(
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

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_,-12.5);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_,0.0);
}

#endif

