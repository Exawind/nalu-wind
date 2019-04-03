/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/MomentumActuatorSrcElemKernel.h"

#ifndef KOKKOS_ENABLE_CUDA

TEST_F(ActuatorSourceKernelHex8Mesh, actuator_source)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::MomentumActuatorSrcElemKernel<sierra::nalu::AlgTraitsHex8>(
        bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_,
        false));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  std::vector<double> rhsExact(24,0.0);
  std::vector<double> lhsExact(24*24,0.0);  

  for (int i=0; i<8; i++){
      for (int j=0; j<3; j++) {
          rhsExact[i*3+j] = (j+1)/8.0;
          lhsExact[(i*3+j)*24 + i*3+j] = 0.1*(j+1)/8.0;
      }
  }

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->lhs_,lhsExact.data());
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_,rhsExact.data());
}

#endif

