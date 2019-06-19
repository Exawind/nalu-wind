/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "node_kernels/ContinuityMassBackwardEulerNodeKernel.h"

TEST_F(ContinuityKernelHex8Mesh, NGP_continuity_mass_backward_euler_node)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  helperObjs.nodeAlg->add_kernel<sierra::nalu::ContinuityMassBackwardEulerNodeKernel>(bulk_);

  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_,-12.5);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_,0.0);
#endif

}
