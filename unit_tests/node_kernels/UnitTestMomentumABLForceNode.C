/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "node_kernels/MomentumABLForceNodeKernel.h"
#include "wind_energy/ABLForcingAlgorithm.h"
#include "node_kernels/UnitTestNodeUtils.h"

#include <vector>

TEST_F(MomentumNodeHex8Mesh, NGP_abl_force)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.realm.ablForcingAlg_ = new unit_test_utils::TestABLForcingAlg(helperObjs.realm);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumABLForceNodeKernel>(bulk_, solnOpts_);

  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_, 8);

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, 1.25, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(helperObjs.linsys->lhs_, 0.0, 1.0e-12);
#endif
}
