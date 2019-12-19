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

#include "node_kernels/MomentumBodyForceNodeKernel.h"

#include <vector>

TEST_F(MomentumNodeHex8Mesh, NGP_momentum_body_force)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  const std::vector<double> forceVector{8.0, 8.0, 8.0};

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumBodyForceNodeKernel>(
    bulk_, forceVector);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_(0), 8);

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, 1.0, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(helperObjs.linsys->lhs_, 0.0, 1.0e-12);
}
