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

#include "node_kernels/MomentumActuatorNodeKernel.h"

#include <vector>

TEST_F(ActuatorSourceKernelHex8Mesh, NGP_momentum_actuator)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  std::mt19937 rng;
  rng.seed(0); // fixed seed

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumActuatorNodeKernel>(
    bulk_->mesh_meta_data());

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  std::vector<double> rhsExact(24, 0.0);
  std::vector<double> lhsExact(24 * 24, 0.0);

  // Exact solution
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 3; j++) {
      rhsExact[i * 3 + j] = (j + 1) / 8.0;
      lhsExact[(i * 3 + j) * 24 + i * 3 + j] = 0.1 * (j + 1) / 8.0;
    }
  }

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, rhsExact.data());
  unit_test_kernel_utils::expect_all_near_2d(
    helperObjs.linsys->lhs_, lhsExact.data());
}
