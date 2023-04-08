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

#include "node_kernels/MomentumCoriolisNodeKernel.h"

TEST_F(MomentumNodeHex8Mesh, NGP_momentum_coriolis)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // simplify ICs
  stk::mesh::field_fill(1.0, *velocity_);
  velocity_->modify_on_host();
  velocity_->sync_to_device();

  stk::mesh::field_fill(1.0, *density_);
  density_->modify_on_host();
  density_->sync_to_device();

  // Setup solution options for default kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;

  // Setup Coriolis specific options.  Mimics the Ekman spiral test
  solnOpts_.earthAngularVelocity_ = 7.2921159e-5;
  solnOpts_.latitude_ = 30.0;
  solnOpts_.eastVector_ = {1.0, 0.0, 0.0};
  solnOpts_.northVector_ = {0.0, 1.0, 0.0};

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumCoriolisNodeKernel>(
    *bulk_, solnOpts_);

  helperObjs.execute();

  sierra::nalu::CoriolisSrc cor(solnOpts_);
  EXPECT_NEAR(cor.upVector_[0], 0.0, tol);
  EXPECT_NEAR(cor.upVector_[1], 0.0, tol);
  EXPECT_NEAR(cor.upVector_[2], 1.0, tol);

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  // Exact solution
  std::vector<double> rhsExact(24, 0.0);
  for (int n = 0; n < 8; ++n) {
    int nnDim = n * 3;
    rhsExact[nnDim + 0] = 0.125 * (+cor.Jxy_ + cor.Jxz_);
    rhsExact[nnDim + 1] = 0.125 * (-cor.Jxy_ + cor.Jyz_);
    rhsExact[nnDim + 2] = 0.125 * (-cor.Jxz_ - cor.Jyz_);
  }
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, rhsExact.data());
}
