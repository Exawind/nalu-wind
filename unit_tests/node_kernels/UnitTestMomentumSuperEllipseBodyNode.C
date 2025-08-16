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
#include <aero/aero_utils/WienerMilenkovic.h>
#include "vs/vector_space.h"

#include "node_kernels/MomentumSuperEllipseBodyNodeKernel.h"

static constexpr double rhs[24] = {
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875,
  7.875        
};

TEST_F(MomentumNodeHex8Mesh, NGP_momentum_super_ellipse)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // simplify ICs
  stk::mesh::field_fill(6.3, *velocity_);
  velocity_->modify_on_host();
  velocity_->sync_to_device();

  stk::mesh::field_fill(1.0, *density_);
  density_->modify_on_host();
  density_->sync_to_device();

  // Setup solution options for default kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumSuperEllipseBodyNodeKernel>(
    *bulk_, solnOpts_);

  helperObjs.execute();

  vs::Vector loc{1.0, -1.0, -1.0};
  vs::Vector orient = wmp::create_wm_param(vs::Vector{1.0, 0.0, 0.0}, stk::math::asin(1.0)*0.5);
  vs::Vector dim{5.0, 5.0, 5.0};

  sierra::nalu::SuperEllipseBodySrc seb(solnOpts_, loc, orient, dim);

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, rhsExact.data());
}
