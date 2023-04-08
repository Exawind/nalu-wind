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

#include "edge_kernels/MomentumABLWallFuncEdgeKernel.h"

#include <cmath>

TEST_F(MomentumABLKernelHex8Mesh, NGP_abl_wall_func)
{
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;

  auto* part = meta_->get_part("surface_5");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 3, part);

  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::MomentumABLWallFuncEdgeKernel<
      sierra::nalu::AlgTraitsQuad4>(
      *meta_, gravity_, z0_, Tref_, kappa_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 12u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 12u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 12u);

  const double lhsExact = ustar_ * kappa_ / std::log(zh_ / z0_) * 0.25;
  const double rhsExact[3] = {-ustar_ * ustar_ * 0.25, 0.0, 0.0};

  // LHS - check diagonal entries
  Kokkos::deep_copy(helperObjs.linsys->hostlhs_, helperObjs.linsys->lhs_);
  for (int i = 0; i < 12; i += 3) {
    EXPECT_NEAR(helperObjs.linsys->hostlhs_(i, i), lhsExact, 1.0e-12);
    EXPECT_NEAR(helperObjs.linsys->hostlhs_(i + 1, i + 1), lhsExact, 1.0e-12);
    EXPECT_NEAR(helperObjs.linsys->hostlhs_(i + 2, i + 2), 0.0, 1.0e-12);
  }

  // Off-diagonal entries in LHS
  for (int i = 0; i < 12; ++i)
    for (int j = 0; j < 12; ++j) {
      if (i == j)
        continue;
      EXPECT_NEAR(helperObjs.linsys->hostlhs_(i, j), 0.0, 1.0e-12);
    }

  // Check RHS
  for (int i = 0; i < 12; ++i)
    EXPECT_NEAR(helperObjs.linsys->hostrhs_(i), rhsExact[i % 3], 1.0e-12);
}
