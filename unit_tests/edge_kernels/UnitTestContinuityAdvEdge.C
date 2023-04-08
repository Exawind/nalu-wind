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

#include "edge_kernels/ContinuityEdgeSolverAlg.h"

namespace {
namespace hex8_golds {
static constexpr double rhs[8] = {
  0.05158341811906, -0.16056815656532, 0.16056815656532, -0.05158341811906,
  0.05158341811906, -0.16056815656532, 0.16056815656532, -0.05158341811906,
};

static constexpr double lhs[8][8] = {
  {
    0.75,
    -0.25,
    -0.25,
    0,
    -0.25,
    0,
    0,
    0,
  },
  {
    -0.25,
    0.75,
    0,
    -0.25,
    0,
    -0.25,
    0,
    0,
  },
  {
    -0.25,
    0,
    0.75,
    -0.25,
    0,
    0,
    -0.25,
    0,
  },
  {
    0,
    -0.25,
    -0.25,
    0.75,
    0,
    0,
    0,
    -0.25,
  },
  {
    -0.25,
    0,
    0,
    0,
    0.75,
    -0.25,
    -0.25,
    0,
  },
  {
    0,
    -0.25,
    0,
    0,
    -0.25,
    0.75,
    0,
    -0.25,
  },
  {
    0,
    0,
    -0.25,
    0,
    -0.25,
    0,
    0.75,
    -0.25,
  },
  {
    0,
    0,
    0,
    -0.25,
    0,
    -0.25,
    -0.25,
    0.75,
  },
};

} // namespace hex8_golds
} // anonymous namespace

TEST_F(ContinuityEdgeHex8Mesh, NGP_advection)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.mdotInterpRhoUTogether_ = true;

  unit_test_utils::EdgeHelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.timeStepN_ = 1.0;
  timeIntegrator.timeStepNm1_ = 1.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  helperObjs.create<sierra::nalu::ContinuityEdgeSolverAlg>(partVec_[0]);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs);
}
