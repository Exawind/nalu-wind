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

#include "edge_kernels/ContinuityOpenEdgeKernel.h"

namespace {
namespace hex8_golds {
static constexpr double rhs[8] = {
  0, 0.056142731673757, 0, 0.011397671913783,
  0, 0.056142731673757, 0, 0.011397671913783,
};

static constexpr double lhs[8][8] = {
  {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  },
  {
    0.25,
    0.25,
    0,
    0,
    0,
    0,
    0,
    0,
  },
  {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  },
  {
    0,
    0,
    0.25,
    0.25,
    0,
    0,
    0,
    0,
  },
  {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  },
  {
    0,
    0,
    0,
    0,
    0.25,
    0.25,
    0,
    0,
  },
  {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  },
  {
    0,
    0,
    0,
    0,
    0,
    0,
    0.25,
    0.25,
  },
};
} // namespace hex8_golds
} // namespace

TEST_F(ContinuityKernelHex8Mesh, NGP_open_edge)
{
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // Setup solution options for velocityRTM queries
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;

  auto* part = meta_->get_part("surface_2");
  const int numDof = 1;
  bool isEdge = true;
  unit_test_utils::FaceElemHelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, stk::topology::HEX_8, numDof, part, isEdge);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.timeStepN_ = 1.0;
  timeIntegrator.timeStepNm1_ = 1.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::ContinuityOpenEdgeKernel<
      sierra::nalu::AlgTraitsQuad4Hex8>(
      *meta_, &solnOpts_, helperObjs.assembleFaceElemSolverAlg->faceDataNeeded_,
      helperObjs.assembleFaceElemSolverAlg->elemDataNeeded_));

  helperObjs.assembleFaceElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}
