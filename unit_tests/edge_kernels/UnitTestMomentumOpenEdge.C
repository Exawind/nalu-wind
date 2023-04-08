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

#include "edge_kernels/MomentumOpenEdgeKernel.h"

namespace {
namespace hex8_golds {
static constexpr double rhs[24] = {
  0,
  0,
  0,
  0,
  0.0063760611306953,
  0,
  0,
  0,
  0,
  -0.057266021854104,
  0.0037477547003379,
  0,
  0,
  0,
  0,
  0,
  0.0063760611306953,
  0,
  0,
  0,
  0,
  -0.057266021854104,
  0.0037477547003379,
  0,
};

static constexpr double lhs[24][24] = {
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    -0.014860258067112,
    0,
    0,
    -0.014860258067112,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
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
    0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0,
    0, 0,     0, 0, 0,      0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0,
    0, 0, 0,     0, 0, 0,      0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0,
    0,
    0,
    0,
    0,
    0,
    -0.044580774201335,
    0,
    0,
    -0.044580774201335,
    0,
    0,
    0,
    0,
    0,
    0,
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
    0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025, 0,
    0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0,      0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025,
    0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
    0,
    0,
    0,
    0,
    -0.014860258067112,
    0,
    0,
    -0.014860258067112,
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
    0, 0,     0, 0, 0,      0, 0, 0, 0, 0, 0, 0,
    0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0,     0, 0, 0,      0, 0, 0, 0, 0, 0,
    0, 0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    -0.044580774201335,
    0,
    0,
    -0.044580774201335,
    0,
    0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0,      0,
    0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025, 0,
  },
  {
    0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025,
  },
};
} // namespace hex8_golds
} // namespace

TEST_F(MomentumEdgeHex8Mesh, NGP_open_edge)
{
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;

  auto* part = meta_->get_part("surface_2");
  bool isEdge = true;
  unit_test_utils::FaceElemHelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, stk::topology::HEX_8, 3, part, isEdge);

  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::MomentumOpenEdgeKernel<sierra::nalu::AlgTraitsQuad4Hex8>(
      *meta_, &solnOpts_, viscosity_,
      helperObjs.assembleFaceElemSolverAlg->faceDataNeeded_,
      helperObjs.assembleFaceElemSolverAlg->elemDataNeeded_,
      sierra::nalu::EntrainmentMethod::COMPUTED));

  helperObjs.assembleFaceElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}
