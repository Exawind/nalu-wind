/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "edge_kernels/MomentumOpenEdgeKernel.h"

namespace {
namespace hex8_golds  {
static constexpr double rhs[24] = {
  0, 0, 0, 0, 0.0063760611306953, 0, 0, 0,
  0, -0.057266021854104, 0.0037477547003379, 0, 0, 0, 0, 0,
  0.0063760611306953, 0, 0, 0, 0, -0.057266021854104, 0.0037477547003379, 0,
};

static constexpr double lhs[24][24] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {-0.014860258067112, 0, 0, -0.014860258067112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, -0.044580774201335, 0, 0, -0.044580774201335, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.014860258067112, 0, 0, -0.014860258067112, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.044580774201335, 0, 0, -0.044580774201335, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, -0.025, },
};
}
}

TEST_F(MomentumKernelHex8Mesh, NGP_open_edge)
{
  if (bulk_.parallel_size() > 1) return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  auto* part = meta_.get_part("surface_2");
  bool isEdge = true;
  unit_test_utils::FaceElemHelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, stk::topology::HEX_8, 3, part, isEdge);

  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::MomentumOpenEdgeKernel<
      sierra::nalu::AlgTraitsQuad4Hex8>(
      meta_, &solnOpts_, viscosity_,
      helperObjs.assembleFaceElemSolverAlg->faceDataNeeded_,
      helperObjs.assembleFaceElemSolverAlg->elemDataNeeded_));

  helperObjs.assembleFaceElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

  helperObjs.check_against_gold_values(24, hex8_golds::lhs, hex8_golds::rhs);
}
