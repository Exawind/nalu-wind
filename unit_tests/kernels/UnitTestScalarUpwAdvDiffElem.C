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

#include "kernel/ScalarUpwAdvDiffElemKernel.h"

namespace {
namespace hex8_golds {
namespace advection_diffusion {

static constexpr double lhs[8][8] = {
  {1.3875e-05, 0.00039642012463318, 0, -0.00040567012463318, -4.625e-06, 0, 0, 0, },
  {-0.00040567012463318, -0.0015903054985327, -0.0012077603738995, 0, 0, -4.625e-06, 0, 0, },
  {0, 0.0011985103738995, 1.3875e-05, -0.0012077603738995, 0, 0, -4.625e-06, 0, },
  {0.00039642012463318, 0, 0.0011985103738995, 0.0016180554985327, 0, 0, 0, -4.625e-06, },
  {-4.625e-06, 0, 0, 0, 1.3875e-05, -9.9507056249943e-05, 0, 9.0257056249943e-05, },
  {0, -4.625e-06, 0, 0, 9.0257056249943e-05, 0.00039340322499977, 0.00028002116874983, 0, },
  {0, 0, -4.625e-06, 0, 0, -0.00028927116874983, 1.3875e-05, 0.00028002116874983, },
  {0, 0, 0, -4.625e-06, -9.9507056249943e-05, 0, -0.00028927116874983, -0.00036565322499977, },
};

static constexpr double rhs[8] = {
  -5.55e-05, 5.55e-05,  -5.55e-05, 5.55e-05,
  5.55e-05, -5.55e-05, 5.55e-05,  -5.55e-05,
};

} // advection_diffusion
} // hex8_golds
} // anonymous namespace

TEST_F(MixtureFractionKernelHex8Mesh, NGP_upw_advection_diffusion)
{
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields(false);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.alphaMap_["mixture_fraction"] = 1.0;
  solnOpts_.alphaUpwMap_["mixture_fraction"] = 1.0;
  solnOpts_.upwMap_["mixture_fraction"] = 0.0;
  solnOpts_.shiftedGradOpMap_["mixture_fraction"] = true;
  solnOpts_.skewSymmetricMap_["mixture_fraction"] = true;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> advKernel(
    new sierra::nalu::ScalarUpwAdvDiffElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, &helperObjs.eqSystem, mixFraction_, dzdx_, viscosity_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Register the kernel for execution
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(advKernel.get());

  // Populate LHS and RHS
  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::advection_diffusion;
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, gold_values::lhs);
}
