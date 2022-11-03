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

#include "kernel/ScalarOpenAdvElemKernel.h"

#if !defined(KOKKOS_ENABLE_GPU)
namespace {
namespace hex8_golds {
static constexpr double rhs[8] = {
  0, -0.0023752535774385, 0.0071257607323155, 0,
  0, -0.0032819663013787, 0.0098458989041362, 0,
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
    0,
    0,
  },
  {
    0,
    0,
    0,
    0,
    0,
    0.0016409831506894,
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
    0.0049229494520681,
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
};
} // namespace hex8_golds
} // namespace
#endif

TEST_F(MixtureFractionKernelHex8Mesh, open_advection)
{
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.cvfemShiftMdot_ = true;
  solnOpts_.cvfemReducedSensPoisson_ = true;
  solnOpts_.activateOpenMdotCorrection_ = true;
  solnOpts_.alphaMap_["mixture_fraction"] = 1.0;
  solnOpts_.alphaUpwMap_["mixture_fraction"] = 1.0;
  solnOpts_.upwMap_["mixture_fraction"] = 0.0;
  solnOpts_.shiftedGradOpMap_["mixture_fraction"] = true;
  solnOpts_.skewSymmetricMap_["mixture_fraction"] = true;

  auto* part = meta_->get_part("surface_2");
  unit_test_utils::FaceElemHelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, stk::topology::HEX_8, 1, part);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.timeStepN_ = 1.0;
  timeIntegrator.timeStepNm1_ = 1.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  std::unique_ptr<sierra::nalu::Kernel> openKernel(
    new sierra::nalu::ScalarOpenAdvElemKernel<sierra::nalu::AlgTraitsQuad4Hex8>(
      *meta_, solnOpts_, &helperObjs.eqSystem, mixFraction_, mixFraction_,
      dzdx_, viscosity_, helperObjs.assembleFaceElemSolverAlg->faceDataNeeded_,
      helperObjs.assembleFaceElemSolverAlg->elemDataNeeded_));

  // Register the kernel for execution
  helperObjs.assembleFaceElemSolverAlg->activeKernels_.push_back(
    openKernel.get());

  // Populate LHS and RHS
  helperObjs.execute();

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
#endif
}
