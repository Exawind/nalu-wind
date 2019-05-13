/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/ContinuityInflowElemKernel.h"

#ifndef KOKKOS_ENABLE_CUDA
namespace {
namespace hex8_golds {

static constexpr double rhs[4] = {
  0, 0, -0.20225424859374, -0.20225424859374,
};

}
} // anonymous namespace
#endif

TEST_F(ContinuityKernelHex8Mesh, inflow)
{
  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.cvfemShiftMdot_ = true;
  solnOpts_.shiftedGradOpMap_["pressure"] = true;
  solnOpts_.cvfemReducedSensPoisson_ = true;
  solnOpts_.mdotInterpRhoUTogether_ = true;
  solnOpts_.activateOpenMdotCorrection_ = true;

  auto* part = meta_.get_part("surface_1");
  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::QUAD_4, 1, part);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.timeStepN_ = 1.0;
  timeIntegrator.timeStepNm1_ = 1.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  std::unique_ptr<sierra::nalu::Kernel> inflowKernel(
    new sierra::nalu::ContinuityInflowElemKernel<sierra::nalu::AlgTraitsQuad4>(
      bulk_, solnOpts_, true,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Register the kernel for execution
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(inflowKernel.get());

  // Populate LHS and RHS
  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 4u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 4u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 4u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<4>(helperObjs.linsys->lhs_, 0.0);
#endif
}
