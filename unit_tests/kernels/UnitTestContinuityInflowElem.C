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

#include "kernel/ContinuityInflowElemKernel.h"

namespace {
namespace hex8_golds {

static constexpr double rhs[4] = {0, 0.11888206453689, 0.11888206453689, 0};
}
} // anonymous namespace

TEST_F(ContinuityKernelHex8Mesh, NGP_inflow)
{
  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.cvfemShiftMdot_ = true;
  solnOpts_.shiftedGradOpMap_["pressure"] = true;
  solnOpts_.cvfemReducedSensPoisson_ = true;
  solnOpts_.mdotInterpRhoUTogether_ = true;
  solnOpts_.activateOpenMdotCorrection_ = true;

  auto* part = meta_->get_part("surface_2");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 1, part);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.timeStepN_ = 1.0;
  timeIntegrator.timeStepNm1_ = 1.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  std::unique_ptr<sierra::nalu::Kernel> inflowKernel(
    new sierra::nalu::ContinuityInflowElemKernel<sierra::nalu::AlgTraitsQuad4>(
      *bulk_, solnOpts_, true,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Register the kernel for execution
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(
    inflowKernel.get());

  // Populate LHS and RHS
  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 4u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 4u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 4u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<4>(helperObjs.linsys->lhs_, 0.0);
}
