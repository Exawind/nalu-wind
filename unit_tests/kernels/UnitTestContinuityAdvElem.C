/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/ContinuityAdvElemKernel.h"

namespace {
namespace hex8_golds {
namespace advection_default {

static constexpr double rhs[8] = {
  0.0515834181190601, -0.1605681565653155,
  -0.0515834181190600,  0.1605681565653155,
  0.0515834181190600, -0.1605681565653155,
  -0.0515834181190601,  0.1605681565653155,
};

static constexpr double lhs[8][8] = {
  {0.421875, -0.046875, -0.078125, -0.046875, -0.046875, -0.078125, -0.046875, -0.078125},
  {-0.046875, 0.421875, -0.046875, -0.078125, -0.078125, -0.046875, -0.078125, -0.046875},
  {-0.078125, -0.046875, 0.421875, -0.046875, -0.046875, -0.078125, -0.046875, -0.078125},
  {-0.046875, -0.078125, -0.046875, 0.421875, -0.078125, -0.046875, -0.078125, -0.046875},
  {-0.046875, -0.078125, -0.046875, -0.078125, 0.421875, -0.046875, -0.078125, -0.046875},
  {-0.078125, -0.046875, -0.078125, -0.046875, -0.046875, 0.421875, -0.046875, -0.078125},
  {-0.046875, -0.078125, -0.046875, -0.078125, -0.078125, -0.046875, 0.421875, -0.046875},
  {-0.078125, -0.046875, -0.078125, -0.046875, -0.046875, -0.078125, -0.046875, 0.421875},
};
} // advection_default

namespace advection_reduced_sensitivities
{

static constexpr double rhs[8] = {
  0.0515834181190601, -0.1605681565653155,
  -0.0515834181190600,  0.1605681565653155,
  0.0515834181190600, -0.1605681565653155,
  -0.0515834181190601,  0.1605681565653155,
};

static constexpr double lhs[8][8] = {
  { 0.75, -0.25, 0, -0.25, -0.25, 0, 0, 0,  },
  { -0.25, 0.75, -0.25, 0, 0, -0.25, 0, 0,  },
  { 0, -0.25, 0.75, -0.25, 0, 0, -0.25, 0,  },
  { -0.25, 0, -0.25, 0.75, 0, 0, 0, -0.25,  },
  { -0.25, 0, 0, 0, 0.75, -0.25, 0, -0.25,  },
  { 0, -0.25, 0, 0, -0.25, 0.75, -0.25, 0,  },
  { 0, 0, -0.25, 0, 0, -0.25, 0.75, -0.25,  },
  { 0, 0, 0, -0.25, -0.25, 0, -0.25, 0.75,  },
};
} // advection_reduced_sensitivities
} // hex8_golds
} // anonymous namespace

#ifndef KOKKOS_ENABLE_CUDA

/// Continuity advection with default Solution options
TEST_F(ContinuityKernelHex8Mesh, advection_default)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.cvfemShiftMdot_ = false;
  solnOpts_.shiftedGradOpMap_["pressure"] = false;
  solnOpts_.cvfemReducedSensPoisson_ = false;
  solnOpts_.mdotInterpRhoUTogether_ = true;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.timeStepN_ = 1.0;
  timeIntegrator.timeStepNm1_ = 1.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> advKernel(
    new sierra::nalu::ContinuityAdvElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Register the kernel for execution
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(advKernel.get());

  // Populate LHS and RHS
  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::advection_default;

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, gold_values::lhs);
}

/// Continuity advection kernel
///
/// `reduced_sens_cvfem_poisson: true`
///
TEST_F(ContinuityKernelHex8Mesh, advection_reduced_sens_cvfem_poisson)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.cvfemShiftMdot_ = false;
  solnOpts_.shiftedGradOpMap_["pressure"] = false;
  solnOpts_.cvfemReducedSensPoisson_ = true;
  solnOpts_.mdotInterpRhoUTogether_ = true;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.timeStepN_ = 1.0;
  timeIntegrator.timeStepNm1_ = 1.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> advKernel(
    new sierra::nalu::ContinuityAdvElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Register the kernel for execution
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(advKernel.get());

  // Populate LHS and RHS
  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::advection_reduced_sensitivities;

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, gold_values::lhs);
}

/// Continuity advection kernel
///
/// `shift_cvfem_poisson: true`
///
TEST_F(ContinuityKernelHex8Mesh, advection_reduced_shift_cvfem_poisson)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.cvfemShiftMdot_ = false;
  solnOpts_.shiftedGradOpMap_["pressure"] = true;
  solnOpts_.cvfemReducedSensPoisson_ = true;
  solnOpts_.mdotInterpRhoUTogether_ = true;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.timeStepN_ = 1.0;
  timeIntegrator.timeStepNm1_ = 1.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> advKernel(
    new sierra::nalu::ContinuityAdvElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Register the kernel for execution
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(advKernel.get());

  // Populate LHS and RHS
  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::advection_reduced_sensitivities;

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, gold_values::lhs);
}

#endif

