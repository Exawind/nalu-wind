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

#include "kernel/MomentumSSTTAMSForcingElemKernel.h"

namespace {
namespace hex8_golds {
namespace TAMS_SST_forcing {
static constexpr double rhs[24] = {0, -0.0024747113019887, 0, 0, 0.016986957760511, 0, 0, 0, 0, 0, -0.00230673068825, 0, 0, 0, 0, 0, 0, 0, 0, -0.0091926025251865, 0, 0, 0, 0, };
} // namespace TAMS_SST_forcing
} // namespace hex8_golds
} // anonymous namespace

TEST_F(TAMSKernelHex8Mesh, NGP_TAMS_SST_forcing)
{
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.includeDivU_ = 0.0;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::MomentumSSTTAMSForcingElemKernel<
      sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, visc_, tvisc_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.currentTime_ = 1.0;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  // Populate LHS and RHS
  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  
  namespace gold_values = ::hex8_golds::TAMS_SST_forcing;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs);
}
