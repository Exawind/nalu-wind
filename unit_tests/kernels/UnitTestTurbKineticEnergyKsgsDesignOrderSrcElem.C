/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/TurbKineticEnergyKsgsDesignOrderSrcElemKernel.h"

#ifndef KOKKOS_ENABLE_CUDA

namespace {
namespace hex8_golds {
namespace TurbKineticEnergyKsgsDesignOrderSrcElemKernel {

static constexpr double lhs[8][8] = {
  {0.14186356569842, 0.047287855232806, 0.015762618410935, 0.047287855232806, 0.047287855232806, 0.015762618410935, 0.0052542061369784, 0.015762618410935, },
  {0.036105468815983, 0.10831640644795, 0.036105468815983, 0.012035156271994, 0.012035156271994, 0.036105468815983, 0.012035156271994, 0.0040117187573315, },
  {0.0098010027279528, 0.029403008183859, 0.088209024551576, 0.029403008183859, 0.0032670009093176, 0.0098010027279528, 0.029403008183859, 0.0098010027279528, },
  {0.039065946609688, 0.013021982203229, 0.039065946609688, 0.11719783982906, 0.013021982203229, 0.0043406607344098, 0.013021982203229, 0.039065946609688, },
  {0.036105468815983, 0.012035156271994, 0.0040117187573315, 0.012035156271994, 0.10831640644795, 0.036105468815983, 0.012035156271994, 0.036105468815983, },
  {0.0092066071139421, 0.027619821341826, 0.0092066071139421, 0.0030688690379807, 0.027619821341826, 0.082859464025479, 0.027619821341826, 0.0092066071139421, },
  {0.0024701116872847, 0.0074103350618541, 0.022231005185562, 0.0074103350618541, 0.0074103350618541, 0.022231005185562, 0.066693015556687, 0.022231005185562, },
  {0.0098010027279528, 0.0032670009093176, 0.0098010027279528, 0.029403008183859, 0.029403008183859, 0.0098010027279528, 0.029403008183859, 0.088209024551576, }
};

static constexpr double rhs[8] = {
  -0.48379389534482,
  -0.35752912547567,
  -0.32180864180464,
  -0.45455303496277,
  -0.36274231628147,
  -0.2682601593739,
  -0.23191864979499,
  -0.32493655628812
};

} // namespace TurbKineticEnergyKsgsDesignOrderSrcElemKernel
} // namespace hex8_golds
} // anonymous namespace

#endif

TEST_F(KsgsKernelHex8Mesh, NGP_turb_kinetic_energy_ksgs_design_order_src_elem_kernel)
{

  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1)
    return;

  fill_mesh_and_init_fields(false,false,false);

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::TurbKineticEnergyKsgsDesignOrderSrcElemKernel<
      sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::TurbKineticEnergyKsgsDesignOrderSrcElemKernel;
  // Why do these need 1.0e-14 while others seem to be fine with 1.0e-15 default?
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs, 1.0e-14);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, gold_values::lhs, 1.0e-14);

#endif
}

