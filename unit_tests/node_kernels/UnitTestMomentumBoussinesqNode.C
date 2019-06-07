/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "node_kernels/MomentumBoussinesqNodeKernel.h"

#include <vector>

TEST_F(MomentumNodeHex8Mesh, NGP_momentum_boussinesq)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> ref_densities(0.8,1.3);
  const std::vector<double> forceVector{8.0, 8.0, 8.0};

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.gravity_.resize(spatialDim_, 0.0);
  solnOpts_.gravity_[2] = -9.81;
  solnOpts_.referenceDensity_ = ref_densities(rng);
  solnOpts_.referenceTemperature_ = 298;
  solnOpts_.thermalExpansionCoeff_ = 1.0;

  unit_test_utils::NodeHelperObjects helperObjs(bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumBoussinesqNodeKernel>(
    bulk_, forceVector, solnOpts_);

  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  double expFac = solnOpts_.referenceDensity_ * solnOpts_.thermalExpansionCoeff_;

  // Exact solution
  std::vector<double> rhsExact(24,0.0);
  for (size_t i=2; i < 24; i += 3)
    rhsExact[i] = -0.125 * solnOpts_.gravity_[2] * expFac * (300.0 - solnOpts_.referenceTemperature_);

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_,rhsExact.data());
#endif
}
