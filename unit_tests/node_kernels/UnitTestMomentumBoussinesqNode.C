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

#include "node_kernels/MomentumBoussinesqNodeKernel.h"

#include <vector>

TEST_F(MomentumNodeHex8Mesh, NGP_momentum_boussinesq)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> ref_densities(0.8, 1.3);

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.gravity_.resize(spatialDim_, 0.0);
  solnOpts_.gravity_[2] = -9.81;
  solnOpts_.referenceDensity_ = ref_densities(rng);
  solnOpts_.referenceTemperature_ = 298;
  solnOpts_.thermalExpansionCoeff_ = 1.0;

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumBoussinesqNodeKernel>(
    *bulk_, solnOpts_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  double expFac =
    solnOpts_.referenceDensity_ * solnOpts_.thermalExpansionCoeff_;

  // Exact solution
  std::vector<double> rhsExact(24, 0.0);
  for (size_t i = 2; i < 24; i += 3)
    rhsExact[i] = -0.125 * solnOpts_.gravity_[2] * expFac *
                  (300.0 - solnOpts_.referenceTemperature_);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, rhsExact.data());
}
