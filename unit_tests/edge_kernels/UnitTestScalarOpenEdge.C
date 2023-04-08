// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <random>

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "edge_kernels/ScalarEdgeOpenSolverAlg.h"
#include "edge_kernels/ScalarOpenEdgeKernel.h"

namespace {
namespace hex8_golds {
static constexpr double rhs[8] = {0, -20, -24.755282581476, 0,
                                  0, -20, -22.795084971875, 0};

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
    10,
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
    10,
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
    10,
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
    10,
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

TEST_F(SSTKernelHex8Mesh, NGP_scalar_edge_open_solver_alg)
{
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;

  auto* part = meta_->get_part("surface_2");
  unit_test_utils::FaceElemHelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, stk::topology::HEX_8, 1, part);

  std::unique_ptr<
    sierra::nalu::ScalarEdgeOpenSolverAlg<sierra::nalu::AlgTraitsQuad4Hex8>>
    kernel(new sierra::nalu::ScalarEdgeOpenSolverAlg<
           sierra::nalu::AlgTraitsQuad4Hex8>(
      *meta_, solnOpts_, tke_, tkebc_, dkdx_, tvisc_,
      helperObjs.assembleFaceElemSolverAlg->faceDataNeeded_,
      helperObjs.assembleFaceElemSolverAlg->elemDataNeeded_));

  helperObjs.assembleFaceElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_scalar_open_edge)
{
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> rand_num(1.0, 2.0);
  std::uniform_real_distribution<double> rand_mdot(-1.0, 1.0);

  const double tkeVal = rand_num(rng);
  const double bcTkeVal = rand_num(rng);
  const double mdotVal = rand_mdot(rng);

  stk::mesh::field_fill(tkeVal, *tke_);
  stk::mesh::field_fill(bcTkeVal, *tkebc_);
  stk::mesh::field_fill(mdotVal, *openMassFlowRate_);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.relaxFactorMap_["turbulent_ke"] = 0.5;

  auto* part = meta_->get_part("surface_5");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 1, part);

  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::ScalarOpenEdgeKernel<sierra::nalu::AlgTraitsQuad4>(
      *meta_, solnOpts_, tke_, tkebc_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 4u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 4u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 4u);

  const double expectedRhs =
    (mdotVal > 0.0) ? (-mdotVal * tkeVal) : (-mdotVal * bcTkeVal);
  const double expectedLhs = (mdotVal > 0.0) ? (mdotVal * 2.0) : 0.0;

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, expectedRhs, 1.0e-12);

  Kokkos::deep_copy(helperObjs.linsys->hostlhs_, helperObjs.linsys->lhs_);
  const auto& lhs = helperObjs.linsys->hostlhs_;

  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(lhs(i, i), expectedLhs, 1.0e-12);
}
