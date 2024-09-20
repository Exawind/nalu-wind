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

#include "node_kernels/BLTGammaM2015NodeKernel.h"

namespace {
namespace hex8_golds {
namespace blt_gamma {
static constexpr double rhs[8] = {
  0, 0, 0, 0.12823826807242, 0, 0, -0.043287170452808, 0.43035793150449,
};

static constexpr double lhs[8][8] = {
  {
    1.5336779974765e-19,
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
    4.6113433001562e-05,
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
    6.4413186123242e-11,
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
    0.13131235277404,
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
    7.8452858117332e-05,
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
    0.001645541661088,
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
    0.12774384574574,
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
    0.7648927575084,
  },
};
} // namespace blt_gamma

} // namespace hex8_golds
} // namespace

TEST_F(SSTKernelHex8Mesh, NGP_blt_gamma_node)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::BLTGammaM2015NodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::blt_gamma;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}
