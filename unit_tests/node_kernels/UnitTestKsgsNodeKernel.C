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

#include "node_kernels/TKEKsgsNodeKernel.h"

namespace {
namespace hex8_golds {
namespace tke_ksgs {
static constexpr double rhs[8] = {
  -0.59750523010263, -0.35120476242195, -0.58458403812095, -0.22719581319874,
  -0.35120476242195, -0.2064329798865,  -0.28427320142228, -0.090566201776693,
};

static constexpr double lhs[8][8] = {
  {
    0.44812892257697,
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
    0.26340357181646,
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
    0.31216473910174,
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
    0.17225002410294,
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
    0.26340357181646,
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
    0.15482473491488,
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
    0.17225002410294,
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
    0.097154884604998,
  },
};
} // namespace tke_ksgs
} // namespace hex8_golds
} // namespace

TEST_F(KsgsKernelHex8Mesh, NGP_tke_ksgs_node)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = false;
  const bool thirdFlag = false;
  fill_mesh_and_init_fields(doPerturb, generateSidesets, thirdFlag);

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKEKsgsNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_ksgs;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}
