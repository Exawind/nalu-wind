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

#include "node_kernels/TKEKENodeKernel.h"
#include "node_kernels/TDRKENodeKernel.h"

namespace {
namespace hex8_golds {
namespace tke_ke {
static constexpr double rhs[8] = {
  -0.251,
  -0.14719631307311828,
  -0.14835082157030577,
  -0.029604928511851453,
  -0.14794631307311829,
  -0.086622875703131569,
  -0.1158765303693139,
  -0.0037418778749914866,
};

static constexpr double lhs[8][8] = {
  {
    0.0005,
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
    0.000125,
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
    0.0005,
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
    0.000125,
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
    0.0005,
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
    0.000125,
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
    0.0005,
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
    0.000125,
  },
};
} // namespace tke_ke

namespace tdr_ke {
static constexpr double rhs[8] = {
  -0.44415022703895984,  -0.24202453432573701,  -0.18769279143193071,
  -0.050569760373841061, -0.2423585799460993,   -0.13070571477449358,
  -0.19014590536955761,  -0.011309382975531565,
};

static constexpr double lhs[8][8] = {
  {
    0.44396628731837412,
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
    0.24200761741533242,
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
    0.18750885171134499,
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
    0.11280579803843291,
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
    0.24217464022551358,
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
    0.13068879786408899,
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
    0.1430786684579321,
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
    0.0770742118579643,
  },
};
} // namespace tdr_ke
} // namespace hex8_golds
} // namespace

TEST_F(KEKernelHex8Mesh, NGP_tke_ke_node)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKEKENodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_ke;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(KEKernelHex8Mesh, NGP_tdr_ke_node)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TDRKENodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tdr_ke;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}
