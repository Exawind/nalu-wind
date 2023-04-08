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

#include "node_kernels/TKEKONodeKernel.h"
#include "node_kernels/SDRKONodeKernel.h"

namespace {
namespace hex8_golds {
namespace tke_ko {
static constexpr double rhs[8] = {
  -0.016368777801313281,  -0.00740039033812396, -0.011709654168113211,
  0.051874097062954767,   -0.00740039033812396, -0.00417063108566584,
  -0.0068417453675951607, 0.053763691195545707,
};

static constexpr double lhs[8][8] = {
  {
    0.0081843889006566403,
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
    0.00370019516906198,
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
    0.0041685949894791578,
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
    0.0021018912401700412,
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
    0.00370019516906198,
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
    0.00208531554283292,
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
    0.0027637516740426134,
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
    0.0014536892633176806,
  },
};
} // namespace tke_ko

namespace sdr_ko {
static constexpr double rhs[8] = {
  -0.035999999999999997, -0.021160269082529031, -0.021160269082529031,
  0.029003272915464909,  -0.021160269082529031, -0.012437694101250944,
  -0.021910289694610025, 0.053914090927629096,
};

static constexpr double lhs[8][8] = {
  {
    0.035999999999999997,
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
    0.021160269082529031,
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
    0.021160269082529031,
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
    0.012437694101250944,
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
    0.021160269082529031,
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
    0.012437694101250944,
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
    0.016507982338594577,
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
    0.0087169431652403921,
  },
};
} // namespace sdr_ko
} // namespace hex8_golds
} // namespace

TEST_F(KOKernelHex8Mesh, NGP_tke_ko_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKEKONodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_ko;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(KOKernelHex8Mesh, NGP_sdr_ko_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRKONodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::sdr_ko;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}
