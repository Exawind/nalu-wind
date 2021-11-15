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

#include "node_kernels/MomentumBodyForceBoxNodeKernel.h"

#include <vector>

namespace {
namespace hex8_golds {
namespace box_outside {

static constexpr double rhs[24] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                                   1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

} // namespace box_outside
} // namespace hex8_golds
} // anonymous namespace

TEST_F(MomentumNodeHex8Mesh, NGP_momentum_body_force_box_inside)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  const std::vector<double> forces{8.0, 8.0, 8.0};
  const std::vector<double> box{0.0, 0.0, 0.0, 10.0, 10.0, 10.0};

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumBodyForceBoxNodeKernel>(
    helperObjs.realm, forces, box);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, 1.0, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(
    helperObjs.linsys->lhs_, 0.0, 1.0e-12);
}

TEST_F(MomentumNodeHex8Mesh, NGP_momentum_body_force_box_outside)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  const std::vector<double> forces{8.0, 8.0, 8.0};
  const std::vector<double> box{0.0, 0.0, 0.0, 10.0, 0.5, 10.0};

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumBodyForceBoxNodeKernel>(
    helperObjs.realm, forces, box);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_(0), 8u);

  namespace gold_values = hex8_golds::box_outside;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<24>(
    helperObjs.linsys->lhs_, 0.0, 1.0e-12);
}
