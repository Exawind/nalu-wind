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

#include "node_kernels/TKESSTNodeKernel.h"
#include "node_kernels/TKESSTDESNodeKernel.h"
#include "node_kernels/SDRSSTNodeKernel.h"
#include "node_kernels/SDRSSTDESNodeKernel.h"

namespace {
namespace hex8_golds {
namespace tke_sst {
static constexpr double rhs[8] = {
  -0.045, -0.026450336353161, -0.037149722161482,
  0.037833723714897, -0.026450336353161, -0.015547117626564,
  -0.02554123547762, 0.044658421964382, };

static constexpr double lhs[8][8] = {
  {0.0225, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0.013225168176581, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0.013225168176581, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0.0077735588132818, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0.013225168176581, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0.0077735588132818, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0.010317488961622, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0.0054480894782752, },
};
} // tke_sst

namespace tke_sst_des {
static constexpr double rhs[8] = {
  -1.1591914445681, -0.68135563570074,
  -1.1341236552933, -0.49442751316802,
  -0.68135563570074, -0.33870892835312,
  -0.55150490139157, -0.20162255473912,
};

static constexpr double lhs[8][8] = {
  {0.86939358342608, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0.51101672677556, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0.60561594548791, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0.33417406945958, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0.51101672677556, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0.25403169626484, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0.33417406945958, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0.17023402848587, },
};
} // tke_sst_des

namespace sdr_sst {
static constexpr double rhs[8] = {
  -0.0414, -0.024334309444908,
  -0.024334309444908, 0.014618955646714,
  -0.024334309444908, -0.013421452431681,
  -0.025196833148802, 0.0071981412261904,
};

static constexpr double lhs[8][8] = {
  {0.0414, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0.024334309444908, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0.024334309444908, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0.014303348216439, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0.024334309444908, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0.013421452431681, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0.018984179689384, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0.0096611889086056, },
};
} // sdr_sst


namespace sdr_sst_des {
static constexpr double rhs[8] = {
  -0.0414, -0.024334309444908,
  -0.024334309444908, 0.014618955646714,
  -0.024334309444908, -0.013421452431681,
  -0.025196833148802, 0.0071981412261904,
};

static constexpr double lhs[8][8] = {
  {0.0414, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0.024334309444908, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0.024334309444908, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0.014303348216439, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0.024334309444908, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0.013421452431681, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0.018984179689384, 0, },
  {0, 0, 0, 0, 0, 0, 0, 0.0096611889086056, },
};
} // sdr_sst_des

} // hex8_golds
}

TEST_F(SSTKernelHex8Mesh, NGP_tke_sst_node)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTNodeKernel>(meta_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_sst;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_tke_sst_des_node)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTDESNodeKernel>(meta_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_sst_des;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_sdr_sst_node)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTNodeKernel>(meta_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::sdr_sst;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_sdr_sst_des_node)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTDESNodeKernel>(meta_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->numSumIntoCalls_(0), 8u);

  // only differs by a production limiting, which is never active in this case
  namespace hex8_golds = hex8_golds::sdr_sst_des;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}
