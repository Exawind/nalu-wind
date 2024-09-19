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
#include "node_kernels/TKESSTLRNodeKernel.h"
#include "node_kernels/TKESSTDESNodeKernel.h"
#include "node_kernels/TKESSTIDDESNodeKernel.h"
#include "node_kernels/TKESSTBLTM2015NodeKernel.h"
#include "node_kernels/SDRSSTNodeKernel.h"
#include "node_kernels/SDRSSTLRNodeKernel.h"
#include "node_kernels/SDRSSTDESNodeKernel.h"
#include "node_kernels/SDRSSTBLTM2015NodeKernel.h"

namespace {
namespace hex8_golds {
namespace tke_sst {
static constexpr double rhs[8] = {
  -0.045,
  -0.026450336353161,
  -0.037149722161482,
  0.037833723714897,
  -0.026450336353161,
  -0.015547117626564,
  -0.02554123547762,
  0.044658421964382,
};

static constexpr double lhs[8][8] = {
  {
    0.0225,
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
    0.013225168176581,
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
    0.013225168176581,
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
    0.0077735588132818,
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
    0.013225168176581,
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
    0.0077735588132818,
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
    0.010317488961622,
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
    0.0054480894782752,
  },
};
} // namespace tke_sst

namespace tke_sst_sust {
static constexpr double rhs[8] = {
  2.7675000000000001,  1.6266956857194192,  1.6159962999110988,
  1.0095285753751273,  1.6266956857194192,  0.95614773403366649,
  0.94615361618261018, 0.61580632549878811,
};
} // namespace tke_sst_sust

namespace tke_sstlr {
static constexpr double rhs[8] = {
  -0.045,
  -0.026450336353161,
  -0.037149722161482,
  0.037833723714897,
  -0.026450336353161,
  -0.0081011105174071153,
  -0.02554123547762,
  0.048161314116169904,
};

static constexpr double lhs[8][8] = {
  {
    0.0225,
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
    0.013225168176581,
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
    0.013225168176581,
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
    0.0077735588132818,
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
    0.013225168176581,
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
    0.0040505552587035577,
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
    0.010317488961622,
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
    0.0039114020054672209,
  },
};
} // namespace tke_sstlr

namespace tke_sst_des {
static constexpr double rhs[8] = {
  -1.1591914445681,  -0.68135563570074, -1.1341236552933,  -0.49442751316802,
  -0.68135563570074, -0.33870892835312, -0.55150490139157, -0.20162255473912,
};

static constexpr double lhs[8][8] = {
  {
    0.86939358342608,
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
    0.51101672677556,
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
    0.60561594548791,
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
    0.33417406945958,
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
    0.51101672677556,
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
    0.25403169626484,
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
    0.33417406945958,
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
    0.17023402848587,
  },
};
} // namespace tke_sst_des

namespace tke_sst_iddes {
static constexpr double rhs[8] = {
  -0.78566371211666, -0.57734551764836, -0.081969668080736, -0.31147981334595,
  -0.20433593046136, -0.19416508662489, -0.026042361300698, -0.037056788165969};

static constexpr double lhs[8][8] = {
  {
    0.5892477840875,
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
    0.43300913823627,
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
    0.043771362853026,
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
    0.22332033598677,
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
    0.15325194784602,
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
    0.14562381496867,
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
    0.015779881252609,
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
    0.061943732501324,
  },
};
} // namespace tke_sst_iddes

namespace tke_sst_des_sust {
static constexpr double rhs[8] = {
  3.4229150667019503, 2.0119389960571135,  1.5591709764645274,
  1.088651351457735,  2.0119389960571135,  1.0001556684351871,
  1.0315739632341863, 0.63878416505705449,
};
} // namespace tke_sst_des_sust


namespace tke_sst_trans {
static constexpr double rhs[8] = {
  -0.004499999980007,
  -0.0026450336235646,
  -0.0037149722043966,
  -0.0013217844303503,
  -0.0026450336153232,
  -0.0015547117509048,
  -0.017568942314324,
  0.0073065228754648,
};

static constexpr double lhs[8][8] = {
  {
    0.00225,
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
    0.0013225168176581,
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
    0.0013225168176581,
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
    0.00077735588132818,
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
    0.0013225168176581,
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
    0.00077735588132818,
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
    0.0070970479374154,
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
    0.0022476729243034,
  },
};
} // // namespace tke_sst_trans

namespace sdr_sst {
static constexpr double rhs[8] = {
  -0.0414,
  -0.024334309444908,
  -0.024334309444908,
  0.014618955646714,
  -0.024334309444908,
  -0.013421452431681,
  -0.025196833148802,
  0.0071981412261904,
};

static constexpr double lhs[8][8] = {
  {
    0.0414,
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
    0.024334309444908,
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
    0.024334309444908,
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
    0.014303348216439,
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
    0.024334309444908,
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
    0.013421452431681,
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
    0.018984179689384,
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
    0.0096611889086056,
  },
};
} // namespace sdr_sst

namespace sdr_sst_sust {
static constexpr double rhs[8] = {
  25.833600000000001, 15.184609093622832, 15.184609093622832,
  8.9542115909208313, 15.184609093622832, 8.374986317368819,
  8.9143958021253162, 5.0713291662427222,
};
} // namespace sdr_sst_sust

namespace sdr_sstlr {
static constexpr double rhs[8] = {
  -0.0414,
  -0.024334309444908,
  -0.024334309444908,
  0.0059865036450417273,
  -0.024334309444908,
  -0.013082261745235498,
  -0.025196833148802,
  0.029552473496357173,
};

static constexpr double lhs[8][8] = {
  {
    0.0414,
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
    0.024334309444908,
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
    0.024334309444908,
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
    0.014303348216439,
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
    0.024334309444908,
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
    0.013082261745235498,
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
    0.018984179689384,
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
    0.0095214597811360858,
  },
};
} // namespace sdr_sstlr

namespace sdr_sst_des {
static constexpr double rhs[8] = {
  -0.0414,
  -0.024334309444908,
  -0.024334309444908,
  0.014618955646714,
  -0.024334309444908,
  -0.013421452431681,
  -0.025196833148802,
  0.0071981412261904,
};

static constexpr double lhs[8][8] = {
  {
    0.0414,
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
    0.024334309444908,
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
    0.024334309444908,
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
    0.014303348216439,
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
    0.024334309444908,
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
    0.013421452431681,
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
    0.018984179689384,
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
    0.0096611889086056,
  },
};
} // namespace sdr_sst_des

namespace sdr_sst_des_sust {
static constexpr double rhs[8] = {
  25.8336,
  15.184609093622832,
  15.184609093622832,
  8.9542115909208313,
  15.184609093622832,
  8.374986317368819,
  8.9143958021253162,
  5.0713291662427222,
};
} // namespace sdr_sst_des_sust

namespace sdr_sst_trans {
static constexpr double rhs[8] = {
  -0.0414,
  -0.024334309444908,
  -0.024334309444908,
  0.00096369609149651,
  -0.024334309444908,
  -0.013421452431681,
  -0.025196833148802,
  -0.0016391627843042,
};

static constexpr double lhs[8][8] = {
  {
    0.0414,
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
    0.024334309444908,
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
    0.024334309444908,
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
    0.014303348216439,
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
    0.024334309444908,
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
    0.013421452431681,
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
    0.018984179689384,
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
    0.0096611889086056,
  },
};
} // namespace sdr_sst_trans

} // namespace hex8_golds
} // namespace

TEST_F(SSTKernelHex8Mesh, NGP_tke_sst_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_sst;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_tke_sst_sust_node)
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

  sierra::nalu::Realm& realm = helperObjs.realm;
  realm.solutionOptions_->turbModelConstantMap_[sierra::nalu::TM_tkeAmb] = 5.0;
  realm.solutionOptions_->turbModelConstantMap_[sierra::nalu::TM_sdrAmb] = 50.0;

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::tke_sst_sust::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::tke_sst::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_tke_sstlr_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTLRNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_sstlr;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_tke_sst_des_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTDESNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_sst_des;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_tke_sst_iddes_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTIDDESNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_sst_iddes;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_tke_sst_des_sust_node)
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

  sierra::nalu::Realm& realm = helperObjs.realm;
  realm.solutionOptions_->turbModelConstantMap_[sierra::nalu::TM_tkeAmb] = 5.0;
  realm.solutionOptions_->turbModelConstantMap_[sierra::nalu::TM_sdrAmb] = 50.0;

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTDESNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::tke_sst_des_sust::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::tke_sst_des::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_tke_sst_trans_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTBLTM2015NodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::tke_sst_trans;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_sdr_sst_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::sdr_sst;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_sdr_sst_sust_node)
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

  sierra::nalu::Realm& realm = helperObjs.realm;
  realm.solutionOptions_->turbModelConstantMap_[sierra::nalu::TM_tkeAmb] = 5.0;
  realm.solutionOptions_->turbModelConstantMap_[sierra::nalu::TM_sdrAmb] = 50.0;

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::sdr_sst_sust::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::sdr_sst::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_sdr_sstlr_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTLRNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::sdr_sstlr;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_sdr_sst_des_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTDESNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  // only differs by a production limiting, which is never active in this case
  namespace hex8_golds = hex8_golds::sdr_sst_des;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_sdr_sst_des_sust_node)
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

  sierra::nalu::Realm& realm = helperObjs.realm;
  realm.solutionOptions_->turbModelConstantMap_[sierra::nalu::TM_tkeAmb] = 5.0;
  realm.solutionOptions_->turbModelConstantMap_[sierra::nalu::TM_sdrAmb] = 50.0;

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTDESNodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  // only differs by a production limiting, which is never active in this case
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::sdr_sst_des_sust::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::sdr_sst_des::lhs, 1.0e-12);
}

TEST_F(SSTKernelHex8Mesh, NGP_sdr_sst_trans_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTBLTM2015NodeKernel>(*meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace hex8_golds = hex8_golds::sdr_sst_trans;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}
