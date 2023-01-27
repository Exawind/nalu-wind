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

#include "node_kernels/SDRSSTAMSNodeKernel.h"
#include "node_kernels/SDRSSTLRAMSNodeKernel.h"
#include "node_kernels/SDRKOAMSNodeKernel.h"
#include "node_kernels/TDRKEAMSNodeKernel.h"
#include "node_kernels/TKESSTAMSNodeKernel.h"
#include "node_kernels/TKESSTLRAMSNodeKernel.h"
#include "node_kernels/TKEKEAMSNodeKernel.h"
#include "node_kernels/TKEKOAMSNodeKernel.h"
#include "node_kernels/MomentumSSTAMSForcingNodeKernel.h"
#include "node_kernels/MomentumKEAMSForcingNodeKernel.h"
#include "node_kernels/MomentumKOAMSForcingNodeKernel.h"

namespace {
namespace hex8_golds {
namespace tke_sstams {
static constexpr double rhs[8] = {
  0.03, 0.03, 0.011797117626563686,  0.01930061419167952,
  0.03, 0.03, 0.0010727535418449535, 0.013845363192687317,
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
    0.0225,
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
    0.0225,
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
    0.0225,
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
    0.0225,
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
    0.0225,
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
    0.02986322059335908,
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
    0.026827992474152702,
  },
};
} // namespace tke_sstams

namespace tke_sstlrams {
static constexpr double rhs[8] = {
  0.03, 0.03, 0.011797117626563686,  0.01930061419167952,
  0.03, 0.03, 0.0010727535418449535, 0.013845363192687317,
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
    0.0225,
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
    0.0225,
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
    0.0225,
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
    0.0225,
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
    0.0225,
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
    0.02986322059335908,
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
    0.026827992474152702,
  },
};
} // namespace tke_sstlrams
namespace tke_keams {
static constexpr double rhs[8] = {
  -0.17510204081633, -0.17510204081633, -0.17514331719359, -0.17512630246215,
  -0.17510204081633, -0.17510204081633, -0.25693986461058, -0.2232051067223,
};

static constexpr double lhs[8][8] = {
  {
    5.1020408163265e-05,
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
    5.1020408163265e-05,
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
    5.1020408163265e-05,
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
    5.1020408163265e-05,
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
    5.1020408163265e-05,
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
    5.1020408163265e-05,
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
    5.1020408163265e-05,
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
    5.1020408163265e-05,
  },
};
} // namespace tke_keams
namespace tke_koams {
static constexpr double rhs[8] = {
  0.03, 0.03, 0.011797117626563686,  0.01930061419167952,
  0.03, 0.03, 0.0010727535418449535, 0.013845363192687317,
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
    0.0225,
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
    0.0225,
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
    0.0225,
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
    0.0225,
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
    0.0225,
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
    0.02986322059335908,
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
    0.026827992474152702,
  },
};
} // namespace tke_koams
namespace sdr_sstams {
static constexpr double rhs[8] = {
  0.0686,
  0.0686,
  0.0686,
  0.0686,
  0.0686,
  0.09006060639111374,
  0.037069601007020843,
  0.064388116448963514,
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
    0.0414,
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
    0.0414,
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
    0.0414,
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
    0.0414,
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
    0.038847416860968854,
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
    0.054948325891780704,
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
    0.047574531285689135,
  },
};
} // namespace sdr_sstams
namespace sdr_sstlrams {
static constexpr double rhs[8] = {
  -0.0084,
  -0.0084,
  -0.017904236132368,
  -0.014739023789053,
  -0.0084,
  -0.00093035080516258,
  -0.037544505045343,
  -0.018962231920743,
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
    0.0414,
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
    0.0414,
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
    0.0414,
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
    0.0414,
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
    0.037865654115188,
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
    0.054948325891781,
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
    0.046886464029246,
  },
};
} // namespace sdr_sstlrams
namespace tdr_keams {
static constexpr double rhs[8] = {
  -0.34878753871849, -0.34876380972278, -0.24834504322869, -0.28177185377029,
  -0.34878753871849, -0.34876380972278, -0.53192598629727, -0.45541636695634,
};

static constexpr double lhs[8][8] = {
  {
    0.45001876935924,
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
    0.45000690486139,
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
    0.32041554937241,
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
    0.36356567137431,
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
    0.45001876935924,
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
    0.45000690486139,
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
    0.48255368682701,
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
    0.47077492362076,
  },
};
} // namespace tdr_keams

namespace sdr_koams {
static constexpr double rhs[8] = {
  0.0030126084335584,  0.0030126084335584, -0.0082258869425877,
  -0.0044833434297577, 0.0030126084335584, 0.0030126084335584,
  -0.021583548165759,  -0.01036790116732,
};

static constexpr double lhs[8][8] = {
  {
    0.036,
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
    0.036,
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
    0.036,
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
    0.036,
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
    0.036,
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
    0.036,
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
    0.047781152949375,
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
    0.042924787958644,
  },
};
} // namespace sdr_koams

namespace forcing_sstams {
static constexpr double rhs[24] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  -0.01722586002226411,
  0};
} // namespace forcing_sstams
namespace forcing_keams {
static constexpr double rhs[24] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  -0.01722586002226411,
  0};
} // namespace forcing_keams
namespace forcing_koams {
static constexpr double rhs[24] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  -0.01722586002226411,
  0};
} // namespace forcing_koams
} // namespace hex8_golds
} // namespace

TEST_F(AMSKernelHex8Mesh, NGP_tke_sstams_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTAMSNodeKernel>(
    *meta_, solnOpts_.get_coordinates_name());

  helperObjs.execute();

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace hex8_golds = hex8_golds::tke_sstams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
#endif
}

TEST_F(AMSKernelHex8Mesh, NGP_tke_sstlrams_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKESSTLRAMSNodeKernel>(
    *meta_, solnOpts_.get_coordinates_name());

  helperObjs.execute();

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace hex8_golds = hex8_golds::tke_sstlrams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
#endif
}

TEST_F(AMSKernelHex8Mesh, NGP_tke_keams_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKEKEAMSNodeKernel>(
    *meta_, solnOpts_.get_coordinates_name());

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace hex8_golds = hex8_golds::tke_keams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
#endif
}

TEST_F(AMSKernelHex8Mesh, NGP_tke_koams_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKEKOAMSNodeKernel>(*meta_);

  helperObjs.execute();

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace hex8_golds = hex8_golds::tke_koams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(AMSKernelHex8Mesh, NGP_sdr_sstams_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTAMSNodeKernel>(
    *meta_, solnOpts_.get_coordinates_name());

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace hex8_golds = hex8_golds::sdr_sstams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
#endif
}
TEST_F(AMSKernelHex8Mesh, NGP_sdr_sstlrams_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRSSTLRAMSNodeKernel>(
    *meta_, solnOpts_.get_coordinates_name());

  helperObjs.execute();

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace hex8_golds = hex8_golds::sdr_sstlrams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
#endif
}
TEST_F(AMSKernelHex8Mesh, NGP_tdr_keams_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TDRKEAMSNodeKernel>(
    *meta_, solnOpts_.get_coordinates_name());

  helperObjs.execute();

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace hex8_golds = hex8_golds::tdr_keams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
#endif
}
TEST_F(AMSKernelHex8Mesh, NGP_sdr_koams_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::nalu::SDRKOAMSNodeKernel>(*meta_);

  helperObjs.execute();

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace hex8_golds = hex8_golds::sdr_koams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}

TEST_F(AMSKernelHex8Mesh, NGP_sstams_forcing)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();
  solnOpts_.eastVector_ = {1.0, 0.0, 0.0};
  solnOpts_.northVector_ = {0.0, 1.0, 0.0};

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumSSTAMSForcingNodeKernel>(
    *bulk_, solnOpts_);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.currentTime_ = 0.0;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  helperObjs.execute();

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace hex8_golds = hex8_golds::forcing_sstams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  // unit_test_kernel_utils::expect_all_near<24>(
  //   helperObjs.linsys->lhs_, 0.0, 1.0e-12);
#endif
}
TEST_F(AMSKernelHex8Mesh, NGP_keams_forcing)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();
  solnOpts_.eastVector_ = {1.0, 0.0, 0.0};
  solnOpts_.northVector_ = {0.0, 1.0, 0.0};

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumSSTAMSForcingNodeKernel>(
    *bulk_, solnOpts_);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.currentTime_ = 0.0;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  helperObjs.execute();

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace hex8_golds = hex8_golds::forcing_keams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  // unit_test_kernel_utils::expect_all_near<24>(
  //   helperObjs.linsys->lhs_, 0.0, 1.0e-12);
#endif
}

TEST_F(AMSKernelHex8Mesh, NGP_koams_forcing)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();
  solnOpts_.eastVector_ = {1.0, 0.0, 0.0};
  solnOpts_.northVector_ = {0.0, 1.0, 0.0};

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  helperObjs.nodeAlg->add_kernel<sierra::nalu::MomentumSSTAMSForcingNodeKernel>(
    *bulk_, solnOpts_);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.currentTime_ = 0.0;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace hex8_golds = hex8_golds::forcing_koams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  // unit_test_kernel_utils::expect_all_near<24>(
  //   helperObjs.linsys->lhs_, 0.0, 1.0e-12);
}
