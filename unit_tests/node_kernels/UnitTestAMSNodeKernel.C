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
#include "node_kernels/TKESSTAMSNodeKernel.h"
#include "node_kernels/MomentumSSTAMSForcingNodeKernel.h"

#if !defined(KOKKOS_ENABLE_GPU)
namespace {
namespace hex8_golds {
namespace tke_ams {
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
} // namespace tke_ams

namespace sdr_ams {
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
} // namespace sdr_ams

namespace forcing_ams {
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
} // namespace forcing_ams
} // namespace hex8_golds
} // namespace

#endif

TEST_F(AMSKernelHex8Mesh, NGP_tke_ams_node)
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

  namespace hex8_golds = hex8_golds::tke_ams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
#endif
}

TEST_F(AMSKernelHex8Mesh, NGP_sdr_ams_node)
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

#if !defined(KOKKOS_ENABLE_GPU)
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace hex8_golds = hex8_golds::sdr_ams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
#endif
}

TEST_F(AMSKernelHex8Mesh, NGP_ams_forcing)
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

  namespace hex8_golds = hex8_golds::forcing_ams;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  // unit_test_kernel_utils::expect_all_near<24>(
  //   helperObjs.linsys->lhs_, 0.0, 1.0e-12);
#endif
}
