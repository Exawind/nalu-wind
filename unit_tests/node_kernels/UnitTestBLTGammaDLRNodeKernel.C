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

#include "node_kernels/BLTGammaDLRNodeKernel.h"
#include "node_kernels/TKESSTBLTDLRNodeKernel.h"

namespace {
namespace hex8_golds {
namespace blt_gamma_dlr {
static constexpr double rhs[8] = {
  0,
  0,
  0,
  0.31832700181246892,
  0,
  0,
  -0.064653649267864766,
  -0.0078747043112279737,
};
static constexpr double lhs_diag[8] = {
  0.013860345093887486, 0.0047886314559874143, 0.0047886314559874143,
  0.32647790271535759,  0.0081469064378713978, 0.0028146869484932352,
  0.19079800579660589,  0.039147127209395671,
};
} // namespace blt_gamma_dlr

namespace tke_sst_blt_dlr {
static constexpr double rhs[8] = {
  -0.0044999999999999997,  -0.0026450336353161293, -0.003714972216148177,
  -0.00078281868639416277, -0.0026450336353161293, -0.0015547117626563683,
  -0.017568942718499798,   0.018424353397961842,
};
static constexpr double lhs_diag[8] = {
  0.0022499999999999998,  0.0013225168176580647, 0.0013225168176580647,
  0.00077735588132818413, 0.0013225168176580647, 0.00077735588132818413,
  0.0070970479374154009,  0.0022476729243034433,
};
} // namespace tke_sst_blt_dlr
} // namespace hex8_golds

void
expect_diag_lhs(const Kokkos::View<double**>& lhs, const double (&gold)[8])
{
  double goldLhs[8][8] = {};
  for (int i = 0; i < 8; ++i)
    goldLhs[i][i] = gold[i];
  unit_test_kernel_utils::expect_all_near<8>(lhs, goldLhs, 1.0e-12);
}
} // namespace

TEST_F(SSTKernelHex8Mesh, NGP_blt_gamma_dlr_node)
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

  const double uInf = 10.0;
  helperObjs.nodeAlg->add_kernel<sierra::kynema_ugf::BLTGammaDLRNodeKernel>(
    *meta_, "velocity", uInf);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace golds = hex8_golds::blt_gamma_dlr;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, golds::rhs, 1.0e-12);
  expect_diag_lhs(helperObjs.linsys->lhs_, golds::lhs_diag);
}

TEST_F(SSTKernelHex8Mesh, NGP_tke_sst_blt_dlr_node)
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

  helperObjs.nodeAlg->add_kernel<sierra::kynema_ugf::TKESSTBLTDLRNodeKernel>(
    *meta_);

  helperObjs.execute();

  Kokkos::deep_copy(
    helperObjs.linsys->hostNumSumIntoCalls_,
    helperObjs.linsys->numSumIntoCalls_);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

  namespace golds = hex8_golds::tke_sst_blt_dlr;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, golds::rhs, 1.0e-12);
  expect_diag_lhs(helperObjs.linsys->lhs_, golds::lhs_diag);
}
