/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "node_kernels/ScalarGclNodeKernel.h"

#ifndef KOKKOS_ENABLE_CUDA
namespace {
namespace gold_values {
namespace scalar_gcl {

static constexpr double rhs[8] =
{ 0.16414997724169, 0.19275120256547, 0.19275120256547, 0.16414997724169, 0.19275120256547, 0.16414997724169, 0.16414997724169, 0.19275120256547 };

} // scalar_gcl
} // gold_values
} // anonymous namespace
#endif

TEST_F(MixtureFractionKernelHex8Mesh, NGP_scalar_gcl_node)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  helperObjs.nodeAlg->add_kernel<sierra::nalu::ScalarGclNodeKernel>(bulk_, mixFraction_);

  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::scalar_gcl::rhs, 1.0e-14);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, 0.0);
#endif

}
