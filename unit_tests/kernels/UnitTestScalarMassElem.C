/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/ScalarMassElemKernel.h"

#ifndef KOKKOS_ENABLE_CUDA

namespace {
namespace hex8_golds {
namespace scalar_mass {

  static constexpr double lhs[8][8] = {
{0.0015342836535592, 0.00051142788451972, 0.00017047596150657, 0.00051142788451972, 0.00051142788451972, 0.00017047596150657, 5.6825320502191e-05, 0.00017047596150657},
{-0.0035176239720477, -0.010552871916143, -0.0035176239720477, -0.0011725413240159, -0.0011725413240159, -0.0035176239720477, -0.0011725413240159, -0.0003908471080053},
{0.00028941835051033, 0.00086825505153098, 0.0026047651545929, 0.00086825505153098, 9.6472783503442e-05, 0.00028941835051033, 0.00086825505153098, 0.00028941835051033},
{-0.0036180084493555, -0.0012060028164518, -0.0036180084493555, -0.010854025348067, -0.0012060028164518, -0.00040200093881728, -0.0012060028164518, -0.0036180084493555},
{-0.0029631933450284, -0.00098773111500948, -0.00032924370500316, -0.00098773111500948, -0.0088895800350853, -0.0029631933450284, -0.00098773111500948, -0.0029631933450284},
{0.00038043765414573, 0.0011413129624372, 0.00038043765414573, 0.00012681255138191, 0.0011413129624372, 0.0034239388873116, 0.0011413129624372, 0.00038043765414573},
{-0.0003583555634414, -0.0010750666903242, -0.0032252000709726, -0.0010750666903242, -0.0010750666903242, -0.0032252000709726, -0.0096756002129179, -0.0032252000709726},
{0.00034184546463258, 0.00011394848821086, 0.00034184546463258, 0.0010255363938977, 0.0010255363938977, 0.00034184546463258, 0.0010255363938977, 0.0030766091816932}
  };


  static constexpr double rhs[8] = {
-0.00091589254110655, -0.0062063621624234, -0.0012984190416109, -0.0055026555161997, -0.005904879831756, -0.0015624935573955, -0.0068051842105984, -0.00142282082153};

} // advection_diffusion
} // hex8_golds
} // anonymous namespace

#endif

/// Scalar advection/diffusion (will use mixture fraction as scalar)
TEST_F(MixtureFractionKernelHex8Mesh, NGP_scalar_mass)
{
  // FIXME: only test on one core
  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1) 
    return;

  fill_mesh_and_init_fields(true);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  int numDof = 1;
  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, numDof, partVec_[0]);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;

  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> massKernel(
    new sierra::nalu::ScalarMassElemKernel<sierra::nalu::AlgTraitsHex8>(
     bulk_, solnOpts_, mixFraction_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_, false));

  // Register the kernel for execution
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(massKernel.get());

  // Populate LHS and RHS
  helperObjs.assembleElemSolverAlg->execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::scalar_mass;
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, gold_values::lhs);
#endif
}

