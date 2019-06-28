/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/TurbKineticEnergyKsgsSrcElemKernel.h"

#ifndef KOKKOS_ENABLE_CUDA

namespace {
namespace hex8_golds {
namespace TurbKineticEnergyKsgsSrcElemKernel {

static constexpr double lhs[8][8] = {
  {0.44812892257697001, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0.26340357181646001, 0, 0, 0, 0, 0, 0, },
  {0, 0, 0.17225002410293999, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0.31216473910173997, 0, 0, 0, 0, },
  {0, 0, 0, 0, 0.26340357181646001, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0.15482473491488, 0, 0, },
  {0, 0, 0, 0, 0, 0, 0.097154884604998,0, },
  {0, 0, 0, 0, 0, 0, 0, 0.17225002410293999, }
};

static constexpr double rhs[8] = {
   -0.59750523010263001,
   -0.35120476242194998,
   -0.22719581319873999,
   -0.58458403812094994,
   -0.35120476242194998,
   -0.2064329798865,
   -0.090566201776693,
   -0.28427320142228002
};

} // namespace TurbKineticEnergyKsgsSrcElemKernel
} // namespace hex8_golds
} // anonymous namespace

#endif

TEST_F(KsgsKernelHex8Mesh, NGP_turb_kinetic_energy_ksgs_src_elem_kernel)
{

  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1)
    return;

  fill_mesh_and_init_fields(false,false,false);

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::TurbKineticEnergyKsgsSrcElemKernel<
      sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

#ifndef KOKKOS_ENABLE_CUDA

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::TurbKineticEnergyKsgsSrcElemKernel;
  // Why do these need 1.0e-14 while others seem to be fine with 1.0e-15 default?
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs, 1.0e-14);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, gold_values::lhs, 1.0e-14);

#endif
}


