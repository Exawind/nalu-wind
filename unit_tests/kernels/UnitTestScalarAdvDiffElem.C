/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/ScalarAdvDiffElemKernel.h"

namespace {
namespace hex8_golds {
namespace advection_diffusion {

  static constexpr double lhs[8][8] = {
    { -0.0000613097250829, 0.0000429113567611, -0.0000257498107039, -0.0001402489548682, -0.0000205431634595, 0.0000140298269036, -0.0000086190271279, -0.0000466184930962,  },
    { -0.0002996386301440, -0.0006885423861745, -0.0006419807726150, -0.0002372547140239, -0.0000965507982442, -0.0002202991073109, -0.0002106385750205, -0.0000778808756134,  },
    { 0.0000163445978461, 0.0004292680425440, 0.0000703036269008, -0.0003567523231394, 0.0000059221861145, 0.0001444645345796, 0.0000286358120453, -0.0001171369110324,  },
    { 0.0003661661425579, 0.0002465304462856, 0.0006269741312273, 0.0007608107866930, 0.0001236406680179, 0.0000826726357635, 0.0002107059133024, 0.0002590406827259,  },
    { -0.0000145009311411, -0.0000287787747695, -0.0000042761980475, 0.0000110494905244, -0.0000232066315811, -0.0000801386088521, -0.0000105582144686, 0.0000405272824199,  },
    { 0.0000447114954398, 0.0000695614483399, 0.0000582132611368, 0.0000265359602858, 0.0001508190513621, 0.0002568672455374, 0.0001914489515611, 0.0000854188029756,  },
    { 0.0000018344412760, -0.0000379149517718, -0.0000026902615610, 0.0000422809376121, 0.0000094438399738, -0.0001021350294235, 0.0000296803441588, 0.0001396382449965,  },
    { -0.0000536073907518, -0.0000330351812149, -0.0000807939763374, -0.0001064211830839, -0.0001495251521836, -0.0000954614971975, -0.0002306552044507, -0.0002829887333757,  },
  };


  static constexpr double rhs[8] = {
    -0.0000023386569023, -0.0000259997889658, 0.0000269085331417, 0.0000040683119211, 0.0000089806982989, 0.0000318761782412, 0.0000172325914019, -0.0000607278671367, };



} // advection_diffusion
} // hex8_golds
} // anonymous namespace

/// Scalar advection/diffusion (will use mixture fraction as scalar)
TEST_F(MixtureFractionKernelHex8Mesh, NGP_advection_diffusion)
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

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> advKernel(
    new sierra::nalu::ScalarAdvDiffElemKernel<sierra::nalu::AlgTraitsHex8>(
     bulk_, solnOpts_, mixFraction_, viscosity_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Register the kernel for execution
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(advKernel.get());

  // Populate LHS and RHS
  helperObjs.assembleElemSolverAlg->execute();

#ifndef KOKKOS_ENABLE_CUDA
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::advection_diffusion;
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, gold_values::lhs);
#endif
}

