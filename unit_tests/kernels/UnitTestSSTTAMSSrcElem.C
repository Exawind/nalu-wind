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

#include "kernel/TurbKineticEnergySSTTAMSSrcElemKernel.h"
#include "kernel/SpecificDissipationRateSSTTAMSSrcElemKernel.h"

namespace {
namespace hex8_golds {
namespace TurbKineticEnergySSTTAMSSrcElemKernel {

static constexpr double lhs[8][8] = {
  {
    0.0096663273107579,
    0.003222109103586,
    0.001074036367862,
    0.003222109103586,
    0.003222109103586,
    0.001074036367862,
    0.00035801212262066,
    0.001074036367862,
  },
  {
    0.0032087706987652,
    0.0096263120962957,
    0.0032087706987652,
    0.0010695902329217,
    0.0010695902329217,
    0.0032087706987652,
    0.0010695902329217,
    0.00035653007764058,
  },
  {
    0.0010993956987652,
    0.0032981870962957,
    0.009894561288887,
    0.0032981870962957,
    0.00036646523292174,
    0.0010993956987652,
    0.0032981870962957,
    0.0010993956987652,
  },
  {
    0.0033382023107579,
    0.001112734103586,
    0.0033382023107579,
    0.010014606932274,
    0.001112734103586,
    0.00037091136786198,
    0.001112734103586,
    0.0033382023107579,
  },
  {
    0.0033382023107579,
    0.001112734103586,
    0.00037091136786198,
    0.001112734103586,
    0.010014606932274,
    0.0033382023107579,
    0.001112734103586,
    0.0033382023107579,
  },
  {
    0.0010993956987652,
    0.0032981870962957,
    0.0010993956987652,
    0.00036646523292174,
    0.0032981870962957,
    0.009894561288887,
    0.0032981870962957,
    0.0010993956987652,
  },
  {
    0.00039627069876522,
    0.0011888120962957,
    0.003566436288887,
    0.0011888120962957,
    0.0011888120962957,
    0.003566436288887,
    0.010699308866661,
    0.003566436288887,
  },
  {
    0.0012288273107579,
    0.00040960910358595,
    0.0012288273107579,
    0.0036864819322736,
    0.0036864819322736,
    0.0012288273107579,
    0.0036864819322736,
    0.011059445796821,
  },
};

static constexpr double rhs[8] = {
  0.025446170565262, 0.026504472559968, 0.019274352239269, 0.015935520016051,
  0.024548317109716, 0.02582851482271,  0.016933151154267, 0.012713784077045,
};

} // namespace TurbKineticEnergySSTTAMSSrcElemKernel

namespace SpecificDissipationRateSSTTAMSSrcElemKernel {

static constexpr double lhs[8][8] = {
  {
    0.019523642870196,
    0.0065078809567322,
    0.0021692936522441,
    0.0065078809567322,
    0.0065078809567322,
    0.0021692936522441,
    0.00072309788408135,
    0.0021692936522441,
  },
  {
    0.006177646884336,
    0.018532940653008,
    0.006177646884336,
    0.002059215628112,
    0.002059215628112,
    0.006177646884336,
    0.002059215628112,
    0.00068640520937066,
  },
  {
    0.0020266292915131,
    0.0060798878745393,
    0.018239663623618,
    0.0060798878745393,
    0.00067554309717104,
    0.0020266292915131,
    0.0060798878745393,
    0.0020266292915131,
  },
  {
    0.0061791188058656,
    0.0020597062686219,
    0.0061791188058656,
    0.018537356417597,
    0.0020597062686219,
    0.00068656875620729,
    0.0020597062686219,
    0.0061791188058656,
  },
  {
    0.0073577479557445,
    0.0024525826519148,
    0.00081752755063828,
    0.0024525826519148,
    0.022073243867233,
    0.0073577479557445,
    0.0024525826519148,
    0.0073577479557445,
  },
  {
    0.0021568095914676,
    0.0064704287744027,
    0.0021568095914676,
    0.00071893653048919,
    0.0064704287744027,
    0.019411286323208,
    0.0064704287744027,
    0.0021568095914676,
  },
  {
    0.00076002464869353,
    0.0022800739460806,
    0.0068402218382417,
    0.0022800739460806,
    0.0022800739460806,
    0.0068402218382417,
    0.020520665514725,
    0.0068402218382417,
  },
  {
    0.0024935033673471,
    0.0008311677891157,
    0.0024935033673471,
    0.0074805101020413,
    0.0074805101020413,
    0.0024935033673471,
    0.0074805101020413,
    0.022441530306124,
  },
};

static constexpr double rhs[8] = {
  0.076960837405051, 0.075871053096668, 0.068709111007399, 0.065654571695131,
  0.086768849334085, 0.084709703584879, 0.072985400890258, 0.069151923952821,
};

} // namespace SpecificDissipationRateSSTTAMSSrcElemKernel
} // namespace hex8_golds
} // anonymous namespace

TEST_F(TAMSKernelHex8Mesh, NGP_turb_kinetic_energy_tams_sst_src_elem)
{

  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::TurbKineticEnergySSTTAMSSrcElemKernel<
      sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_,
      false));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::TurbKineticEnergySSTTAMSSrcElemKernel;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, gold_values::lhs);
}

TEST_F(TAMSKernelHex8Mesh, NGP_specific_dissipation_rate_tams_sst_src_elem)
{

  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::SpecificDissipationRateSSTTAMSSrcElemKernel<
      sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_,
      false));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::SpecificDissipationRateSSTTAMSSrcElemKernel;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, gold_values::lhs);
}

