/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/TurbKineticEnergySSTSrcElemKernel.h"
#include "kernel/TurbKineticEnergySSTDESSrcElemKernel.h"
#include "kernel/SpecificDissipationRateSSTSrcElemKernel.h"

namespace {
namespace hex8_golds {
namespace TurbKineticEnergySSTSrcElemKernel {

static constexpr double lhs[8][8] = {
  {
    0.0069752671206081166, 0.0023250890402027055, 0.00077502968006756855,
    0.0023250890402027055, 0.0023250890402027055, 0.00077502968006756855,
    0.00025834322668918952, 0.00077502968006756855,
  },
  {
    0.0017833984706739155, 0.005350195412021746, 0.0017833984706739155,
    0.00059446615689130516, 0.00059446615689130516, 0.0017833984706739155,
    0.00059446615689130516, 0.00019815538563043504,
  },
  {
    0.00047062403152685068, 0.0014118720945805519, 0.0042356162837416562,
    0.0014118720945805519, 0.00015687467717561689, 0.00047062403152685068,
    0.0014118720945805519, 0.00047062403152685068,
  },
  {
    0.00185533509705029, 0.00061844503235009669, 0.00185533509705029,
    0.0055660052911508696, 0.00061844503235009669, 0.00020614834411669889,
    0.00061844503235009669, 0.00185533509705029,
  },
  {
    0.0018553350970502902, 0.0006184450323500968, 0.00020614834411669891,
    0.0006184450323500968, 0.0055660052911508705, 0.0018553350970502902,
    0.0006184450323500968, 0.0018553350970502902,
  },
  {
    0.00047062403152685068, 0.0014118720945805519, 0.00047062403152685068,
    0.00015687467717561689, 0.0014118720945805519, 0.0042356162837416562,
    0.0014118720945805519, 0.00047062403152685068,
  },
  {
    0.00013065390263746531, 0.00039196170791239594, 0.0011758851237371878,
    0.00039196170791239594, 0.00039196170791239594, 0.0011758851237371878,
    0.0035276553712115634, 0.0011758851237371878,
  },
  {
    0.00052603049446954737, 0.00017534349815651577, 0.00052603049446954737,
    0.0015780914834086419, 0.0015780914834086419, 0.00052603049446954737,
    0.0015780914834086419, 0.0047342744502259261,
  },
};

static constexpr double rhs[8] = {
  -0.034715637556614845, -0.020697436813089803, -0.014471024253348201,
  -0.026571541126838166, -0.026997836698597691, -0.014793264290619728,
  -0.009761402630737508, -0.020407355252450767,
};

} // namespace TurbKineticEnergySSTSrcElemKernel

namespace TurbKineticEnergySSTDESSrcElemKernel {

static constexpr double lhs[8][8] = {
  {
    0.18162464537165954, 0.060541548457219853, 0.020180516152406618,
    0.060541548457219853, 0.060541548457219853, 0.020180516152406618,
    0.0067268387174688722, 0.020180516152406618,
  },
  {
    0.045307794491460572, 0.13592338347438171, 0.045307794491460572,
    0.01510259816382019, 0.01510259816382019, 0.045307794491460572,
    0.01510259816382019, 0.0050341993879400634,
  },
  {
    0.012383713186335632, 0.037151139559006896, 0.11145341867702067,
    0.037151139559006896, 0.0041279043954452104, 0.012383713186335632,
    0.037151139559006896, 0.012383713186335632,
  },
  {
    0.050131830062481619, 0.016710610020827209, 0.050131830062481619,
    0.15039549018744486, 0.016710610020827209, 0.0055702033402757357,
    0.016710610020827209, 0.050131830062481619,
  },
  {
    0.045307794491460572, 0.01510259816382019, 0.0050341993879400634,
    0.01510259816382019, 0.13592338347438171, 0.045307794491460572,
    0.01510259816382019, 0.045307794491460572,
  },
  {
    0.01090404787216135, 0.03271214361648405, 0.01090404787216135,
    0.0036346826240537832, 0.03271214361648405, 0.098136430849452144,
    0.03271214361648405, 0.01090404787216135,
  },
  {
    0.0029833026770613251, 0.008949908031183975, 0.026849724093551925,
    0.008949908031183975, 0.008949908031183975, 0.026849724093551925,
    0.080549172280655779, 0.026849724093551925,
  },
  {
    0.012383713186335632, 0.0041279043954452104, 0.012383713186335632,
    0.037151139559006896, 0.037151139559006896, 0.012383713186335632,
    0.037151139559006896, 0.11145341867702067,
  },
};

static constexpr double rhs[8] = {
  -0.93004487921220791, -0.67850029644065035, -0.61831603722608175,
  -0.88075166717605236, -0.68371348724645631, -0.48143775480877343,
  -0.42776941006248004, -0.62144395170956546,
};

} // namespace TurbKineticEnergySSTDESSrcElemKernel

namespace SpecificDissipationRateSSTSrcElemKernel {

static constexpr double lhs[8][8] = {
  {
    0.014088352257161414, 0.0046961174190538043, 0.0015653724730179349,
    0.0046961174190538043, 0.0046961174190538043, 0.0015653724730179349,
    0.0005217908243393116, 0.0015653724730179349,
  },
  {
    0.0034334662835608129, 0.01030039885068244, 0.0034334662835608129,
    0.0011444887611869376, 0.0011444887611869376, 0.0034334662835608129,
    0.0011444887611869376, 0.00038149625372897923,
  },
  {
    0.00086754973541695217, 0.0026026492062508565, 0.00780794761875257,
    0.0026026492062508565, 0.00028918324513898408, 0.00086754973541695217,
    0.0026026492062508565, 0.00086754973541695217,
  },
  {
    0.003434284360902973, 0.0011447614536343243, 0.003434284360902973,
    0.010302853082708919, 0.0011447614536343243, 0.00038158715121144144,
    0.0011447614536343243, 0.003434284360902973,
  },
  {
    0.0040893531148636746, 0.0013631177049545581, 0.00045437256831818606,
    0.0013631177049545581, 0.012268059344591024, 0.0040893531148636746,
    0.0013631177049545581, 0.0040893531148636746,
  },
  {
    0.00092327669310721344, 0.0027698300793216404, 0.00092327669310721344,
    0.00030775889770240448, 0.0027698300793216404, 0.0083094902379649213,
    0.0027698300793216404, 0.00092327669310721344,
  },
  {
    0.00025058674982000032, 0.00075176024946000102, 0.0022552807483800031,
    0.00075176024946000102, 0.00075176024946000102, 0.0022552807483800031,
    0.0067658422451400083, 0.0022552807483800031,
  },
  {
    0.0010674069479120892, 0.00035580231597069639, 0.0010674069479120892,
    0.0032022208437362675, 0.0032022208437362675, 0.0010674069479120892,
    0.0032022208437362675, 0.0096066625312088028,
  },
};

static constexpr double rhs[8] = {
  -0.0234919792716402,    -0.015488938741822928, -0.012701346228598396,
  -0.019963104832714122,  -0.013804481873609888, -0.01064155214884124,
  -0.0097857771214982167, -0.01451935596563356,
};

} // namespace SpecificDissipationRateSSTSrcElemKernel
} // namespace hex8_golds
} // anonymous namespace

#ifndef KOKKOS_ENABLE_CUDA

TEST_F(SSTKernelHex8Mesh, turbkineticenergysstsrcelem)
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
    new sierra::nalu::TurbKineticEnergySSTSrcElemKernel<
      sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_,
      false));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::TurbKineticEnergySSTSrcElemKernel;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, gold_values::lhs);
}

TEST_F(SSTKernelHex8Mesh, turbkineticenergysstdessrcelem)
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
    new sierra::nalu::TurbKineticEnergySSTDESSrcElemKernel<
      sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_,
      false));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::TurbKineticEnergySSTDESSrcElemKernel;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, gold_values::lhs);
}

TEST_F(SSTKernelHex8Mesh, specificdissipationratesstsrcelem)
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
    new sierra::nalu::SpecificDissipationRateSSTSrcElemKernel<
      sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, helperObjs.assembleElemSolverAlg->dataNeededByKernels_,
      false));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  namespace gold_values = hex8_golds::SpecificDissipationRateSSTSrcElemKernel;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, gold_values::lhs);
}

#endif

