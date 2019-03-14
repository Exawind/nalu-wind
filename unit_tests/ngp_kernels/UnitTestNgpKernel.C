/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"
#include "ngp_kernels/UTNgpKernelUtils.h"

#include "AlgTraits.h"

#include "stk_ngp/Ngp.hpp"

#include <memory>

using TeamType = sierra::nalu::DeviceTeamHandleType;
using ShmemType = sierra::nalu::DeviceShmem;

TEST_F(Hex8MeshWithNSOFields, NGPKernelBasic)
{
  using AlgTraitsHex8 = sierra::nalu::AlgTraitsHex8;
  using TestContinuityKernel =
    unit_test_ngp_kernels::TestContinuityKernel<AlgTraitsHex8>;

  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  unit_test_utils::HelperObjects helperObjs(bulk, stk::topology::HEX_8, 1, partVec[0]);
  auto* assembleElemSolverAlg = helperObjs.assembleElemSolverAlg;
  auto& dataNeeded = assembleElemSolverAlg->dataNeededByKernels_;

  std::unique_ptr<TestContinuityKernel> testKernel(
    new TestContinuityKernel(bulk, dataNeeded));
  assembleElemSolverAlg->activeKernels_.push_back(testKernel.get());

  EXPECT_EQ(3u, dataNeeded.get_fields().size());
  EXPECT_EQ(1u, assembleElemSolverAlg->activeKernels_.size());

  auto* ngpTestKernel = assembleElemSolverAlg->activeKernels_[0]->create_on_device();

  ASSERT_NE(ngpTestKernel, nullptr);
}

void kernel_runalg_test(
  stk::mesh::BulkData& bulk,
  sierra::nalu::AssembleElemSolverAlgorithm& solverAlg)
{
  auto* testKernel = solverAlg.activeKernels_[0]->create_on_device();

  solverAlg.run_algorithm(
    bulk,
    KOKKOS_LAMBDA(sierra::nalu::SharedMemData<TeamType, ShmemType> & smdata) {
      testKernel->execute(smdata.simdlhs, smdata.simdrhs, *smdata.prereqData[0]);
    });
}

TEST_F(Hex8MeshWithNSOFields, NGPKernelRunAlg)
{
  using AlgTraitsHex8 = sierra::nalu::AlgTraitsHex8;
  using TestContinuityKernel =
    unit_test_ngp_kernels::TestContinuityKernel<AlgTraitsHex8>;

  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  unit_test_utils::HelperObjects helperObjs(bulk, stk::topology::HEX_8, 1, partVec[0]);
  auto* assembleElemSolverAlg = helperObjs.assembleElemSolverAlg;
  auto& dataNeeded = assembleElemSolverAlg->dataNeededByKernels_;

  std::unique_ptr<TestContinuityKernel> testKernel(
    new TestContinuityKernel(bulk, dataNeeded));
  assembleElemSolverAlg->activeKernels_.push_back(testKernel.get());

  EXPECT_EQ(3u, dataNeeded.get_fields().size());
  EXPECT_EQ(1u, assembleElemSolverAlg->activeKernels_.size());

  kernel_runalg_test(bulk, *assembleElemSolverAlg);
}

TEST_F(Hex8MeshWithNSOFields, NGPKernelExecute)
{
  using AlgTraitsHex8 = sierra::nalu::AlgTraitsHex8;
  using TestContinuityKernel =
    unit_test_ngp_kernels::TestContinuityKernel<AlgTraitsHex8>;

  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  unit_test_utils::HelperObjects helperObjs(bulk, stk::topology::HEX_8, 1, partVec[0]);
  auto* assembleElemSolverAlg = helperObjs.assembleElemSolverAlg;
  auto& dataNeeded = assembleElemSolverAlg->dataNeededByKernels_;

  std::unique_ptr<TestContinuityKernel> testKernel(
    new TestContinuityKernel(bulk, dataNeeded));
  assembleElemSolverAlg->activeKernels_.push_back(testKernel.get());

  EXPECT_EQ(3u, dataNeeded.get_fields().size());
  EXPECT_EQ(1u, assembleElemSolverAlg->activeKernels_.size());

  assembleElemSolverAlg->execute();
}
