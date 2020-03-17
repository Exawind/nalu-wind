// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include <gtest/gtest.h>

#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"
#include "ngp_kernels/UTNgpKernelUtils.h"

#include "AlgTraits.h"

#include "stk_mesh/base/Ngp.hpp"

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
  assembleElemSolverAlg->activeKernels_.clear();
  testKernel->free_on_device();
}

void kernel_runalg_test(
  stk::mesh::BulkData& bulk,
  sierra::nalu::AssembleElemSolverAlgorithm& solverAlg)
{
  auto* testKernel = solverAlg.activeKernels_[0]->create_on_device();

  solverAlg.run_algorithm(
    bulk,
    KOKKOS_LAMBDA(sierra::nalu::SharedMemData<TeamType, ShmemType> & smdata) {
      testKernel->execute(smdata.simdlhs, smdata.simdrhs, smdata.simdPrereqData);
    });

  solverAlg.activeKernels_[0]->free_on_device();
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
  assembleElemSolverAlg->activeKernels_.clear();
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

  helperObjs.execute();
}
