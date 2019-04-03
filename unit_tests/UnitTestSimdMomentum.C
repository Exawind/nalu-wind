#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include <SolutionOptions.h>
#include <TimeIntegrator.h>
#include <kernel/MomentumAdvDiffElemKernel.h>
#include <nso/MomentumNSOElemKernel.h>
#include <ElemDataRequests.h>

#ifndef KOKKOS_ENABLE_CUDA

TEST_F(Hex8MeshWithNSOFields, twoMomentumKernels)
{
  fill_mesh_and_initialize_test_fields("generated:20x20x20");

  sierra::nalu::SolutionOptions solnOpts;
  solnOpts.meshMotion_ = false;
  solnOpts.meshDeformation_ = false;
  solnOpts.externalMeshDeformation_ = false;
  solnOpts.includeDivU_ = 0.0;

  const unsigned numDof = 3;
  unit_test_utils::HelperObjects helperObjs(bulk, stk::topology::HEX_8, numDof, meta.get_part("block_1"));

  std::unique_ptr<sierra::nalu::Kernel> advDiffKernel(
     new sierra::nalu::MomentumAdvDiffElemKernel<sierra::nalu::AlgTraitsHex8>(
        bulk, solnOpts, velocity, viscosity,
        helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  std::unique_ptr<sierra::nalu::Kernel> nsoKernel(
     new sierra::nalu::MomentumNSOElemKernel<sierra::nalu::AlgTraitsHex8>(
        bulk, solnOpts, velocity, Gju, viscosity, 0.0, 0.0,
        helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(advDiffKernel.get());
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(nsoKernel.get());

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;

  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, all_local);
  const unsigned numElems = stk::mesh::count_selected_entities(all_local, elemBuckets);

  double startTime = stk::wall_time();

  helperObjs.assembleElemSolverAlg->execute();

  double elapsedTimeSimd = stk::wall_time() - startTime;
  std::cout<<"numElems: "<<numElems<<", elapsedTime Hex8MeshWithNSOFields.twoMomentumKernels: "<<elapsedTimeSimd<<std::endl;

  EXPECT_EQ(numElems, helperObjs.linsys->numSumIntoCalls_);
}

#endif

