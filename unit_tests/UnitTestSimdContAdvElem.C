#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <stk_simd/Simd.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <UnitTestHelperObjects.h>

#include "UnitTestUtils.h"

#include <SolutionOptions.h>
#include <TimeIntegrator.h>
#include <kernel/ContinuityAdvElemKernel.h>
#include <ElemDataRequests.h>
#include <ScratchViews.h>

#ifndef KOKKOS_ENABLE_CUDA

TEST_F(Hex8MeshWithNSOFields, continuityAdvElem)
{
  fill_mesh_and_initialize_test_fields("generated:10x10x10");

  sierra::nalu::SolutionOptions solnOpts;
  solnOpts.meshMotion_ = false;
  solnOpts.meshDeformation_ = false;
  solnOpts.externalMeshDeformation_ = false;
  solnOpts.cvfemShiftMdot_ = false;
  solnOpts.shiftedGradOpMap_["pressure"] = false;
  solnOpts.cvfemReducedSensPoisson_ = false;
  solnOpts.mdotInterpRhoUTogether_ = true;

  unit_test_utils::HelperObjects helperObjs(bulk, stk::topology::HEX_8, 1, partVec[0]);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.timeStepN_ = 1.0;
  timeIntegrator.timeStepNm1_ = 1.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  std::unique_ptr<sierra::nalu::Kernel> advKernel(
    new sierra::nalu::ContinuityAdvElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk, solnOpts, helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(advKernel.get());

  double startTime = stk::wall_time();

  helperObjs.assembleElemSolverAlg->execute();

  double elapsedTimeSimd = stk::wall_time() - startTime;

  stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, all_local);
  const unsigned numElems = stk::mesh::count_selected_entities(all_local, elemBuckets);

  std::cerr<<"numElems: "<<numElems<<", elapsedTime Hex8MeshWithNSOFields.continuityAdvElem: "<<elapsedTimeSimd<<std::endl;

  EXPECT_EQ(numElems, helperObjs.linsys->numSumIntoCalls_);
}

#endif

