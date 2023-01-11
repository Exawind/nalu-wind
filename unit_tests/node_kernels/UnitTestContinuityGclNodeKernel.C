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

#include "KokkosInterface.h"
#include "NGPInstance.h"
#include "node_kernels/ContinuityGclNodeKernel.h"

TEST_F(ContinuityKernelHex8Mesh, NGP_continuity_gcl_node)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = 0.0;
  timeIntegrator.gamma3_ = 0.0;

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  helperObjs.nodeAlg->add_kernel<sierra::nalu::ContinuityGclNodeKernel>(*bulk_);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, 9.375);
  unit_test_kernel_utils::expect_all_near<8>(helperObjs.linsys->lhs_, 0.0);
}

template <class T>
inline T*
my_create(const T& hostObj)
{
  const std::string debuggingName(typeid(T).name());
  T* obj = sierra::nalu::kokkos_malloc_on_device<T>(debuggingName);

  // Create local copy for capture on device
  const T hostCopy(hostObj);
  Kokkos::parallel_for(
    debuggingName, 1, KOKKOS_LAMBDA(const int) {
      printf("before placement new\n");
      new (obj) T();
      printf("after placement new\n");
      *obj = hostCopy;
      printf("after assignment\n");
    });
  return obj;
}

void
test_kernel_on_device(const sierra::nalu::ContinuityGclNodeKernel& kernel)
{
  sierra::nalu::ContinuityGclNodeKernel* deviceKernel = my_create(kernel);
}

TEST_F(ContinuityKernelHex8Mesh, NGP_continuity_gcl_node_kernel)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  sierra::nalu::ContinuityGclNodeKernel kernel(*bulk_);

  test_kernel_on_device(kernel);
}
