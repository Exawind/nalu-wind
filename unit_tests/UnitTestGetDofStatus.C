#include <gtest/gtest.h>

#include <UnitTestHelperObjects.h>
#include <stk_mesh/base/BulkData.hpp>

#include "UnitTestUtils.h"

#include <TpetraLinearSystem.h>
#include <TpetraLinearSystemHelpers.h>

#if !defined(KOKKOS_ENABLE_GPU)

class DofStatusHex8Mesh : public Hex8Mesh
{
};

TEST_F(DofStatusHex8Mesh, getDofStatus_basic)
{
  fill_mesh_and_initialize_test_fields("generated:1x1x2");

  unit_test_utils::HelperObjects helperObjs(
    bulk, stk::topology::HEX_8, 1, partVec[0]);

  stk::mesh::Entity node1 = bulk->get_entity(stk::topology::NODE_RANK, 1);
  if (bulk->is_valid(node1)) {
    if (bulk->bucket(node1).owned()) {
      EXPECT_EQ(
        sierra::nalu::DS_OwnedDOF,
        sierra::nalu::getDofStatus_impl(node1, helperObjs.realm));
    }
  }
}

TEST_F(DofStatusHex8Mesh, getDofStatus_shared)
{
  fill_mesh_and_initialize_test_fields("generated:10x10x10");

  unit_test_utils::HelperObjects helperObjs(
    bulk, stk::topology::HEX_8, 1, partVec[0]);

  stk::mesh::Selector shared = bulk->mesh_meta_data().globally_shared_part();
  const stk::mesh::BucketVector& sharedBuckets =
    bulk->get_buckets(stk::topology::NODE_RANK, shared);

  for (const stk::mesh::Bucket* bptr : sharedBuckets) {
    const stk::mesh::Bucket& bucket = *bptr;
    for (stk::mesh::Entity node : bucket) {
      if (bucket.owned()) {
        EXPECT_EQ(
          sierra::nalu::DS_OwnedDOF,
          sierra::nalu::getDofStatus_impl(node, helperObjs.realm));
      } else {
        EXPECT_EQ(
          sierra::nalu::DS_SharedNotOwnedDOF,
          sierra::nalu::getDofStatus_impl(node, helperObjs.realm));
      }
    }
  }
}

#endif
