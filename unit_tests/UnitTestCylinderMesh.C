#include "gtest/gtest.h"

#include "stk_util/environment/WallTime.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

#include "UnitTestUtils.h"
#include "UnitTestRealm.h"

#include "KokkosInterface.h"
#include "ElemDataRequests.h"

void
test_cylinder_mesh_field_values(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::FieldBase* coordField,
  const stk::mesh::FieldBase* testField)
{
  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  stk::mesh::Selector all_local =
    meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& nodeBuckets =
    bulk.get_buckets(stk::topology::NODE_RANK, all_local);

  stk::mesh::NgpMesh ngpMesh(bulk);
  stk::mesh::NgpField<double>& ngpCoordField =
    stk::mesh::get_updated_ngp_field<double>(*coordField);
  stk::mesh::NgpField<double>& ngpTestField =
    stk::mesh::get_updated_ngp_field<double>(*testField);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  auto team_exec = sierra::nalu::get_device_team_policy(
    nodeBuckets.size(), bytes_per_team, bytes_per_thread);

  const double xDelta = 0.1;
  const double yDelta = 0.2;
  const double zDelta = 0.3;

  ngpCoordField.sync_to_device();

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
      const stk::mesh::NgpMesh::BucketType& bkt =
        ngpMesh.get_bucket(stk::topology::NODE_RANK, team.league_rank());

      const size_t bucketLen = bkt.size();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, bucketLen), [&](const size_t& bktIndex) {
          stk::mesh::Entity node = bkt[bktIndex];
          stk::mesh::FastMeshIndex nodeIndex = ngpMesh.fast_mesh_index(node);

          ngpTestField(nodeIndex, 0) = ngpCoordField(nodeIndex, 0) + xDelta;
          ngpTestField(nodeIndex, 1) = ngpCoordField(nodeIndex, 1) + yDelta;
          ngpTestField(nodeIndex, 2) = ngpCoordField(nodeIndex, 2) + zDelta;
        });
    });

  ngpTestField.modify_on_device();
  ngpTestField.sync_to_host();

  const double tol = 1.0e-12;
  for (const stk::mesh::Bucket* bkt : nodeBuckets) {
    for (stk::mesh::Entity node : *bkt) {
      const double* coordFieldData = reinterpret_cast<const double*>(
        stk::mesh::field_data(*coordField, node));
      const double* testFieldData = reinterpret_cast<const double*>(
        stk::mesh::field_data(*testField, node));
      EXPECT_NEAR(testFieldData[0], (coordFieldData[0] + xDelta), tol);
      EXPECT_NEAR(testFieldData[1], (coordFieldData[1] + yDelta), tol);
      EXPECT_NEAR(testFieldData[2], (coordFieldData[2] + zDelta), tol);
    }
  }
}

TEST_F(CylinderMesh, basic_setup)
{
  const double innerRadius = 1.0;
  const double outerRadius = 2.0;
  fill_mesh_and_initialize_test_fields(20, 20, 20, innerRadius, outerRadius);

  test_cylinder_mesh_field_values(*bulk, coordField, testField);

  const bool dumpTheMesh = true;
  if (dumpTheMesh) {
    unit_test_utils::dump_mesh(*bulk, {});
  }
}
