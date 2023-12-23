#include "gtest/gtest.h"

#include "stk_util/environment/WallTime.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

#include "UnitTestUtils.h"
#include "UnitTestRealm.h"

#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "ElemDataRequests.h"
#include "FieldManager.h"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Types.hpp"

void
test_ngp_mesh_1(
  const stk::mesh::BulkData& bulk, const stk::mesh::NgpMesh& ngpMesh)
{
  stk::topology elemTopo = stk::topology::HEX_8;

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  stk::mesh::Selector all_local =
    meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& elemBuckets =
    bulk.get_buckets(stk::topology::ELEM_RANK, all_local);
  unsigned numStkBuckets = elemBuckets.size();
  unsigned numStkElements = 0;
  for (const stk::mesh::Bucket* b : elemBuckets) {
    numStkElements += b->size();
  }
  unsigned expectedNodesPerElem = elemTopo.num_nodes();

  Kokkos::View<unsigned*, sierra::nalu::MemSpace> ngpResults("ngpResults", 2);
  Kokkos::View<unsigned*, sierra::nalu::MemSpace>::HostMirror hostResults =
    Kokkos::create_mirror_view(ngpResults);
  Kokkos::deep_copy(ngpResults, hostResults);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  auto team_exec = sierra::nalu::get_device_team_policy(
    elemBuckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
      const stk::mesh::NgpMesh::BucketType& b =
        ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());
      ++ngpResults(0);

      const size_t bucketLen = b.size();
      const size_t simdBucketLen = sierra::nalu::get_num_simd_groups(bucketLen);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, simdBucketLen),
        [&](const size_t& bktIndex) {
          int numSimdElems =
            sierra::nalu::get_length_of_next_simd_group(bktIndex, bucketLen);

          for (int simdElemIndex = 0; simdElemIndex < numSimdElems;
               ++simdElemIndex) {
            stk::mesh::Entity element =
              b[bktIndex * sierra::nalu::simdLen + simdElemIndex];
            stk::mesh::FastMeshIndex elemIndex =
              ngpMesh.fast_mesh_index(element);
            if (
              ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex).size() ==
              expectedNodesPerElem) {
              unsigned one = 1;
              Kokkos::atomic_add(&ngpResults(1), one);
            }
          }
        });
    });

  Kokkos::deep_copy(hostResults, ngpResults);
  EXPECT_EQ(numStkBuckets, hostResults(0));
  EXPECT_EQ(numStkElements, hostResults(1));
}

TEST(NgpMesh, NGPMesh)
{
  const std::string meshSpec("generated:2x2x2");

  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());

  test_ngp_mesh_1(realm.bulk_data(), realm.ngp_mesh());
}

void
test_ngp_mesh_field_values(
  sierra::nalu::FieldManager& fieldManager,
  const stk::mesh::BulkData& bulk,
  sierra::nalu::VectorFieldType* velocity,
  sierra::nalu::GenericFieldType* massFlowRate)
{
  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  stk::mesh::Selector all_local =
    meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& elemBuckets =
    bulk.get_buckets(stk::topology::ELEM_RANK, all_local);

  auto ngpVelocity =
    fieldManager.get_device_smart_field<double, tags::READ_WRITE>("velocity");
  auto ngpMassFlowRate =
    fieldManager.get_device_smart_field<double, tags::READ_WRITE>(
      "mass_flow_rate_scs");
  stk::mesh::NgpMesh ngpMesh(bulk);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  auto team_exec = sierra::nalu::get_device_team_policy(
    elemBuckets.size(), bytes_per_team, bytes_per_thread);

  const double xVel = 1.0;
  const double yVel = 2.0;
  const double zVel = 3.0;
  const double flowRate = 4.0;

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
      const stk::mesh::NgpMesh::BucketType& b =
        ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

      const size_t bucketLen = b.size();
      const size_t simdBucketLen = sierra::nalu::get_num_simd_groups(bucketLen);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, simdBucketLen),
        [&](const size_t& bktIndex) {
          int numSimdElems =
            sierra::nalu::get_length_of_next_simd_group(bktIndex, bucketLen);

          for (int simdElemIndex = 0; simdElemIndex < numSimdElems;
               ++simdElemIndex) {
            stk::mesh::Entity element =
              b[bktIndex * sierra::nalu::simdLen + simdElemIndex];
            stk::mesh::FastMeshIndex elemIndex =
              ngpMesh.fast_mesh_index(element);
            ngpMassFlowRate.get(elemIndex, 0) = flowRate;

            stk::mesh::NgpMesh::ConnectedNodes nodes =
              ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex);
            for (unsigned n = 0; n < nodes.size(); ++n) {
              ngpVelocity.get(ngpMesh, nodes[n], 0) = xVel;
              ngpVelocity.get(ngpMesh, nodes[n], 1) = yVel;
              ngpVelocity.get(ngpMesh, nodes[n], 2) = zVel;
            }
          }
        });
    });

  auto flowRateData = fieldManager.get_legacy_smart_field<double, tags::READ>(
    "mass_flow_rate_scs");
  auto velocityData =
    fieldManager.get_legacy_smart_field<double, tags::READ>("velocity");
  const double tol = 1.0e-16;
  for (const stk::mesh::Bucket* b : elemBuckets) {
    for (stk::mesh::Entity elem : *b) {
      EXPECT_NEAR(flowRate, *flowRateData(elem), tol);

      const stk::mesh::Entity* nodes = bulk.begin_nodes(elem);
      const unsigned numNodes = bulk.num_nodes(elem);
      for (unsigned n = 0; n < numNodes; ++n) {
        EXPECT_NEAR(xVel, velocityData(nodes[n])[0], tol);
        EXPECT_NEAR(yVel, velocityData(nodes[n])[1], tol);
        EXPECT_NEAR(zVel, velocityData(nodes[n])[2], tol);
      }
    }
  }
}

TEST_F(Hex8MeshWithNSOFields, NGPMeshField)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  test_ngp_mesh_field_values(*fieldManager, *bulk, velocity, massFlowRate);
}

struct TestKernelWithNgpField
{
  KOKKOS_FUNCTION TestKernelWithNgpField() : ngpField(), num(0)
  {
    printf("TestKernelWithNgpField def ctor\n");
  }

  KOKKOS_FUNCTION TestKernelWithNgpField(const TestKernelWithNgpField& src)
    : ngpField(src.ngpField), num(src.num)
  {
    printf("TestKernelWithNgpField copy ctor\n");
  }

  KOKKOS_FUNCTION ~TestKernelWithNgpField()
  {
    printf("TestKernelWithNgpField dtor\n");
  }

  KOKKOS_FUNCTION unsigned get_num() const /*override*/ { return num; }

  stk::mesh::NgpField<double> ngpField;
  unsigned num = 0;
};

void
test_ngp_field_placement_new()
{
  TestKernelWithNgpField hostObj;
  hostObj.num = 42;

  printf(
    "sizeof(TestKernelWithNgpField): %lu, sizeof(NgpField): %lu\n",
    sizeof(TestKernelWithNgpField), sizeof(stk::mesh::NgpField<double>));
  std::string debugName("TestKernelWithNgpField");
  TestKernelWithNgpField* devicePtr = static_cast<TestKernelWithNgpField*>(
    Kokkos::kokkos_malloc<stk::ngp::MemSpace>(
      debugName, sizeof(TestKernelWithNgpField)));

  int constructionFinished = 0;
  Kokkos::parallel_reduce(
    sierra::nalu::DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const unsigned& i, int& localFinished) {
      new (devicePtr) TestKernelWithNgpField(hostObj);
      localFinished = 1;
    },
    constructionFinished);
  EXPECT_EQ(1, constructionFinished);

  int numFromDevice = 0;
  Kokkos::parallel_reduce(
    sierra::nalu::DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(const unsigned& i, int& localNum) {
      localNum = devicePtr->get_num();
    },
    numFromDevice);
  EXPECT_EQ(42, numFromDevice);
}

TEST(DevicePlacementNew, NGP_structWithNgpField)
{
  test_ngp_field_placement_new();
}
