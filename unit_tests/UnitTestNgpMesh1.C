#include "gtest/gtest.h"

#include "stk_util/environment/WallTime.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/GetEntities.hpp"

#include "UnitTestUtils.h"
#include "UnitTestRealm.h"

#include "SimdInterface.h"
#include "ElemDataRequests.h"

#include "stk_ngp/Ngp.hpp"

void test_ngp_mesh_1(const stk::mesh::BulkData& bulk, ngp::Mesh& ngpMesh)
{
  stk::topology elemTopo = stk::topology::HEX_8;

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, all_local);
  unsigned numStkBuckets = elemBuckets.size();
  unsigned numStkElements = 0;
  for(const stk::mesh::Bucket* b : elemBuckets) {
    numStkElements += b->size();
  }
  unsigned expectedNodesPerElem = elemTopo.num_nodes();

  Kokkos::View<unsigned*,sierra::nalu::MemSpace> ngpResults("ngpResults", 2);
  Kokkos::View<unsigned*,sierra::nalu::MemSpace>::HostMirror hostResults = Kokkos::create_mirror_view(ngpResults);
  Kokkos::deep_copy(ngpResults, hostResults);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  auto team_exec = sierra::nalu::get_device_team_policy(elemBuckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    const ngp::Mesh::BucketType& b = ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());
    ++ngpResults(0);

    const size_t bucketLen   = b.size();
    const size_t simdBucketLen = sierra::nalu::get_num_simd_groups(bucketLen);
 
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&](const size_t& bktIndex)
    {
      int numSimdElems = sierra::nalu::get_length_of_next_simd_group(bktIndex, bucketLen);
 
      for(int simdElemIndex=0; simdElemIndex<numSimdElems; ++simdElemIndex) {
        stk::mesh::Entity element = b[bktIndex*sierra::nalu::simdLen + simdElemIndex];
        stk::mesh::FastMeshIndex elemIndex = ngpMesh.fast_mesh_index(element);
        if (ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex).size() == expectedNodesPerElem) {
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

void test_ngp_mesh_field_values(const stk::mesh::BulkData& bulk, 
                                VectorFieldType* velocity, 
                                GenericFieldType* massFlowRate)
{
  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, all_local);

  ngp::Mesh ngpMesh(bulk);
  ngp::Field<double> ngpVelocity(bulk, *velocity);
  ngp::Field<double> ngpMassFlowRate(bulk, *massFlowRate);

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  auto team_exec = sierra::nalu::get_device_team_policy(elemBuckets.size(), bytes_per_team, bytes_per_thread);

  const double xVel = 1.0;
  const double yVel = 2.0;
  const double zVel = 3.0;
  const double flowRate = 4.0;

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    const ngp::Mesh::BucketType& b = ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

    const size_t bucketLen   = b.size();
    const size_t simdBucketLen = sierra::nalu::get_num_simd_groups(bucketLen);
 
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&](const size_t& bktIndex)
    {
      int numSimdElems = sierra::nalu::get_length_of_next_simd_group(bktIndex, bucketLen);
 
      for(int simdElemIndex=0; simdElemIndex<numSimdElems; ++simdElemIndex) {
        stk::mesh::Entity element = b[bktIndex*sierra::nalu::simdLen + simdElemIndex];
        stk::mesh::FastMeshIndex elemIndex = ngpMesh.fast_mesh_index(element);
        ngpMassFlowRate.get(elemIndex, 0) = flowRate;

        ngp::Mesh::ConnectedNodes nodes = ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex);
        for (unsigned n = 0; n < nodes.size(); ++n)
        {
          ngpVelocity.get(ngpMesh, nodes[n], 0) = xVel;
          ngpVelocity.get(ngpMesh, nodes[n], 1) = yVel;
          ngpVelocity.get(ngpMesh, nodes[n], 2) = zVel;
        }
      }
    });
  });

  ngpVelocity.modify_on_device();
  ngpMassFlowRate.modify_on_device();
  ngpVelocity.copy_device_to_host();
  ngpMassFlowRate.copy_device_to_host();

  const double tol = 1.0e-16;
  for (const stk::mesh::Bucket* b : elemBuckets)
  {
    for (stk::mesh::Entity elem : *b)
    {
      const double* flowRateData = stk::mesh::field_data(*massFlowRate, elem);
      EXPECT_NEAR(flowRate, *flowRateData, tol);

      const stk::mesh::Entity* nodes = bulk.begin_nodes(elem);
      const unsigned numNodes = bulk.num_nodes(elem);
      for (unsigned n = 0; n < numNodes; ++n)
      {
        const double* velocityData = stk::mesh::field_data(*velocity, nodes[n]);
        EXPECT_NEAR(xVel, velocityData[0], tol);
        EXPECT_NEAR(yVel, velocityData[1], tol);
        EXPECT_NEAR(zVel, velocityData[2], tol);
      }
    }
  }
}

TEST_F(Hex8MeshWithNSOFields, NGPMeshField)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  test_ngp_mesh_field_values(bulk, velocity, massFlowRate);
}
