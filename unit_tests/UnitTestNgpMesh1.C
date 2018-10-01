#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include "UnitTestUtils.h"

#include <SimdInterface.h>
#include <ElemDataRequests.h>

#include <stk_ngp/Ngp.hpp>

void do_the_test(const stk::mesh::BulkData& bulk)
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

  ngp::Mesh ngpMesh(bulk);

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
          ++ngpResults(1);
        }
      }
    });
  });

  Kokkos::deep_copy(hostResults, ngpResults);
  EXPECT_EQ(numStkBuckets, hostResults(0));
  EXPECT_EQ(numStkElements, hostResults(1));
}

TEST_F(Hex8MeshWithNSOFields, ngpMesh1)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  do_the_test(bulk);
}

