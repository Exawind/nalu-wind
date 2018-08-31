#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include <SimdInterface.h>
#include <ElemDataRequests.h>
#include <SharedMemData.h>
#include <ScratchViews.h>

#include <stk_ngp/Ngp.hpp>

TEST_F(Hex8MeshWithNSOFields, SharedMemData)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");
  stk::topology elemTopo = stk::topology::HEX_8;

  sierra::nalu::ElemDataRequests dataReq;
  auto meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo);
  dataReq.add_cvfem_volume_me(meSCV);

  dataReq.add_gathered_nodal_field(*velocity, 3);
  dataReq.add_gathered_nodal_field(*pressure, 1);

  EXPECT_EQ(2u, dataReq.get_fields().size());

  stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& elemBuckets = bulk.get_buckets(stk::topology::ELEM_RANK, all_local);

  const int numNodes = elemTopo.num_nodes();
  const int rhsSize = numNodes;
  const int lhsSize = rhsSize * rhsSize;

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::calculate_shared_mem_bytes_per_thread(lhsSize, rhsSize, rhsSize,
                                                                      meta.spatial_dimension(), dataReq);
  const bool interleaveMEViews = false;

  ngp::Mesh ngpMesh(bulk);
  int totalNumFields = meta.get_fields().size();

  auto team_exec = sierra::nalu::get_device_team_policy(elemBuckets.size(), bytes_per_team, bytes_per_thread);
  Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team)
  {
    stk::mesh::Bucket& b = *elemBuckets[team.league_rank()];

    sierra::nalu::SharedMemData smdata(team, ngpMesh, totalNumFields, dataReq, numNodes, rhsSize);

    const size_t bucketLen   = b.size();
    const size_t simdBucketLen = sierra::nalu::get_num_simd_groups(bucketLen);
 
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&](const size_t& bktIndex)
    {
      int numSimdElems = sierra::nalu::get_length_of_next_simd_group(bktIndex, bucketLen);
      smdata.numSimdElems = numSimdElems;
 
      for(int simdElemIndex=0; simdElemIndex<numSimdElems; ++simdElemIndex) {
        stk::mesh::Entity element = b[bktIndex*sierra::nalu::simdLen + simdElemIndex];
        stk::mesh::FastMeshIndex elemIndex = ngpMesh.fast_mesh_index(element);
        smdata.elemNodes[simdElemIndex] = ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex);
        sierra::nalu::fill_pre_req_data(dataReq, ngpMesh, stk::topology::ELEM_RANK, element,
                          *smdata.prereqData[simdElemIndex], interleaveMEViews);
      }

      sierra::nalu::copy_and_interleave(smdata.prereqData, numSimdElems, smdata.simdPrereqData, interleaveMEViews);
    });
  });
}

