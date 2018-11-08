#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include <SimdInterface.h>
#include <ElemDataRequestsGPU.h>
#include <ScratchViewsNGP.h>
#include <kernel/Kernel.h>

#include <stk_ngp/Ngp.hpp>

using TeamType = sierra::nalu::DeviceTeamHandleType;
using ShmemType = sierra::nalu::DeviceShmem;

class TestKernel
{
  public:
    TestKernel(unsigned vOrdinal, unsigned pOrdinal)
     : velocityOrdinal(vOrdinal), pressureOrdinal(pOrdinal)
   {}

  KOKKOS_FUNCTION
  void execute(
    sierra::nalu::SharedMemView<double**,ShmemType> &lhs,
    sierra::nalu::SharedMemView<double*,ShmemType> &rhs,
    sierra::nalu::ScratchViewsNGP<double> &scratchViews) const
  {
    printf("TestKernel::execute!!\n");
  }

private:
  unsigned velocityOrdinal;
  unsigned pressureOrdinal;
};

typedef Kokkos::DualView<int*, Kokkos::LayoutRight, sierra::nalu::DeviceSpace> IntViewType;

void do_the_test(stk::mesh::BulkData& bulk, sierra::nalu::ScalarFieldType* pressure, sierra::nalu::VectorFieldType* velocity)
{
  stk::topology elemTopo = stk::topology::HEX_8;
  sierra::nalu::ElemDataRequests dataReq;
  auto meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo);
  dataReq.add_cvfem_volume_me(meSCV);

  dataReq.add_gathered_nodal_field(*velocity, 3);
  dataReq.add_gathered_nodal_field(*pressure, 1);

  EXPECT_EQ(2u, dataReq.get_fields().size());

  sierra::nalu::ElemDataRequestsGPU dataNGP(dataReq);

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();

  const int numNodes = elemTopo.num_nodes();
  const int rhsSize = numNodes;
  const int lhsSize = rhsSize * rhsSize;

  const unsigned velocityOrdinal = velocity->mesh_meta_data_ordinal();
  const unsigned pressureOrdinal = pressure->mesh_meta_data_ordinal();

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::calculate_shared_mem_bytes_per_thread(lhsSize, rhsSize, rhsSize,
                                                                      meta.spatial_dimension(), dataNGP);
  const bool interleaveMEViews = false;

  int numResults = 5;
  IntViewType result("result", numResults);

  Kokkos::deep_copy(result.h_view, 0);
  result.template modify<typename IntViewType::host_mirror_space>();
  result.template sync<typename IntViewType::execution_space>();

  ngp::Mesh ngpMesh(bulk);
  int totalNumFields = meta.get_fields().size();

  TestKernel testKernel(velocityOrdinal, pressureOrdinal);

  int threadsPerTeam = 1;
  auto team_exec = sierra::nalu::get_device_team_policy(ngpMesh.num_buckets(stk::topology::ELEM_RANK), bytes_per_team, bytes_per_thread, threadsPerTeam);

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    const ngp::Mesh::BucketType& b = ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

    sierra::nalu::ScratchViewsNGP<double> scrviews(team, ngpMesh.get_spatial_dimension(), totalNumFields, numNodes, dataNGP);

    sierra::nalu::SharedMemView<double**,ShmemType> simdlhs =
        sierra::nalu::get_shmem_view_2D<double,TeamType,ShmemType>(team, rhsSize, rhsSize);
    sierra::nalu::SharedMemView<double*,ShmemType> simdrhs =
        sierra::nalu::get_shmem_view_1D<double,TeamType,ShmemType>(team, rhsSize);

    NGP_ThrowAssert(scrviews.total_bytes() != 0);
    const size_t bucketLen   = b.size();
 
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bucketLen), [&](const size_t& bktIndex)
    {
        stk::mesh::Entity element = b[bktIndex];
        sierra::nalu::fill_pre_req_data(dataNGP, ngpMesh, stk::topology::ELEM_RANK, element,
                          scrviews, interleaveMEViews);
        auto& velocityView = scrviews.get_scratch_view_2D(velocityOrdinal);
        auto& pressureView = scrviews.get_scratch_view_1D(pressureOrdinal);

        result.d_view(0) = (pressureView(0) - 1.0) < 1.e-9 ? 1 : 0;
        result.d_view(1) = (pressureView(7) - 1.0) < 1.e-9 ? 1 : 0;
        result.d_view(2) = (velocityView(6,0) - 1.0) < 1.e-9 ? 1 : 0;
        result.d_view(3) = (velocityView(6,1) - 0.0) < 1.e-9 ? 1 : 0;
        result.d_view(4) = (velocityView(6,2) - 0.0) < 1.e-9 ? 1 : 0;

        testKernel.execute(simdlhs, simdrhs, scrviews);
    });
  });

  result.modify<IntViewType::execution_space>();
  result.sync<IntViewType::host_mirror_space>();

  for(int i=0; i<numResults; ++i) {
    EXPECT_EQ(1, result.h_view(i));
  }
}

TEST_F(Hex8MeshWithNSOFields, ScratchViews)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  do_the_test(bulk, pressure, velocity);
}

