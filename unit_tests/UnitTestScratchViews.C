#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include <SimdInterface.h>
#include <ElemDataRequestsGPU.h>
#include <ScratchViews.h>
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
    sierra::nalu::SharedMemView<double**,ShmemType> & /* lhs */,
    sierra::nalu::SharedMemView<double*,ShmemType> & /* rhs */,
    sierra::nalu::ScratchViews<sierra::nalu::DoubleType,TeamType,ShmemType> & /* scratchViews */) const
  {
    printf("TestKernel::execute!!\n");
  }

  KOKKOS_FUNCTION
  void execute(
    sierra::nalu::SharedMemView<double**,ShmemType> & /* lhs */,
    sierra::nalu::SharedMemView<double*,ShmemType> & /* rhs */,
    sierra::nalu::ScratchViews<double,TeamType,ShmemType> & /* scratchViews */) const
  {
    printf("TestKernel::execute!!\n");
  }

  void execute(
    sierra::nalu::SharedMemView<sierra::nalu::DoubleType**,ShmemType> & /* lhs */,
    sierra::nalu::SharedMemView<sierra::nalu::DoubleType*,ShmemType> & rhs,
    sierra::nalu::ScratchViews<double,TeamType,ShmemType> &  scratchViews) const
  {
    auto& v_vel = scratchViews.get_scratch_view_2D(velocityOrdinal);
    auto& v_pres = scratchViews.get_scratch_view_1D(pressureOrdinal);

    rhs(0) += v_vel(0, 0) + v_pres(0);
  }

private:
  unsigned velocityOrdinal;
  unsigned pressureOrdinal;
};

typedef Kokkos::DualView<int*, Kokkos::LayoutRight, sierra::nalu::DeviceSpace> IntViewType;

void do_the_test(stk::mesh::BulkData& bulk, sierra::nalu::ScalarFieldType* pressure, sierra::nalu::VectorFieldType* velocity)
{
  stk::topology elemTopo = stk::topology::HEX_8;
  sierra::nalu::ElemDataRequests dataReq(bulk.mesh_meta_data());
  auto meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo);
  dataReq.add_cvfem_volume_me(meSCV);

  dataReq.add_gathered_nodal_field(*velocity, 3);
  dataReq.add_gathered_nodal_field(*pressure, 1);

  EXPECT_EQ(2u, dataReq.get_fields().size());

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  ngp::FieldManager fieldMgr(bulk);

  sierra::nalu::ElemDataRequestsGPU dataNGP(fieldMgr, dataReq, meta.get_fields().size());

  const int numNodes = elemTopo.num_nodes();
  const int rhsSize = numNodes;
  const int lhsSize = rhsSize * rhsSize;

  const unsigned velocityOrdinal = velocity->mesh_meta_data_ordinal();
  const unsigned pressureOrdinal = pressure->mesh_meta_data_ordinal();

  const int bytes_per_team = 0;
  const int bytes_per_thread =
    (sierra::nalu::calculate_shared_mem_bytes_per_thread(
       lhsSize, rhsSize, rhsSize, meta.spatial_dimension(), dataNGP) +
     (rhsSize + lhsSize) * sizeof(double) * sierra::nalu::simdLen);

  const bool interleaveMEViews = false;

  int numResults = 5;
  IntViewType result("result", numResults);

  Kokkos::deep_copy(result.h_view, 0);
  result.template modify<typename IntViewType::host_mirror_space>();
  result.template sync<typename IntViewType::execution_space>();

  ngp::Mesh ngpMesh(bulk);

  TestKernel testKernel(velocityOrdinal, pressureOrdinal);

  int threads_per_team = 1;
  auto team_exec = sierra::nalu::get_device_team_policy(ngpMesh.num_buckets(stk::topology::ELEM_RANK), bytes_per_team, bytes_per_thread, threads_per_team);

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    const ngp::Mesh::BucketType& b = ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

    sierra::nalu::ScratchViews<double,TeamType,ShmemType> scrviews(team, ngpMesh.get_spatial_dimension(), numNodes, dataNGP);

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

TEST_F(Hex8MeshWithNSOFields, NGPScratchViews)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  do_the_test(bulk, pressure, velocity);
}

void do_assemble_elem_solver_test(
  stk::mesh::BulkData& bulk,
  sierra::nalu::AssembleElemSolverAlgorithm& solverAlg,
  unsigned velocityOrdinal,
  unsigned pressureOrdinal)
{
  TestKernel testKernel(velocityOrdinal, pressureOrdinal);

  solverAlg.run_algorithm(
    bulk,
    KOKKOS_LAMBDA(sierra::nalu::SharedMemData<TeamType, ShmemType> & smdata) {
       testKernel.execute(smdata.simdlhs, smdata.simdrhs, *smdata.prereqData[0]);
    });
}

TEST_F(Hex8MeshWithNSOFields, NGPAssembleElemSolver)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  unit_test_utils::HelperObjects helperObjs(bulk, stk::topology::HEX_8, 1, partVec[0]);
  auto* assembleElemSolverAlg = helperObjs.assembleElemSolverAlg;
  auto& dataNeeded = assembleElemSolverAlg->dataNeededByKernels_;

  auto meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(stk::topology::HEX_8);
  dataNeeded.add_cvfem_volume_me(meSCV);
  dataNeeded.add_gathered_nodal_field(*velocity, 3);
  dataNeeded.add_gathered_nodal_field(*pressure, 1);

  EXPECT_EQ(2u, dataNeeded.get_fields().size());

  do_assemble_elem_solver_test(
    bulk, *assembleElemSolverAlg, velocity->mesh_meta_data_ordinal(),
    pressure->mesh_meta_data_ordinal());
}

void do_the_smdata_test(stk::mesh::BulkData& bulk, sierra::nalu::ScalarFieldType* pressure, sierra::nalu::VectorFieldType* velocity)
{
  stk::topology elemTopo = stk::topology::HEX_8;
  sierra::nalu::ElemDataRequests dataReq(bulk.mesh_meta_data());
  auto meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo);
  dataReq.add_cvfem_volume_me(meSCV);

  dataReq.add_gathered_nodal_field(*velocity, 3);
  dataReq.add_gathered_nodal_field(*pressure, 1);

  EXPECT_EQ(2u, dataReq.get_fields().size());

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  ngp::FieldManager fieldMgr(bulk);

  sierra::nalu::ElemDataRequestsGPU dataNGP(fieldMgr, dataReq, meta.get_fields().size());

  const int numNodes = elemTopo.num_nodes();
  const int rhsSize = numNodes;
  const int lhsSize = rhsSize * rhsSize;

  const unsigned velocityOrdinal = velocity->mesh_meta_data_ordinal();
  const unsigned pressureOrdinal = pressure->mesh_meta_data_ordinal();

  const int bytes_per_team = 0;
  const int bytes_per_thread =
    (sierra::nalu::calculate_shared_mem_bytes_per_thread(
       lhsSize, rhsSize, rhsSize, meta.spatial_dimension(), dataNGP) +
     (rhsSize + lhsSize) * sizeof(double) * sierra::nalu::simdLen);

  const bool interleaveMEViews = false;

  int numResults = 5;
  IntViewType result("result", numResults);

  Kokkos::deep_copy(result.h_view, 0);
  result.template modify<typename IntViewType::host_mirror_space>();
  result.template sync<typename IntViewType::execution_space>();

  ngp::Mesh ngpMesh(bulk);

  TestKernel testKernel(velocityOrdinal, pressureOrdinal);

  int threads_per_team = 1;
  auto team_exec = sierra::nalu::get_device_team_policy(ngpMesh.num_buckets(stk::topology::ELEM_RANK), bytes_per_team, bytes_per_thread, threads_per_team);

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team)
  {
    const ngp::Mesh::BucketType& b = ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

    sierra::nalu::SharedMemData<TeamType,ShmemType> smdata(team, ngpMesh.get_spatial_dimension(), dataNGP, numNodes, rhsSize);

    NGP_ThrowAssert(smdata.simdPrereqData.total_bytes() != 0);
    const size_t bucketLen   = b.size();
 
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bucketLen), [&](const size_t& bktIndex)
    {
        stk::mesh::Entity element = b[bktIndex];
        sierra::nalu::fill_pre_req_data(dataNGP, ngpMesh, stk::topology::ELEM_RANK, element,
                          *smdata.prereqData[0], interleaveMEViews);

#ifndef KOKKOS_ENABLE_CUDA
//no copy-interleave needed on GPU since no simd.
        sierra::nalu::copy_and_interleave(smdata.prereqData, 1, smdata.simdPrereqData, interleaveMEViews);
#endif

        auto& velocityView = smdata.prereqData[0]->get_scratch_view_2D(velocityOrdinal);
        auto& pressureView = smdata.prereqData[0]->get_scratch_view_1D(pressureOrdinal);

        result.d_view(0) = (pressureView(0) - 1.0) < 1.e-9 ? 1 : 0;
        result.d_view(1) = (pressureView(7) - 1.0) < 1.e-9 ? 1 : 0;
        result.d_view(2) = (velocityView(6,0) - 1.0) < 1.e-9 ? 1 : 0;
        result.d_view(3) = (velocityView(6,1) - 0.0) < 1.e-9 ? 1 : 0;
        result.d_view(4) = (velocityView(6,2) - 0.0) < 1.e-9 ? 1 : 0;

        testKernel.execute(smdata.lhs, smdata.rhs, smdata.simdPrereqData);
    });
  });

  result.modify<IntViewType::execution_space>();
  result.sync<IntViewType::host_mirror_space>();

  for(int i=0; i<numResults; ++i) {
    EXPECT_EQ(1, result.h_view(i));
  }
}

TEST_F(Hex8MeshWithNSOFields, NGPSharedMemData)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  do_the_smdata_test(bulk, pressure, velocity);
}

#ifdef KOKKOS_ENABLE_CUDA

using DeviceSpace = Kokkos::Cuda;
using DeviceShmem = DeviceSpace::scratch_memory_space;
using DynamicScheduleType = Kokkos::Schedule<Kokkos::Dynamic>;
using DeviceTeamHandleType = Kokkos::TeamPolicy<DeviceSpace, DynamicScheduleType>::member_type;
template <typename T, typename SHMEM>
using SharedMemView = Kokkos::View<T, Kokkos::LayoutRight, SHMEM, Kokkos::MemoryUnmanaged>;

void do_kokkos_test()
{

  const int bytes_per_team = 2048;
  const int bytes_per_thread = 2048;
  int threads_per_team = 1;
  unsigned N = 1;

  Kokkos::TeamPolicy<DeviceSpace> team_exec(N, threads_per_team);

  Kokkos::parallel_for(
    team_exec.set_scratch_size(
      1, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread)),
    KOKKOS_LAMBDA(const DeviceTeamHandleType& team)
    {
      int len = 50;
      SharedMemView<double*,DeviceShmem> shview =
        Kokkos::subview(SharedMemView<double**,DeviceShmem>(
                          team.team_scratch(1), team.team_size(), len),
                        team.team_rank(), Kokkos::ALL());
    });
}

TEST_F(Hex8MeshWithNSOFields, NGPteamSize)
{
  do_kokkos_test();
}

#endif
