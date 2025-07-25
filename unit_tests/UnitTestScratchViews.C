#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include <master_element/MasterElementRepo.h>
#include <SimdInterface.h>
#include <ElemDataRequestsGPU.h>
#include <ScratchViews.h>
#include <kernel/Kernel.h>
#include <ngp_utils/NgpFieldManager.h>

#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpMesh.hpp>

using TeamType = sierra::nalu::DeviceTeamHandleType;
using ShmemType = sierra::nalu::DeviceShmem;

class TestKernel
{
public:
  TestKernel(unsigned vOrdinal, unsigned pOrdinal)
    : velocityOrdinal(vOrdinal), pressureOrdinal(pOrdinal)
  {
  }

  KOKKOS_FUNCTION
  void execute(
    sierra::nalu::SharedMemView<double**, ShmemType>& /* lhs */,
    sierra::nalu::SharedMemView<double*, ShmemType>& /* rhs */,
    sierra::nalu::
      ScratchViews<DoubleType, TeamType, ShmemType>& /* scratchViews */) const
  {
  }

  KOKKOS_FUNCTION
  void execute(
    sierra::nalu::SharedMemView<DoubleType**, ShmemType>& /* lhs */,
    sierra::nalu::SharedMemView<DoubleType*, ShmemType>& rhs,
    sierra::nalu::ScratchViews<DoubleType, TeamType, ShmemType>& scratchViews)
    const
  {
    auto& v_vel = scratchViews.get_scratch_view_2D(velocityOrdinal);
    auto& v_pres = scratchViews.get_scratch_view_1D(pressureOrdinal);

    rhs(0) += v_vel(0, 0) + v_pres(0);
  }

private:
  unsigned velocityOrdinal;
  unsigned pressureOrdinal;
};

typedef Kokkos::DualView<int*, Kokkos::LayoutRight, sierra::nalu::DeviceSpace>
  IntViewType;
typedef Kokkos::
  DualView<double*, Kokkos::LayoutRight, sierra::nalu::DeviceSpace>
    DoubleTypeView;

void
do_the_test(
  stk::mesh::BulkData& bulk,
  const sierra::nalu::FieldManager& fieldManager,
  sierra::nalu::ScalarFieldType* pressure,
  sierra::nalu::VectorFieldType* velocity)
{
  stk::topology elemTopo = stk::topology::HEX_8;
  sierra::nalu::ElemDataRequests dataReq(bulk.mesh_meta_data());
  const int nodesPerElement = sierra::nalu::AlgTraitsHex8::nodesPerElement_;
  auto meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element_on_dev(
      sierra::nalu::AlgTraitsHex8::topo_);
  dataReq.add_cvfem_volume_me(meSCV);

  auto* coordsField = bulk.mesh_meta_data().coordinate_field();

  dataReq.add_coordinates_field(
    *coordsField, 3, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_gathered_nodal_field(*velocity, 3);
  dataReq.add_gathered_nodal_field(*pressure, 1);
  dataReq.add_master_element_call(
    sierra::nalu::SCV_VOLUME, sierra::nalu::CURRENT_COORDINATES);

  EXPECT_EQ(3u, dataReq.get_fields().size());

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();

  sierra::nalu::ElemDataRequestsGPU dataNGP(fieldManager, dataReq);

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

  int numResults = 7;
  IntViewType result("result", numResults);

  Kokkos::deep_copy(result.h_view, 0);
  result.template modify<typename IntViewType::host_mirror_space>();
  result.template sync<typename IntViewType::execution_space>();

  stk::mesh::NgpMesh ngpMesh(bulk);

  TestKernel testKernel(velocityOrdinal, pressureOrdinal);

  int threads_per_team = 1;
  auto team_exec = sierra::nalu::get_device_team_policy(
    ngpMesh.num_buckets(stk::topology::ELEM_RANK), bytes_per_team,
    bytes_per_thread, threads_per_team);

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
      const stk::mesh::NgpMesh::BucketType& b =
        ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

      sierra::nalu::ScratchViews<DoubleType, TeamType, ShmemType> scrviews(
        team, ngpMesh.get_spatial_dimension(), nodesPerElement, dataNGP);

      sierra::nalu::SharedMemView<DoubleType**, ShmemType> simdlhs =
        sierra::nalu::get_shmem_view_2D<DoubleType, TeamType, ShmemType>(
          team, rhsSize, rhsSize);
      sierra::nalu::SharedMemView<DoubleType*, ShmemType> simdrhs =
        sierra::nalu::get_shmem_view_1D<DoubleType, TeamType, ShmemType>(
          team, rhsSize);
      for (int i = 0; i < rhsSize; i++) {
        simdrhs(i) = 0.0;
      }

      STK_NGP_ThrowAssert(scrviews.total_bytes() != 0);
      const size_t bucketLen = b.size();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, bucketLen), [&](const size_t& bktIndex) {
          stk::mesh::Entity element = b[bktIndex];
          sierra::nalu::fill_pre_req_data(
            dataNGP, ngpMesh, stk::topology::ELEM_RANK, element, scrviews);
          auto& velocityView = scrviews.get_scratch_view_2D(velocityOrdinal);
          auto& pressureView = scrviews.get_scratch_view_1D(pressureOrdinal);
          result.d_view(0) =
            stk::simd::get_data((pressureView(0) - 1.0), 0) < 1.e-9 ? 1 : 0;
          result.d_view(1) =
            stk::simd::get_data((pressureView(7) - 1.0), 0) < 1.e-9 ? 1 : 0;
          result.d_view(2) =
            stk::simd::get_data((velocityView(6, 0) - 1.0), 0) < 1.e-9 ? 1 : 0;
          result.d_view(3) =
            stk::simd::get_data((velocityView(6, 1) - 0.0), 0) < 1.e-9 ? 1 : 0;
          result.d_view(4) =
            stk::simd::get_data((velocityView(6, 2) - 0.0), 0) < 1.e-9 ? 1 : 0;

          testKernel.execute(simdlhs, simdrhs, scrviews);
        });

      result.d_view(5) =
        stk::simd::get_data((simdrhs(0) - 16.0), 0) < 1.e-9 ? 1 : 0;
      result.d_view(6) =
        stk::simd::get_data((simdrhs(1) - 0.0), 0) < 1.e-9 ? 1 : 0;
    });

  result.modify<IntViewType::execution_space>();
  result.sync<IntViewType::host_mirror_space>();

  for (int i = 0; i < numResults; ++i) {
    EXPECT_EQ(1, result.h_view(i));
  }
}

TEST_F(Hex8MeshWithNSOFields, NGPScratchViews)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  const int nDim = 3;
  const double velVec[nDim] = {1.0, 0.0, 0.0};

  const int numStates = 2;
  stk::mesh::MetaData& meta = bulk->mesh_meta_data();
  const sierra::nalu::FieldManager fieldManager(meta, numStates);

  stk::mesh::EntityVector nodes;
  stk::mesh::get_entities(*bulk, stk::topology::NODE_RANK, nodes);

  for (stk::mesh::Entity node : nodes) {
    double* fieldData =
      static_cast<double*>(stk::mesh::field_data(*velocity, node));
    for (int d = 0; d < nDim; ++d) {
      fieldData[d] = velVec[d];
    }
  }

  do_the_test(*bulk, fieldManager, pressure, velocity);
}

void
do_assemble_elem_solver_test(
  stk::mesh::BulkData& bulk,
  sierra::nalu::AssembleElemSolverAlgorithm& solverAlg,
  unsigned velocityOrdinal,
  unsigned pressureOrdinal)
{
  TestKernel testKernel(velocityOrdinal, pressureOrdinal);

  solverAlg.run_algorithm(
    bulk,
    KOKKOS_LAMBDA(sierra::nalu::SharedMemData<TeamType, ShmemType> & smdata) {
      auto& scv_volume =
        smdata.simdPrereqData.get_me_views(sierra::nalu::CURRENT_COORDINATES)
          .scv_volume;

      printf(
        "SCV volume = %f; expected = 0.125\n",
        stk::simd::get_data(scv_volume(4), 0));
      testKernel.execute(smdata.simdlhs, smdata.simdrhs, smdata.simdPrereqData);
    });
}

TEST_F(Hex8MeshWithNSOFields, NGPAssembleElemSolver)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");

  unit_test_utils::HelperObjects helperObjs(
    bulk, stk::topology::HEX_8, 1, partVec[0]);
  auto* assembleElemSolverAlg = helperObjs.assembleElemSolverAlg;
  auto& dataNeeded = assembleElemSolverAlg->dataNeededByKernels_;

  auto meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element_on_dev(
      sierra::nalu::AlgTraitsHex8::topo_);
  dataNeeded.add_cvfem_volume_me(meSCV);
  auto* coordsField = bulk->mesh_meta_data().coordinate_field();

  dataNeeded.add_coordinates_field(
    *coordsField, 3, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(*velocity, 3);
  dataNeeded.add_gathered_nodal_field(*pressure, 1);
  dataNeeded.add_master_element_call(
    sierra::nalu::SCV_VOLUME, sierra::nalu::CURRENT_COORDINATES);

  EXPECT_EQ(3u, dataNeeded.get_fields().size());

  do_assemble_elem_solver_test(
    *bulk, *assembleElemSolverAlg, velocity->mesh_meta_data_ordinal(),
    pressure->mesh_meta_data_ordinal());
}

void
do_the_smdata_test(
  stk::mesh::BulkData& bulk,
  const std::shared_ptr<sierra::nalu::FieldManager> fieldManager,
  sierra::nalu::ScalarFieldType* pressure,
  sierra::nalu::VectorFieldType* velocity)
{
  stk::topology elemTopo = stk::topology::HEX_8;
  sierra::nalu::ElemDataRequests dataReq(bulk.mesh_meta_data());
  auto meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element_on_dev(
      sierra::nalu::AlgTraitsHex8::topo_);
  dataReq.add_cvfem_volume_me(meSCV);

  auto* coordsField = bulk.mesh_meta_data().coordinate_field();

  dataReq.add_coordinates_field(
    *coordsField, 3, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_gathered_nodal_field(*velocity, 3);
  dataReq.add_gathered_nodal_field(*pressure, 1);
  dataReq.add_master_element_call(
    sierra::nalu::SCV_VOLUME, sierra::nalu::CURRENT_COORDINATES);

  EXPECT_EQ(3u, dataReq.get_fields().size());

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();

  sierra::nalu::ElemDataRequestsGPU dataNGP(*fieldManager, dataReq);

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

  int numResults = sierra::nalu::AlgTraitsHex8::numScvIp_;
  DoubleTypeView scv_check("scv_volume", numResults);
  Kokkos::deep_copy(scv_check.h_view, 0.0);
  scv_check.template modify<typename DoubleTypeView::host_mirror_space>();
  scv_check.template sync<typename DoubleTypeView::execution_space>();

  stk::mesh::NgpMesh ngpMesh(bulk);

  TestKernel testKernel(velocityOrdinal, pressureOrdinal);

  int threads_per_team = 1;
  auto team_exec = sierra::nalu::get_device_team_policy(
    ngpMesh.num_buckets(stk::topology::ELEM_RANK), bytes_per_team,
    bytes_per_thread, threads_per_team);

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType& team) {
      const stk::mesh::NgpMesh::BucketType& b =
        ngpMesh.get_bucket(stk::topology::ELEM_RANK, team.league_rank());

      sierra::nalu::SharedMemData<TeamType, ShmemType> smdata(
        team, ngpMesh.get_spatial_dimension(), dataNGP, numNodes, rhsSize);

      STK_NGP_ThrowAssert(smdata.simdPrereqData.total_bytes() != 0);
      const size_t bucketLen = b.size();

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, bucketLen), [&](const size_t& bktIndex) {
          stk::mesh::Entity element = b[bktIndex];
          sierra::nalu::fill_pre_req_data(
            dataNGP, ngpMesh, stk::topology::ELEM_RANK, element,
            *smdata.prereqData[0]);

#if !defined(KOKKOS_ENABLE_GPU)
          // no copy-interleave needed on GPU since no simd.
          sierra::nalu::copy_and_interleave(
            smdata.prereqData, 1, smdata.simdPrereqData);
#endif
          sierra::nalu::fill_master_element_views(
            dataNGP, smdata.simdPrereqData);

          // Copy over SCV volume to temporary array for checking on host
          //
          // This assumes that the test mesh has 8 (2x2x2) elements, and so we
          // save off one integration point from each element for checks on GPU.
          auto& scv_volume = smdata.simdPrereqData
                               .get_me_views(sierra::nalu::CURRENT_COORDINATES)
                               .scv_volume;
          scv_check.d_view(bktIndex % numResults) =
            stk::simd::get_data(scv_volume(bktIndex % numResults), 0);

          testKernel.execute(
            smdata.simdlhs, smdata.simdrhs, smdata.simdPrereqData);
        });
    });

  scv_check.modify<DoubleTypeView::execution_space>();
  scv_check.sync<DoubleTypeView::host_mirror_space>();

  for (int i = 0; i < numResults; ++i)
    EXPECT_NEAR(scv_check.h_view(i), 0.125, 1.0e-8);
}

TEST_F(Hex8MeshWithNSOFields, NGPSharedMemData)
{
  if (stk::parallel_machine_size(comm) == 1) {
    fill_mesh_and_initialize_test_fields("generated:2x2x2");

    do_the_smdata_test(*bulk, fieldManager, pressure, velocity);
  }
}

#if defined(KOKKOS_ENABLE_GPU)
using DeviceSpace = Kokkos::DefaultExecutionSpace;
using DeviceShmem = DeviceSpace::scratch_memory_space;
using DynamicScheduleType = Kokkos::Schedule<Kokkos::Dynamic>;
using DeviceTeamHandleType =
  Kokkos::TeamPolicy<DeviceSpace, DynamicScheduleType>::member_type;
template <typename T, typename SHMEM>
using SharedMemView =
  Kokkos::View<T, Kokkos::LayoutRight, SHMEM, Kokkos::MemoryUnmanaged>;

void
do_kokkos_test()
{

  const int bytes_per_team = 2048;
  const int bytes_per_thread = 2048;
  int threads_per_team = 1;
  unsigned N = 1;

  Kokkos::TeamPolicy<DeviceSpace> team_exec(N, threads_per_team);

  Kokkos::parallel_for(
    team_exec.set_scratch_size(
      1, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread)),
    KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {
      int len = 50;
      SharedMemView<double*, DeviceShmem> shview = Kokkos::subview(
        SharedMemView<double**, DeviceShmem>(
          team.team_scratch(1), team.team_size(), len),
        team.team_rank(), Kokkos::ALL());
    });
}

TEST_F(Hex8MeshWithNSOFields, NGPteamSize) { do_kokkos_test(); }

#endif
