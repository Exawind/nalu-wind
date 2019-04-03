#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include <master_element/MasterElementFactory.h>

#include <SimdInterface.h>
#include <ElemDataRequests.h>
#include <ElemDataRequestsGPU.h>
#include <SharedMemData.h>
#include <ScratchViews.h>

void do_the_test_gpu(const sierra::nalu::ElemDataRequests& dataReq,
                     stk::mesh::BulkData& bulk)
{
  unsigned totalNumFields_guess = 10;
  ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU ngpDataReq(fieldMgr, dataReq, totalNumFields_guess);

  unsigned numCorrectTests = 0;
  int threadsPerTeam = 1;
  size_t bytesPerTeam = 0;
  size_t bytesPerThread = 0;
  auto team_exec = sierra::nalu::get_device_team_policy(1, bytesPerTeam, bytesPerThread, threadsPerTeam);
  Kokkos::parallel_reduce(team_exec, KOKKOS_LAMBDA(const sierra::nalu::DeviceTeamHandleType&  /* team */, unsigned& localNumTests)
  {
    if (ngpDataReq.get_fields().size() == 3) {
      ++localNumTests;
    }

    if (ngpDataReq.get_coordinates_fields().size() == 2) {
      ++localNumTests;
    }

    if (ngpDataReq.get_coordinates_types().size() == 2) {
      ++localNumTests;
    }

    if (ngpDataReq.get_coordinates_types()(0) == sierra::nalu::CURRENT_COORDINATES) {
      ++localNumTests;
    }

    if (ngpDataReq.get_coordinates_types()(1) == sierra::nalu::MODEL_COORDINATES) {
      ++localNumTests;
    }
  }, numCorrectTests);

  EXPECT_EQ(5u, numCorrectTests);
}

TEST_F(Hex8MeshWithNSOFields, NGPElemDataRequests)
{
  fill_mesh_and_initialize_test_fields("generated:2x2x2");
  stk::topology elemTopo = stk::topology::HEX_8;

  sierra::nalu::ElemDataRequests dataReq(bulk.mesh_meta_data());
  auto meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo);
  dataReq.add_cvfem_volume_me(meSCV);

  dataReq.add_gathered_nodal_field(*velocity, 3);
  dataReq.add_gathered_nodal_field(*pressure, 1);

  dataReq.add_coordinates_field(*bulk.mesh_meta_data().coordinate_field(), 3, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_coordinates_field(*bulk.mesh_meta_data().coordinate_field(), 3, sierra::nalu::MODEL_COORDINATES);

  EXPECT_EQ(3u, dataReq.get_fields().size());

  do_the_test_gpu(dataReq, bulk);
}

