// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <gtest/gtest.h>
#include <UnitTestUtils.h>
#include <aero/actuator/ActuatorSearch.h>
#include <stk_io/StkMeshIoBroker.hpp>
#include <NaluEnv.h>
#include <UnitTestUtils.h>

namespace sierra {
namespace nalu {

// These functions are only designed to run on host

namespace {

class ActuatorSearchTest : public ::testing::Test
{
public:
  ActuatorSearchTest()
    : nProcs(NaluEnv::self().parallel_size()),
      myRank(NaluEnv::self().parallel_rank()),
      nPoints(nProcs * 4),
      partNames({"block_1"}),
      ioBroker(MPI_COMM_WORLD),
      coarsePointIds("crsPointIds", 0),
      coarseElemIds("crsElemIds", 0),
      points("points", nPoints),
      radii("radii", nPoints),
      isLocal("isLocal", nPoints),
      localParallelRedundancy("localParallelRedundancy", nPoints),
      nx("nx"),
      slabSize(4)
  {
  }

  void SetUp()
  {
    nx(0) = 2;
    nx(1) = 2;
    nx(2) = 1;
    ASSERT_EQ(slabSize, (unsigned)nx(0) * nx(1));
    ASSERT_EQ(static_cast<unsigned>(nPoints), (int)slabSize * nProcs);
    Kokkos::parallel_for(
      "populateValues", ActFixRangePolicy(0, nPoints), ACTUATOR_LAMBDA(int i) {
        // place point in center of elements for easy parallel mapping
        points(i, 0) = i % nx(0) + 0.5;
        points(i, 1) = (i / nx(0)) % nx(1) + 0.5;
        points(i, 2) = i / (nx(0) * nx(1)) + 0.5;
        radii(i) = 1e-3;
      });
    const std::string meshSpec = "generated:" + std::to_string(nx(0)) + "x" +
                                 std::to_string(nx(1)) + "x" +
                                 std::to_string(nProcs);
    ioBroker.add_mesh_database(meshSpec, stk::io::READ_MESH);
    ioBroker.create_input_mesh();
    ioBroker.populate_bulk_data();
  }

  const unsigned nProcs;
  const unsigned myRank;
  const int nPoints;
  std::vector<std::string> partNames;
  stk::io::StkMeshIoBroker ioBroker;
  ActScalarU64Dv coarsePointIds;
  ActScalarU64Dv coarseElemIds;
  ActFixVectorDbl points;
  ActFixScalarDbl radii;
  ActFixScalarBool isLocal;
  ActFixScalarInt localParallelRedundancy;
  Kokkos::View<int[3], ActuatorFixedMemLayout, ActuatorFixedMemSpace> nx;
  const unsigned slabSize;
};

TEST_F(ActuatorSearchTest, NGP_createBoundingSpheres)
{
  auto spheres = CreateBoundingSpheres(points, radii);
  for (int i = 0; i < nPoints; i++) {
    auto center = spheres[i].first.center();
    EXPECT_DOUBLE_EQ(points(i, 0), center.get_x_min());
    EXPECT_DOUBLE_EQ(points(i, 1), center.get_y_min());
    EXPECT_DOUBLE_EQ(points(i, 2), center.get_z_min());
    EXPECT_NEAR(
      radii(i), spheres[i].first.get_x_max() - center.get_x_min(), 1e-7);
    // confirm point location matches elem id
    int recreatePointID = (int)center.get_x_min() +
                          (int)center.get_y_min() * nx(0) +
                          (int)center.get_z_min() * nx(0) * nx(1);
    EXPECT_EQ(i, recreatePointID);
  }
}

TEST_F(ActuatorSearchTest, NGP_createElementBoxes)
{
  stk::mesh::BulkData& stkBulk = ioBroker.bulk_data();
  typedef stk::mesh::Field<double, stk::mesh::Cartesian> CoordFieldType;
  CoordFieldType* coordField =
    stkBulk.mesh_meta_data().get_field<CoordFieldType>(
      stk::topology::NODE_RANK, "coordinates");
  EXPECT_TRUE(coordField != nullptr);
  try {
    auto elemVec = CreateElementBoxes(stkBulk, partNames);
    // ioBroker uses slab decomposition in z
    EXPECT_EQ(slabSize, elemVec.size());
    // confirm element center matches corresponding point's location
    for (auto&& elem : elemVec) {
      const double cX = (elem.first.get_x_max() + elem.first.get_x_min()) * 0.5;
      const double cY = (elem.first.get_y_max() + elem.first.get_y_min()) * 0.5;
      const double cZ = (elem.first.get_z_max() + elem.first.get_z_min()) * 0.5;
      EXPECT_NEAR(points(elem.second.id() - 1, 0), cX, 1e-7)
        << "Error with point Id: " << elem.second.id() - 1;
      EXPECT_NEAR(points(elem.second.id() - 1, 1), cY, 1e-7)
        << "Error with point Id: " << elem.second.id() - 1;
      EXPECT_NEAR(points(elem.second.id() - 1, 2), cZ, 1e-7)
        << "Error with point Id: " << elem.second.id() - 1;
    }
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
}

TEST_F(ActuatorSearchTest, NGP_executeCoarseSearch)
{
  stk::mesh::BulkData& stkBulk = ioBroker.bulk_data();
  auto spheres = CreateBoundingSpheres(points, radii);
  auto elemBoxes = CreateElementBoxes(stkBulk, partNames);

  try {
    ExecuteCoarseSearch(
      spheres, elemBoxes, coarsePointIds, coarseElemIds, stk::search::KDTREE);
    // Each search should find one slab of element/point pairs
    EXPECT_EQ(
      slabSize, static_cast<unsigned>(coarsePointIds.view_host().extent_int(0)))
      << "Coarse Search result size: "
      << coarsePointIds.view_host().extent_int(0) << " on rank: " << myRank;
    for (unsigned i = 0; i < coarsePointIds.extent(0); i++) {
      const uint64_t thePt = coarsePointIds.h_view(i);
      const uint64_t theElem = coarseElemIds.h_view(i);
      EXPECT_EQ(thePt + 1, theElem)
        << "rank: " << myRank << " point: " << thePt << " elem: " << theElem;
    }
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
}

TEST_F(ActuatorSearchTest, NGP_executeFineSearch)
{
  stk::mesh::BulkData& stkBulk = ioBroker.bulk_data();
  // increase radius to hit multiple elems in coarse search
  ActFixScalarDbl radii2("radii2", nPoints);
  ActFixVectorDbl localCoords("localCoords", nPoints);
  for (unsigned i = 0; i < radii2.extent(0); i++) {
    radii2(i) = 2.0;
  }
  auto spheres = CreateBoundingSpheres(points, radii2);
  auto elemBoxes = CreateElementBoxes(stkBulk, partNames);
  ExecuteCoarseSearch(
    spheres, elemBoxes, coarsePointIds, coarseElemIds, stk::search::KDTREE);
  ActFixElemIds matchElemIds("matchElemIds", nPoints);
  // this case should match coarse search
  try {
    ExecuteFineSearch(
      stkBulk, coarsePointIds, coarseElemIds, points, matchElemIds, localCoords,
      isLocal, localParallelRedundancy);
    unsigned numLocal = 0;
    for (unsigned i = 0; i < points.extent(0); i++) {
      if (isLocal(i)) {
        numLocal++;
        EXPECT_EQ(i, matchElemIds(i) - 1)
          << "rank: " << myRank << " point: " << i
          << " elem: " << matchElemIds(i);
      }
    }
    EXPECT_EQ(slabSize, numLocal) << "rank: " << myRank;
  } catch (std::exception const& err) {
    FAIL() << err.what();
  }
}

} // namespace

} // namespace nalu
} // namespace sierra
