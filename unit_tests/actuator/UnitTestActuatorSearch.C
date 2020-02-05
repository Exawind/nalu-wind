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
#include <actuator/ActuatorSearch.h>
#include <stk_io/StkMeshIoBroker.hpp>
#include <NaluEnv.h>

namespace sierra{
namespace nalu{

// These functions are only designed to run on host

namespace{

const int nPoints = 2;

class ActuatorSearchTest : public ::testing::Test{
protected:
  ActuatorSearchTest():
    partNames({"block_1"}),
    ioBroker(MPI_COMM_WORLD),
    points("points",nPoints),
    radii("radii",nPoints),
    dx("dx"){}

  void SetUp(){
    dx(0)=0.1; dx(1)=0.11; dx(2) = 0.111;
    Kokkos::parallel_for("populateValues", ActFixRangePolicy(0, nPoints),
      KOKKOS_LAMBDA(int i){
        for(int j=0; j<3; j++){
          points(i,j) = dx(j)*i;
        }
        radii(i) = (double)i;
    });
    const std::string meshSpec = "generated:2x2x" + std::to_string(NaluEnv::self().parallel_size());
    ioBroker.add_mesh_database(meshSpec, stk::io::READ_MESH);
    ioBroker.create_input_mesh();
    ioBroker.populate_bulk_data();
  }


  std::vector<std::string> partNames;
  stk::io::StkMeshIoBroker ioBroker;
  ActFixVectorDbl points;
  ActFixScalarDbl radii;
  Kokkos::View<double[3], ActuatorFixedMemLayout, ActuatorFixedMemSpace> dx;
};

TEST_F(ActuatorSearchTest, createBoundingSpheres){
  auto spheres = CreateBoundingSpheres(points, radii);
  for( int i = 0; i< nPoints; i++){
      auto center = spheres[i].first.center();
      EXPECT_DOUBLE_EQ(points(i,0), center.get_x_min());
      EXPECT_DOUBLE_EQ(points(i,1), center.get_y_min());
      EXPECT_DOUBLE_EQ(points(i,2), center.get_z_min());
      EXPECT_DOUBLE_EQ(radii(i), spheres[i].first.get_x_max()-center.get_x_min());
  }
}

TEST_F(ActuatorSearchTest, createElementBoxes){
  stk::mesh::MetaData& stkMeta = ioBroker.meta_data();
  stk::mesh::BulkData& stkBulk = ioBroker.bulk_data();
  typedef stk::mesh::Field<double,stk::mesh::Cartesian> CoordFieldType;
  CoordFieldType* coordField = stkBulk.mesh_meta_data().get_field<CoordFieldType>(stk::topology::NODE_RANK, "coordinates");
  EXPECT_TRUE(coordField != nullptr);
  try{
    auto elemVec = CreateElementBoxes(stkMeta, stkBulk, partNames);
    EXPECT_EQ(4, elemVec.size());
  }
  catch(std::exception const & err){
    FAIL() << err.what();
  }

}

TEST_F(ActuatorSearchTest, executeCoarseSearch){
  stk::mesh::MetaData& stkMeta = ioBroker.meta_data();
  stk::mesh::BulkData& stkBulk = ioBroker.bulk_data();
  auto spheres = CreateBoundingSpheres(points, radii);
  auto elemBoxes = CreateElementBoxes(stkMeta, stkBulk, partNames);

  try{
    auto results = ExecuteCoarseSearch(spheres, elemBoxes, stk::search::KDTREE);
    // TODO(psakiev) check this
  }
  catch(std::exception const & err){
    FAIL() << err.what();
  }

  SUCCEED();
}

}

} //namespace nalu
} //namespace sierra
