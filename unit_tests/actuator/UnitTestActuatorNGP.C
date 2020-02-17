// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/UnitTestActuatorNGP.h>
#include <actuator/ActuatorParsing.h>
#include <stk_io/StkMeshIoBroker.hpp>
#include <NaluEnv.h>
#include <gtest/gtest.h>

namespace sierra{
namespace nalu{

namespace{
TEST(ActuatorNGP, testExecuteOnHostOnly){
  stk::mesh::MetaData stkMeta(3);
  stk::mesh::BulkData stkBulk(stkMeta, MPI_COMM_WORLD);
  ActuatorMeta actMeta(1, stkBulk);
  ActuatorInfoNGP infoTurb0;
  infoTurb0.turbineName_ = "Turbine0";
  infoTurb0.numPoints_ = 20;
  actMeta.add_turbine(infoTurb0);
  TestActuatorHostOnly actuator(actMeta);
  ASSERT_NO_THROW(actuator.execute());
  const ActuatorBulk& actBulk = actuator.actuator_bulk();
  EXPECT_DOUBLE_EQ(3.0, actBulk.epsilon_.h_view(1,0));
  EXPECT_DOUBLE_EQ(6.0, actBulk.epsilon_.h_view(1,1));
  EXPECT_DOUBLE_EQ(9.0, actBulk.epsilon_.h_view(1,2));

  EXPECT_DOUBLE_EQ(1.0, actBulk.pointCentroid_.h_view(1,0));
  EXPECT_DOUBLE_EQ(0.5, actBulk.pointCentroid_.h_view(1,1));
  EXPECT_DOUBLE_EQ(0.25, actBulk.pointCentroid_.h_view(1,2));

  EXPECT_DOUBLE_EQ(2.5, actBulk.velocity_.h_view(1,0));
  EXPECT_DOUBLE_EQ(5.0, actBulk.velocity_.h_view(1,1));
  EXPECT_DOUBLE_EQ(7.5, actBulk.velocity_.h_view(1,2));

  EXPECT_DOUBLE_EQ(3.1, actBulk.actuatorForce_.h_view(1,0));
  EXPECT_DOUBLE_EQ(6.2, actBulk.actuatorForce_.h_view(1,1));
  EXPECT_DOUBLE_EQ(9.3, actBulk.actuatorForce_.h_view(1,2));
}

TEST(ActuatorNGP, testExecuteOnHostAndDevice){
  stk::mesh::MetaData stkMeta(3);
  stk::mesh::BulkData stkBulk(stkMeta, MPI_COMM_WORLD);
  ActuatorMeta actMeta(1, stkBulk);
  ActuatorInfoNGP infoTurb0;
  infoTurb0.turbineName_ = "Turbine0";
  infoTurb0.numPoints_ = 20;
  actMeta.add_turbine(infoTurb0);
  TestActuatorHostDev actuator(actMeta);
  ASSERT_NO_THROW(actuator.execute());
  const ActuatorBulkMod& actBulk = actuator.actuator_bulk();
  const double expectVal = actBulk.velocity_.h_view(1,1)*actBulk.pointCentroid_.h_view(1,0);
  EXPECT_DOUBLE_EQ(expectVal, actBulk.scalar_.h_view(1));
}

//-----------------------------------------------------------------
class ActuatorNGPOnMesh : public ::testing::Test{
protected:
  std::string inputFileSurrogate_;
  stk::mesh::MetaData stkMeta_;
  stk::mesh::BulkData stkBulk_;
  const double tol_;
  const VectorFieldType* coordinates_{nullptr};
  VectorFieldType* velocity_{nullptr};
  VectorFieldType* actuatorForce_{nullptr};

ActuatorNGPOnMesh():
  stkMeta_(3),
  stkBulk_(stkMeta_, MPI_COMM_WORLD),
  tol_(1e-8),
  coordinates_(nullptr),
  velocity_(&stkMeta_.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity")),
  actuatorForce_(&stkMeta_.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuatorForce"))
{
  stk::mesh::put_field_on_mesh(*velocity_, stkMeta_.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(*actuatorForce_, stkMeta_.universal_part(), 3, nullptr);

}

void SetUp(){
  const std::string meshSpec = "generated:5x5x5";
  unit_test_utils::fill_hex8_mesh(meshSpec, stkBulk_);
  coordinates_ = static_cast<const VectorFieldType*>(stkMeta_.coordinate_field());
  const stk::mesh::Selector selector = stkMeta_.locally_owned_part() | stkMeta_.globally_shared_part();
  const auto& buckets  = stkBulk_.get_buckets(stk::topology::NODE_RANK, selector);
  for(const stk::mesh::Bucket* bptr : buckets){
    for(stk::mesh::Entity node : *bptr){
      const double* coords = stk::mesh::field_data(*coordinates_, node);
      double* vel = stk::mesh::field_data(*velocity_, node);
      double* aF = stk::mesh::field_data(*actuatorForce_, node);
      for(int i=0; i<3; i++){
        vel[i] = coords[i];
        aF[i] = 0.0;
      }
    }
  }
}


};

TEST_F(ActuatorNGPOnMesh, testSearchAndInterpolate){
  inputFileSurrogate_ =
      "actuator:\n"
      "  type: ActLinePointDrag\n"
      "  n_turbines_glob: 1\n"
      "  search_method: stk_kdtree\n"
      "  search_target_part: [block_1]\n"
      ;
  YAML::Node y_actuator = YAML::Load(inputFileSurrogate_);
  ActuatorMeta actMeta = actuator_parse(y_actuator, stkBulk_);

  // more parse stuff to be implemented
  ActuatorInfoNGP actInfo;
  actInfo.numPoints_= 3;
  actMeta.add_turbine(actInfo);

  // construct object and allocate memory
  TestActuatorSearchInterp actuator(actMeta);

  // what gets called in the time loop
  actuator.execute();

  // check results
  auto actBulk = actuator.actuator_bulk();
  const int nTotal = actBulk.actuatorMeta_.numPointsTotal_;
  auto points = actBulk.pointCentroid_.template view<Kokkos::HostSpace>();
  auto vel = actBulk.velocity_.template view<Kokkos::HostSpace>();
  auto force = actBulk.actuatorForce_.template view<Kokkos::HostSpace>();
  for(int i=0; i<nTotal; i++){
    EXPECT_NEAR(1.0+1.5*i, vel(i,0), tol_);
    EXPECT_NEAR(2.5,       vel(i,1), tol_);
    EXPECT_NEAR(2.5,       vel(i,2), tol_);
    for(int j=0; j<3; j++){
      EXPECT_NEAR(vel(i,j)*1.2, force(i,j), tol_);
    }
  }


}

}

} //namespace nalu
} //namespace sierra
