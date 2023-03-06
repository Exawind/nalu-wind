// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"

#include "AlgTraits.h"
#include "ngp_algorithms/GeometryInteriorAlg.h"
#include "ngp_algorithms/GeometryBoundaryAlg.h"
#include "ngp_algorithms/WallFuncGeometryAlg.h"
#include "ngp_algorithms/GeometryAlgDriver.h"
#include "gcl/MeshVelocityAlg.h"
#include "utils/StkHelpers.h"

#include "mesh_motion/FrameBase.h"
#include "mesh_motion/MotionRotationKernel.h"

#include "UnitTestRealm.h"
#include "UnitTestUtils.h"

#ifndef KOKKOS_ENABLE_GPU

namespace {

std::vector<double>
transform(
  const sierra::nalu::mm::TransMatType& transMat,
  const sierra::nalu::mm::ThreeDVecType& xyz)
{
  std::vector<double> transCoord(3, 0.0);

  // perform matrix multiplication between transformation matrix
  // and original coordinates to obtain transformed coordinates
  for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; d++) {
    transCoord[d] = transMat[d * sierra::nalu::mm::matSize + 0] * xyz[0] +
                    transMat[d * sierra::nalu::mm::matSize + 1] * xyz[1] +
                    transMat[d * sierra::nalu::mm::matSize + 2] * xyz[2] +
                    transMat[d * sierra::nalu::mm::matSize + 3];
  }

  return transCoord;
}

namespace hex8_golds_x_rot {
namespace mesh_velocity {
static constexpr double swept_vol[12] = {0.0,    -0.0625, 0.0,     -0.0625,
                                         0.0,    0.0625,  0.0,     0.0625,
                                         0.0625, 0.0625,  -0.0625, -0.0625};

static constexpr double face_vel_mag[12] = {0.0,   -0.125, 0.0,    -0.125,
                                            0.0,   0.125,  0.0,    0.125,
                                            0.125, 0.125,  -0.125, -0.125};
} // namespace mesh_velocity
} // namespace hex8_golds_x_rot

namespace hex8_golds_y_rot {
namespace mesh_velocity {
static constexpr double swept_vol[12] = {0.0625,  0.0,    -0.0625, 0.0,
                                         -0.0625, 0.0,    0.0625,  0.0,
                                         -0.0625, 0.0625, 0.0625,  -0.0625};

static constexpr double face_vel_mag[12] = {0.125,  0.0,   -0.125, 0.0,
                                            -0.125, 0.0,   0.125,  0.0,
                                            -0.125, 0.125, 0.125,  -0.125};
} // namespace mesh_velocity
} // namespace hex8_golds_y_rot
} // namespace

TEST_F(TestKernelHex8Mesh, mesh_velocity_x_rot)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  // declare relevant fields
  dnvField_ = &(meta_->declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume", 3));
  stk::mesh::put_field_on_mesh(*dnvField_, meta_->universal_part(), nullptr);

  VectorFieldType* meshDisp_ = &(meta_->declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement", 3));
  stk::mesh::put_field_on_mesh(*meshDisp_, meta_->universal_part(), nullptr);

  VectorFieldType* cCoords_ = &(meta_->declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates"));
  stk::mesh::put_field_on_mesh(*cCoords_, meta_->universal_part(), nullptr);

  const auto& meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
      stk::topology::HEX_8);
  GenericFieldType* sweptVolume_ = &(meta_->declare_field<GenericFieldType>(
    stk::topology::ELEM_RANK, "swept_face_volume", 3));
  stk::mesh::put_field_on_mesh(
    *sweptVolume_, meta_->universal_part(), meSCS->num_integration_points(),
    nullptr);

  GenericFieldType* faceVelMag_ = &(meta_->declare_field<GenericFieldType>(
    stk::topology::ELEM_RANK, "face_velocity_mag", 2));
  stk::mesh::put_field_on_mesh(
    *faceVelMag_, meta_->universal_part(), meSCS->num_integration_points(),
    nullptr);

  fill_mesh_and_init_fields();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.5;   // first time step size
  timeIntegrator.timeStepNm1_ = 0.0; // second time step size
  timeIntegrator.currentTime_ = 0.5; // current time
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;
  // Force computation of edge area vector
  helperObjs.realm.realmUsesEdges_ = false;
  helperObjs.realm.solutionOptions_->meshMotion_ = true;

  sierra::nalu::GeometryAlgDriver geomAlgDriver(helperObjs.realm);
  geomAlgDriver.register_elem_algorithm<sierra::nalu::GeometryInteriorAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "geometry");
  geomAlgDriver.register_elem_algorithm<sierra::nalu::MeshVelocityAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "mesh_vel");

  // First set the mesh displacement corresponding to rotation about x-axis
  // create a yaml node describing rotation
  const std::string rotInfo = "omega: -3.141592653589793  \n"
                              "axis: [1.0,0.0,0.0]        \n"
                              "centroid: [0.5,0.5,0.5]    \n";
  YAML::Node rotNode = YAML::Load(rotInfo);
  sierra::nalu::MotionRotationKernel rotClass(rotNode);

  VectorFieldType* meshDispNp1 =
    &(meshDisp_->field_of_state(stk::mesh::StateNP1));
  VectorFieldType* meshDispN = &(meshDisp_->field_of_state(stk::mesh::StateN));
  VectorFieldType* meshDispNm1 =
    &(meshDisp_->field_of_state(stk::mesh::StateNM1));

  {
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
    for (const auto* b : bkts) {
      for (const auto node : *b) {
        double* dispNp1 = stk::mesh::field_data(*meshDispNp1, node);
        double* dispN = stk::mesh::field_data(*meshDispN, node);
        double* dispNm1 = stk::mesh::field_data(*meshDispNm1, node);
        double* ccoord = stk::mesh::field_data(*cCoords_, node);
        double* mcoord = stk::mesh::field_data(*coordinates_, node);
        dispNm1[0] = 0.0;
        dispNm1[1] = 0.0;
        dispNm1[2] = 0.0;

        // transform data structures to confirm to mesh motion
        sierra::nalu::mm::ThreeDVecType mX;
        for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; ++d)
          mX[d] = mcoord[d];

        sierra::nalu::mm::TransMatType transMat =
          rotClass.build_transformation(0.0, mX);
        std::vector<double> rot_xyz = transform(transMat, mX);

        dispN[0] = rot_xyz[0] - mX[0];
        dispN[1] = rot_xyz[1] - mX[1];
        dispN[2] = rot_xyz[2] - mX[2];

        transMat = rotClass.build_transformation(0.5, mX);
        rot_xyz = transform(transMat, mX);

        dispNp1[0] = rot_xyz[0] - mX[0];
        dispNp1[1] = rot_xyz[1] - mX[1];
        dispNp1[2] = rot_xyz[2] - mX[2];

        ccoord[0] = rot_xyz[0];
        ccoord[1] = rot_xyz[1];
        ccoord[2] = rot_xyz[2];
      }
    }
  }
  geomAlgDriver.execute();

  const double tol = 1.0e-15;
  namespace gold_values = ::hex8_golds_x_rot::mesh_velocity;
  {
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::ELEM_RANK, sel);
    int counter = 0;
    for (const auto* b : bkts) {
      const double* sv = stk::mesh::field_data(*sweptVolume_, *b, 0);
      const double* fvm = stk::mesh::field_data(*faceVelMag_, *b, 0);
      for (int i = 0; i < 12; i++) {
        EXPECT_NEAR(gold_values::swept_vol[i], sv[i], tol);
        counter++;
        EXPECT_NEAR(gold_values::face_vel_mag[i], fvm[i], tol);
        counter++;
      }
    }
    EXPECT_EQ(counter, 24);
  } // namespace =::hex8_golds_x_rot::mesh_velocity;
}

TEST_F(TestKernelHex8Mesh, mesh_velocity_y_rot)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  // declare relevant fields
  dnvField_ = &(meta_->declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume", 3));
  stk::mesh::put_field_on_mesh(*dnvField_, meta_->universal_part(), nullptr);

  VectorFieldType* meshDisp_ = &(meta_->declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement", 3));
  stk::mesh::put_field_on_mesh(*meshDisp_, meta_->universal_part(), nullptr);

  VectorFieldType* cCoords_ = &(meta_->declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates"));
  stk::mesh::put_field_on_mesh(*cCoords_, meta_->universal_part(), nullptr);

  const auto& meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
      stk::topology::HEX_8);
  GenericFieldType* sweptVolume_ = &(meta_->declare_field<GenericFieldType>(
    stk::topology::ELEM_RANK, "swept_face_volume", 3));
  stk::mesh::put_field_on_mesh(
    *sweptVolume_, meta_->universal_part(), meSCS->num_integration_points(),
    nullptr);

  GenericFieldType* faceVelMag_ = &(meta_->declare_field<GenericFieldType>(
    stk::topology::ELEM_RANK, "face_velocity_mag", 2));
  stk::mesh::put_field_on_mesh(
    *faceVelMag_, meta_->universal_part(), meSCS->num_integration_points(),
    nullptr);

  fill_mesh_and_init_fields();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.5;   // first time step size
  timeIntegrator.timeStepNm1_ = 0.0; // second time step size
  timeIntegrator.currentTime_ = 0.5; // current time
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;
  // Force computation of edge area vector
  helperObjs.realm.realmUsesEdges_ = false;
  helperObjs.realm.solutionOptions_->meshMotion_ = true;

  sierra::nalu::GeometryAlgDriver geomAlgDriver(helperObjs.realm);
  geomAlgDriver.register_elem_algorithm<sierra::nalu::GeometryInteriorAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "geometry");
  geomAlgDriver.register_elem_algorithm<sierra::nalu::MeshVelocityAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "mesh_vel");

  // First set the mesh displacement corresponding to rotation about x-axis
  // create a yaml node describing rotation
  const std::string rotInfo = "omega: -3.141592653589793  \n"
                              "axis: [0.0,1.0,0.0]        \n"
                              "centroid: [0.5,0.5,0.5]    \n";
  YAML::Node rotNode = YAML::Load(rotInfo);
  sierra::nalu::MotionRotationKernel rotClass(rotNode);

  VectorFieldType* meshDispNp1 =
    &(meshDisp_->field_of_state(stk::mesh::StateNP1));
  VectorFieldType* meshDispN = &(meshDisp_->field_of_state(stk::mesh::StateN));
  VectorFieldType* meshDispNm1 =
    &(meshDisp_->field_of_state(stk::mesh::StateNM1));

  {
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
    for (const auto* b : bkts) {
      for (const auto node : *b) {
        double* dispNp1 = stk::mesh::field_data(*meshDispNp1, node);
        double* dispN = stk::mesh::field_data(*meshDispN, node);
        double* dispNm1 = stk::mesh::field_data(*meshDispNm1, node);
        double* ccoord = stk::mesh::field_data(*cCoords_, node);
        double* mcoord = stk::mesh::field_data(*coordinates_, node);
        dispNm1[0] = 0.0;
        dispNm1[1] = 0.0;
        dispNm1[2] = 0.0;

        // transform data structures to confirm to mesh motion
        sierra::nalu::mm::ThreeDVecType mX;
        for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; ++d)
          mX[d] = mcoord[d];

        sierra::nalu::mm::TransMatType transMat =
          rotClass.build_transformation(0.0, mX);
        std::vector<double> rot_xyz = transform(transMat, mX);

        dispN[0] = rot_xyz[0] - mX[0];
        dispN[1] = rot_xyz[1] - mX[1];
        dispN[2] = rot_xyz[2] - mX[2];

        transMat = rotClass.build_transformation(0.5, mX);
        rot_xyz = transform(transMat, mX);

        dispNp1[0] = rot_xyz[0] - mX[0];
        dispNp1[1] = rot_xyz[1] - mX[1];
        dispNp1[2] = rot_xyz[2] - mX[2];

        ccoord[0] = rot_xyz[0];
        ccoord[1] = rot_xyz[1];
        ccoord[2] = rot_xyz[2];
      }
    }
  }
  geomAlgDriver.execute();

  const double tol = 1.0e-15;
  namespace gold_values = ::hex8_golds_y_rot::mesh_velocity;
  {
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::ELEM_RANK, sel);
    int counter = 0;
    for (const auto* b : bkts) {
      const double* sv = stk::mesh::field_data(*sweptVolume_, *b, 0);
      const double* fvm = stk::mesh::field_data(*faceVelMag_, *b, 0);
      for (int i = 0; i < 12; i++) {
        EXPECT_NEAR(gold_values::swept_vol[i], sv[i], tol);
        counter++;
        EXPECT_NEAR(gold_values::face_vel_mag[i], fvm[i], tol);
        counter++;
      }
    }
    EXPECT_EQ(counter, 24);
  } // namespace =::hex8_golds_y_rot::mesh_velocity;
}

TEST_F(TestKernelHex8Mesh, mesh_velocity_y_rot_scs_center)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  // declare relevant fields
  dnvField_ = &(meta_->declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume", 3));
  stk::mesh::put_field_on_mesh(*dnvField_, meta_->universal_part(), nullptr);

  VectorFieldType* meshDisp_ = &(meta_->declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement", 3));
  stk::mesh::put_field_on_mesh(*meshDisp_, meta_->universal_part(), nullptr);

  VectorFieldType* cCoords_ = &(meta_->declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates"));
  stk::mesh::put_field_on_mesh(*cCoords_, meta_->universal_part(), nullptr);

  const auto& meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
      stk::topology::HEX_8);
  GenericFieldType* sweptVolume_ = &(meta_->declare_field<GenericFieldType>(
    stk::topology::ELEM_RANK, "swept_face_volume", 3));
  stk::mesh::put_field_on_mesh(
    *sweptVolume_, meta_->universal_part(), meSCS->num_integration_points(),
    nullptr);

  GenericFieldType* faceVelMag_ = &(meta_->declare_field<GenericFieldType>(
    stk::topology::ELEM_RANK, "face_velocity_mag", 2));
  stk::mesh::put_field_on_mesh(
    *faceVelMag_, meta_->universal_part(), meSCS->num_integration_points(),
    nullptr);

  fill_mesh_and_init_fields();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = 0.25;   // first time step size
  timeIntegrator.timeStepNm1_ = 0.0;  // second time step size
  timeIntegrator.currentTime_ = 0.25; // current time
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;
  // Force computation of edge area vector
  helperObjs.realm.realmUsesEdges_ = false;
  helperObjs.realm.solutionOptions_->meshMotion_ = true;

  sierra::nalu::GeometryAlgDriver geomAlgDriver(helperObjs.realm);
  geomAlgDriver.register_elem_algorithm<sierra::nalu::GeometryInteriorAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "geometry");
  geomAlgDriver.register_elem_algorithm<sierra::nalu::MeshVelocityAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "mesh_vel");

  // First set the mesh displacement corresponding to rotation about x-axis
  // create a yaml node describing rotation
  const std::string rotInfo = "omega: -3.141592653589793  \n"
                              "axis: [0.0,1.0,0.0]        \n"
                              "centroid: [0.5,0.5,0.75]   \n";
  YAML::Node rotNode = YAML::Load(rotInfo);
  sierra::nalu::MotionRotationKernel rotClass(rotNode);

  VectorFieldType* meshDispNp1 =
    &(meshDisp_->field_of_state(stk::mesh::StateNP1));
  VectorFieldType* meshDispN = &(meshDisp_->field_of_state(stk::mesh::StateN));
  VectorFieldType* meshDispNm1 =
    &(meshDisp_->field_of_state(stk::mesh::StateNM1));

  {
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
    for (const auto* b : bkts) {
      for (const auto node : *b) {
        double* dispNp1 = stk::mesh::field_data(*meshDispNp1, node);
        double* dispN = stk::mesh::field_data(*meshDispN, node);
        double* dispNm1 = stk::mesh::field_data(*meshDispNm1, node);
        double* ccoord = stk::mesh::field_data(*cCoords_, node);
        double* mcoord = stk::mesh::field_data(*coordinates_, node);
        dispNm1[0] = 0.0;
        dispNm1[1] = 0.0;
        dispNm1[2] = 0.0;

        // transform data structures to confirm to mesh motion
        sierra::nalu::mm::ThreeDVecType mX;
        for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; ++d)
          mX[d] = mcoord[d];

        sierra::nalu::mm::TransMatType transMat =
          rotClass.build_transformation(0.0, mX);
        std::vector<double> rot_xyz = transform(transMat, mX);

        dispN[0] = rot_xyz[0] - mX[0];
        dispN[1] = rot_xyz[1] - mX[1];
        dispN[2] = rot_xyz[2] - mX[2];

        transMat = rotClass.build_transformation(0.5, mX);
        rot_xyz = transform(transMat, mX);

        dispNp1[0] = rot_xyz[0] - mcoord[0];
        dispNp1[1] = rot_xyz[1] - mcoord[1];
        dispNp1[2] = rot_xyz[2] - mcoord[2];

        ccoord[0] = rot_xyz[0];
        ccoord[1] = rot_xyz[1];
        ccoord[2] = rot_xyz[2];
      }
    }
  }
  geomAlgDriver.execute();

  const double tol = 1.0e-15;
  namespace gold_values = ::hex8_golds_y_rot::mesh_velocity;
  {
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::ELEM_RANK, sel);
    int counter = 0;
    for (const auto* b : bkts) {
      const double* sv = stk::mesh::field_data(*sweptVolume_, *b, 0);
      const double* fvm = stk::mesh::field_data(*faceVelMag_, *b, 0);
      for (int i = 0; i < 12; i++) {
        // check only for scs through the center of which rotation axis passes
        // in addition to all scs perpendicular to rotation axis - total 6 scs
        if (
          (i == 1) || (i == 3) || (i == 4) || (i == 5) || (i == 6) ||
          (i == 7)) {
          EXPECT_NEAR(0.0, sv[i], tol);
          counter++;
          EXPECT_NEAR(0.0, fvm[i], tol);
          counter++;
        }
      }
    }
    EXPECT_EQ(counter, 12);
  } // namespace =::hex8_golds_y_rot::mesh_velocity;
}

#endif // KOKKOS_ENABLE_GPU
