#include <gtest/gtest.h>
#include <limits>

#include "mesh_motion/MeshMotionAlg.h"
#include "mesh_motion/MeshTransformationAlg.h"
#include "mesh_motion/MotionRotationKernel.h"
#include "mesh_motion/MotionScalingKernel.h"
#include "mesh_motion/MotionTranslationKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

#include "UnitTestRealm.h"
#include "UnitTestUtils.h"

#include <string>

namespace {
// create a yaml node describing mesh motion
const std::string mInfo = "mesh_transformation:               \n"
                          "  - name: scale                    \n"
                          "    mesh_parts: [ all_blocks ]     \n"
                          "    motion:                        \n"
                          "     - type: scaling               \n"
                          "       factor: [1.2, 1.0, 1.2]     \n"
                          "                                   \n"
                          "mesh_motion:                       \n"
                          "  - name: trans_rot                \n"
                          "    mesh_parts: [ block_1 ]        \n"
                          "    motion:                        \n"
                          "     - type: rotation              \n"
                          "       omega: 3.0                  \n"
                          "       axis: [0.0, 0.0, 1.0]       \n"
                          "                                   \n"
                          "     - type: translation           \n"
                          "       start_time: 15.0            \n"
                          "       end_time: 25.0              \n"
                          "       velocity: [2.0, 0.0, 0.0]   \n";

// define YAML nodes at different levels
const YAML::Node yamlNode = YAML::Load(mInfo);
const YAML::Node mesh_transformation = yamlNode["mesh_transformation"];
const YAML::Node mesh_motion = yamlNode["mesh_motion"];

const YAML::Node frame_first = mesh_transformation[0];
const YAML::Node scale = frame_first["motion"];
const YAML::Node scaleNode = scale[0];

const YAML::Node frame_second = mesh_motion[0];
const YAML::Node rot_trans = frame_second["motion"];
const YAML::Node rotNode = rot_trans[0];
const YAML::Node transNode = rot_trans[1];

const double testTol = 1e-12;

sierra::nalu::mm::TransMatType
eval_transformation(sierra::nalu::Realm& realm, double time, const double* xyz)
{
  // transform data structures to confirm to mesh motion
  sierra::nalu::mm::ThreeDVecType vecX;
  for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; d++)
    vecX[d] = xyz[d];

  // perform scaling transformation
  sierra::nalu::MotionScalingKernel scaleClass(realm.meta_data(), scaleNode);
  sierra::nalu::mm::TransMatType compTrans =
    scaleClass.build_transformation(time, vecX);

  // perform rotation transformation
  sierra::nalu::MotionRotationKernel rotClass(rotNode);
  sierra::nalu::mm::TransMatType tempMat =
    rotClass.build_transformation(time, vecX);
  compTrans = rotClass.add_motion(tempMat, compTrans);

  // perform translation transformation
  sierra::nalu::MotionTranslationKernel transClass(transNode);
  tempMat = transClass.build_transformation(time, vecX);

  return rotClass.add_motion(tempMat, compTrans);
}

std::vector<double>
eval_coords(const sierra::nalu::mm::TransMatType& transMat, const double* xyz)
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

std::vector<double>
eval_vel(
  const double time,
  const sierra::nalu::mm::TransMatType& transMat,
  const double* mxyz,
  const double* cxyz)
{
  std::vector<double> vel(3, 0.0);
  sierra::nalu::mm::ThreeDVecType motionVel;

  // transform data structures to confirm to mesh motion
  sierra::nalu::mm::ThreeDVecType vecMX;
  sierra::nalu::mm::ThreeDVecType vecCX;
  for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; d++) {
    vecMX[d] = mxyz[d];
    vecCX[d] = cxyz[d];
  }

  // perform rotation transformation
  sierra::nalu::MotionRotationKernel rotClass(rotNode);
  motionVel = rotClass.compute_velocity(time, transMat, vecMX, vecCX);

  for (size_t d = 0; d < vel.size(); d++)
    vel[d] += motionVel[d];

  // perform translation transformation
  const double startTime = transNode["start_time"].as<double>();
  const double endTime = transNode["end_time"].as<double>();

  if ((time >= (startTime - testTol)) && (time <= (endTime + testTol))) {
    sierra::nalu::MotionTranslationKernel transClass(transNode);
    motionVel = transClass.compute_velocity(time, transMat, vecMX, vecCX);

    for (size_t d = 0; d < vel.size(); d++)
      vel[d] += motionVel[d];
  }

  return vel;
}
} // namespace

TEST(meshMotion, NGP_initialize)
{
  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();
  realm.solutionOptions_->meshTransformation_ = true;
  realm.solutionOptions_->meshMotion_ = true;

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.secondOrderTimeAccurate_ = false;
  realm.timeIntegrator_ = &timeIntegrator;

  // register mesh motion fields and initialize coordinate fields
  realm.register_nodal_fields(&(realm.meta_data().universal_part()));

  // create field to copy coordinates
  // NOTE: This is done to allow computation of gold values later on
  // because mesh_transformation changes the field - coordinates
  int nDim = realm.meta_data().spatial_dimension();
  VectorFieldType* modelCoordsCopy =
    &(realm.meta_data().declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates_copy"));
  stk::mesh::put_field_on_mesh(
    *modelCoordsCopy, realm.meta_data().universal_part(), nDim, nullptr);

  // create mesh
  const std::string meshSpec("generated:2x2x2");
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());
  realm.init_current_coordinates();

  // copy coordinates to copy coordinates
  VectorFieldType* modelCoords = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");

  // get the parts in the current motion frame
  stk::mesh::Selector sel =
    stk::mesh::Selector(realm.meta_data().universal_part()) &
    (realm.meta_data().locally_owned_part() |
     realm.meta_data().globally_shared_part());
  const auto& bkts =
    realm.bulk_data().get_buckets(stk::topology::NODE_RANK, sel);

  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* mxyz = stk::mesh::field_data(*modelCoords, node);
      double* cxyz = stk::mesh::field_data(*modelCoordsCopy, node);

      for (int d = 0; d < nDim; ++d) {
        cxyz[d] = mxyz[d];
      }
    } // end for loop - in index
  }   // end for loop - bkts

  // create mesh transformation algorithm class
  std::unique_ptr<sierra::nalu::MeshTransformationAlg> meshTransformationAlg;
  meshTransformationAlg.reset(new sierra::nalu::MeshTransformationAlg(
    realm.bulk_data(), mesh_transformation));

  // create mesh motion algorithm class
  std::unique_ptr<sierra::nalu::MeshMotionAlg> meshMotionAlg;
  meshMotionAlg.reset(
    new sierra::nalu::MeshMotionAlg(realm.bulk_data(), mesh_motion));

  // initialize and execute mesh motion algorithm
  const double currTime = 0.0;
  meshTransformationAlg->initialize(currTime);
  meshMotionAlg->initialize(currTime);

  // get fields to be tested
  auto* currCoords = realm.fieldManager_->get_field_ptr<VectorFieldType*>("current_coordinates");
  auto* meshVelocity = realm.fieldManager_->get_field_ptr<VectorFieldType*>( "mesh_velocity");

  // sync coordinates to host
  currCoords->sync_to_host();
  meshVelocity->sync_to_host();

  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* oxyz = stk::mesh::field_data(*modelCoordsCopy, node);
      double* xyz = stk::mesh::field_data(*currCoords, node);
      double* vel = stk::mesh::field_data(*meshVelocity, node);

      sierra::nalu::mm::TransMatType transMat =
        eval_transformation(realm, currTime, oxyz);

      std::vector<double> gold_norm_xyz = eval_coords(transMat, oxyz);
      std::vector<double> gold_norm_vel =
        eval_vel(currTime, transMat, oxyz, &gold_norm_xyz[0]);

      EXPECT_NEAR(xyz[0], gold_norm_xyz[0], testTol);
      EXPECT_NEAR(xyz[1], gold_norm_xyz[1], testTol);
      EXPECT_NEAR(xyz[2], gold_norm_xyz[2], testTol);

      EXPECT_NEAR(vel[0], gold_norm_vel[0], testTol);
      EXPECT_NEAR(vel[1], gold_norm_vel[1], testTol);
      EXPECT_NEAR(vel[2], gold_norm_vel[2], testTol);
    } // end for loop - in index
  }   // end for loop - bkts
}

TEST(meshMotion, NGP_execute)
{
  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();
  realm.solutionOptions_->meshTransformation_ = true;
  realm.solutionOptions_->meshMotion_ = true;

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.secondOrderTimeAccurate_ = false;
  realm.timeIntegrator_ = &timeIntegrator;

  // register mesh motion fields and initialize coordinate fields
  realm.register_nodal_fields(&(realm.meta_data().universal_part()));

  // create field to copy coordinates
  // NOTE: This is done to allow computation of gold values later on
  // because mesh_transformation changes the field - coordinates
  int nDim = realm.meta_data().spatial_dimension();
  VectorFieldType* modelCoordsCopy =
    &(realm.meta_data().declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates_copy"));
  stk::mesh::put_field_on_mesh(
    *modelCoordsCopy, realm.meta_data().universal_part(), nDim, nullptr);

  // create mesh
  const std::string meshSpec("generated:2x2x2");
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());
  realm.init_current_coordinates();

  // copy coordinates to copy coordinates
  VectorFieldType* modelCoords = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");

  // get the parts in the current motion frame
  stk::mesh::Selector sel =
    stk::mesh::Selector(realm.meta_data().universal_part()) &
    (realm.meta_data().locally_owned_part() |
     realm.meta_data().globally_shared_part());
  const auto& bkts =
    realm.bulk_data().get_buckets(stk::topology::NODE_RANK, sel);

  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* mxyz = stk::mesh::field_data(*modelCoords, node);
      double* cxyz = stk::mesh::field_data(*modelCoordsCopy, node);

      for (int d = 0; d < nDim; ++d) {
        cxyz[d] = mxyz[d];
      }
    } // end for loop - in index
  }   // end for loop - bkts

  // create mesh transformation algorithm class
  std::unique_ptr<sierra::nalu::MeshTransformationAlg> meshTransformationAlg;
  meshTransformationAlg.reset(new sierra::nalu::MeshTransformationAlg(
    realm.bulk_data(), mesh_transformation));

  // create mesh motion algorithm class
  std::unique_ptr<sierra::nalu::MeshMotionAlg> meshMotionAlg;
  meshMotionAlg.reset(
    new sierra::nalu::MeshMotionAlg(realm.bulk_data(), mesh_motion));

  // initialize and execute mesh motion algorithm
  double currTime = 0.0;
  meshTransformationAlg->initialize(currTime);
  meshMotionAlg->initialize(currTime);

  // execute mesh motion algorithm
  currTime = 30.0;
  meshMotionAlg->execute(currTime);

  // get fields to be tested
  VectorFieldType* currCoords = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates");
  VectorFieldType* meshVelocity = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_velocity");

  // sync modified coordinates and velocity to host
  currCoords->sync_to_host();
  meshVelocity->sync_to_host();

  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* oxyz = stk::mesh::field_data(*modelCoordsCopy, node);
      double* xyz = stk::mesh::field_data(*currCoords, node);
      double* vel = stk::mesh::field_data(*meshVelocity, node);

      sierra::nalu::mm::TransMatType transMat =
        eval_transformation(realm, currTime, oxyz);

      std::vector<double> gold_norm_xyz = eval_coords(transMat, oxyz);
      std::vector<double> gold_norm_vel =
        eval_vel(currTime, transMat, oxyz, &gold_norm_xyz[0]);

      EXPECT_NEAR(xyz[0], gold_norm_xyz[0], testTol);
      EXPECT_NEAR(xyz[1], gold_norm_xyz[1], testTol);
      EXPECT_NEAR(xyz[2], gold_norm_xyz[2], testTol);

      EXPECT_NEAR(vel[0], gold_norm_vel[0], testTol);
      EXPECT_NEAR(vel[1], gold_norm_vel[1], testTol);
      EXPECT_NEAR(vel[2], gold_norm_vel[2], testTol);
    } // end for loop - in index
  }   // end for loop - bkts
}
