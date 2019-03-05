#include <gtest/gtest.h>
#include <limits>

#include "mesh_motion/MeshMotionAlg.h"
#include "mesh_motion/MotionRotation.h"
#include "mesh_motion/MotionScaling.h"
#include "mesh_motion/MotionTranslation.h"
#include "Realm.h"
#include "SolutionOptions.h"

#include "UnitTestRealm.h"
#include "UnitTestUtils.h"

#include <string>

namespace {
  // create a yaml node describing mesh motion
  const std::string mInfo =
    "mesh_motion:                       \n"
    "                                   \n"
    "  - name: scale                    \n"
    "    frame: inertial                \n"
    "    motion:                        \n"
    "     - type: scaling               \n"
    "       factor: [1.2,  1.0, 1.2]    \n"
    "                                   \n"
    "  - name: trans_rot                \n"
    "    mesh_parts: [ block_1 ]        \n"
    "    frame: non_inertial            \n"
    "    reference: scale               \n"
    "    motion:                        \n"
    "     - type: rotation              \n"
    "       omega: 3.0                  \n"
    "       axis: [0.0, 0.0, 1.0]       \n"
    "                                   \n"
    "     - type: translation           \n"
    "       start_time: 15.0            \n"
    "       end_time: 25.0              \n"
    "       velocity: [2.0, 0.0, 0.0]   \n"
    ;

  // define YAML nodes at different levels
  const YAML::Node meshMotionNode = YAML::Load(mInfo);
  const YAML::Node frames         = meshMotionNode["mesh_motion"];

  const YAML::Node frame_first = frames[0];
  const YAML::Node scale       = frame_first["motion"];
  const YAML::Node scaleNode   = scale[0];

  const YAML::Node frame_second = frames[1];
  const YAML::Node rot_trans    = frame_second["motion"];
  const YAML::Node rotNode      = rot_trans[0];
  const YAML::Node transNode    = rot_trans[1];

  const double testTol = 1e-12;

  sierra::nalu::MotionBase::TransMatType eval_transformation(
    double time,
    const double* xyz)
  {
    // initialize composite trasnformation matrix
    sierra::nalu::MotionBase::TransMatType comp_trans
      = sierra::nalu::MotionBase::identityMat_;

    // perform scaling transformation
    sierra::nalu::MotionScaling scaleClass(scaleNode);
    scaleClass.build_transformation(time, xyz);
    comp_trans = scaleClass.add_motion(scaleClass.get_trans_mat(), comp_trans);

    // perform rotation transformation
    sierra::nalu::MotionRotation rotClass(rotNode);
    rotClass.build_transformation(time, xyz);
    comp_trans = rotClass.add_motion(rotClass.get_trans_mat(), comp_trans);

    // perform translation transformation
    const double startTime = transNode["start_time"].as<double>();
    const double endTime   = transNode["end_time"].as<double>();

    if( time >= (startTime-testTol) )
    {
      if( time >= (endTime+testTol) )
        time = endTime;

      sierra::nalu::MotionTranslation transClass(transNode);
      transClass.build_transformation(time, xyz);
      comp_trans = transClass.add_motion(transClass.get_trans_mat(), comp_trans);
    }

    return comp_trans;
  }

  std::vector<double> eval_coords(
    const sierra::nalu::MotionBase::TransMatType& transMat,
    const double* xyz )
  {
    std::vector<double> transCoord(3,0.0);

    // perform matrix multiplication between transformation matrix
    // and original coordinates to obtain transformed coordinates
    for (int d = 0; d < sierra::nalu::MotionBase::threeDVecSize; d++) {
      transCoord[d] = transMat[d][0]*xyz[0]
                     +transMat[d][1]*xyz[1]
                     +transMat[d][2]*xyz[2]
                     +transMat[d][3];
    }

    return transCoord;
  }

  std::vector<double> eval_vel(
    const double time,
    const sierra::nalu::MotionBase::TransMatType& transMat,
    double* xyz )
  {
    std::vector<double> vel(3,0.0);

    // perform rotation transformation
    sierra::nalu::MotionRotation rotClass(rotNode);
    sierra::nalu::MotionBase::ThreeDVecType motionVel =
      rotClass.compute_velocity(time, transMat, xyz);

    for (size_t d = 0; d < vel.size(); d++)
      vel[d] += motionVel[d];

    // perform translation transformation
    const double startTime = transNode["start_time"].as<double>();
    const double endTime   = transNode["end_time"].as<double>();

    if( (time >= (startTime-testTol)) && (time <= (endTime+testTol)) )
    {
      sierra::nalu::MotionTranslation transClass(transNode);
      motionVel = transClass.compute_velocity(time, transMat, xyz);

      for (size_t d = 0; d < vel.size(); d++)
        vel[d] += motionVel[d];
    }

    return vel;
  }
}

TEST(meshMotion, meshMotionAlg_initialize)
{
  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();
  realm.solutionOptions_->meshMotion_ = true;

  // register mesh motion fields and initialize coordinate fields
  realm.register_nodal_fields( &(realm.meta_data().universal_part()) );
  realm.init_current_coordinates();

  // create mesh
  const std::string meshSpec("generated:2x2x2");
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());

  // create mesh motion algorithm class
  std::unique_ptr<sierra::nalu::MeshMotionAlg> meshMotionAlg;
  meshMotionAlg.reset(
    new sierra::nalu::MeshMotionAlg( realm.bulk_data(), meshMotionNode["mesh_motion"] ));

  // get relevant fields
  VectorFieldType* modelCoords = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* currCoords = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates");
  VectorFieldType* meshVelocity = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_velocity");

  // get the parts in the current motion frame
  stk::mesh::Selector sel = stk::mesh::Selector(realm.meta_data().universal_part())
    & (realm.meta_data().locally_owned_part() | realm.meta_data().globally_shared_part());
  const auto& bkts = realm.bulk_data().get_buckets(stk::topology::NODE_RANK, sel);

  /////////////////////////////////////////////////////////////
  // initialize and execute mesh motion algorithm
  const double currTime = 0.0;
  meshMotionAlg->initialize(currTime);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* oxyz = stk::mesh::field_data( *modelCoords, node);
      double*  xyz = stk::mesh::field_data(  *currCoords, node);
      double*  vel = stk::mesh::field_data(*meshVelocity, node);

      sierra::nalu::MotionBase::TransMatType transMat =
        eval_transformation(currTime, oxyz);

      std::vector<double> gold_norm_xyz = eval_coords(transMat, oxyz);
      std::vector<double> gold_norm_vel = eval_vel(currTime, transMat, &gold_norm_xyz[0]);

      EXPECT_NEAR(xyz[0], gold_norm_xyz[0], testTol);
      EXPECT_NEAR(xyz[1], gold_norm_xyz[1], testTol);
      EXPECT_NEAR(xyz[2], gold_norm_xyz[2], testTol);

      EXPECT_NEAR(vel[0], gold_norm_vel[0], testTol);
      EXPECT_NEAR(vel[1], gold_norm_vel[1], testTol);
      EXPECT_NEAR(vel[2], gold_norm_vel[2], testTol);
    } // end for loop - in index
  } // end for loop - bkts
}

TEST(meshMotion, meshMotionAlg_execute)
{
  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();
  realm.solutionOptions_->meshMotion_ = true;

  // register mesh motion fields and initialize coordinate fields
  realm.register_nodal_fields( &(realm.meta_data().universal_part()) );
  realm.init_current_coordinates();

  // create mesh
  const std::string meshSpec("generated:2x2x2");
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());

  // create mesh motion algorithm class
  const YAML::Node meshMotionNode = YAML::Load(mInfo);
  std::unique_ptr<sierra::nalu::MeshMotionAlg> meshMotionAlg;
  meshMotionAlg.reset(
    new sierra::nalu::MeshMotionAlg( realm.bulk_data(), meshMotionNode["mesh_motion"] ));

  // get relevant fields
  VectorFieldType* modelCoords = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* currCoords = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates");
  VectorFieldType* meshVelocity = realm.meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_velocity");

  // get the parts in the current motion frame
  stk::mesh::Selector sel = stk::mesh::Selector(realm.meta_data().universal_part())
    & (realm.meta_data().locally_owned_part() | realm.meta_data().globally_shared_part());
  const auto& bkts = realm.bulk_data().get_buckets(stk::topology::NODE_RANK, sel);

  /////////////////////////////////////////////////////////////
  // execute mesh motion algorithm
  double currTime = 0.0;
  meshMotionAlg->initialize(currTime);

  // execute mesh motion algorithm
  currTime = 30.0;
  meshMotionAlg->execute(currTime);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* oxyz = stk::mesh::field_data( *modelCoords, node);
      double*  xyz = stk::mesh::field_data(  *currCoords, node);
      double*  vel = stk::mesh::field_data(*meshVelocity, node);

      sierra::nalu::MotionBase::TransMatType transMat =
        eval_transformation(currTime, oxyz);

      std::vector<double> gold_norm_xyz = eval_coords(transMat, oxyz);
      std::vector<double> gold_norm_vel = eval_vel(currTime, transMat, &gold_norm_xyz[0]);

      EXPECT_NEAR(xyz[0], gold_norm_xyz[0], testTol);
      EXPECT_NEAR(xyz[1], gold_norm_xyz[1], testTol);
      EXPECT_NEAR(xyz[2], gold_norm_xyz[2], testTol);

      EXPECT_NEAR(vel[0], gold_norm_vel[0], testTol);
      EXPECT_NEAR(vel[1], gold_norm_vel[1], testTol);
      EXPECT_NEAR(vel[2], gold_norm_vel[2], testTol);
    } // end for loop - in index
  } // end for loop - bkts
}
