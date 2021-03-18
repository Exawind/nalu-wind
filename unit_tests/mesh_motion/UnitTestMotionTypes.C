#include <gtest/gtest.h>
#include <limits>

#include "mesh_motion/MotionDeformingInteriorKernel.h"
#include "mesh_motion/MotionRotationKernel.h"
#include "mesh_motion/MotionScalingKernel.h"
#include "mesh_motion/MotionTranslationKernel.h"

#include "UnitTestRealm.h"

namespace {

  const double testTol = 1e-14;

  std::vector<double> transform(
    const sierra::nalu::mm::TransMatType& transMat,
    const sierra::nalu::mm::ThreeDVecType& xyz)
  {
    std::vector<double> transCoord(3,0.0);

    // perform matrix multiplication between transformation matrix
    // and original coordinates to obtain transformed coordinates
    for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; d++) {
      transCoord[d] = transMat[d*sierra::nalu::mm::matSize+0]*xyz[0]
                     +transMat[d*sierra::nalu::mm::matSize+1]*xyz[1]
                     +transMat[d*sierra::nalu::mm::matSize+2]*xyz[2]
                     +transMat[d*sierra::nalu::mm::matSize+3];
    }

    return transCoord;
  }

}

TEST(meshMotion, rotation_omega)
{
  // create a yaml node describing rotation
  const std::string rotInfo =
    "omega: 3.0              \n"
    "centroid: [0.3,0.5,0.0] \n"
    ;

  YAML::Node rotNode = YAML::Load(rotInfo);

  // initialize the mesh rotation class
  sierra::nalu::MotionRotationKernel rotClass(rotNode);

  // build transformation
  const double time = 3.5;
  sierra::nalu::mm::ThreeDVecType xyz{2.5,1.5,6.5};
  sierra::nalu::mm::TransMatType transMat = rotClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  const double gold_norm_x =  0.133514518380489;
  const double gold_norm_y = -1.910867599933667;
  const double gold_norm_z =  6.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  sierra::nalu::mm::ThreeDVecType tmp;
  sierra::nalu::mm::ThreeDVecType vel = rotClass.compute_velocity(time, transMat, tmp, xyz);

  const double gold_norm_vx = -3.0;
  const double gold_norm_vy =  6.6;
  const double gold_norm_vz =  0.0;

  EXPECT_NEAR(vel[0], gold_norm_vx, testTol);
  EXPECT_NEAR(vel[1], gold_norm_vy, testTol);
  EXPECT_NEAR(vel[2], gold_norm_vz, testTol);
}

TEST(meshMotion, rotation_angle)
{
  // create a yaml node describing rotation
  const std::string rotInfo =
    "angle: 180            \n"
    "centroid: [0.3,0.5,0.0] \n"
    ;

  YAML::Node rotNode = YAML::Load(rotInfo);

  // initialize the mesh rotation class
  sierra::nalu::MotionRotationKernel rotClass(rotNode);

  // build transformation
  const double time = 0.0;
  sierra::nalu::mm::ThreeDVecType xyz{2.5,1.5,6.5};
  sierra::nalu::mm::TransMatType transMat = rotClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  const double gold_norm_x = -1.9;
  const double gold_norm_y = -0.5;
  const double gold_norm_z =  6.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);
}

TEST(meshMotion, scaling)
{
  // create a yaml node describing scaling
  const std::string scaleInfo =
    "factor: [2.0,2.0,1.0] \n"
    "centroid: [0.3,0.5,0.0] \n"
    ;

  YAML::Node scaleNode = YAML::Load(scaleInfo);

  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();

  // initialize the mesh scaling class
  sierra::nalu::MotionScalingKernel scaleClass(realm.meta_data(), scaleNode);

  // build transformation
  const double time = 0.0;
  sierra::nalu::mm::ThreeDVecType xyz{2.5,1.5,6.5};
  sierra::nalu::mm::TransMatType transMat = scaleClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  const double gold_norm_x = 4.7;
  const double gold_norm_y = 2.5;
  const double gold_norm_z = 6.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);
}

TEST(meshMotion, translation_velocity)
{
  // create a yaml node describing translation
  const std::string transInfo =
    "start_time: 15.0        \n"
    "end_time: 25.0          \n"
    "velocity: [1.5, 3.5, 2] \n"
    ;

  YAML::Node transNode = YAML::Load(transInfo);

  // initialize the mesh translation class
  sierra::nalu::MotionTranslationKernel transClass(transNode);

  // build transformation at t = 10.0
  double time = 10.0;
  sierra::nalu::mm::ThreeDVecType xyz{2.5,1.5,6.5};
  sierra::nalu::mm::TransMatType transMat = transClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  double gold_norm_x = xyz[0];
  double gold_norm_y = xyz[1];
  double gold_norm_z = xyz[2];

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  // build transformation at t = 20.0
  time = 20.0;
  transMat = transClass.build_transformation(time, xyz);
  norm = transform(transMat, xyz);

  gold_norm_x = 10.0;
  gold_norm_y = 19.0;
  gold_norm_z = 16.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  // build transformation at t = 30.0
  time = 30.0;
  transMat = transClass.build_transformation(time, xyz);
  norm = transform(transMat, xyz);
  
  gold_norm_x = 17.5;
  gold_norm_y = 36.5;
  gold_norm_z = 26.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);
}

TEST(meshMotion, translation_displacement)
{
  // create a yaml node describing translation
  const std::string transInfo =
    "displacement: [1.5, 3.5, 2] \n"
    ;

  YAML::Node transNode = YAML::Load(transInfo);

  // initialize the mesh translation class
  sierra::nalu::MotionTranslationKernel transClass(transNode);

  // build transformation
  const double time = 0.0;
  sierra::nalu::mm::ThreeDVecType xyz{2.5,1.5,6.5};
  sierra::nalu::mm::TransMatType transMat = transClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);
  
  const double gold_norm_x = 4.0;
  const double gold_norm_y = 5.0;
  const double gold_norm_z = 8.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);
}

TEST(meshMotion, deform_interior_outside_node)
{
  // create a yaml node describing translation
  const std::string deformInfo =
    "xyz_min: [0,0,0]         \n"
    "xyz_max: [15,5,5]        \n"
    "amplitude: [1.5,0.0,1.5] \n"
    "frequency: [0.1,0.0,0.1] \n"
    "centroid: [7.5,2.5,2.5]  \n"
    ;

  YAML::Node deformNode = YAML::Load(deformInfo);

  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();

  // initialize the mesh translation class
  sierra::nalu::MotionDeformingInteriorKernel deformClass(realm.meta_data(),deformNode);

  // build transformation
  const double time = 1.66666667;
  sierra::nalu::mm::ThreeDVecType xyz{9,7,3.5};
  sierra::nalu::mm::TransMatType transMat = deformClass.build_transformation(time, xyz);
  std::vector<double> currCoord = transform(transMat, xyz);

  const double gold_norm_x = 9.0;
  const double gold_norm_y = 7.0;
  const double gold_norm_z = 3.5;

  EXPECT_NEAR(currCoord[0], gold_norm_x, testTol);
  EXPECT_NEAR(currCoord[1], gold_norm_y, testTol);
  EXPECT_NEAR(currCoord[2], gold_norm_z, testTol);

  sierra::nalu::mm::ThreeDVecType tmp;
  sierra::nalu::mm::ThreeDVecType vel = deformClass.compute_velocity(time, transMat, xyz, tmp);

  const double gold_norm_vx = 0.0;
  const double gold_norm_vy = 0.0;
  const double gold_norm_vz = 0.0;

  EXPECT_NEAR(vel[0], gold_norm_vx, testTol);
  EXPECT_NEAR(vel[1], gold_norm_vy, testTol);
  EXPECT_NEAR(vel[2], gold_norm_vz, testTol);
}

TEST(meshMotion, deform_interior_inside_node)
{
  // create a yaml node describing translation
  const std::string deformInfo =
    "xyz_min: [0,0,0]         \n"
    "xyz_max: [15,5,5]        \n"
    "amplitude: [1.5,0.0,1.5] \n"
    "frequency: [0.1,0.0,0.1] \n"
    "centroid: [7.5,2.5,2.5]  \n"
    ;

  YAML::Node deformNode = YAML::Load(deformInfo);

  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();

  // initialize the mesh translation class
  sierra::nalu::MotionDeformingInteriorKernel deformClass(realm.meta_data(),deformNode);

  // build transformation
  const double time = 1.66666667;
  sierra::nalu::mm::ThreeDVecType xyz{9.0,4,1.5};
  sierra::nalu::mm::TransMatType transMat = deformClass.build_transformation(time, xyz);
  std::vector<double> currCoord = transform(transMat, xyz);

  const double gold_norm_x = 9.7500000027207;
  const double gold_norm_y = 4.0;
  const double gold_norm_z = 0.749999997279301;

  EXPECT_NEAR(currCoord[0], gold_norm_x, testTol);
  EXPECT_NEAR(currCoord[1], gold_norm_y, testTol);
  EXPECT_NEAR(currCoord[2], gold_norm_z, testTol);

  sierra::nalu::mm::ThreeDVecType tmp;
  sierra::nalu::mm::ThreeDVecType vel = deformClass.compute_velocity(time, transMat, xyz, tmp);

  const double gold_norm_vx = 0.816209714892358;
  const double gold_norm_vy = 0.0;
  const double gold_norm_vz = -0.816209714892358;

  EXPECT_NEAR(vel[0], gold_norm_vx, testTol);
  EXPECT_NEAR(vel[1], gold_norm_vy, testTol);
  EXPECT_NEAR(vel[2], gold_norm_vz, testTol);

}
