#include <gtest/gtest.h>
#include <limits>

#include "mesh_motion/MotionRotation.h"
#include "mesh_motion/MotionScaling.h"
#include "mesh_motion/MotionTranslation.h"

namespace {

  const double testTol = 1e-14;

  std::vector<double> transform(
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
  sierra::nalu::MotionRotation rotClass(rotNode);

  // build transformation
  const double time = 3.5;
  double xyz[3] = {2.5,1.5,6.5};
  rotClass.build_transformation(time, xyz);

  std::vector<double> norm = transform(rotClass.get_trans_mat(), xyz);

  const double gold_norm_x =  0.133514518380489;
  const double gold_norm_y = -1.910867599933667;
  const double gold_norm_z =  6.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  sierra::nalu::MotionBase::ThreeDVecType vel =
    rotClass.compute_velocity(time, rotClass.get_trans_mat(), &xyz[0]);

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
  sierra::nalu::MotionRotation rotClass(rotNode);

  // build transformation
  const double time = 0.0;
  double xyz[3] = {2.5,1.5,6.5};
  rotClass.build_transformation(time, xyz);

  std::vector<double> norm = transform(rotClass.get_trans_mat(), xyz);

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

  // initialize the mesh scaling class
  sierra::nalu::MotionScaling scaleClass(scaleNode);

  // build transformation
  const double time = 0.0;
  double xyz[3] = {2.5,1.5,6.5};
  scaleClass.build_transformation(time, xyz);

  std::vector<double> norm = transform(scaleClass.get_trans_mat(), xyz);

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
  sierra::nalu::MotionTranslation transClass(transNode);

  // build transformation at t = 10.0
  double time = 10.0;
  double xyz[3] = {2.5,1.5,6.5};
  transClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transClass.get_trans_mat(), xyz);

  double gold_norm_x = xyz[0];
  double gold_norm_y = xyz[1];
  double gold_norm_z = xyz[2];

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  // build transformation at t = 20.0
  time = 20.0;
  transClass.build_transformation(time, xyz);
  norm = transform(transClass.get_trans_mat(), xyz);

  gold_norm_x = 10.0;
  gold_norm_y = 19.0;
  gold_norm_z = 16.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  // build transformation at t = 30.0
  time = 30.0;
  transClass.build_transformation(time, xyz);
  norm = transform(transClass.get_trans_mat(), xyz);
  
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
  sierra::nalu::MotionTranslation transClass(transNode);

  // build transformation
  const double time = 0.0;
  double xyz[3] = {2.5,1.5,6.5};
  transClass.build_transformation(time, xyz);

  std::vector<double> norm = transform(transClass.get_trans_mat(), xyz);
  
  const double gold_norm_x = 4.0;
  const double gold_norm_y = 5.0;
  const double gold_norm_z = 8.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);
}
