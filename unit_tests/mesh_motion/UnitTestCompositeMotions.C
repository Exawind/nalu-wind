#include <gtest/gtest.h>
#include <limits>

#include "mesh_motion/MotionRotation.h"
#include "mesh_motion/MotionTranslation.h"

#include<iostream>

namespace {

  // create a yaml node describing mesh motion
  const std::string mInfo =
    "    motion:                        \n"
    "     - type: rotation              \n"
    "       omega: 3.0                  \n"
    "       centroid: [0.3,0.5,0.0]     \n"
    "       axis: [0.0, 0.0, 1.0]       \n"
    "                                   \n"
    "     - type: translation           \n"
    "       velocity: [2.0, 0.0, 0.0]   \n"
    ;

  // define YAML nodes at different levels
  const YAML::Node motionNode = YAML::Load(mInfo);
  const YAML::Node motion     = motionNode["motion"];
  const YAML::Node rotNode    = motion[0];
  const YAML::Node transNode  = motion[1];

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

TEST(meshMotion, composite_motions)
{
  // build transformation
  const double time = 3.5;
  double xyz[3] = {2.5,1.5,6.5};

  // initialize composite trasnformation matrix
  sierra::nalu::MotionBase::TransMatType comp_trans
    = sierra::nalu::MotionBase::identityMat_;

  // initialize the mesh rotation class
  sierra::nalu::MotionRotation rotClass(rotNode);
  rotClass.build_transformation(time, xyz);
  std::vector<double> rotCoord = transform(rotClass.get_trans_mat(), xyz);
  comp_trans = rotClass.add_motion(rotClass.get_trans_mat(), comp_trans);

  // compute rotational velocity in absence of translation
  sierra::nalu::MotionBase::ThreeDVecType rotVel =
    rotClass.compute_velocity(time, rotClass.get_trans_mat(), xyz, &rotCoord[0]);

  // initialize the mesh translation class
  sierra::nalu::MotionTranslation transClass(transNode);
  transClass.build_transformation(time, xyz);
  comp_trans = transClass.add_motion(transClass.get_trans_mat(), comp_trans);
  std::vector<double> newCoord = transform(comp_trans, xyz);

  // compute rotational velocity in absence of translation
  sierra::nalu::MotionBase::ThreeDVecType compVelRot =
    rotClass.compute_velocity(time, comp_trans, xyz, &newCoord[0]);

  // ensure the rotational componenets of the velocity remains same
  EXPECT_NEAR(compVelRot[0], rotVel[0], testTol);
  EXPECT_NEAR(compVelRot[1], rotVel[1], testTol);
  EXPECT_NEAR(compVelRot[2], rotVel[2], testTol);
}
