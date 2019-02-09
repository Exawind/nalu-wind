
#include "mesh_motion/MotionRotation.h"

#include <NaluEnv.h>
#include <NaluParsing.h>

#include <cmath>

namespace sierra{
namespace nalu{

MotionRotation::MotionRotation(const YAML::Node& node)
  : MotionBase()
{
  load(node);
}

void MotionRotation::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  double eps = std::numeric_limits<double>::epsilon();

  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_-eps;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_+eps;

  // rotation could be based on angular velocity or angle
  get_if_present(node, "omega", omega_, omega_);

  get_if_present(node, "angle", angle_, angle_);

  // default approach is to use omega
  useOmega_ = ( node["angle"] ? false : true);

  if( node["axis"] )
    axis_ = node["axis"].as<ThreeDVecType>();
  else
    NaluEnv::self().naluOutputP0() << "MotionRotation: axis of rotation not supplied; will use 0,0,1" << std::endl;

  // get origin based on if it was defined or is to be computed
  if( node["centroid"] )
    origin_ = node["centroid"].as<ThreeDVecType>();
}

void MotionRotation::build_transformation(
  double time,
  const double*  /* xyz */)
{
  if(time < (startTime_)) return;

  double motionTime = (time < endTime_)? time : endTime_;

  // determine current angle
  double curr_angle = 0.0;
  if (useOmega_)
    curr_angle = omega_*(motionTime-startTime_);
  else
    curr_angle = angle_*M_PI/180;

  rotation_mat(curr_angle);
}

void MotionRotation::rotation_mat(const double angle)
{
  reset_mat(transMat_);

  // Build matrix for translating object to cartesian origin
  transMat_[0][3] = -origin_[0];
  transMat_[1][3] = -origin_[1];
  transMat_[2][3] = -origin_[2];

  // Build matrix for rotating object
  // compute magnitude of axis around which to rotate
  double mag = 0.0;
  for (int d=0; d < threeDVecSize; d++)
      mag += axis_[d] * axis_[d];
  mag = std::sqrt(mag);

  // build quaternion based on angle and axis of rotation
  const double cosang = std::cos(0.5*angle);
  const double sinang = std::sin(0.5*angle);
  const double q0 = cosang;
  const double q1 = sinang * axis_[0]/mag;
  const double q2 = sinang * axis_[1]/mag;
  const double q3 = sinang * axis_[2]/mag;

  // rotation matrix based on quaternion
  TransMatType currTransMat = {};
  // 1st row
  currTransMat[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
  currTransMat[0][1] = 2.0*(q1*q2 - q0*q3);
  currTransMat[0][2] = 2.0*(q0*q2 + q1*q3);
  // 2nd row
  currTransMat[1][0] = 2.0*(q1*q2 + q0*q3);
  currTransMat[1][1] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
  currTransMat[1][2] = 2.0*(q2*q3 - q0*q1);
  // 3rd row
  currTransMat[2][0] = 2.0*(q1*q3 - q0*q2);
  currTransMat[2][1] = 2.0*(q0*q1 + q2*q3);
  currTransMat[2][2] = q0*q0 - q1*q1 - q2*q2 + q3*q3;
  // 4th row
  currTransMat[3][3] = 1.0;

  // composite addition of motions in current group
  transMat_ = add_motion(currTransMat,transMat_);

  // Build matrix for translating object back to its origin
  reset_mat(currTransMat);
  currTransMat[0][3] = origin_[0];
  currTransMat[1][3] = origin_[1];
  currTransMat[2][3] = origin_[2];

  // composite addition of motions
  transMat_ = add_motion(currTransMat,transMat_);
}

MotionBase::ThreeDVecType MotionRotation::compute_velocity(
  const double time,
  const TransMatType& compTrans,
  const double* xyz )
{
  ThreeDVecType vel = {};

  if( (time < startTime_) || (time > endTime_) ) return vel;

  // construct unit vector
  ThreeDVecType unitVec = {};

  double mag = 0.0;
  for (int d=0; d < threeDVecSize; d++)
    mag += axis_[d] * axis_[d];
  mag = std::sqrt(mag);

  unitVec[0] = axis_[0]/mag;
  unitVec[1] = axis_[1]/mag;
  unitVec[2] = axis_[2]/mag;

  // transform the origin of the rotating body
  ThreeDVecType transOrigin = {};
  for (int d = 0; d < threeDVecSize; d++) {
    transOrigin[d] = compTrans[d][0]*origin_[0]
                    +compTrans[d][1]*origin_[1]
                    +compTrans[d][2]*origin_[2]
                    +compTrans[d][3];
  }

  // compute relative coords and vector omega (dimension 3) for general cross product
  ThreeDVecType relCoord = {};
  ThreeDVecType vecOmega = {};
  for (int d=0; d < threeDVecSize; d++) {
    relCoord[d] = xyz[d] - transOrigin[d];
    vecOmega[d] = omega_*unitVec[d];
  }

  // cross product v = \omega \cross \x
  vel[0] = vecOmega[1]*relCoord[2] - vecOmega[2]*relCoord[1];
  vel[1] = vecOmega[2]*relCoord[0] - vecOmega[0]*relCoord[2];
  vel[2] = vecOmega[0]*relCoord[1] - vecOmega[1]*relCoord[0];

  return vel;
}

} // nalu
} // sierra
