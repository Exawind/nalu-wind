
#include "mesh_motion/MotionRotation.h"

#include <NaluEnv.h>

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
  if(node["start_time"])
    startTime_ = node["start_time"].as<double>();

  if(node["end_time"])
    endTime_ = node["end_time"].as<double>();

  // rotation could be based on angular velocity or angle
  if(node["omega"]){
    useOmega_ = true;
    omega_ = node["omega"].as<double>();
  }
  if(node["angle"])
  {
    useOmega_ = false;
    angle_ = node["angle"].as<double>();
  }

  if( node["axis"] )
    axis_ = node["axis"].as<threeDVecType>();
  else
    NaluEnv::self().naluOutputP0() << "MotionRotation: axis of rotation not supplied; will use 0,0,1" << std::endl;

  // get origin based on if it was defined or is to be computed
  if( node["centroid"] )
    origin_ = node["centroid"].as<threeDVecType>();
}

void MotionRotation::build_transformation(
  const double time,
  const double* xyz)
{
  if( (time >= (startTime_-eps_)) && (time <= (endTime_+eps_)) )
  {
    // determine current angle
    double curr_angle = 0.0;
    if (useOmega_)
      curr_angle = omega_*(time-startTime_);
    else
      curr_angle = angle_*M_PI/180;

    rotation_mat(curr_angle);
  }
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
  transMatType curr_trans_mat_ = {};
  // 1st row
  curr_trans_mat_[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
  curr_trans_mat_[0][1] = 2.0*(q1*q2 - q0*q3);
  curr_trans_mat_[0][2] = 2.0*(q0*q2 + q1*q3);
  // 2nd row
  curr_trans_mat_[1][0] = 2.0*(q1*q2 + q0*q3);
  curr_trans_mat_[1][1] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
  curr_trans_mat_[1][2] = 2.0*(q2*q3 - q0*q1);
  // 3rd row
  curr_trans_mat_[2][0] = 2.0*(q1*q3 - q0*q2);
  curr_trans_mat_[2][1] = 2.0*(q0*q1 + q2*q3);
  curr_trans_mat_[2][2] = q0*q0 - q1*q1 - q2*q2 + q3*q3;
  // 4th row
  curr_trans_mat_[3][3] = 1.0;

  // composite addition of motions in current group
  transMat_ = add_motion(curr_trans_mat_,transMat_);

  // Build matrix for translating object back to its origin
  reset_mat(curr_trans_mat_);
  curr_trans_mat_[0][3] = origin_[0];
  curr_trans_mat_[1][3] = origin_[1];
  curr_trans_mat_[2][3] = origin_[2];

  // composite addition of motions
  transMat_ = add_motion(curr_trans_mat_,transMat_);
}

MotionBase::threeDVecType MotionRotation::compute_velocity(
  double time,
  const transMatType& comp_trans,
  double* xyz )
{
  threeDVecType vel = {};

  if( (time >= (startTime_-eps_)) && (time <= (endTime_+eps_)) )
  {
    // construct unit vector
    threeDVecType unitVec = {};

    double mag = 0.0;
    for (int d=0; d < threeDVecSize; d++)
      mag += axis_[d] * axis_[d];
    mag = std::sqrt(mag);

    unitVec[0] = axis_[0]/mag;
    unitVec[1] = axis_[1]/mag;
    unitVec[2] = axis_[2]/mag;

    // transform the origin of the rotating body
    threeDVecType trans_origin = {};
    for (int d = 0; d < threeDVecSize; d++) {
      trans_origin[d] = comp_trans[d][0]*origin_[0]
                       +comp_trans[d][1]*origin_[1]
                       +comp_trans[d][2]*origin_[2]
                       +comp_trans[d][3];
    }

    // compute relative coords and vector omega (dimension 3) for general cross product
    threeDVecType relCoord = {};
    threeDVecType vecOmega = {};
    for (int d=0; d < threeDVecSize; d++) {
      relCoord[d] = xyz[d] - trans_origin[d];
      vecOmega[d] = omega_*unitVec[d];
    }

    // cross product v = \omega \cross \x
    vel[0] = vecOmega[1]*relCoord[2] - vecOmega[2]*relCoord[1];
    vel[1] = vecOmega[2]*relCoord[0] - vecOmega[0]*relCoord[2];
    vel[2] = vecOmega[0]*relCoord[1] - vecOmega[1]*relCoord[0];
  }

  return vel;
}

} // nalu
} // sierra
