
#include "mesh_motion/MotionRotationKernel.h"

#include <NaluEnv.h>
#include <NaluParsing.h>

namespace sierra{
namespace nalu{

MotionRotationKernel::MotionRotationKernel(const YAML::Node& node)
  : NgpMotionKernel<MotionRotationKernel>()
{
  load(node);
}

void MotionRotationKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_-DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_+DBL_EPSILON;

  // rotation could be based on angular velocity or angle
  get_if_present(node, "omega", omega_, omega_);

  get_if_present(node, "angle", angle_, angle_);

  // default approach is to use omega
  useOmega_ = ( node["angle"] ? false : true);

  if( node["axis"] ) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      axis_[d] = node["axis"][d].as<double>();
  }
  else
    NaluEnv::self().naluOutputP0() << "MotionRotationKernel: axis of rotation not supplied; will use 0,0,1" << std::endl;

  // get origin based on if it was defined or is to be computed
  if( node["centroid"] ) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      origin_[d] = node["centroid"][d].as<double>();
  }
}

void MotionRotationKernel::build_transformation(
  const double& time,
  const ThreeDVecType& /* xyz */,
  TransMatType& transMat)
{
  reset_mat(transMat);

  if(time < (startTime_)) return;
  double motionTime = (time < endTime_)? time : endTime_;

  // determine current angle
  double angle = 0.0;
  if (useOmega_)
    angle = omega_*(motionTime-startTime_);
  else
    angle = angle_*M_PI/180;

  // Build matrix for translating object to cartesian origin
  TransMatType tempMat = {};
  reset_mat(tempMat);
  tempMat[0][3] = -origin_[0];
  tempMat[1][3] = -origin_[1];
  tempMat[2][3] = -origin_[2];

  // Build matrix for rotating object
  // compute magnitude of axis around which to rotate
  double mag = 0.0;
  for (int d=0; d < nalu_ngp::NDimMax; d++)
      mag += axis_[d] * axis_[d];
  mag = stk::math::sqrt(mag);

  // build quaternion based on angle and axis of rotation
  const double cosang = stk::math::cos(0.5*angle);
  const double sinang = stk::math::sin(0.5*angle);
  const double q0 = cosang;
  const double q1 = sinang * axis_[0]/mag;
  const double q2 = sinang * axis_[1]/mag;
  const double q3 = sinang * axis_[2]/mag;

  // rotation matrix based on quaternion
  TransMatType tempMat2 = {};
  // 1st row
  tempMat2[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
  tempMat2[0][1] = 2.0*(q1*q2 - q0*q3);
  tempMat2[0][2] = 2.0*(q0*q2 + q1*q3);
  // 2nd row
  tempMat2[1][0] = 2.0*(q1*q2 + q0*q3);
  tempMat2[1][1] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
  tempMat2[1][2] = 2.0*(q2*q3 - q0*q1);
  // 3rd row
  tempMat2[2][0] = 2.0*(q1*q3 - q0*q2);
  tempMat2[2][1] = 2.0*(q0*q1 + q2*q3);
  tempMat2[2][2] = q0*q0 - q1*q1 - q2*q2 + q3*q3;
  // 4th row
  tempMat2[3][3] = 1.0;

  // composite addition of motions in current group
  TransMatType tempMat3 = {};
  add_motion(tempMat2,tempMat,tempMat3);

  // Build matrix for translating object back to its origin
  reset_mat(tempMat);
  tempMat[0][3] = origin_[0];
  tempMat[1][3] = origin_[1];
  tempMat[2][3] = origin_[2];

  // composite addition of motions
  add_motion(tempMat,tempMat3,transMat);
}

void MotionRotationKernel::compute_velocity(
  const double& time,
  const TransMatType& compTrans,
  const ThreeDVecType& /* mxyz */,
  const ThreeDVecType& cxyz,
  ThreeDVecType& vel )
{
  if((time < startTime_) || (time > endTime_)) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      vel[d] = 0.0;

    return;
  }

  // construct unit vector
  ThreeDVecType unitVec = {};

  double mag = 0.0;
  for (int d=0; d < nalu_ngp::NDimMax; d++)
    mag += axis_[d] * axis_[d];
  mag = stk::math::sqrt(mag);

  unitVec[0] = axis_[0]/mag;
  unitVec[1] = axis_[1]/mag;
  unitVec[2] = axis_[2]/mag;

  // transform the origin of the rotating body
  ThreeDVecType transOrigin = {};
  for (int d = 0; d < nalu_ngp::NDimMax; d++) {
    transOrigin[d] = compTrans[d][0]*origin_[0]
                    +compTrans[d][1]*origin_[1]
                    +compTrans[d][2]*origin_[2]
                    +compTrans[d][3];
  }

  // compute relative coords and vector omega (dimension 3) for general cross product
  ThreeDVecType relCoord = {};
  ThreeDVecType vecOmega = {};
  for (int d=0; d < nalu_ngp::NDimMax; d++) {
    relCoord[d] = cxyz[d] - transOrigin[d];
    vecOmega[d] = omega_*unitVec[d];
  }

  // cross product v = \omega \cross \x
  vel[0] = vecOmega[1]*relCoord[2] - vecOmega[2]*relCoord[1];
  vel[1] = vecOmega[2]*relCoord[0] - vecOmega[0]*relCoord[2];
  vel[2] = vecOmega[0]*relCoord[1] - vecOmega[1]*relCoord[0];
}

} // nalu
} // sierra
