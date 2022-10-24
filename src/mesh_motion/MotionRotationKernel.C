
#include "mesh_motion/MotionRotationKernel.h"

#include <NaluEnv.h>
#include <NaluParsing.h>

namespace sierra {
namespace nalu {

MotionRotationKernel::MotionRotationKernel(const YAML::Node& node)
  : NgpMotionKernel<MotionRotationKernel>()
{
  load(node);
}

void
MotionRotationKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_ - DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_ + DBL_EPSILON;

  // rotation could be based on angular velocity or angle
  get_if_present(node, "omega", omega_, omega_);

  get_if_present(node, "angle", angle_, angle_);

  // default approach is to use omega
  useOmega_ = (node["angle"] ? false : true);

  if (node["axis"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      axis_[d] = node["axis"][d].as<double>();
  } else
    NaluEnv::self().naluOutputP0()
      << "MotionRotationKernel: axis of rotation not supplied; will use 0,0,1"
      << std::endl;

  // get origin based on if it was defined or is to be computed
  if (node["centroid"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      origin_[d] = node["centroid"][d].as<double>();
  }
}

KOKKOS_FUNCTION
mm::TransMatType
MotionRotationKernel::build_transformation(
  const double& time, const mm::ThreeDVecType& /* xyz */)
{
  mm::TransMatType transMat;

  if (time < (startTime_))
    return transMat;
  double motionTime = (time < endTime_) ? time : endTime_;

  // determine current angle
  double angle = 0.0;
  if (useOmega_)
    angle = omega_ * (motionTime - startTime_);
  else
    angle = angle_ * M_PI / 180;

  // Build matrix for translating object to cartesian origin
  transMat[0 * mm::matSize + 3] = -origin_[0];
  transMat[1 * mm::matSize + 3] = -origin_[1];
  transMat[2 * mm::matSize + 3] = -origin_[2];

  // Build matrix for rotating object
  // compute magnitude of axis around which to rotate
  double mag = 0.0;
  for (int d = 0; d < nalu_ngp::NDimMax; d++)
    mag += axis_[d] * axis_[d];
  mag = stk::math::sqrt(mag);

  // build quaternion based on angle and axis of rotation
  const double cosang = stk::math::cos(0.5 * angle);
  const double sinang = stk::math::sin(0.5 * angle);
  const double q0 = cosang;
  const double q1 = sinang * axis_[0] / mag;
  const double q2 = sinang * axis_[1] / mag;
  const double q3 = sinang * axis_[2] / mag;

  // rotation matrix based on quaternion
  mm::TransMatType tempMat;
  // 1st row
  tempMat[0 * mm::matSize + 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
  tempMat[0 * mm::matSize + 1] = 2.0 * (q1 * q2 - q0 * q3);
  tempMat[0 * mm::matSize + 2] = 2.0 * (q0 * q2 + q1 * q3);
  // 2nd row
  tempMat[1 * mm::matSize + 0] = 2.0 * (q1 * q2 + q0 * q3);
  tempMat[1 * mm::matSize + 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3;
  tempMat[1 * mm::matSize + 2] = 2.0 * (q2 * q3 - q0 * q1);
  // 3rd row
  tempMat[2 * mm::matSize + 0] = 2.0 * (q1 * q3 - q0 * q2);
  tempMat[2 * mm::matSize + 1] = 2.0 * (q0 * q1 + q2 * q3);
  tempMat[2 * mm::matSize + 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

  // composite addition of motions in current group
  transMat = add_motion(tempMat, transMat);

  // Build matrix for translating object back to its origin
  tempMat = mm::TransMatType::I();
  tempMat[0 * mm::matSize + 3] = origin_[0];
  tempMat[1 * mm::matSize + 3] = origin_[1];
  tempMat[2 * mm::matSize + 3] = origin_[2];

  // composite addition of motions
  return add_motion(tempMat, transMat);
}

KOKKOS_FUNCTION
mm::ThreeDVecType
MotionRotationKernel::compute_velocity(
  const double& time,
  const mm::TransMatType& compTrans,
  const mm::ThreeDVecType& /* mxyz */,
  const mm::ThreeDVecType& cxyz)
{
  mm::ThreeDVecType vel;

  if ((time < startTime_) || (time > endTime_))
    return vel;

  // construct unit vector
  mm::ThreeDVecType unitVec;

  double mag = 0.0;
  for (int d = 0; d < nalu_ngp::NDimMax; d++)
    mag += axis_[d] * axis_[d];
  mag = stk::math::sqrt(mag);

  unitVec[0] = axis_[0] / mag;
  unitVec[1] = axis_[1] / mag;
  unitVec[2] = axis_[2] / mag;

  // transform the origin of the rotating body
  mm::ThreeDVecType transOrigin;
  for (int d = 0; d < nalu_ngp::NDimMax; d++) {
    transOrigin[d] = compTrans[d * mm::matSize + 0] * origin_[0] +
                     compTrans[d * mm::matSize + 1] * origin_[1] +
                     compTrans[d * mm::matSize + 2] * origin_[2] +
                     compTrans[d * mm::matSize + 3];
  }

  // compute relative coords and vector omega (dimension 3) for general cross
  // product
  mm::ThreeDVecType relCoord;
  mm::ThreeDVecType vecOmega;
  for (int d = 0; d < nalu_ngp::NDimMax; d++) {
    relCoord[d] = cxyz[d] - transOrigin[d];
    vecOmega[d] = omega_ * unitVec[d];
  }

  // cross product v = \omega \cross \x
  vel[0] = vecOmega[1] * relCoord[2] - vecOmega[2] * relCoord[1];
  vel[1] = vecOmega[2] * relCoord[0] - vecOmega[0] * relCoord[2];
  vel[2] = vecOmega[0] * relCoord[1] - vecOmega[1] * relCoord[0];
  return vel;
}

} // namespace nalu
} // namespace sierra
