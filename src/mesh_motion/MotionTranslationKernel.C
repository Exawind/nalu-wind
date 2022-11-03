#include "mesh_motion/MotionTranslationKernel.h"

#include <NaluParsing.h>

#include <cmath>

namespace sierra {
namespace nalu {

MotionTranslationKernel::MotionTranslationKernel(const YAML::Node& node)
  : NgpMotionKernel<MotionTranslationKernel>()
{
  load(node);
}

void
MotionTranslationKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_ - DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_ + DBL_EPSILON;

  // translation could be based on velocity or displacement
  if (node["velocity"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      velocity_[d] = node["velocity"][d].as<double>();
  }

  if (node["displacement"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      displacement_[d] = node["displacement"][d].as<double>();
  }

  // default approach is to use a constant displacement
  useVelocity_ = (node["velocity"] ? true : false);
}

KOKKOS_FUNCTION
mm::TransMatType
MotionTranslationKernel::build_transformation(
  const double& time, const mm::ThreeDVecType& /* mxyz */)
{
  mm::TransMatType transMat;

  if (time < (startTime_))
    return transMat;
  double motionTime = (time < endTime_) ? time : endTime_;

  // determine translation based on user defined input
  if (useVelocity_) {
    for (int d = 0; d < nalu_ngp::NDimMax; d++)
      transMat[d * mm::matSize + 3] = velocity_[d] * (motionTime - startTime_);
  } else {
    for (int d = 0; d < nalu_ngp::NDimMax; d++)
      transMat[d * mm::matSize + 3] = displacement_[d];
  }
  return transMat;
}

KOKKOS_FUNCTION
mm::ThreeDVecType
MotionTranslationKernel::compute_velocity(
  const double& time,
  const mm::TransMatType& /* compTrans */,
  const mm::ThreeDVecType& /* mxyz */,
  const mm::ThreeDVecType& /* cxyz */)
{
  if ((time < startTime_) || (time > endTime_))
    return mm::ThreeDVecType{0, 0, 0};
  else
    return velocity_;
}

} // namespace nalu
} // namespace sierra
