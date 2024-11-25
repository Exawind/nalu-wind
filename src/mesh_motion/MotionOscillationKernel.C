
#include "mesh_motion/MotionOscillationKernel.h"

#include <NaluEnv.h>
#include <NaluParsing.h>

namespace sierra {
namespace nalu {

MotionOscillationKernel::MotionOscillationKernel(const YAML::Node& node)
  : NgpMotionKernel<MotionOscillationKernel>()
{
  load(node);
}

void
MotionOscillationKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_ - DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_ + DBL_EPSILON;

  // Oscillation based on period and amplitude
  get_if_present(node, "period", period_, period_);
  get_if_present(node, "amplitude", amplitude_, amplitude_);

  if (node["direction"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      direction_[d] = node["direction"][d].as<double>();
  } else
    NaluEnv::self().naluOutputP0() << "MotionOscillationKernel: direction of "
                                      "Oscillation not supplied; will use 0,0,1"
                                   << std::endl;
}

KOKKOS_FUNCTION
mm::TransMatType
MotionOscillationKernel::build_transformation(
  const double& time, const mm::ThreeDVecType& /* xyz */)
{
  mm::TransMatType transMat;

  if (time < (startTime_))
    return transMat;
  double motionTime = (time < endTime_) ? time : endTime_;

  // determine current angle within periodic function
  const double angle =
    2.0 * M_PI / period_ * (stk::math::max(0.0, motionTime - startTime_));
  // determine displacement along oscillation direction
  const double disp = amplitude_ * stk::math::sin(angle);

  // get magnitude of oscillation direction vector
  double mag = 0.0;
  for (int d = 0; d < nalu_ngp::NDimMax; d++)
    mag += direction_[d] * direction_[d];
  mag = stk::math::sqrt(mag);

  // determine translation in each direction
  for (int d = 0; d < nalu_ngp::NDimMax; d++)
    transMat[d * mm::matSize + 3] = disp * direction_[d] / mag;
  return transMat;
}

KOKKOS_FUNCTION
mm::ThreeDVecType
MotionOscillationKernel::compute_velocity(
  const double& time,
  const mm::TransMatType& /* compTrans */,
  const mm::ThreeDVecType& /* mxyz */,
  const mm::ThreeDVecType& /* cxyz */)
{
  if ((time < startTime_) || (time > endTime_))
    return mm::ThreeDVecType{0, 0, 0};
  else {
    // determine current angle within periodic function
    const double omega = 2.0 * M_PI / period_;
    const double angle = omega * time;
    // determine velocity along oscillation direction
    const double vel_1D = amplitude_ * omega * stk::math::cos(angle);

    // get magnitude of oscillation direction vector
    double mag = 0.0;
    for (int d = 0; d < nalu_ngp::NDimMax; d++)
      mag += direction_[d] * direction_[d];
    mag = stk::math::sqrt(mag);

    // determine translation in each direction
    mm::ThreeDVecType mesh_vel{0, 0, 0};
    for (int d = 0; d < nalu_ngp::NDimMax; d++)
      mesh_vel[d] = vel_1D * direction_[d] / mag;
    return mesh_vel;
  }
}

} // namespace nalu
} // namespace sierra
