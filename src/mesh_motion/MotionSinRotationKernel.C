#include "mesh_motion/MotionSinRotationKernel.h"

#include <NaluEnv.h>
#include <NaluParsing.h>

namespace sierra{
namespace nalu{

MotionSinRotationKernel::MotionSinRotationKernel(const YAML::Node& node)
    : MotionRotationKernel(node)
{
  load(node);
}

void MotionSinRotationKernel::load(const YAML::Node& node)
{

  get_if_present(node, "amplitude", amplt_, amplt_);
  amplt_ = amplt_*M_PI/180;

  get_if_present(node, "phase", phase_, phase_);
  phase_ = phase_*M_PI/180.0;

}

double MotionSinRotationKernel::get_cur_angle(const double time)
{
  double motionTime = (time < endTime_)? time : endTime_;

  // determine current angle
  angle_ = amplt_ * stk::math::sin(omega_ * (motionTime-startTime_) + phase_ );

  return angle_;
}

double MotionSinRotationKernel::get_cur_ang_vel(const double time)
{
    double motionTime = (time < endTime_)? time : endTime_;

    // determine current angular velocity
    ang_vel_ = amplt_ * omega_ *
        stk::math::cos(omega_ * (motionTime-startTime_) + phase_ );
    
    return ang_vel_;
}


} // nalu
} // sierra
