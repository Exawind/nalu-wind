
#include "mesh_motion/MotionTranslation.h"

#include <cmath>

namespace sierra{
namespace nalu{

MotionTranslation::MotionTranslation(const YAML::Node& node)
  : MotionBase()
{
  load(node);
}

void MotionTranslation::load(const YAML::Node& node)
{
  if(node["start_time"])
    startTime_ = node["start_time"].as<double>();

  if(node["end_time"])
    endTime_ = node["end_time"].as<double>();

  // rotation could be based on angular velocity or angle
  if(node["velocity"]){
    useVelocity_ = true;
    velocity_ = node["velocity"].as<threeDVecType>();
  }
  if(node["displacement"])
  {
    useVelocity_ = false;
    displacement_ = node["displacement"].as<threeDVecType>();
  }
}

void MotionTranslation::build_transformation(
  const double time,
  const double* xyz)
{
  if( (time >= (startTime_-eps_)) && (time <= (endTime_+eps_)) )
  {
    // determine translation based on user defined input
    if (useVelocity_)
    {
      threeDVecType curr_disp = {};
      for (int d=0; d < threeDVecSize; d++)
        curr_disp[d] = velocity_[d]*(time-startTime_);

      translation_mat(curr_disp);
    }
    else
      translation_mat(displacement_);
  }
}

void MotionTranslation::translation_mat(const threeDVecType& curr_disp)
{
  reset_mat(transMat_);

  // Build matrix for translating object
  transMat_[0][3] = curr_disp[0];
  transMat_[1][3] = curr_disp[1];
  transMat_[2][3] = curr_disp[2];
}

MotionBase::threeDVecType MotionTranslation::compute_velocity(
  double time,
  const transMatType& comp_trans,
  double* xyz )
{
  threeDVecType vel = {};

  if( (time >= (startTime_-eps_)) && (time <= (endTime_+eps_)) )
    vel = velocity_;

  return vel;
}

} // nalu
} // sierra
