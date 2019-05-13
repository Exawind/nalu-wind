
#include "mesh_motion/MotionTranslation.h"

#include <NaluParsing.h>

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
  // perturb start and end times with a small value for
  // accurate comparison with floats
  double eps = std::numeric_limits<double>::epsilon();

  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_-eps;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_+eps;

  // translation could be based on velocity or displacement
  if( node["velocity"] )
     velocity_ = node["velocity"].as<ThreeDVecType>();

  if( node["displacement"] )
    displacement_ = node["displacement"].as<ThreeDVecType>();

  // default approach is to use a constant displacement
  useVelocity_ = ( node["velocity"] ? true : false);
}

void MotionTranslation::build_transformation(
  const double time,
  const double* /* mxyz */ )
{
  if(time < (startTime_)) return;

  double motionTime = (time < endTime_)? time : endTime_;

  // determine translation based on user defined input
  if (useVelocity_)
  {
    ThreeDVecType curr_disp = {};
    for (int d=0; d < threeDVecSize; d++)
      curr_disp[d] = velocity_[d]*(motionTime-startTime_);

    translation_mat(curr_disp);
  }
  else
    translation_mat(displacement_);
}

void MotionTranslation::translation_mat(const ThreeDVecType& curr_disp)
{
  reset_mat(transMat_);

  // Build matrix for translating object
  transMat_[0][3] = curr_disp[0];
  transMat_[1][3] = curr_disp[1];
  transMat_[2][3] = curr_disp[2];
}

MotionBase::ThreeDVecType MotionTranslation::compute_velocity(
  const double time,
  const TransMatType&  /* compTrans */,
  const double* /* mxyz */,
  const double* /* cxyz */ )
{
  ThreeDVecType vel = {};

  if( (time >= startTime_) && (time <= endTime_) )
    vel = velocity_;

  return vel;
}

} // nalu
} // sierra
