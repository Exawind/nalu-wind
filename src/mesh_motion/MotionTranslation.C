
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
  get_if_present(node, "start_time", startTime_, startTime_);

  get_if_present(node, "end_time", endTime_, endTime_);

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
  const double* xyz)
{
  double eps = std::numeric_limits<double>::epsilon();

  if( (time >= (startTime_-eps)) && (time <= (endTime_+eps)) )
  {
    // determine translation based on user defined input
    if (useVelocity_)
    {
      ThreeDVecType curr_disp = {};
      for (int d=0; d < threeDVecSize; d++)
        curr_disp[d] = velocity_[d]*(time-startTime_);

      translation_mat(curr_disp);
    }
    else
      translation_mat(displacement_);
  }
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
  double time,
  const TransMatType& comp_trans,
  double* xyz )
{
  ThreeDVecType vel = {};

  double eps = std::numeric_limits<double>::epsilon();

  if( (time >= (startTime_-eps)) && (time <= (endTime_+eps)) )
    vel = velocity_;

  return vel;
}

} // nalu
} // sierra
