
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
  get_if_present(node, "velocity", velocity_, velocity_);

  get_if_present(node, "displacement", displacement_, displacement_);

  // default approach is to use a constant displacement
  useVelocity_ = ( node["velocity"] ? true : false);
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
