
#include "mesh_motion/MotionScaling.h"

#include <cmath>

namespace sierra{
namespace nalu{

MotionScaling::MotionScaling(const YAML::Node& node)
  : MotionBase()
{
  load(node);
}

void MotionScaling::load(const YAML::Node& node)
{
  if(node["start_time"])
    startTime_ = node["start_time"].as<double>();

  if(node["end_time"])
    endTime_ = node["end_time"].as<double>();

  // scaling could be based on velocity or factor
  if(node["velocity"]){
    useVelocity_ = true;
    velocity_ = node["velocity"].as<threeDVecType>();
  }
  if(node["factor"])
  {
    useVelocity_ = false;
    factor_ = node["factor"].as<threeDVecType>();
  }

  // get origin based on if it was defined or is to be computed
  if( computeCentroid_ )
    origin_ = computedCentroid_;
  else if( node["centroid"] )
    origin_ = node["centroid"].as<threeDVecType>();
}

void MotionScaling::build_transformation(
  const double time,
  const double* xyz)
{
  if( (time >= (startTime_-eps_)) && (time <= (endTime_+eps_)) )
  {
    // determine current displacement
    threeDVecType factor = {};
    if (useVelocity_)
      for (int d=0; d < threeDVecSize; d++)
        factor[d] = velocity_[d]*(time-startTime_);
    else
      factor = factor_;

    scaling_mat(factor);
  }
}

void MotionScaling::scaling_mat(const threeDVecType& factor)
{
  reset_mat(transMat_);

  // Build matrix for translating object to cartesian origin
  transMat_[0][3] = -origin_[0];
  transMat_[1][3] = -origin_[1];
  transMat_[2][3] = -origin_[2];

  // Build matrix for scaling object
  transMatType curr_trans_mat_ = {};

  curr_trans_mat_[0][0] = factor[0];
  curr_trans_mat_[1][1] = factor[1];
  curr_trans_mat_[2][2] = factor[2];
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

MotionBase::threeDVecType MotionScaling::compute_velocity(
  double time,
  const transMatType& comp_trans,
  double* xyz )
{
  threeDVecType vel = {};

  if( (time >= (startTime_-eps_)) && (time <= (endTime_+eps_)) )
  {
    // transform the origin of the rotating body
    threeDVecType trans_origin = {};
    for (int d = 0; d < threeDVecSize; d++) {
      trans_origin[d] = comp_trans[d][0]*origin_[0]
                       +comp_trans[d][1]*origin_[1]
                       +comp_trans[d][2]*origin_[2]
                       +comp_trans[d][3];
    }

    for (int d=0; d < threeDVecSize; d++)
    {
      int signum = (-eps_ < xyz[d]-trans_origin[d]) -
                           (xyz[d]-trans_origin[d] < eps_);

      vel[d] = signum * velocity_[d];
    }
  }

  return vel;
}

} // nalu
} // sierra
