
#include "mesh_motion/MotionScaling.h"

#include <NaluParsing.h>

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
  get_if_present(node, "start_time", startTime_, startTime_);

  get_if_present(node, "end_time", endTime_, endTime_);

  if( node["factor"] )
    factor_ = node["factor"].as<threeDVecType>();

  // get origin based on if it was defined or is to be computed
  if( node["centroid"] )
    origin_ = node["centroid"].as<threeDVecType>();
}

void MotionScaling::build_transformation(
  const double time,
  const double* xyz)
{
  scaling_mat(factor_);
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

} // nalu
} // sierra
