
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
  if( node["factor"] )
    factor_ = node["factor"].as<ThreeDVecType>();

  // get origin based on if it was defined or is to be computed
  if( node["centroid"] )
    origin_ = node["centroid"].as<ThreeDVecType>();
}

void MotionScaling::build_transformation(
  const double  /* time */,
  const double*  /* xyz */)
{
  scaling_mat(factor_);
}

void MotionScaling::scaling_mat(const ThreeDVecType& factor)
{
  reset_mat(transMat_);

  // Build matrix for translating object to cartesian origin
  transMat_[0][3] = -origin_[0];
  transMat_[1][3] = -origin_[1];
  transMat_[2][3] = -origin_[2];

  // Build matrix for scaling object
  TransMatType currTransMat = {};

  currTransMat[0][0] = factor[0];
  currTransMat[1][1] = factor[1];
  currTransMat[2][2] = factor[2];
  currTransMat[3][3] = 1.0;

  // composite addition of motions in current group
  transMat_ = add_motion(currTransMat,transMat_);

  // Build matrix for translating object back to its origin
  reset_mat(currTransMat);
  currTransMat[0][3] = origin_[0];
  currTransMat[1][3] = origin_[1];
  currTransMat[2][3] = origin_[2];

  // composite addition of motions
  transMat_ = add_motion(currTransMat,transMat_);
}

} // nalu
} // sierra
