#include "mesh_motion/MotionScalingKernel.h"

#include<FieldTypeDef.h>
#include <NaluParsing.h>

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cmath>

namespace sierra{
namespace nalu{

MotionScalingKernel::MotionScalingKernel(
  stk::mesh::MetaData& meta,
  const YAML::Node& node)
  : NgpMotionKernel<MotionScalingKernel>()
{
  load(node);

  if( useRate_ ) {
    // declare divergence of mesh velocity for this motion
    isDeforming_ = true;
    ScalarFieldType *divV = &(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_velocity"));
    stk::mesh::put_field_on_mesh(*divV, meta.universal_part(), nullptr);
    stk::mesh::field_fill(0.0, *divV);
  }
}

void MotionScalingKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_-DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_+DBL_EPSILON;

  // translation could be based on rate or factor
  if( node["rate"] ) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      rate_[d] = node["rate"][d].as<double>();
  }

  // default approach is to use a constant displacement
  useRate_ = ( node["rate"] ? true : false);

  if( node["factor"] ) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      factor_[d] = node["factor"][d].as<double>();
  }

  // get origin based on if it was defined or is to be computed
  if( node["centroid"] ) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      origin_[d] = node["centroid"][d].as<double>();
  }
}

void MotionScalingKernel::build_transformation(
  const double time,
  const double* /* xyz */)
{
  if(time < (startTime_)) return;

  double motionTime = (time < endTime_)? time : endTime_;

  // determine translation based on user defined input
  if (useRate_)
  {
    ThreeDVecType curr_fac = {};

    for (int d=0; d < nalu_ngp::NDimMax; d++)
      curr_fac[d] = rate_[d]*(motionTime-startTime_) + 1.0;

    scaling_mat(curr_fac);
  }
  else
    scaling_mat(factor_);
}

void MotionScalingKernel::scaling_mat(const ThreeDVecType& factor)
{
  reset_mat(transMat_);

  // Build matrix for translating object to cartesian origin
  TransMatType tempMat = {};
  reset_mat(tempMat);
  tempMat[0][3] = -origin_[0];
  tempMat[1][3] = -origin_[1];
  tempMat[2][3] = -origin_[2];

  // Build matrix for scaling object
  TransMatType tempMat2 = {};
  reset_mat(tempMat2);
  tempMat2[0][0] = factor[0];
  tempMat2[1][1] = factor[1];
  tempMat2[2][2] = factor[2];
  tempMat2[3][3] = 1.0;

  // composite addition of motions in current group
  TransMatType tempMat3 = {};
  add_motion(tempMat2,tempMat,tempMat3);

  // Build matrix for translating object back to its origin
  reset_mat(tempMat);
  tempMat[0][3] = origin_[0];
  tempMat[1][3] = origin_[1];
  tempMat[2][3] = origin_[2];

  // composite addition of motions
  add_motion(tempMat,tempMat3,transMat_);
}

void MotionScalingKernel::compute_velocity(
  const double time,
  const TransMatType& compTrans,
  const double* mxyz,
  const double* /* cxyz */,
  ThreeDVecType& vel )
{
  if((time < startTime_) || (time > endTime_)) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      vel[d] = 0.0;

    return;
  }

  // transform the origin of the scaling body
  ThreeDVecType transOrigin = {};
  for (int d = 0; d < nalu_ngp::NDimMax; d++) {
    transOrigin[d] = compTrans[d][0]*origin_[0]
                    +compTrans[d][1]*origin_[1]
                    +compTrans[d][2]*origin_[2]
                    +compTrans[d][3];
  }

  for (int d=0; d < nalu_ngp::NDimMax; d++)
    vel[d] = rate_[d] * (mxyz[d]-transOrigin[d]);
}

} // nalu
} // sierra
