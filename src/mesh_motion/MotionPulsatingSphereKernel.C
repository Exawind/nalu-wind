#include "mesh_motion/MotionPulsatingSphereKernel.h"

#include<FieldTypeDef.h>
#include <NaluParsing.h>

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

namespace sierra{
namespace nalu{

MotionPulsatingSphereKernel::MotionPulsatingSphereKernel(
  stk::mesh::MetaData& meta,
  const YAML::Node& node)
  : NgpMotionKernel<MotionPulsatingSphereKernel>()
{
  load(node);

  // declare divergence of mesh velocity for this motion
  isDeforming_ = true;
  ScalarFieldType *divV = &(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_velocity"));
  stk::mesh::put_field_on_mesh(*divV, meta.universal_part(), nullptr);
  stk::mesh::field_fill(0.0, *divV);
}

void MotionPulsatingSphereKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_-DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_+DBL_EPSILON;

  get_if_present(node, "amplitude", amplitude_, amplitude_);

  get_if_present(node, "frequency", frequency_, frequency_);

  // get origin based on if it was defined
  if( node["centroid"] ) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      origin_[d] = node["centroid"][d].as<double>();
  }
}

mm::TransMatType MotionPulsatingSphereKernel::build_transformation(
  const double& time,
  const mm::ThreeDVecType& xyz)
{
  mm::TransMatType transMat;

  if(time < (startTime_)) return transMat;
  double currTime = (time < endTime_)? time : endTime_;

  double radius = stk::math::sqrt(stk::math::pow(xyz[0]-origin_[0],2)
                                 +stk::math::pow(xyz[1]-origin_[1],2)
                                 +stk::math::pow(xyz[2]-origin_[2],2));
  double curr_radius = radius + amplitude_*(1 - stk::math::cos(2*M_PI*frequency_*currTime));

  double uniform_scaling = curr_radius/radius;
  if(radius == 0.0) uniform_scaling = 1.0;

  // Build matrix for translating object to cartesian origin
  transMat[0*mm::matSize+3] = -origin_[0];
  transMat[1*mm::matSize+3] = -origin_[1];
  transMat[2*mm::matSize+3] = -origin_[2];

  // Build matrix for scaling object
  mm::TransMatType tempMat;
  tempMat[0*mm::matSize+0] = uniform_scaling;
  tempMat[1*mm::matSize+1] = uniform_scaling;
  tempMat[2*mm::matSize+2] = uniform_scaling;

  // composite addition of motions in current group
  transMat = add_motion(tempMat,transMat);

  // Build matrix for translating object back to its origin
  tempMat = mm::TransMatType::I();
  tempMat[0*mm::matSize+3] = origin_[0];
  tempMat[1*mm::matSize+3] = origin_[1];
  tempMat[2*mm::matSize+3] = origin_[2];

  // composite addition of motions
  return add_motion(tempMat,transMat);
}

mm::ThreeDVecType MotionPulsatingSphereKernel::compute_velocity(
  const double& time,
  const mm::TransMatType&  /* compTrans */,
  const mm::ThreeDVecType& mxyz,
  const mm::ThreeDVecType& /* cxyz */)
{
  mm::ThreeDVecType vel;

  if((time < startTime_) || (time > endTime_))
    return vel;

  double radius = stk::math::sqrt(stk::math::pow(mxyz[0]-origin_[0],2)
                                 +stk::math::pow(mxyz[1]-origin_[1],2)
                                 +stk::math::pow(mxyz[2]-origin_[2],2));

  double pulsatingVelocity =
    amplitude_ * stk::math::sin(2*M_PI*frequency_*time) * 2*M_PI*frequency_ / radius;

  // account for zero radius
  if(radius == 0) pulsatingVelocity = 0;

  for (int d=0; d < nalu_ngp::NDimMax; d++)
    vel[d] = pulsatingVelocity * (mxyz[d]-origin_[d]);

  return vel;
}

} // nalu
} // sierra
