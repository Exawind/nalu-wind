#include "mesh_motion/MotionPulsatingSphereKernel.h"

#include<FieldTypeDef.h>
#include <NaluParsing.h>

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cmath>

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

void MotionPulsatingSphereKernel::build_transformation(
  const double time,
  const double* xyz)
{
  if(time < (startTime_)) return;

  double motionTime = (time < endTime_)? time : endTime_;

  scaling_mat(motionTime,xyz);
}

void MotionPulsatingSphereKernel::scaling_mat(
  const double time,
  const double* xyz)
{
  reset_mat(transMat_);

  double radius = std::sqrt( std::pow(xyz[0]-origin_[0],2)
                             +std::pow(xyz[1]-origin_[1],2)
                             +std::pow(xyz[2]-origin_[2],2));

  double curr_radius = radius + amplitude_*(1 - std::cos(2*M_PI*frequency_*time));

  double uniform_scaling = curr_radius/radius;
  if(radius == 0.0) uniform_scaling = 1.0;

  // Build matrix for translating object to cartesian origin
  TransMatType tempMat = {};
  reset_mat(tempMat);
  tempMat[0][3] = -origin_[0];
  tempMat[1][3] = -origin_[1];
  tempMat[2][3] = -origin_[2];

  // Build matrix for scaling object
  TransMatType tempMat2 = {};
  reset_mat(tempMat2);
  tempMat2[0][0] = uniform_scaling;
  tempMat2[1][1] = uniform_scaling;
  tempMat2[2][2] = uniform_scaling;
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

void MotionPulsatingSphereKernel::compute_velocity(
  const double time,
  const TransMatType&  /* compTrans */,
  const double* mxyz,
  const double* /* cxyz */,
  ThreeDVecType& vel )
{
  if((time < startTime_) || (time > endTime_)) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      vel[d] = 0.0;

    return;
  }

  double radius = std::sqrt( std::pow(mxyz[0]-origin_[0],2)
                             +std::pow(mxyz[1]-origin_[1],2)
                             +std::pow(mxyz[2]-origin_[2],2));

  double pulsatingVelocity =
    amplitude_ * std::sin(2*M_PI*frequency_*time) * 2*M_PI*frequency_ / radius;

  // account for zero radius
  if(radius == 0) pulsatingVelocity = 0;

  for (int d=0; d < nalu_ngp::NDimMax; d++)
    vel[d] = pulsatingVelocity * (mxyz[d]-origin_[d]);
}

} // nalu
} // sierra
