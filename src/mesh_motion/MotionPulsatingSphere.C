#include "mesh_motion/MotionPulsatingSphere.h"

#include <NaluParsing.h>
#include "utils/ComputeVectorDivergence.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cmath>

namespace sierra{
namespace nalu{

MotionPulsatingSphere::MotionPulsatingSphere(
  stk::mesh::MetaData& meta,
  const YAML::Node& node)
  : MotionBase()
{
  load(node);

  // declare divergence of mesh velocity for this motion
  ScalarFieldType *divV = &(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_velocity"));
  stk::mesh::put_field_on_mesh(*divV, meta.universal_part(), nullptr);
  stk::mesh::field_fill(0.0, *divV);
}

void MotionPulsatingSphere::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  double eps = std::numeric_limits<double>::epsilon();

  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_-eps;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_+eps;

  get_if_present(node, "amplitude", amplitude_, amplitude_);

  get_if_present(node, "frequency", frequency_, frequency_);

  // get origin based on if it was defined
  if( node["centroid"] )
    origin_ = node["centroid"].as<ThreeDVecType>();
}

void MotionPulsatingSphere::build_transformation(
  const double time,
  const double* xyz)
{
  if(time < (startTime_)) return;

  double motionTime = (time < endTime_)? time : endTime_;

  scaling_mat(motionTime,xyz);
}

void MotionPulsatingSphere::scaling_mat(
  const double time,
  const double* xyz)
{
  reset_mat(transMat_);

  double radius = std::sqrt( std::pow(xyz[0]-origin_[0],2)
                            +std::pow(xyz[1]-origin_[1],2)
                            +std::pow(xyz[2]-origin_[2],2));

  double curr_radius = radius + amplitude_*(1 - std::cos(2*M_PI*frequency_*time));

  double uniform_scaling = curr_radius/radius;

  // Build matrix for translating object to cartesian origin
  transMat_[0][3] = -origin_[0];
  transMat_[1][3] = -origin_[1];
  transMat_[2][3] = -origin_[2];

  // Build matrix for scaling object
  TransMatType currTransMat = {};

  currTransMat[0][0] = uniform_scaling;
  currTransMat[1][1] = uniform_scaling;
  currTransMat[2][2] = uniform_scaling;
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

MotionBase::ThreeDVecType MotionPulsatingSphere::compute_velocity(
  const double time,
  const TransMatType&  /* compTrans */,
  const double* xyz )
{
  ThreeDVecType vel = {};

  if( (time < startTime_) || (time > endTime_) ) return vel;

  double radius = std::sqrt( std::pow(xyz[0]-origin_[0],2)
                            +std::pow(xyz[1]-origin_[1],2)
                            +std::pow(xyz[2]-origin_[2],2));

  double pulsatingVelocity =
    amplitude_ * std::sin(2*M_PI*frequency_*time) * 2*M_PI*frequency_ / radius;

  double eps = std::numeric_limits<double>::epsilon();

  for (int d=0; d < threeDVecSize; d++)
  {
    int signum = (-eps < xyz[d]-origin_[d]) - (xyz[d]-origin_[d] < eps);
    vel[d] = signum * pulsatingVelocity * (xyz[d]-origin_[d]);
  }

  return vel;
}

void MotionPulsatingSphere::post_work(
  stk::mesh::BulkData& bulk,
  stk::mesh::PartVector& partVec,
  stk::mesh::PartVector& partVecBc,
  bool& computedMeshVelDiv)
{
  if(computedMeshVelDiv) return;

  // compute divergence of mesh velocity
  VectorFieldType* meshVelocity = bulk.mesh_meta_data().get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_velocity");

  ScalarFieldType* meshDivVelocity = bulk.mesh_meta_data().get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "div_mesh_velocity");

  compute_vector_divergence(bulk, partVec, partVecBc, meshVelocity, meshDivVelocity, true);
  computedMeshVelDiv = true;
}

} // nalu
} // sierra
