#include "mesh_motion/MotionScalingKernel.h"

#include <FieldTypeDef.h>
#include <NaluParsing.h>

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cmath>

namespace sierra {
namespace nalu {

MotionScalingKernel::MotionScalingKernel(
  stk::mesh::MetaData& meta, const YAML::Node& node)
  : NgpMotionKernel<MotionScalingKernel>()
{
  load(node);

  if (useRate_) {
    // declare divergence of mesh velocity for this motion
    isDeforming_ = true;
    ScalarFieldType* divV = &(meta.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "div_mesh_velocity"));
    stk::mesh::put_field_on_mesh(*divV, meta.universal_part(), nullptr);
    stk::mesh::field_fill(0.0, *divV);
  }
}

void
MotionScalingKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_ - DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_ + DBL_EPSILON;

  // translation could be based on rate or factor
  if (node["rate"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      rate_[d] = node["rate"][d].as<double>();
  }

  // default approach is to use a constant displacement
  useRate_ = (node["rate"] ? true : false);

  if (node["factor"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      factor_[d] = node["factor"][d].as<double>();
  }

  // get origin based on if it was defined or is to be computed
  if (node["centroid"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      origin_[d] = node["centroid"][d].as<double>();
  }
}

KOKKOS_FUNCTION
mm::TransMatType
MotionScalingKernel::build_transformation(
  const double& time, const mm::ThreeDVecType& /* xyz */)
{
  mm::TransMatType transMat;

  if (time < (startTime_))
    return transMat;
  double motionTime = (time < endTime_) ? time : endTime_;

  // Build matrix for translating object to cartesian origin
  transMat[0 * mm::matSize + 3] = -origin_[0];
  transMat[1 * mm::matSize + 3] = -origin_[1];
  transMat[2 * mm::matSize + 3] = -origin_[2];

  // Determine scaling based on user defined input
  mm::TransMatType tempMat;
  if (useRate_) {
    for (int d = 0; d < nalu_ngp::NDimMax; d++)
      tempMat[d * mm::matSize + d] = rate_[d] * (motionTime - startTime_) + 1.0;
  } else {
    for (int d = 0; d < nalu_ngp::NDimMax; d++)
      tempMat[d * mm::matSize + d] = factor_[d];
  }

  // composite addition of motions in current group
  transMat = add_motion(tempMat, transMat);

  // Build matrix for translating object back to its origin
  tempMat = mm::TransMatType::I();
  tempMat[0 * mm::matSize + 3] = origin_[0];
  tempMat[1 * mm::matSize + 3] = origin_[1];
  tempMat[2 * mm::matSize + 3] = origin_[2];

  // composite addition of motions
  return add_motion(tempMat, transMat);
}

KOKKOS_FUNCTION
mm::ThreeDVecType
MotionScalingKernel::compute_velocity(
  const double& time,
  const mm::TransMatType& compTrans,
  const mm::ThreeDVecType& mxyz,
  const mm::ThreeDVecType& /* cxyz */)
{
  mm::ThreeDVecType vel;

  if ((time < startTime_) || (time > endTime_))
    return vel;

  // transform the origin of the scaling body
  mm::ThreeDVecType transOrigin;
  for (int d = 0; d < nalu_ngp::NDimMax; d++) {
    transOrigin[d] = compTrans[d * mm::matSize + 0] * origin_[0] +
                     compTrans[d * mm::matSize + 1] * origin_[1] +
                     compTrans[d * mm::matSize + 2] * origin_[2] +
                     compTrans[d * mm::matSize + 3];
  }

  for (int d = 0; d < nalu_ngp::NDimMax; d++)
    vel[d] = rate_[d] * (mxyz[d] - transOrigin[d]);

  return vel;
}

} // namespace nalu
} // namespace sierra
