#include "mesh_motion/MotionDeformingInteriorKernel.h"

#include <NaluParsing.h>
#include "utils/ComputeVectorDivergence.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cmath>

namespace sierra {
namespace nalu {

MotionDeformingInteriorKernel::MotionDeformingInteriorKernel(
  stk::mesh::MetaData& meta, const YAML::Node& node)
  : NgpMotionKernel<MotionDeformingInteriorKernel>()
{
  load(node);

  // declare divergence of mesh velocity for this motion
  isDeforming_ = true;
  ScalarFieldType* divV = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "div_mesh_velocity"));
  stk::mesh::put_field_on_mesh(*divV, meta.universal_part(), nullptr);
  stk::mesh::field_fill(0.0, *divV);
}

void
MotionDeformingInteriorKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_ - DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_ + DBL_EPSILON;

  // get lower bounds of deforming part of mesh
  if (!node["xyz_min"])
    NaluEnv::self().naluOutputP0() << "MotionDeformingInteriorKernel: Need to "
                                      "define lower bounds of mesh that deform"
                                   << std::endl;
  for (int d = 0; d < nalu_ngp::NDimMax; ++d)
    xyzMin_[d] = node["xyz_min"][d].as<double>();

  // get lower bounds of deforming part of mesh
  if (!node["xyz_max"])
    NaluEnv::self().naluOutputP0() << "MotionDeformingInteriorKernel: Need to "
                                      "define upper bounds of mesh that deform"
                                   << std::endl;
  for (int d = 0; d < nalu_ngp::NDimMax; ++d)
    xyzMax_[d] = node["xyz_max"][d].as<double>();

  // get amplitude it was defined
  if (node["amplitude"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      amplitude_[d] = node["amplitude"][d].as<double>();
  }

  // get frequency it was defined
  if (node["frequency"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      frequency_[d] = node["frequency"][d].as<double>();
  }

  // get origin based on if it was defined or is to be computed
  if (node["centroid"]) {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      origin_[d] = node["centroid"][d].as<double>();
  }
}

KOKKOS_FUNCTION
mm::TransMatType
MotionDeformingInteriorKernel::build_transformation(
  const double& time, const mm::ThreeDVecType& xyz)
{
  mm::TransMatType transMat;

  if (time < (startTime_))
    return transMat;
  double motionTime = (time < endTime_) ? time : endTime_;

  // return identity matrix if point is outside bounds
  if (
    (xyz[0] <= xyzMin_[0]) || (xyz[0] >= xyzMax_[0]) ||
    (xyz[1] <= xyzMin_[1]) || (xyz[1] >= xyzMax_[1]) ||
    (xyz[2] <= xyzMin_[2]) || (xyz[2] >= xyzMax_[2]))
    return transMat;

  // initialize variables
  mm::ThreeDVecType radius;
  mm::ThreeDVecType curr_radius;
  mm::ThreeDVecType scaling;

  // Build matrix for translating object to cartesian origin
  transMat[0 * mm::matSize + 3] = -origin_[0];
  transMat[1 * mm::matSize + 3] = -origin_[1];
  transMat[2 * mm::matSize + 3] = -origin_[2];

  // Build matrix for scaling object
  mm::TransMatType tempMat;
  for (int d = 0; d < nalu_ngp::NDimMax; d++) {
    radius[d] = stk::math::abs(xyz[d] - origin_[d]);

    curr_radius[d] =
      radius[d] + amplitude_[d] *
                    (1 - stk::math::cos(2 * M_PI * frequency_[d] * motionTime));

    scaling[d] = curr_radius[d] / radius[d];
    if (radius[d] <= DBL_EPSILON)
      scaling[d] = 1.0;

    tempMat[d * mm::matSize + d] = scaling[d];
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
MotionDeformingInteriorKernel::compute_velocity(
  const double& time,
  const mm::TransMatType& /* compTrans */,
  const mm::ThreeDVecType& mxyz,
  const mm::ThreeDVecType& /* cxyz */)
{
  mm::ThreeDVecType vel;

  // return zero velocity if point is outside bounds or time limits
  if (
    (time < startTime_) || (time > endTime_) || (mxyz[0] <= xyzMin_[0]) ||
    (mxyz[0] >= xyzMax_[0]) || (mxyz[1] <= xyzMin_[1]) ||
    (mxyz[1] >= xyzMax_[1]) || (mxyz[2] <= xyzMin_[2]) ||
    (mxyz[2] >= xyzMax_[2]))
    return vel;

  // initialize variables
  mm::ThreeDVecType radius;
  mm::ThreeDVecType osclVelocity;

  for (int d = 0; d < nalu_ngp::NDimMax; d++) {
    radius[d] = stk::math::abs(mxyz[d] - origin_[d]);

    osclVelocity[d] = amplitude_[d] *
                      stk::math::sin(2 * M_PI * frequency_[d] * time) * 2 *
                      M_PI * frequency_[d] / radius[d];
    if (radius[d] <= DBL_EPSILON)
      osclVelocity[d] = 0.0;

    vel[d] = osclVelocity[d] * (mxyz[d] - origin_[d]);
  }

  return vel;
}

} // namespace nalu
} // namespace sierra
