#include "mesh_motion/MotionDeformingInterior.h"

#include <NaluParsing.h>
#include "utils/ComputeVectorDivergence.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cmath>

namespace sierra{
namespace nalu{

MotionDeformingInterior::MotionDeformingInterior(
  stk::mesh::MetaData& meta,
  const YAML::Node& node)
  : MotionBase()
{
  load(node);

  // flag to denote if motion deforms elements
  isDeforming_ = true;

  // declare divergence of mesh velocity for this motion
  ScalarFieldType *divV = &(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_velocity"));
  stk::mesh::put_field_on_mesh(*divV, meta.universal_part(), nullptr);
  stk::mesh::field_fill(0.0, *divV);
}

void MotionDeformingInterior::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  double eps = std::numeric_limits<double>::epsilon();

  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_-eps;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_+eps;

  // get lower bounds of deforming part of mesh
  if( !node["xyz_min"] )
    NaluEnv::self().naluOutputP0() << "MotionDeformingInterior: Need to define lower bounds of mesh that deform" << std::endl;
  xyzMin_ = node["xyz_min"].as<ThreeDVecType>();

  // get upper bounds of deforming part of mesh
  if( !node["xyz_max"] )
    NaluEnv::self().naluOutputP0() << "MotionDeformingInterior: Need to define upper bounds of mesh that deform" << std::endl;
  xyzMax_ = node["xyz_max"].as<ThreeDVecType>();

  // get amplitude it was defined
  if( node["amplitude"] )
    amplitude_ = node["amplitude"].as<ThreeDVecType>();

  // get amplitude it was defined
  if( node["frequency"] )
    frequency_ = node["frequency"].as<ThreeDVecType>();

  // get origin based on if it was defined
  if( node["centroid"] )
    origin_ = node["centroid"].as<ThreeDVecType>();
}

void MotionDeformingInterior::build_transformation(
  const double time,
  const double* xyz)
{
  if(time < (startTime_)) return;

  double motionTime = (time < endTime_)? time : endTime_;

  scaling_mat(motionTime,xyz);
}

void MotionDeformingInterior::scaling_mat(
  const double time,
  const double* xyz)
{
  reset_mat(transMat_);

  // return identity matrix if point is outside bounds
  if( (xyz[0] <= xyzMin_[0]) || (xyz[0] >= xyzMax_[0]) ||
      (xyz[1] <= xyzMin_[1]) || (xyz[1] >= xyzMax_[1]) ||
      (xyz[2] <= xyzMin_[2]) || (xyz[2] >= xyzMax_[2])  )
    return;

  // initialize variables
  double eps = std::numeric_limits<double>::epsilon();

  ThreeDVecType radius      = {};
  ThreeDVecType curr_radius = {};
  ThreeDVecType scaling     = {};

  for (int d=0; d < threeDVecSize; d++)
  {
    radius[d] = std::abs(xyz[d]-origin_[d]);

    curr_radius[d] = radius[d] + amplitude_[d]*(1 - std::cos(2*M_PI*frequency_[d]*time));

    scaling[d] = curr_radius[d]/radius[d];
    if(radius[d] <= eps) scaling[d] = 1.0;
  }

  // Build matrix for translating object to cartesian origin
  transMat_[0][3] = -origin_[0];
  transMat_[1][3] = -origin_[1];
  transMat_[2][3] = -origin_[2];

  // Build matrix for scaling object
  TransMatType currTransMat = {};

  currTransMat[0][0] = scaling[0];
  currTransMat[1][1] = scaling[1];
  currTransMat[2][2] = scaling[2];
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

MotionBase::ThreeDVecType MotionDeformingInterior::compute_velocity(
  const double time ,
  const TransMatType&  /* compTrans */,
  const double* mxyz ,
  const double* /* xyz */ )
{
  ThreeDVecType vel = {};

  // return zero velocity if point is outside bounds or time limits
  if( (time < startTime_) || (time > endTime_) ||
      (mxyz[0] <= xyzMin_[0]) || (mxyz[0] >= xyzMax_[0]) ||
      (mxyz[1] <= xyzMin_[1]) || (mxyz[1] >= xyzMax_[1]) ||
      (mxyz[2] <= xyzMin_[2]) || (mxyz[2] >= xyzMax_[2])  )
    return vel;

  // initialize variables
  double eps = std::numeric_limits<double>::epsilon();

  ThreeDVecType radius       = {};
  ThreeDVecType osclVelocity = {};

  for (int d=0; d < threeDVecSize; d++)
  {
    radius[d] = std::abs(mxyz[d]-origin_[d]);

    osclVelocity[d] =
      amplitude_[d] * std::sin(2*M_PI*frequency_[d]*time) * 2*M_PI*frequency_[d] / radius[d];

    if(radius[d] <= eps) osclVelocity[d] = 0.0;

    vel[d] = osclVelocity[d] * (mxyz[d]-origin_[d]);
  }

  return vel;
}

void MotionDeformingInterior::post_compute_geometry(
  stk::mesh::BulkData& bulk,
  stk::mesh::PartVector& partVec,
  stk::mesh::PartVector& partVecBc,
  bool& computedMeshVelDiv)
{
  if(computedMeshVelDiv) return;

  // compute divergence of mesh velocity
  ScalarFieldType* meshDivVelocity = bulk.mesh_meta_data().get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "div_mesh_velocity");

  GenericFieldType* faceVelMag = bulk.mesh_meta_data().get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "face_velocity_mag");
  if(faceVelMag == NULL) {
    std::cerr << "Using edge algorithm for mesh vel div" << std::endl;
    faceVelMag = bulk.mesh_meta_data().get_field<GenericFieldType>(
      stk::topology::EDGE_RANK, "edge_face_velocity_mag");
    compute_edge_scalar_divergence(bulk, partVec, partVecBc, faceVelMag, meshDivVelocity);
  } else {
    std::cerr << "Using element algorithm for mesh vel div" << std::endl;
    compute_scalar_divergence(bulk, partVec, partVecBc, faceVelMag, meshDivVelocity);
  }
  computedMeshVelDiv = true;
}

} // nalu
} // sierra
