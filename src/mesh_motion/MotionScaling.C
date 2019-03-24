
#include "mesh_motion/MotionScaling.h"

#include <NaluParsing.h>
#include "utils/ComputeVectorDivergence.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cmath>

namespace sierra{
namespace nalu{

MotionScaling::MotionScaling(
  stk::mesh::MetaData& meta,
  const YAML::Node& node)
  : MotionBase()
{
  load(node);

  if( useRate_ ) {
    // declare divergence of mesh velocity for this motion
    ScalarFieldType *divV = &(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_velocity"));
    stk::mesh::put_field_on_mesh(*divV, meta.universal_part(), nullptr);
    stk::mesh::field_fill(0.0, *divV);
  }
}

void MotionScaling::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  double eps = std::numeric_limits<double>::epsilon();

  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_-eps;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_+eps;

  // translation could be based on rate or factor
  if( node["rate"] )
     rate_ = node["rate"].as<ThreeDVecType>();

  // default approach is to use a constant displacement
  useRate_ = ( node["rate"] ? true : false);

  if( node["factor"] )
    factor_ = node["factor"].as<ThreeDVecType>();

  // get origin based on if it was defined or is to be computed
  if( node["centroid"] )
    origin_ = node["centroid"].as<ThreeDVecType>();
}

void MotionScaling::build_transformation(
  const double time,
  const double* /* xyz */)
{
  if(time < (startTime_)) return;

  double motionTime = (time < endTime_)? time : endTime_;

  // determine translation based on user defined input
  if (useRate_)
  {
    ThreeDVecType curr_fac = {};

    for (int d=0; d < threeDVecSize; d++)
      curr_fac[d] = rate_[d]*(motionTime-startTime_) + 1.0;

    scaling_mat(curr_fac);
  }
  else
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

MotionBase::ThreeDVecType MotionScaling::compute_velocity(
  const double time,
  const TransMatType& compTrans,
  const double* mxyz,
  const double* /* cxyz */ )
{
  ThreeDVecType vel = {};

  if( (time < startTime_) || (time > endTime_) ) return vel;

  // transform the origin of the scaling body
  ThreeDVecType transOrigin = {};
  for (int d = 0; d < threeDVecSize; d++) {
    transOrigin[d] = compTrans[d][0]*origin_[0]
                    +compTrans[d][1]*origin_[1]
                    +compTrans[d][2]*origin_[2]
                    +compTrans[d][3];
  }

  for (int d=0; d < threeDVecSize; d++)
    vel[d] = rate_[d] * (mxyz[d]-transOrigin[d]);

  return vel;
}

void MotionScaling::post_compute_geometry(
  stk::mesh::BulkData& bulk,
  stk::mesh::PartVector& partVec,
  stk::mesh::PartVector& partVecBc,
  bool& computedMeshVelDiv)
{
  if(computedMeshVelDiv || !useRate_) return;

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
