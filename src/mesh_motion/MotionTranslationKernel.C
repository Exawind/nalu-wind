#include "mesh_motion/MotionTranslationKernel.h"

#include <NaluParsing.h>

#include <cmath>

namespace sierra{
namespace nalu{

MotionTranslationKernel::MotionTranslationKernel(const YAML::Node& node)
  : NgpMotionKernel<MotionTranslationKernel>()
{
  load(node);
}

void MotionTranslationKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_-DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_+DBL_EPSILON;

  // translation could be based on velocity or displacement
  if( node["velocity"] ) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      velocity_[d] = node["velocity"][d].as<DblType>();
  }

  if( node["displacement"] ) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      displacement_[d] = node["displacement"][d].as<DblType>();
  }

  // default approach is to use a constant displacement
  useVelocity_ = (node["velocity"] ? true : false);
}

void MotionTranslationKernel::build_transformation(
  const DblType time,
  const DblType* /* mxyz */ )
{
  if(time < (startTime_)) return;

  DblType motionTime = (time < endTime_)? time : endTime_;

  // determine translation based on user defined input
  if (useVelocity_)
  {
    ThreeDVecType curr_disp = {};
    for (int d=0; d < nalu_ngp::NDimMax; d++)
      curr_disp[d] = velocity_[d]*(motionTime-startTime_);

    translation_mat(curr_disp);
  }
  else
    translation_mat(displacement_);
}

void MotionTranslationKernel::translation_mat(const ThreeDVecType& curr_disp)
{
  reset_mat(transMat_);

  // Build matrix for translating object
  transMat_[0][3] = curr_disp[0];
  transMat_[1][3] = curr_disp[1];
  transMat_[2][3] = curr_disp[2];
}

void MotionTranslationKernel::compute_velocity(
  const DblType time,
  const TransMatType&  /* compTrans */,
  const DblType* /* mxyz */,
  const DblType* /* cxyz */,
  ThreeDVecType& vel )
{
  if((time < startTime_) || (time > endTime_)) {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      vel[d] = 0.0;
  }
  else {
    for (int d=0; d < nalu_ngp::NDimMax; ++d)
      vel[d] = velocity_[d];
  }
}

} // nalu
} // sierra
