#ifndef MOTIONSCALING_H
#define MOTIONSCALING_H

#include "NgpMotion.h"

namespace stk {
namespace mesh {
class MetaData;
}
}

namespace sierra{
namespace nalu{

class MotionScalingKernel : public NgpMotionKernel<MotionScalingKernel>
{
public:
  MotionScalingKernel(
    stk::mesh::MetaData&,
    const YAML::Node&);

  KOKKOS_FUNCTION
  MotionScalingKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MotionScalingKernel() = default;

  /** Function to compute motion-specific transformation matrix
   *
   * @param[in] time Current time
   * @param[in] xyz  Coordinates
   * @return Transformation matrix
   */
  KOKKOS_FUNCTION
  virtual mm::TransMatType build_transformation(
    const double& time,
    const mm::ThreeDVecType& xyz);

  /** Function to compute motion-specific velocity
   *
   * @param[in]  time      Current time
   * @param[in]  compTrans Transformation matrix
   *                       including all motions
   * @param[in]  mxyz      Model coordinates
   * @param[in]  cxyz      Transformed coordinates
   * @return Velocity vector associated with coordinates
   */
  KOKKOS_FUNCTION
  virtual mm::ThreeDVecType compute_velocity(
    const double& time,
    const mm::TransMatType& compTrans,
    const mm::ThreeDVecType& mxyz,
    const mm::ThreeDVecType& cxyz);

private:
  void load(const YAML::Node&);

  mm::ThreeDVecType factor_;
  mm::ThreeDVecType rate_;

  bool useRate_ = false;
};


} // nalu
} // sierra

#endif /* MOTIONSCALING_H */
