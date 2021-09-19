#ifndef MOTIONSINROTATIONKERNEL_H
#define MOTIONSINROTATIONKERNEL_H

#include "NgpMotion.h"

namespace sierra{
namespace nalu{

class MotionSinRotationKernel : public NgpMotionKernel<MotionSinRotationKernel>
{
public:
  MotionSinRotationKernel(const YAML::Node&);

  KOKKOS_FUNCTION
  MotionSinRotationKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MotionSinRotationKernel() = default;

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

  mm::ThreeDVecType axis_{0.0,0.0,1.0};

  double amplt_{0.0};
  double omega_{0.0};
  double angle_{0.0};
  double ang_vel_{0.0};
  
};

} // nalu
} //sierra

#endif /* MOTIONSINROTATIONKERNEL_H */
