#ifndef MOTIONPRESCRIBEDKERNEL_H
#define MOTIONPRESCRIBEDKERNEL_H

#include "NgpMotion.h"
#include <array>
#include <vector>

namespace sierra {
namespace nalu {

struct MotionValues
{
  double time_value = 0.0;
  std::array<double, 3> x_disp = {-1e16, -1e16, -1e16};
  std::array<double, 3> vel = {0.0, 0.0, 0.0};
  std::array<double, 3> angular_vel = {0.0, 0.0, 0.0};
  std::array<double, 3> angular_disp = {-1e16, -1e16, -1e16};
};

class MotionPrescribedKernel : public NgpMotionKernel<MotionPrescribedKernel>
{
public:
  MotionPrescribedKernel(const YAML::Node&);

  MotionPrescribedKernel() = default;

  virtual ~MotionPrescribedKernel() = default;

  /** Function to compute motion-specific transformation matrix
   *
   * @param[in] time Current time
   * @param[in] xyz  Coordinates
   * @return Transformation matrix
   */
  KOKKOS_FUNCTION
  virtual mm::TransMatType
  build_transformation(const double& time, const mm::ThreeDVecType& xyz);

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
  Kokkos::View<double**> defined_motion_values_;
};

} // namespace nalu
} // namespace sierra

#endif /* MOTIONPRESCRIBEDKERNEL_H */
