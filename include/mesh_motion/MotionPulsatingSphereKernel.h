#ifndef MOTIONPULSATINGSPHERE_H
#define MOTIONPULSATINGSPHERE_H

#include "NgpMotion.h"

namespace YAML { class Node; }

namespace sierra{
namespace nalu{

class MotionPulsatingSphereKernel : public NgpMotionKernel<MotionPulsatingSphereKernel>
{
public:
  MotionPulsatingSphereKernel(
    stk::mesh::MetaData&,
    const YAML::Node&);

  KOKKOS_FUNCTION
  MotionPulsatingSphereKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MotionPulsatingSphereKernel() = default;

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

  double amplitude_{0.0};
  double frequency_{0.0};
};

} // nalu
} // sierra

#endif /* MOTIONPULSATINGSPHERE_H */
