#ifndef MOTIONOscillationKERNEL_H
#define MOTIONOscillationKERNEL_H

#include "NgpMotion.h"

namespace sierra {
namespace nalu {

class MotionOscillationKernel : public NgpMotionKernel<MotionOscillationKernel>
{
public:
  MotionOscillationKernel(const YAML::Node&);

  MotionOscillationKernel() = default;

  virtual ~MotionOscillationKernel() = default;

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

  mm::ThreeDVecType direction_{0.0, 0.0, 1.0};

  double period_{0.0};
  double amplitude_{0.0};

};

} // namespace nalu
} // namespace sierra

#endif /* MOTIONOscillationKERNEL_H */
