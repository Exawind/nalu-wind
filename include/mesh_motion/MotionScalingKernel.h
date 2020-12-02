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
   * @param[in]  time     Current time
   * @param[in]  xyz      Coordinates
   * @param[out] transMat Transformation matrix
   */
  KOKKOS_FUNCTION
  virtual void build_transformation(
    const double& time,
    const ThreeDVecType& xyz,
    TransMatType& transMat);

  /** Function to compute motion-specific velocity
   *
   * @param[in]  time       Current time
   * @param[in]  compTrans  Transformation matrix
   *                        including all motions
   * @param[in]  mxyz       Model coordinates
   * @param[in]  cxyz       Transformed coordinates
   * @param[out] vel        Velocity associated with coordinates
   */
  KOKKOS_FUNCTION
  virtual void compute_velocity(
    const double& time,
    const TransMatType& compTrans,
    const ThreeDVecType& mxyz,
    const ThreeDVecType& cxyz,
    ThreeDVecType& vel);

private:
  void load(const YAML::Node&);

  ThreeDVecType factor_ = {0.0,0.0,0.0};
  ThreeDVecType rate_ = {0.0,0.0,0.0};

  bool useRate_ = false;
};


} // nalu
} // sierra

#endif /* MOTIONSCALING_H */
