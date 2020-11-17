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

  KOKKOS_FUNCTION
  virtual void build_transformation(const double, const double* = nullptr);

  /** Function to compute motion-specific velocity
   *
   * @param[in]  time       Current time
   * @param[in]  compTrans  Transformation matrix
   *                        for points other than xyz
   * @param[in]  mxyz       Model coordinates
   * @param[in]  mxyz       Transformed coordinates
   * @param[out] vel        Velocity associated with coordinates
   */
  KOKKOS_FUNCTION
  virtual void compute_velocity(
    const double time,
    const TransMatType& compTrans,
    const double* mxyz,
    const double* cxyz,
    ThreeDVecType& vel);

private:
  void load(const YAML::Node&);

  void scaling_mat(const ThreeDVecType&);

  ThreeDVecType factor_ = {0.0,0.0,0.0};
  ThreeDVecType rate_ = {0.0,0.0,0.0};

  bool useRate_ = false;
};


} // nalu
} // sierra

#endif /* MOTIONSCALING_H */
