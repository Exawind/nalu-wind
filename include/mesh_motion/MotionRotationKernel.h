#ifndef MOTIONROTATIONKERNEL_H
#define MOTIONROTATIONKERNEL_H

#include "NgpMotion.h"

namespace sierra{
namespace nalu{

class MotionRotationKernel : public NgpMotionKernel<MotionRotationKernel>
{
public:
  MotionRotationKernel(const YAML::Node&);

  KOKKOS_FUNCTION
  MotionRotationKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MotionRotationKernel() = default;

  KOKKOS_FUNCTION
  virtual void build_transformation(const DblType, const DblType* = nullptr);

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
    const DblType time,
    const TransMatType& compTrans,
    const DblType* mxyz,
    const DblType* cxyz,
    ThreeDVecType& vel);

private:
  void load(const YAML::Node&);

  void rotation_mat(const DblType);

  ThreeDVecType axis_ = {0.0,0.0,1.0};

  DblType omega_{0.0};
  DblType angle_{0.0};

  bool useOmega_ = true;
};

} // nalu
} //sierra

#endif /* MOTIONROTATIONKERNEL_H */
