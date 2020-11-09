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

  void rotation_mat(const double);

  ThreeDVecType axis_ = {0.0,0.0,1.0};

  double omega_{0.0};
  double angle_{0.0};

  bool useOmega_ = true;
};

} // nalu
} //sierra

#endif /* MOTIONROTATIONKERNEL_H */
