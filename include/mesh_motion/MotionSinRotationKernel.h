#ifndef MOTIONSINROTATIONKERNEL_H
#define MOTIONSINROTATIONKERNEL_H

#include "NgpMotion.h"
#include "MotionRotationKernel.h"

namespace sierra{
namespace nalu{

class MotionSinRotationKernel : public MotionRotationKernel
{
public:
  MotionSinRotationKernel(const YAML::Node&);

  KOKKOS_FUNCTION
  MotionSinRotationKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MotionSinRotationKernel() = default;

private:
  void load(const YAML::Node&);

  double get_cur_angle(const double time);

  double get_cur_ang_vel(const double time);

  double amplt_{0.0};
  double ang_vel_{0.0};
  double phase_{0.0};

};

} // nalu
} //sierra

#endif /* MOTIONSINROTATIONKERNEL_H */
