#ifndef MOTIONROTATION_H
#define MOTIONROTATION_H

#include "MotionBase.h"

namespace sierra{
namespace nalu{

class MotionRotation : public MotionBase
{
public:
  MotionRotation(const YAML::Node&);

  virtual ~MotionRotation()
  {
  }

  virtual void build_transformation(const double, const double* = nullptr);

  /** Function to compute motion-specific velocity
   *
   * @param[in] time           Current time
   * @param[in] compTrans      Transformation matrix
   *                           for points other than xyz
   * @param[in] mxyz           Model coordinates
   * @param[in] mxyz           Transformed coordinates
   */
  virtual ThreeDVecType compute_velocity(
    const double time,
    const TransMatType& compTrans,
    const double* mxyz,
    const double* cxyz );

private:
  MotionRotation() = delete;
  MotionRotation(const MotionRotation&) = delete;

  void load(const YAML::Node&);

  void rotation_mat(const double);

  ThreeDVecType axis_ = {{0.0,0.0,1.0}};

  double omega_{0.0};
  double angle_{0.0};

  bool useOmega_ = true;
};


} // nalu
} //sierra

#endif /* MOTIONROTATION_H */
