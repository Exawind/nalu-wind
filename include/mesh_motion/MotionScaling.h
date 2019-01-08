#ifndef MOTIONSCALING_H
#define MOTIONSCALING_H

#include "MotionBase.h"

namespace sierra{
namespace nalu{

class MotionScaling : public MotionBase
{
public:
  MotionScaling(const YAML::Node&);

  virtual ~MotionScaling() {}

  virtual void build_transformation(const double, const double* = nullptr);

  /** Function to compute motion-specific velocity
   *
   * @param[in] time           Current time
   * @param[in] comp_trans_mat Transformation matrix
   *                           for points other than xyz
   * @param[in] xyz            Transformed coordinates
   */
  virtual threeDVecType compute_velocity(
    double time,
    const transMatType& comp_trans,
    double* xyz );

private:
  MotionScaling() = delete;
  MotionScaling(const MotionScaling&) = delete;

  void load(const YAML::Node&);

  void scaling_mat(const threeDVecType&);

  threeDVecType factor_ = {{0.0,0.0,0.0}};
  threeDVecType velocity_ = {{0.0,0.0,0.0}};

  bool useVelocity_ = false;
};


} // nalu
} // sierra

#endif /* MOTIONSCALING_H */
