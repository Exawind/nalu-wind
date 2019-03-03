#ifndef MOTIONSCALING_H
#define MOTIONSCALING_H

#include "MotionBase.h"

namespace sierra{
namespace nalu{

class MotionScaling : public MotionBase
{
public:
  MotionScaling(const YAML::Node&);

  virtual ~MotionScaling()
  {

  }

  virtual void build_transformation(const double, const double* = nullptr);

  /** Function to compute motion-specific velocity
   *
   * @param[in] time           Current time
   * @param[in] compTrans      Transformation matrix
   *                           for points other than xyz
   * @param[in] xyz            Transformed coordinates
   */
  virtual ThreeDVecType compute_velocity(
    const double /* time */,
    const TransMatType& /* compTrans */,
    const double* /* xyz */)
  {
    throw std::runtime_error(
      "MotionScaling:compute_velocity() Scaling is not setup to be used as a non-inertial motion");
  }

private:
  MotionScaling() = delete;
  MotionScaling(const MotionScaling&) = delete;

  void load(const YAML::Node&);

  void scaling_mat(const ThreeDVecType&);

  ThreeDVecType factor_ = {{0.0,0.0,0.0}};
};


} // nalu
} // sierra

#endif /* MOTIONSCALING_H */
