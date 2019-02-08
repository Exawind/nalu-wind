#ifndef MOTIONPULSATINGSPHERE_H
#define MOTIONPULSATINGSPHERE_H

#include "MotionBase.h"

namespace sierra{
namespace nalu{

class MotionPulsatingSphere : public MotionBase
{
public:
  MotionPulsatingSphere(const YAML::Node&);

  virtual ~MotionPulsatingSphere() {}

  virtual void build_transformation(const double, const double*);

  /** Function to compute motion-specific velocity
   *
   * @param[in] time           Current time
   * @param[in] comp_trans_mat Transformation matrix
   *                           for points other than xyz
   * @param[in] xyz            Transformed coordinates
   */
  virtual ThreeDVecType compute_velocity(
    const double time,
    const TransMatType& compTrans,
    const double* xyz );

private:
  MotionPulsatingSphere() = delete;
  MotionPulsatingSphere(const MotionPulsatingSphere&) = delete;

  void load(const YAML::Node&);

  void scaling_mat(const double, const double*);

  double amplitude_{1.0};
  double frequency_{1.0};
};

} // nalu
} // sierra

#endif /* MOTIONPULSATINGSPHERE_H */
