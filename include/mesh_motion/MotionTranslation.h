#ifndef MOTIONTRANSLATION_H
#define MOTIONTRANSLATION_H

#include "MotionBase.h"

namespace sierra{
namespace nalu{

class MotionTranslation : public MotionBase
{
public:
  MotionTranslation(const YAML::Node&);

  virtual ~MotionTranslation() {}

  virtual void build_transformation(const double, const double* = nullptr);

  /** Function to compute motion-specific velocity
   *
   * @param[in] time           Current time
   * @param[in] comp_trans_mat Transformation matrix
   *                           for points other than xyz
   * @param[in] xyz            Transformed coordinates
   */
  virtual ThreeDVecType compute_velocity(
    double time,
    const TransMatType& comp_trans,
    double* xyz );

private:
  MotionTranslation() = delete;
  MotionTranslation(const MotionTranslation&) = delete;

  void load(const YAML::Node&);

  void translation_mat(const ThreeDVecType&);

  ThreeDVecType displacement_ = {{0.0,0.0,0.0}};
  ThreeDVecType velocity_ = {{0.0,0.0,0.0}};

  bool useVelocity_ = false;
};

} // nalu
} // sierra

#endif /* MOTIONTRANSLATION_H */
