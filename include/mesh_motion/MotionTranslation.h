#ifndef MOTIONTRANSLATION_H
#define MOTIONTRANSLATION_H

#include "MotionBase.h"

namespace sierra{
namespace nalu{

class MotionTranslation : public MotionBase
{
public:
  MotionTranslation(const YAML::Node&);

  virtual ~MotionTranslation()
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
