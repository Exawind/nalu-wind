#ifndef MOTIONTRANSLATIONKERNEL_H
#define MOTIONTRANSLATIONKERNEL_H

#include "NgpMotion.h"

namespace sierra{
namespace nalu{

class MotionTranslationKernel : public NgpMotionKernel<MotionTranslationKernel>
{
public:
  MotionTranslationKernel(const YAML::Node&);

  KOKKOS_FUNCTION
  MotionTranslationKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MotionTranslationKernel() = default;

  /** Function to compute motion-specific transformation matrix
   *
   * @param[in]  time     Current time
   * @param[in]  xyz      Coordinates
   * @param[out] transMat Transformation matrix
   */
  KOKKOS_FUNCTION
  virtual void build_transformation(
    const double& time,
    const ThreeDVecType& xyz,
    TransMatType& transMat);

  /** Function to compute motion-specific velocity
   *
   * @param[in]  time       Current time
   * @param[in]  compTrans  Transformation matrix
   *                        including all motions
   * @param[in]  mxyz       Model coordinates
   * @param[in]  cxyz       Transformed coordinates
   * @param[out] vel        Velocity associated with coordinates
   */
  KOKKOS_FUNCTION
  virtual void compute_velocity(
    const double& time,
    const TransMatType& compTrans,
    const ThreeDVecType& mxyz,
    const ThreeDVecType& cxyz,
    ThreeDVecType& vel);

private:
  void load(const YAML::Node&);

  ThreeDVecType displacement_ = {0.0,0.0,0.0};
  ThreeDVecType velocity_ = {0.0,0.0,0.0};

  bool useVelocity_ = false;
};

} // nalu
} // sierra

#endif /* MOTIONTRANSLATIONKERNEL_H */
