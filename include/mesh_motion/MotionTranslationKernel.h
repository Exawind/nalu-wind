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

  void translation_mat(const ThreeDVecType&);

  ThreeDVecType displacement_ = {0.0,0.0,0.0};
  ThreeDVecType velocity_ = {0.0,0.0,0.0};

  bool useVelocity_ = false;
};

} // nalu
} // sierra

#endif /* MOTIONTRANSLATIONKERNEL_H */
