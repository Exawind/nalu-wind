#ifndef MOTIONPULSATINGSPHERE_H
#define MOTIONPULSATINGSPHERE_H

#include "NgpMotion.h"

namespace YAML { class Node; }

namespace sierra{
namespace nalu{

class MotionPulsatingSphereKernel : public NgpMotionKernel<MotionPulsatingSphereKernel>
{
public:
  MotionPulsatingSphereKernel(
    stk::mesh::MetaData&,
    const YAML::Node&);

  KOKKOS_FUNCTION
  MotionPulsatingSphereKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MotionPulsatingSphereKernel() = default;

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

  double amplitude_{0.0};
  double frequency_{0.0};
};

} // nalu
} // sierra

#endif /* MOTIONPULSATINGSPHERE_H */
