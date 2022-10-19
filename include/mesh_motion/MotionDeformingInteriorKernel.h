#ifndef MOTIONDEFORMINGINTERIORKERNEL_H
#define MOTIONDEFORMINGINTERIORKERNEL_H

#include "NgpMotion.h"

namespace sierra {
namespace nalu {

class MotionDeformingInteriorKernel
  : public NgpMotionKernel<MotionDeformingInteriorKernel>
{
public:
  MotionDeformingInteriorKernel(stk::mesh::MetaData&, const YAML::Node&);

  MotionDeformingInteriorKernel() = default;

  virtual ~MotionDeformingInteriorKernel() = default;

  /** Function to compute motion-specific transformation matrix
   *
   * @param[in] time Current time
   * @param[in] xyz  Coordinates
   * @return Transformation matrix
   */
  KOKKOS_FUNCTION
  virtual mm::TransMatType
  build_transformation(const double& time, const mm::ThreeDVecType& xyz);

  /** Function to compute motion-specific velocity
   *
   * @param[in]  time      Current time
   * @param[in]  compTrans Transformation matrix
   *                       including all motions
   * @param[in]  mxyz      Model coordinates
   * @param[in]  cxyz      Transformed coordinates
   * @return Velocity vector associated with coordinates
   */
  KOKKOS_FUNCTION
  virtual mm::ThreeDVecType compute_velocity(
    const double& time,
    const mm::TransMatType& compTrans,
    const mm::ThreeDVecType& mxyz,
    const mm::ThreeDVecType& cxyz);

  /** perform post compute geometry work for this motion
   *
   * @param[in] computedMeshVelDiv flag to denote if divergence of
   *                               mesh velocity already computed
   */
  void post_compute_geometry(
    stk::mesh::BulkData&,
    stk::mesh::PartVector&,
    stk::mesh::PartVector&,
    bool& computedMeshVelDiv);

private:
  void load(const YAML::Node&);

  mm::ThreeDVecType xyzMin_;
  mm::ThreeDVecType xyzMax_;

  mm::ThreeDVecType amplitude_{0.0, 0.0, 0.0};
  mm::ThreeDVecType frequency_{0.0, 0.0, 0.0};
};

} // namespace nalu
} // namespace sierra

#endif /* MOTIONDEFORMINGINTERIORKERNEL_H */
