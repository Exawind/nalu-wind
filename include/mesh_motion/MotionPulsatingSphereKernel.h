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

  KOKKOS_FUNCTION
  virtual void build_transformation(const DblType, const DblType*);

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

  /** perform post compute geometry work for this motion
   *
   * @param[in] computedMeshVelDiv flag to denote if divergence of
   *                               mesh velocity already computed
   */
  void post_compute_geometry(
    stk::mesh::BulkData&,
    stk::mesh::PartVector&,
    stk::mesh::PartVector&,
    bool& computedMeshVelDiv );

private:
  void load(const YAML::Node&);

  void scaling_mat(const DblType, const DblType*);

  DblType amplitude_{0.0};
  DblType frequency_{0.0};
};

} // nalu
} // sierra

#endif /* MOTIONPULSATINGSPHERE_H */
