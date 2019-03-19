#ifndef MOTIONPULSATINGSPHERE_H
#define MOTIONPULSATINGSPHERE_H

#include "MotionBase.h"

namespace sierra{
namespace nalu{

class MotionPulsatingSphere : public MotionBase
{
public:
  MotionPulsatingSphere(
    stk::mesh::MetaData&,
    const YAML::Node&);

  virtual ~MotionPulsatingSphere()
  {
  }

  virtual void build_transformation(const double, const double*);

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
  MotionPulsatingSphere() = delete;
  MotionPulsatingSphere(const MotionPulsatingSphere&) = delete;

  void load(const YAML::Node&);

  void scaling_mat(const double, const double*);

  double amplitude_{0.0};
  double frequency_{0.0};
};

} // nalu
} // sierra

#endif /* MOTIONPULSATINGSPHERE_H */
