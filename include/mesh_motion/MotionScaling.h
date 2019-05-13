#ifndef MOTIONSCALING_H
#define MOTIONSCALING_H

#include "MotionBase.h"

namespace sierra{
namespace nalu{

class MotionScaling : public MotionBase
{
public:
  MotionScaling(
    stk::mesh::MetaData&,
    const YAML::Node&);

  virtual ~MotionScaling()
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
  MotionScaling() = delete;
  MotionScaling(const MotionScaling&) = delete;

  void load(const YAML::Node&);

  void scaling_mat(const ThreeDVecType&);

  ThreeDVecType factor_ = {{0.0,0.0,0.0}};
  ThreeDVecType rate_ = {{0.0,0.0,0.0}};

  bool useRate_ = false;
};


} // nalu
} // sierra

#endif /* MOTIONSCALING_H */
