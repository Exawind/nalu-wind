#ifndef MOTIONBASE_H
#define MOTIONBASE_H

#include <FieldTypeDef.h>

#include "yaml-cpp/yaml.h"

#include <algorithm>
#include <cassert>
#include <cfloat>

namespace sierra{
namespace nalu{

class MotionBase
{
public:
  //! Define matrix type alias
  static constexpr int transMatSize = 4;
  using TransMatType = std::array<std::array<double, transMatSize>, transMatSize>;

  //! Define 3D vector type alias
  static constexpr int threeDVecSize = 3;
  using ThreeDVecType = std::array<double, threeDVecSize>;

  MotionBase()
  {
  }

  virtual ~MotionBase()
  {
  }

  virtual void build_transformation(const double, const double* = nullptr) = 0;

  /** Function to compute motion-specific velocity
   *
   * @param[in] time           Current time
   * @param[in] compTrans      Transformation matrix
   *                           for points other than xyz
   * @param[in] xyz            Transformed coordinates
   */
  virtual ThreeDVecType compute_velocity(
    const double time,
    const TransMatType& compTrans,
    const double* xyz ) = 0;

  /** Composite addition of motions
   *
   * @param[in] motionL Left matrix in composite transformation of matrices
   * @param[in] motionR Right matrix in composite transformation of matrices
   * @return    4x4 matrix representing composite addition of motions
   */
  TransMatType add_motion(
    const TransMatType& motionL,
    const TransMatType& motionR);

  const TransMatType& get_trans_mat() const
  {
    return transMat_;
  }

  void set_computed_centroid( std::vector<double>& centroid )
  {
    std::copy_n(centroid.begin(), threeDVecSize, origin_.begin());
  }

  virtual void post_work(
    stk::mesh::BulkData&,
    stk::mesh::PartVector&,
    stk::mesh::PartVector&,
    bool&)
  {
  }

  static const TransMatType identityMat_;

protected:
  void reset_mat(TransMatType& mat)
  {
    mat = identityMat_;
  }

  /** Transformation matrix
   *
   * A 4x4 matrix that combines rotation, translation, scaling,
   * allowing representation of all affine transformations
   */
  TransMatType transMat_ = identityMat_;

  /** Computed centroid
   *
   * A 3x1 vector storing the centroid computed respective
   * to a collection of parts defined in the input file
   */
  ThreeDVecType origin_ = {{0.0,0.0,0.0}};

  double startTime_{0.0};
  double endTime_{std::numeric_limits<double>::max()};

private:
    MotionBase(const MotionBase&) = delete;
};

} // nalu
} // sierra

#endif /* MOTIONBASE_H */
