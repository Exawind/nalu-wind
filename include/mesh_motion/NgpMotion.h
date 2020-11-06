#ifndef NGPMOTION_H
#define NGPMOTION_H

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <vector>
#include <array>

#include "NGPInstance.h"

namespace YAML { class Node; }

namespace stk {
namespace mesh {
class MetaData;
class BulkData;
class Part;

typedef std::vector<Part*> PartVector;
}
}

namespace sierra{
namespace nalu{

struct NgpMotionTraits
{
  static constexpr int NDimMax = 3;
  using DblType = double;
  using ShmemType = DeviceShmem;
};

class NgpMotion
{
public:
  //! Define matrix type alias
  using TransMatType = std::array<std::array<double, NgpMotionTraits::NDimMax+1>, NgpMotionTraits::NDimMax+1>;

  //! Define 3D vector type alias
  using ThreeDVecType = std::array<double, NgpMotionTraits::NDimMax>;

  KOKKOS_FORCEINLINE_FUNCTION
  NgpMotion() = default;

  KOKKOS_FUNCTION
  virtual ~NgpMotion() {}

  virtual NgpMotion* create_on_device() = 0;

  virtual void free_on_device() = 0;

  virtual void build_transformation(const double, const double* = nullptr) = 0;

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
    const double* cxyz ) = 0;

  virtual void post_compute_geometry(
    stk::mesh::BulkData&,
    stk::mesh::PartVector&,
    stk::mesh::PartVector&,
    bool&)
  {
  }

  /** Composite addition of motions
   *
   * @param[in] motionL Left matrix in composite transformation of matrices
   * @param[in] motionR Right matrix in composite transformation of matrices
   * @return    4x4 matrix representing composite addition of motions
   */
  KOKKOS_FORCEINLINE_FUNCTION
  TransMatType add_motion(
    const TransMatType& motionL,
    const TransMatType& motionR)
  {
    TransMatType comp_trans_mat_ = {};

    for (int r = 0; r < NgpMotionTraits::NDimMax+1; r++)
      for (int c = 0; c < NgpMotionTraits::NDimMax+1; c++)
        for (int k = 0; k < NgpMotionTraits::NDimMax+1; k++) {
          comp_trans_mat_[r][c] += motionL[r][k] * motionR[k][c];
        }

    return comp_trans_mat_;
  }

  void set_computed_centroid( std::vector<double>& centroid )
  {
    std::copy_n(centroid.begin(), NgpMotionTraits::NDimMax, origin_.begin());
  }

  const TransMatType& get_trans_mat() const
  {
    return transMat_;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr TransMatType identity_mat()
  {
    return {{{{1,0,0,0}},
             {{0,1,0,0}},
             {{0,0,1,0}},
             {{0,0,0,1}}}};
  }

protected:
  KOKKOS_FORCEINLINE_FUNCTION
  void reset_mat(TransMatType& mat)
  {
    mat = identity_mat();
  }

  /** Transformation matrix
   *
   * A 4x4 matrix that combines rotation, translation, scaling,
   * allowing representation of all affine transformations
   */
  TransMatType transMat_ = identity_mat();

  /** Centroid
   *
   * A 3x1 vector storing the centroid as computed
   * to a collection of parts or as defined in the input file
   */
  ThreeDVecType origin_ = {{0.0,0.0,0.0}};

  double startTime_{0.0};
  double endTime_{std::numeric_limits<double>::max()};
};

template<typename T>
class NgpMotionKernel : public NgpMotion
{
public:
  KOKKOS_FORCEINLINE_FUNCTION
  NgpMotionKernel() = default;

  KOKKOS_FUNCTION
  virtual ~NgpMotionKernel() = default;

  virtual NgpMotion* create_on_device() final
  {
    free_on_device();
    deviceCopy_ = nalu_ngp::create<T>(*dynamic_cast<T*>(this));
    return deviceCopy_;
  }

  virtual void free_on_device() final
  {
    if (deviceCopy_ != nullptr) {
      nalu_ngp::destroy<T>(dynamic_cast<T*>(deviceCopy_));
      deviceCopy_ = nullptr;
    }
  }

  T* device_copy() const { return deviceCopy_; }

private:
  T* deviceCopy_{nullptr};
};

} // nalu
} // sierra

#endif /* NGPMOTION_H */
