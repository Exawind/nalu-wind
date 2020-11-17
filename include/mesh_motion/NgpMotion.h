#ifndef NGPMOTION_H
#define NGPMOTION_H

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <vector>
#include <array>

#include "NGPInstance.h"
#include "ngp_utils/NgpTypes.h"

namespace YAML { class Node; }

namespace sierra{
namespace nalu{

class NgpMotion
{
public:
  //! Define matrix type alias
  using TransMatType = double [nalu_ngp::NDimMax+1][nalu_ngp::NDimMax+1];

  //! Define 3D vector type alias
  using ThreeDVecType = double [nalu_ngp::NDimMax];

  KOKKOS_FORCEINLINE_FUNCTION
  NgpMotion() = default;

  KOKKOS_FUNCTION
  virtual ~NgpMotion() {}

  virtual NgpMotion* create_on_device() = 0;

  virtual void free_on_device() = 0;

  virtual void build_transformation(const double, const double* = nullptr) = 0;

  /** Function to compute motion-specific velocity
   *
   * @param[in]  time       Current time
   * @param[in]  compTrans  Transformation matrix
   *                        for points other than xyz
   * @param[in]  mxyz       Model coordinates
   * @param[in]  mxyz       Transformed coordinates
   * @param[out] vel        Velocity associated with coordinates
   */
  virtual void compute_velocity(
    const double time,
    const TransMatType& compTrans,
    const double* mxyz,
    const double* cxyz,
    ThreeDVecType& vel) = 0;

  /** Composite addition of motions
   *
   * @param[in]  motionL        Left matrix in composite transformation of matrices
   * @param[in]  motionR        Right matrix in composite transformation of matrices
   * @param[out] comp_trans_mat 4x4 matrix representing composite addition of motions
   */
  KOKKOS_FORCEINLINE_FUNCTION
  void add_motion(
    const TransMatType& motionL,
    const TransMatType& motionR,
    TransMatType& comp_trans_mat)
  {
    for (int r = 0; r < nalu_ngp::NDimMax+1; r++) {
      for (int c = 0; c < nalu_ngp::NDimMax+1; c++) {
        comp_trans_mat[r][c] = 0.0;
        for (int k = 0; k < nalu_ngp::NDimMax+1; k++) {
          comp_trans_mat[r][c] += motionL[r][k] * motionR[k][c];
        }
      }
    }
  }

  void set_computed_centroid( std::vector<double>& centroid )
  {
    for (int d = 0; d < nalu_ngp::NDimMax; ++d)
      origin_[d] = centroid[d];
  }

  const TransMatType& get_trans_mat() const
  {
    return transMat_;
  }

  bool is_deforming()
  {
    return isDeforming_;
  }

  /** Reset matrix to an identity matrix
   *
   * @param[in]  mat  4x4 matrix
   */
  KOKKOS_FORCEINLINE_FUNCTION
  static void reset_mat(TransMatType& mat)
  {
    for (int r = 0; r < nalu_ngp::NDimMax+1; r++) {
      for (int c = 0; c < nalu_ngp::NDimMax+1; c++) {
        if(r == c)
          mat[r][c] = 1.0;
        else
          mat[r][c] = 0.0;
      }
    }
  }

  /** Make matrix into an identity matrix
   *
   * @param[in]  dest_mat  4x4 matrix to be copied into
   * @param[in]  src_mat   4x4 matrix to be copied from
   */
  KOKKOS_FORCEINLINE_FUNCTION
  static void copy_mat(
    TransMatType& dest_mat,
    const TransMatType& src_mat)
  {
    for (int r = 0; r < nalu_ngp::NDimMax+1; r++) {
      for (int c = 0; c < nalu_ngp::NDimMax+1; c++) {
        dest_mat[r][c] = src_mat[r][c];
      }
    }
  }

protected:
  /** Transformation matrix
   *
   * A 4x4 matrix that combines rotation, translation, scaling,
   * allowing representation of all affine transformations
   */
  TransMatType transMat_ = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

  /** Centroid
   *
   * A 3x1 vector storing the centroid as computed
   * to a collection of parts or as defined in the input file
   */
  ThreeDVecType origin_ = {0.0,0.0,0.0};

  double startTime_{0.0};
  double endTime_{DBL_MAX};

  bool isDeforming_ = false;
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
