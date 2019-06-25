/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ABLSRCINTERP_H
#define ABLSRCINTERP_H

#include "KokkosInterface.h"
#include "stk_util/util/ReportHandler.hpp"

#include <vector>

namespace sierra {
namespace nalu {

namespace abl_impl {

using Array1D = Kokkos::View<double*, MemSpace>;
using Array2D = Kokkos::View<double*[3], MemSpace>;

KOKKOS_FORCEINLINE_FUNCTION
unsigned abl_find_index(
  const Array1D& xinp,
  const double& xout)
{
  const unsigned npts = xinp.extent(0);
  for (unsigned i = 1; i < npts; ++i) {
    if (xout <= xinp(i)) {
      return (i - 1);
    }
  }
  return (npts - 2);
}
}

/** NGP-friendly source interpolation class for use with ABL forcing term
 *
 */
class ABLScalarInterpolator
{
public:
  using Array1D = abl_impl::Array1D;
  KOKKOS_FORCEINLINE_FUNCTION ABLScalarInterpolator() = default;
  KOKKOS_FORCEINLINE_FUNCTION ~ABLScalarInterpolator() = default;

  ABLScalarInterpolator(
    const std::vector<double>& xinp,
    const std::vector<double>& yinp
  ) : xinp_("ABLScalarX", xinp.size()),
      yinp_("ABLScalarY", yinp.size()),
      xinpHost_(Kokkos::create_mirror_view(xinp_)),
      yinpHost_(Kokkos::create_mirror_view(yinp_)),
      numPts_(xinp_.size())
  {
    for (unsigned i=0; i < numPts_; ++i) {
      xinpHost_(i) = xinp[i];
      yinpHost_(i) = yinp[i];
    }
    Kokkos::deep_copy(xinp_, xinpHost_);
    Kokkos::deep_copy(yinp_, yinpHost_);
  }

  /** Update the source array on device
   */
  void update_view_on_device(const std::vector<double>& yinp)
  {
    for (unsigned i=0; i < numPts_; ++i) {
      yinpHost_(i) = yinp[i];
    }
    Kokkos::deep_copy(yinp_, yinpHost_);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(const double& xout, double& yout) const
  {
    // Forcing only at hub-height
    if (numPts_ == 1)
      yout = yinp_(0);

    if (xout <= xinp_(0))
      // Constant forcing below first specified height
      yout = yinp_(0);
    else if (xout >= xinp_(numPts_ - 1))
      // Constant forcing above last specified height
      yout = yinp_(numPts_ - 1);
    else {
      // Linearly interpolate source term within user-specified heights
      auto idx = abl_impl::abl_find_index(xinp_, xout);
      double fac = (xout - xinp_(idx)) / (xinp_(idx + 1) - xinp_(idx));
      yout = (1.0 - fac) * yinp_(idx) + fac * yinp_(idx + 1);
    }
  }

private:
  //! Height array (device view)
  Array1D xinp_;
  //! Source array (e.g., temperature)
  Array1D yinp_;

  //! Height array (host view)
  Array1D::HostMirror xinpHost_;
  //! Source array (host view)
  Array1D::HostMirror yinpHost_;

  //! Number of user-specified heights
  unsigned numPts_;
};

/** NGP-friendly source interpolation class for use with ABL forcing term
 *
 */
class ABLVectorInterpolator
{
public:
  using Array1D = abl_impl::Array1D;
  using Array2D = abl_impl::Array2D;
  KOKKOS_FORCEINLINE_FUNCTION ABLVectorInterpolator() = default;
  KOKKOS_FORCEINLINE_FUNCTION ~ABLVectorInterpolator() = default;

  ABLVectorInterpolator(
    const std::vector<double>& xinp,
    const std::vector<std::vector<double>>& yinp
  ) : xinp_("ABLVectorX", xinp.size()),
      yinp_("ABLVectorY", yinp.size()),
      xinpHost_(Kokkos::create_mirror_view(xinp_)),
      yinpHost_(Kokkos::create_mirror_view(yinp_)),
      numPts_(xinp.size())
  {
    for (unsigned i=0; i < numPts_; ++i) {
      xinpHost_(i) = xinp[i];
      for (int d=0; d < ndim; ++d)
        yinpHost_(i, d) = yinp[d][i];
    }
    Kokkos::deep_copy(xinp_, xinpHost_);
    Kokkos::deep_copy(yinp_, yinpHost_);
  }

  /** Update the source array on device
   */
  void update_view_on_device(const std::vector<std::vector<double>>& yinp)
  {
    for (unsigned i=0; i < numPts_; ++i) {
      for (int d=0; d < ndim; ++d)
        yinpHost_(i, d) = yinp[d][i];
    }
    Kokkos::deep_copy(yinp_, yinpHost_);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(const double& xout, double* yout) const
  {
    // Forcing only at hub-height
    if (numPts_ == 1)
      for (int d=0; d < ndim; d++)
        yout[d] = yinp_(0, d);

    if (xout <= xinp_(0))
      // constant forcing below first specified height
      for (int d=0; d < ndim; d++)
        yout[d] = yinp_(0, d);
    else if (xout >= xinp_(numPts_ - 1)) {
      // constant forcing above last specified height
      const int idx = numPts_ - 1;
      for (int d=0; d < ndim; d++)
        yout[d] = yinp_(idx, d);
    }
    else {
      // interpolate within user-provided range of inputs
      auto idx = abl_impl::abl_find_index(xinp_, xout);
      double fac = (xout - xinp_(idx)) / (xinp_(idx + 1) - xinp_(idx));
      for (int d=0; d < ndim; d++)
        yout[d] = (1.0 - fac) * yinp_(idx, d) + fac * yinp_(idx + 1, d);
    }
  }

private:
  static constexpr int ndim = 3;

  //! Height array (device view)
  Array1D xinp_;
  //! Source vector array (2-D device view)
  Array2D yinp_;

  Array1D::HostMirror xinpHost_;
  Array2D::HostMirror yinpHost_;

  //! Number of user-specified heights
  unsigned numPts_;
};

}  // nalu
}  // sierra


#endif /* ABLSRCINTERP_H */
