// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Tri33DCVFEM_h
#define Tri33DCVFEM_h

#include <master_element/MasterElement.h>

#include <AlgTraits.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>

namespace sierra {
namespace nalu {

// 3D Tri 3
class Tri3DSCS : public MasterElement
{
public:
  KOKKOS_FUNCTION
  Tri3DSCS();
  KOKKOS_FUNCTION virtual ~Tri3DSCS() {}

  using AlgTraits = AlgTraitsTri3;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int ordinal = 0) const final;

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType**, DeviceShmem>& areav) override;

  virtual void determinant(
    const SharedMemView<double**>& coords,
    SharedMemView<double**>& areav) override;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void
  tri_shape_fcn(const double* par_coord, SharedMemView<SCALAR**, SHMEM>& shpfc);

  void
  tri_shape_fcn(const int npts, const double* par_coord, double* shape_fcn);

  double isInElement(
    const double* elemNodalCoord,
    const double* pointCoord,
    double* isoParCoord) override;

  double parametric_distance(const std::array<double, 3>& x);

  void interpolatePoint(
    const int& nComp,
    const double* isoParCoord,
    const double* field,
    double* result) override;

  void general_shape_fcn(
    const int numIp, const double* isoParCoord, double* shpfc) override;

  void general_normal(
    const double* isoParCoord, const double* coords, double* normal) override;

  virtual const double* integration_locations() const final { return intgLoc_; }
  virtual const double* integration_location_shift() const final
  {
    return intgLocShift_;
  }

protected:
  KOKKOS_FUNCTION virtual void
  shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

  KOKKOS_FUNCTION virtual void
  shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void
  shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

private:
  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScsIp_;

  // define ip node mappings; ordinal size = 1
  const int ipNodeMap_[3] = {0, 1, 2};

  // standard integration location
  static constexpr double seven36ths = 7.0 / 36.0;
  static constexpr double eleven18ths = 11.0 / 18.0;
  const double intgLoc_[6] = {seven36ths,  seven36ths,   // surf 1
                              eleven18ths, seven36ths,   // surf 2
                              seven36ths,  eleven18ths}; // surf 3

  // shifted
  const double intgLocShift_[6] = {0.00, 0.00,  // surf 1
                                   1.00, 0.00,  // surf 2
                                   0.00, 1.00}; // surf 3

  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scs(
    const SharedMemView<DBLTYPE**, SHMEM>& coords,
    SharedMemView<DBLTYPE**, SHMEM>& areav);
};

} // namespace nalu
} // namespace sierra

#endif
