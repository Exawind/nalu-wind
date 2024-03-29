// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Quad43DSCS_h
#define Quad43DSCS_h

#include "master_element/MasterElement.h"

#include <array>

namespace sierra {
namespace nalu {

// 3D Quad 4
class Quad3DSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsQuad4;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION
  Quad3DSCS();
  KOKKOS_FUNCTION virtual ~Quad3DSCS() {}

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int ordinal = 0) const final;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType**, DeviceShmem>& areav) override;

  virtual void determinant(
    const SharedMemView<double**>& coords,
    SharedMemView<double**>& areav) override;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void quad4_shape_fcn(
    const double* isoParCoord, SharedMemView<SCALAR**, SHMEM>& shpfc);

  double isInElement(
    const double* elemNodalCoord,
    const double* pointCoord,
    double* isoParCoord) override;

  void interpolatePoint(
    const int& nComp,
    const double* isoParCoord,
    const double* field,
    double* result) override;

  void general_shape_fcn(
    const int numIp, const double* isoParCoord, double* shpfc) override;

  void general_normal(
    const double* isoParCoord, const double* coords, double* normal) override;

  void non_unit_face_normal(
    const double* par_coord,
    const double* elem_nodal_coor,
    double* normal_vector);

  double parametric_distance(const std::array<double, 3>& x);

  const double elemThickness_;

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
  static const int nDim_ = AlgTraits::nDim_;
  static const int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static const int numIntPoints_ = AlgTraits::numScsIp_;
  static constexpr double scaleToStandardIsoFac_ = 2.0;

  // define ip node mappings; ordinal size = 1
  const int ipNodeMap_[nodesPerElement_] = {0, 1, 2, 3};

  // standard integration location
  const double intgLoc_[8] = {-0.25, -0.25, // surf 1
                              0.25,  -0.25, // surf 2
                              0.25,  0.25,  // surf 3
                              -0.25, 0.25}; // surf 4

  // shifted
  const double intgLocShift_[8] = {-0.50, -0.50, // surf 1
                                   0.50,  -0.50, // surf 2
                                   0.50,  0.50,  // surf 3
                                   -0.50, 0.50}; // surf 4

  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scs(
    const SharedMemView<DBLTYPE**, SHMEM>& coords,
    SharedMemView<DBLTYPE**, SHMEM>& areav) const;
};

} // namespace nalu
} // namespace sierra

#endif
