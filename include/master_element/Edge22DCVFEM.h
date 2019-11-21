// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef Edge22DCVFEM_h 
#define Edge22DCVFEM_h 

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

namespace sierra{
namespace nalu{

// edge 2d
class Edge2DSCS : public MasterElement
{
public:
  KOKKOS_FUNCTION
  Edge2DSCS();
  KOKKOS_FUNCTION
  virtual ~Edge2DSCS() = default;
  using AlgTraits = AlgTraitsEdge_2D;
  using MasterElement::shape_fcn;
  using MasterElement::determinant;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  KOKKOS_FUNCTION virtual void determinant(
    SharedMemView<DoubleType**, DeviceShmem> &coords,
    SharedMemView<DoubleType**, DeviceShmem> &area);

  KOKKOS_FUNCTION virtual void shape_fcn(
     SharedMemView<DoubleType**, DeviceShmem> &shpfc);

  KOKKOS_FUNCTION virtual void shifted_shape_fcn(
    SharedMemView<DoubleType**, DeviceShmem> &shpfc);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  double isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord);
  
  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result);

  void general_shape_fcn(
    const int numIp,
    const double *isoParCoord,
    double *shpfc);

  void general_normal(
    const double *isoParCoord,
    const double *coords,
    double *normal);

  virtual const double* integration_locations() const final {
    return intgLoc_;
  }
  virtual const double* integration_location_shift() const final {
    return intgLocShift_;
  }
  double parametric_distance(const std::array<double,2> &x);

  const double elemThickness_;  

private :
  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScsIp_;
  static constexpr double scaleToStandardIsoFac_ = 2.0;
  const int ipNodeMap_[2] = {0,1};
  const double intgLoc_[2] = {-0.25, 0.25};
  const double intgLocShift_[2] = {-0.50, 0.50};

};

} // namespace nalu
} // namespace Sierra

#endif
