/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Edge22DCVFEM_h 
#define Edge22DCVFEM_h 

#include <master_element/MasterElement.h>

#include <AlgTraits.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include <vector>
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
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

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

  double parametric_distance(const std::vector<double> &x);

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
