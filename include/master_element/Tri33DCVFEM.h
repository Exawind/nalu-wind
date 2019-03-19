/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


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


namespace sierra{
namespace nalu{

// 3D Tri 3
class Tri3DSCS : public MasterElement
{
public:

  KOKKOS_FUNCTION
  Tri3DSCS();
  KOKKOS_FUNCTION
  virtual ~Tri3DSCS() = default;

  using AlgTraits = AlgTraitsTri3;
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

   void tri_shape_fcn(
     const int npts,
     const double *par_coord,
     double* shape_fcn);

   double isInElement(
     const double *elemNodalCoord,
     const double *pointCoord,
     double *isoParCoord);

  double parametric_distance(
    const std::array<double,3> &x);

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
private:
  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScsIp_;

  // define ip node mappings; ordinal size = 1
  const int ipNodeMap_[3] = {0, 1, 2};

  // standard integration location
  static constexpr double seven36ths = 7.0/36.0;
  static constexpr double eleven18ths = 11.0/18.0;
  const double intgLoc_[6] = {
   seven36ths,   seven36ths,  // surf 1
   eleven18ths,  seven36ths,  // surf 2
   seven36ths,   eleven18ths};// surf 3

  // shifted
  const double intgLocShift_[6] = {
    0.00,   0.00, // surf 1
    1.00,   0.00, // surf 2
    0.00,   1.00};// surf 3
};

} // namespace nalu
} // namespace Sierra

#endif
