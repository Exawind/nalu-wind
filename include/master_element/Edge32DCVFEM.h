/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Edge32DCVFEM_h
#define Edge32DCVFEM_h

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

namespace stk {
  struct topology;
}

namespace sierra{
namespace nalu{

// edge 2d
class Edge32DSCS : public MasterElement
{
public:

  using AlgTraits = AlgTraitsEdge3_2D;

  Edge32DSCS();
  virtual ~Edge32DSCS() {}
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  const int * ipNodeMap(int ordinal = 0);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result);

private:
  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScsIp_;
  static constexpr int numQuad_ = 2;


  const double scsDist_ = std::sqrt(3.0)/3.0;
  const double scsEndLoc_[4] = { -1.0, -scsDist_, scsDist_, +1.0};

  const double gaussAbscissaeShift_[numIntPoints_] = {-1.0, -1.0, 0.0, 0.0, +1.0, +1.0};

  const double intgLocShift_[numIntPoints_] = {
    -1.00,  -1.00,
     0.00,   0.00,
     1.00,   1.00};

  int ipNodeMap_[numIntPoints_];
  double intgLoc_[numIntPoints_];
  double ipWeight_[numIntPoints_];
  double gaussWeight_[numIntPoints_];


  void set_quadrature_rule();
  double tensor_product_weight(const int s1Node, const int s1Ip) const;
  double gauss_point_location(const int nodeOrdinal, const int gaussPointOrdinal) const;

  void area_vector(
    const double *POINTER_RESTRICT coords,
    const double s,
    double *POINTER_RESTRICT areaVector) const;

};

} // namespace nalu
} // namespace Sierra

#endif
