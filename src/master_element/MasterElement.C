/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>

#include <master_element/MasterElementHO.h>
#include <master_element/MasterElementUtils.h>

#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/QuadratureRule.h>
#include <AlgTraits.h>

#include <NaluEnv.h>
#include <FORTRAN_Proto.h>

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <iostream>

#include <cmath>
#include <limits>
#include <array>
#include <map>
#include <memory>

namespace sierra{
namespace nalu{

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MasterElement::MasterElement()
  : nDim_(0),
    nodesPerElement_(0),
    numIntPoints_(0),
    scaleToStandardIsoFac_(1.0)
{
  // nothing else
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MasterElement::~MasterElement()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- isoparametric_mapping -------------------------------------------
//--------------------------------------------------------------------------
double
MasterElement::isoparametric_mapping( 
  const double b,
  const double a,
  const double xi) const
{
  return xi*(b-a)/2.0 +(a+b)/2.0;
}

//--------------------------------------------------------------------------
//-------- within_tolerance ------------------------------------------------
//--------------------------------------------------------------------------
bool 
MasterElement::within_tolerance( const double & val, const double & tol )
{
  return (std::abs(val)<tol);
}

//--------------------------------------------------------------------------
//-------- vector_norm_sq --------------------------------------------------
//--------------------------------------------------------------------------
double 
MasterElement::vector_norm_sq( const double * vect, int len )
{
  double norm_sq = 0.0;
  for (int i=0; i<len; i++) {
    norm_sq += vect[i]*vect[i];
  }
  return norm_sq;
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Edge2DSCS::Edge2DSCS()
  : MasterElement(),
    elemThickness_(0.01)
{
  nDim_ = 2;
  nodesPerElement_ = 2;
  numIntPoints_ = 2;
  scaleToStandardIsoFac_ = 2.0;

  // define ip node mappings; ordinal size = 1
  ipNodeMap_.resize(2);
  ipNodeMap_[0] = 0;
  ipNodeMap_[1] = 1;

  intgLoc_.resize(2);
  intgLoc_[0]  =  -0.25; intgLoc_[1]  = 0.25;
 
  intgLocShift_.resize(2);
  intgLocShift_[0]  =  -0.50; intgLocShift_[1]  = 0.50; 
  
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Edge2DSCS::~Edge2DSCS()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Edge2DSCS::ipNodeMap(
  int /*ordinal*/)
{
  // define ip->node mappings for each face (single ordinal); 
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void Edge2DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(edge2d_scs_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_,
      coords, areav );

  // fake check
  *error = (lerr == 0) ? 0.0 : 1.0;
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Edge2DSCS::shape_fcn(double *shpfc)
{
  for ( int i =0; i < nodesPerElement_; ++i ) {
    int j = 2*i;
    shpfc[j  ] = 0.5-intgLoc_[i];
    shpfc[j+1] = 0.5+intgLoc_[i];
  }
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Edge2DSCS::shifted_shape_fcn(double *shpfc)
{
  for ( int i =0; i< nodesPerElement_; ++i ) {
    int j = 2*i;
    shpfc[j  ] = 0.5-intgLocShift_[i];
    shpfc[j+1] = 0.5+intgLocShift_[i];
  }
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
Edge2DSCS::isInElement(
    const double * elem_nodal_coor,     // (2,2)
    const double * point_coor,          // (2)
	  double * par_coor ) 
{
  // elem_nodal_coor has the endpoints of the line
  // segment defining this element.  Set the first
  // endpoint to zero.  This means subtrace the
  // first endpoint from the second.
  const double X1 = elem_nodal_coor[1]-elem_nodal_coor[0];
  const double X2 = elem_nodal_coor[3]-elem_nodal_coor[2];

  // Now subtract the first endpoint from the target point
  const double P1 = point_coor[0] - elem_nodal_coor[0];
  const double P2 = point_coor[1] - elem_nodal_coor[2];

  // Now find the projection along the line of the point
  // This is the parametric coordinate in range (0,1)
  const double norm2 = X1*X1 + X2*X2;
  
  const double xi = (P1*X1 + P2*X2) / norm2;
  // rescale to (-1,1)
  par_coor[0] = 2*xi - 1;

  // Now find the projection from the point to a perpenducular
  // line.  This gives the distance from the point to the element.
  const double alpha = std::abs(P1*X2 - P2*X1) / norm2;
  if (2 == nDim_) 
    par_coor[1] = alpha;

  std::vector<double> x(2);
  x[0] = par_coor[0];
  x[1] = alpha;
  const double dist = parametric_distance(x);

  return dist;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
Edge2DSCS::parametric_distance(const std::vector<double> &x)
{
  double dist = std::fabs(x[0]);
  if (elemThickness_ < x[1] && dist < 1.0+x[1]) 
    dist = 1+x[1];
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
Edge2DSCS::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result )
{
  double xi = isoParCoord[0]; 
  for ( int i = 0; i < nComp; i++ ) {
    // Base 'field array' index for ith component
    int b = 2*i;
    result[i] = 0.5*(1.0-xi) * field[b+0] +
      0.5*(1.0+xi) * field[b+1];
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Edge2DSCS::general_shape_fcn(
  const int numIp,
  const double *isoParCoord,
  double *shpfc)
{
  const double npe = nodesPerElement_;
  for ( int ip = 0; ip < numIp; ++ip ) {
    int j = npe*ip;
    shpfc[j  ] = 0.5*(1.0-isoParCoord[ip]);
    shpfc[j+1] = 0.5*(1.0+isoParCoord[ip]);
  }
}

//--------------------------------------------------------------------------
//-------- general_normal --------------------------------------------------
//--------------------------------------------------------------------------
void
Edge2DSCS::general_normal(
  const double */*isoParCoord*/,
  const double *coords,
  double *normal)
{
  // can be only linear
  const double dx  = coords[2] - coords[0];
  const double dy  = coords[3] - coords[1];
  const double mag = std::sqrt(dx*dx + dy*dy);

  normal[0] =  dy/mag;
  normal[1] = -dx/mag;
}
} // namespace nalu
} // namespace sierra
