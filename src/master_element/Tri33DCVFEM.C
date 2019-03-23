/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <master_element/Tri33DCVFEM.h>

#include <AlgTraits.h>

#include <NaluEnv.h>
#include <FORTRAN_Proto.h>

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
KOKKOS_FUNCTION
Tri3DSCS::Tri3DSCS()
  : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;

  // shifted
  MasterElement::intgLocShift_.assign(intgLocShift_, 6+intgLocShift_);
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Tri3DSCS::ipNodeMap(
  int /*ordinal*/) const
{
  // define ip->node mappings for each face (single ordinal); 
  return ipNodeMap_;
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void Tri3DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  int lerr = 0;

  const int npe = nodesPerElement_;
  const int nint = numIntPoints_;
  SIERRA_FORTRAN(tri3d_scs_det)
    ( &nelem, &npe, &nint,
      coords, areav );

  // fake check
  *error = (lerr == 0) ? 0.0 : 1.0;
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Tri3DSCS::shape_fcn(double *shpfc)
{
  tri_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Tri3DSCS::shifted_shape_fcn(double *shpfc)
{
  tri_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- tri_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
void
Tri3DSCS::tri_shape_fcn(
  const int     npts,
  const double *isoParCoord,
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int threej = 3*j;
    const int k = 2*j;
    const double xi = isoParCoord[k];
    const double eta = isoParCoord[k+1];
    shape_fcn[    threej] = 1.0 - xi - eta;
    shape_fcn[1 + threej] = xi;
    shape_fcn[2 + threej] = eta;
  }
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
Tri3DSCS::isInElement(
    const double * elem_nodal_coor,
    const double * point_coor,
	  double * par_coor ) 
{
  // always intended for 3D...
  const int npar_coord = 3;
  // Translate element so that (x,y,z) coordinates of the
  // first node
  double x[2] = { elem_nodal_coor[1] - elem_nodal_coor[0],
                  elem_nodal_coor[2] - elem_nodal_coor[0] };
  double y[2] = { elem_nodal_coor[4] - elem_nodal_coor[3],
                  elem_nodal_coor[5] - elem_nodal_coor[3] };
  double z[2] = { elem_nodal_coor[7] - elem_nodal_coor[6],
                  elem_nodal_coor[8] - elem_nodal_coor[6] };

  // Translate position of point in same manner

  double xp = point_coor[0] - elem_nodal_coor[0];
  double yp = point_coor[1] - elem_nodal_coor[3];
  double zp = point_coor[2] - elem_nodal_coor[6];

  // Set new nodal coordinates with Node 1 at origin and with new
  // x and y axes lying in the plane of the element
  double len12 = std::sqrt( x[0]*x[0] + y[0]*y[0] + z[0] *z[0] );
  double len13 = std::sqrt( x[1]*x[1] + y[1]*y[1] + z[1] *z[1] );

  double xnew[3];
  double ynew[3];
  double znew[3];

  // Use cross-product of 12 and 13 to find enclosed angle and
  // direction of new z-axis

  znew[0] = y[0]*z[1] - y[1]*z[0];
  znew[1] = x[1]*z[0] - x[0]*z[1];
  znew[2] = x[0]*y[1] - x[1]*y[0];

  double Area2 = std::sqrt( znew[0]*znew[0] + znew[1]*znew[1] +
                            znew[2]*znew[2] );

  // find sin of angle
  double sin_theta = Area2 / ( len12 * len13 ) ;

  // find cosine of angle
  double cos_theta = (x[0]*x[1] + y[0]*y[1] + z[0]*z[1])/(len12 * len13);

  // nodal coordinates of nodes 2 and 3 in new system
  // (coordinates of node 1 are identically 0.0)
  double x_nod_new[2] = { len12, len13*cos_theta};
  double y_nod_new[2] = {  0.0, len13*sin_theta};

  // find direction cosines transform position of
  // point to be checked into new coordinate system

  // direction cosines of new x axis along side 12

  xnew[0] = x[0]/len12;
  xnew[1] = y[0]/len12;
  xnew[2] = z[0]/len12;

  // direction cosines of new z axis
  znew[0] = znew[0]/Area2;
  znew[1] = znew[1]/Area2;
  znew[2] = znew[2]/Area2;

  // direction cosines of new y-axis (cross-product of znew and xnew)
  ynew[0] = znew[1]*xnew[2] - xnew[1]*znew[2];
  ynew[1] = xnew[0]*znew[2] - znew[0]*xnew[2];
  ynew[2] = znew[0]*xnew[1] - xnew[0]*znew[1];

  // compute transformed coordinates of point
  // (coordinates in xnew,ynew,znew)
  double xpnew = xnew[0]*xp + xnew[1]*yp + xnew[2]*zp;
  double ypnew = ynew[0]*xp + ynew[1]*yp + ynew[2]*zp;
  double zpnew = znew[0]*xp + znew[1]*yp + znew[2]*zp;

  // Find parametric coordinates of point and check that
  // it lies in the element
  par_coor[0] = 1. - xpnew / x_nod_new[0] +
		 ypnew*( x_nod_new[1] - x_nod_new[0] ) / Area2;
  par_coor[1] = ( xpnew*y_nod_new[1] - ypnew*x_nod_new[1] ) / Area2;

  if (3 == npar_coord) par_coor[2] = zpnew/std::sqrt(Area2);

  std::array<double,3> w = { par_coor[0], par_coor[1], zpnew/std::sqrt(Area2) };

  par_coor[0] = w[1];
  par_coor[1] = 1.0-w[0]-w[1];

  const double dist = parametric_distance(w);

  return dist;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double 
Tri3DSCS::parametric_distance(
  const std::array<double,3> &x)
{
  const double ELEM_THICK = 0.01;
  const double X=x[0] - 1./3.;
  const double Y=x[1] - 1./3.;
  const double dist0 = -3*X;
  const double dist1 = -3*Y;
  const double dist2 =  3*(X+Y);
  double dist = std::max(std::max(dist0,dist1),dist2);
  const double y = std::fabs(x[2]);
  if (ELEM_THICK < y && dist < 1+y) dist = 1+y;
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
Tri3DSCS::interpolatePoint(
  const int  & ncomp_field,
  const double * isoParCoord,
  const double * field,
  double * result )
{
  const double r = isoParCoord[0];
  const double s = isoParCoord[1];
  const double t = 1.0 - r - s;

  for ( int i = 0; i < ncomp_field; i++ ) {
    int b = 3*i;  //Base 'field array' index for ith component
    result[i] = t*field[b] + r*field[b+1] + s*field[b+2];
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Tri3DSCS::general_shape_fcn(
  const int numIp,
  const double *isoParCoord,
  double *shpfc)
{
  tri_shape_fcn(numIp, isoParCoord, shpfc);
}


//--------------------------------------------------------------------------
//-------- general_normal --------------------------------------------------
//--------------------------------------------------------------------------
void
Tri3DSCS::general_normal(
  const double */*isoParCoord*/,
  const double *coords,
  double *normal)
{
  // can be only linear
  const double ax  = coords[3] - coords[0];
  const double ay  = coords[4] - coords[1];
  const double az  = coords[5] - coords[2];
  const double bx  = coords[6] - coords[0];
  const double by  = coords[7] - coords[1];
  const double bz  = coords[8] - coords[2];

  normal[0] = ( ay*bz - az*by );
  normal[1] = ( az*bx - ax*bz );
  normal[2] = ( ax*by - ay*bx );

  const double mag = std::sqrt( normal[0]*normal[0] +
                                normal[1]*normal[1] +
                                normal[2]*normal[2] );
  normal[0] /= mag;
  normal[1] /= mag;
  normal[2] /= mag;
}
} // namespace nalu
} // namespace sierra
