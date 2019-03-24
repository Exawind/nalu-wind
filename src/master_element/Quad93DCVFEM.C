/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <master_element/Quad93DCVFEM.h>

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/TensorOps.h>

#include <element_promotion/QuadratureRule.h>

#include <FORTRAN_Proto.h>

#include <cmath>
#include <iostream>

namespace sierra{
namespace nalu{

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Quad93DSCS::Quad93DSCS()
  : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::numIntPoints_ = numIntPoints_;
  MasterElement::nodesPerElement_ = nodesPerElement_;

  

  // set up integration rule and relevant maps on scs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  eval_shape_functions_at_ips();
  eval_shape_derivs_at_ips();
  eval_shape_functions_at_shifted_ips();
  eval_shape_derivs_at_shifted_ips();
}

//--------------------------------------------------------------------------
//-------- set_interior_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::set_interior_info()
{
  const int nodeMap[9] = {
                               0, 4, 1,   // bottom row of nodes
                               7, 8, 5,   // middle row of nodes
                               3, 6, 2    // top row of nodes
                             };

  auto tensor_map_2D = [=] (int i, int j) { return nodeMap[i+nodes1D_*j]; };

  //1D integration rule per sub-control volume

   // define ip node mappings

   // tensor product nodes (3x3) x tensor product quadrature (2x2)
   int vector_index_2D = 0; int scalar_index = 0;
   for (int l = 0; l < nodes1D_; ++l) {
     for (int k = 0; k < nodes1D_; ++k) {
       for (int j = 0; j < numQuad_; ++j) {
         for (int i = 0; i < numQuad_; ++i) {
           //integration point location
           intgLoc_[vector_index_2D]     = gauss_point_location(k,i);
           intgLoc_[vector_index_2D + 1] = gauss_point_location(l,j);

           intgLocShift_[vector_index_2D]     = shifted_gauss_point_location(k,i);
           intgLocShift_[vector_index_2D + 1] = shifted_gauss_point_location(l,j);

           //weight
           ipWeight_[scalar_index] = tensor_product_weight(k,l,i,j);

           //sub-control volume association
           ipNodeMap_[scalar_index] = tensor_map_2D(k,l);

           // increment indices
           ++scalar_index;
           vector_index_2D += surfaceDimension_;
         }
       }
     }
   }
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::eval_shape_functions_at_ips()
{
  quad9_shape_fcn(numIntPoints_, intgLoc_, shapeFunctions_);
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::eval_shape_derivs_at_ips()
{
  quad9_shape_deriv(numIntPoints_, intgLoc_, shapeDerivs_);
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::eval_shape_functions_at_shifted_ips()
{
  quad9_shape_fcn(numIntPoints_, intgLocShift_, shapeFunctionsShift_);
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::eval_shape_derivs_at_shifted_ips()
{
  quad9_shape_deriv(numIntPoints_, intgLocShift_, shapeDerivsShift_);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::shape_fcn(double* shpfc)
{
  for (int ip = 0; ip < numIntPoints_ * nodesPerElement_; ++ip) {
    shpfc[ip] = shapeFunctions_[ip];
  }
}
//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::shifted_shape_fcn(double* shpfc)
{
  for (int ip = 0; ip < numIntPoints_ * nodesPerElement_; ++ip) {
    shpfc[ip] = shapeFunctionsShift_[ip];
  }
}

//--------------------------------------------------------------------------
//-------- gauss_point_location --------------------------------------------
//--------------------------------------------------------------------------
double
Quad93DSCS::gauss_point_location(
  int nodeOrdinal,
  int gaussPointOrdinal) const
{
  return isoparametric_mapping( scsEndLoc_[nodeOrdinal+1],
    scsEndLoc_[nodeOrdinal],
    gaussAbscissae_[gaussPointOrdinal] );
}

//--------------------------------------------------------------------------
//-------- gauss_point_location --------------------------------------------
//--------------------------------------------------------------------------
double
Quad93DSCS::shifted_gauss_point_location(
  int nodeOrdinal,
  int gaussPointOrdinal) const
{
  return gaussAbscissaeShift_[nodeOrdinal*numQuad_ + gaussPointOrdinal];
}

//--------------------------------------------------------------------------
//-------- tensor_product_weight -------------------------------------------
//--------------------------------------------------------------------------
double
Quad93DSCS::tensor_product_weight(
  int s1Node, int s2Node,
  int s1Ip, int s2Ip) const
{
  // surface integration
  const double Ls1 = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
  const double Ls2 = scsEndLoc_[s2Node+1]-scsEndLoc_[s2Node];
  const double isoparametricArea = Ls1 * Ls2; 
  const double weight = isoparametricArea * gaussWeight_[s1Ip] * gaussWeight_[s2Ip];
  return weight;
}

//--------------------------------------------------------------------------
//-------- quad9_shape_fcn -------------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::quad9_shape_fcn(
  int  numIntPoints,
  const double *intgLoc,
  double *shpfc) const
{
  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    int nineIp = 9*ip; // nodes per element is always 9
    int k = 2*ip;
    const double s = intgLoc[k];
    const double t = intgLoc[k+1];

    const double one_m_s = 1.0 - s;
    const double one_p_s = 1.0 + s;
    const double one_m_t = 1.0 - t;
    const double one_p_t = 1.0 + t;

    const double one_m_ss = 1.0 - s * s;
    const double one_m_tt = 1.0 - t * t;

    shpfc[nineIp  ] =  0.25 * s * t *  one_m_s *  one_m_t;
    shpfc[nineIp+1] = -0.25 * s * t *  one_p_s *  one_m_t;
    shpfc[nineIp+2] =  0.25 * s * t *  one_p_s *  one_p_t;
    shpfc[nineIp+3] = -0.25 * s * t *  one_m_s *  one_p_t;
    shpfc[nineIp+4] = -0.50 *     t *  one_p_s *  one_m_s * one_m_t;
    shpfc[nineIp+5] =  0.50 * s     *  one_p_t *  one_m_t * one_p_s;
    shpfc[nineIp+6] =  0.50 *     t *  one_p_s *  one_m_s * one_p_t;
    shpfc[nineIp+7] = -0.50 * s     *  one_p_t *  one_m_t * one_m_s;
    shpfc[nineIp+8] =  one_m_ss * one_m_tt;
  }
}


//--------------------------------------------------------------------------
//-------- quad9_shape_deriv -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::quad9_shape_deriv(
  int numIntPoints,
  const double *intgLoc,
  double *deriv) const
{
  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    const int grad_offset = surfaceDimension_ * nodesPerElement_ * ip; // nodes per element is always 9
    const int vector_offset = surfaceDimension_ * ip;
    int node; int offset;

    const double s = intgLoc[vector_offset+0];
    const double t = intgLoc[vector_offset+1];

    const double s2 = s*s;
    const double t2 = t*t;

    node = 0;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 - 2.0 * s * t - t2 + t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t - 2.0 * s * t - s2 + s);

    node = 1;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 - 2.0 * s * t + t2 - t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t + 2.0 * s * t - s2 - s);

    node = 2;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 + 2.0 * s * t + t2 + t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t + 2.0 * s * t + s2 + s);

    node = 3;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 + 2.0 * s * t - t2 - t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t - 2.0 * s * t + s2 - s);

    node = 4;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 - 2.0 * s * t);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t - s2 - 2.0 * t + 1.0);

    node = 5;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 + t2 - 2.0 * s - 1.0);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t + 2.0 * s * t);

    node = 6;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 + 2.0 * s * t);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t + s2 - 2.0 * t - 1.0);

    node = 7;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 - t2 - 2.0 * s + 1.0);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t - 2.0 * s * t);

    node = 8;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 2.0 * s * t2 - 2.0 * s;
    deriv[offset+1] = 2.0 * s2 * t - 2.0 * t;
  }
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Quad93DSCS::ipNodeMap(
  int /*ordinal*/) const
{
  // define ip->node mappings for each face (single ordinal);
  return ipNodeMap_;
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double * /* error */)
{
  std::array<double,3> areaVector;

  for (int k = 0; k < nelem; ++k) {
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;
    const int vector_elem_offset = nDim_ * numIntPoints_ * k;

    for (int ip = 0; ip < numIntPoints_; ++ip) {
      const int grad_offset = surfaceDimension_ * nodesPerElement_ * ip;
      const int offset = nDim_ * ip + vector_elem_offset;

      //compute area vector for this ip
      area_vector( &coords[coord_elem_offset],
                   &shapeDerivs_[grad_offset],
                   areaVector.data() );

      // apply quadrature weight and orientation (combined as weight)
      for (int j = 0; j < nDim_; ++j) {
        areav[offset+j]  = ipWeight_[ip] * areaVector[j];
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
Quad93DSCS::isInElement(
  const double *elemNodalCoord,
  const double *pointCoord,
  double *isoParCoord )
{
  const double isInElemConverged = 1.0e-16;
  // Translate element so that (x,y,z) coordinates of the first node are (0,0,0)

  double x[3] = { elemNodalCoord[1] - elemNodalCoord[0],
                  elemNodalCoord[2] - elemNodalCoord[0],
                  elemNodalCoord[3] - elemNodalCoord[0] };

  double y[3] = { elemNodalCoord[10] - elemNodalCoord[9],
                  elemNodalCoord[11] - elemNodalCoord[9],
                  elemNodalCoord[12] - elemNodalCoord[9] };

  double z[3] = { elemNodalCoord[19] - elemNodalCoord[18],
                  elemNodalCoord[20] - elemNodalCoord[18],
                  elemNodalCoord[21] - elemNodalCoord[18] };

  // (xp,yp,zp) is the point at which we're searching for (xi,eta,d)
  // (must translate this also)
  // d = (scaled) distance in (x,y,z) space from point (xp,yp,zp) to the
  //     surface defined by the face element (the distance is scaled by
  //     the length of the non-unit normal vector; rescaling of d is done
  //     following the NR iteration below).

  double xp = pointCoord[0] - elemNodalCoord[0];
  double yp = pointCoord[1] - elemNodalCoord[9];
  double zp = pointCoord[2] - elemNodalCoord[18];

  // Newton-Raphson iteration for (xi,eta,d)

  double jdet;
  double j[9];
  double gn[3];
  double xcur[3];          // current (x,y,z) point on element surface
  double normal[3];        // (non-unit) normal computed at xcur

  // Solution vector solcur[3] = {xi,eta,d}
  double solcur[3] = {-0.5,-0.5,-0.5};     // initial guess
  double deltasol[] = {1.0,1.0, 1.0};

  unsigned i = 0;
  const unsigned MAX_NR_ITER = 100;

  do
  {
    // Update guess vector
    solcur[0] += deltasol[0];
    solcur[1] += deltasol[1];
    solcur[2] += deltasol[2];

    interpolatePoint(3,solcur,elemNodalCoord,xcur);

    // Translate xcur ((x,y,z) point corresponding
    // to current (xi,eta) guess)

    xcur[0] -= elemNodalCoord[0];
    xcur[1] -= elemNodalCoord[9];
    xcur[2] -= elemNodalCoord[18];

    non_unit_face_normal(solcur,elemNodalCoord,normal);

    gn[0] = xcur[0] - xp + solcur[2] * normal[0];
    gn[1] = xcur[1] - yp + solcur[2] * normal[1];
    gn[2] = xcur[2] - zp + solcur[2] * normal[2];

    // Mathematica-generated code for the jacobian

    j[0]=0.125*(-2.*(-1.+solcur[1])*x[0]+(2.*(1.+solcur[1])*(x[1]-x[2])+solcur[2]*(-(y[1]*z[0])+y[2]*z[0]+y[0]*z[1]-y[0]*z[2])));

    j[1]=0.125*(-2.*(1.+solcur[0])*x[0]+2.*(1.+solcur[0])*x[1]-2.*(-1.+solcur[0])*x[2]+(solcur[2]*(y[2]*(z[0]-z[1])+(-y[0]+y[1])*z[2])));

    j[2]= normal[0];

    j[3]=0.125*(-2.*(-1.+solcur[1])*y[0]+(2.*(1.+solcur[1])*(y[1]-y[2])+solcur[2]*(x[1]*z[0]-x[2]*z[0]-x[0]*z[1]+x[0]*z[2])));

    j[4]=0.125*(-2.*(1.+solcur[0])*y[0]+2.*(1.+solcur[0])*y[1]-2.*(-1.+solcur[0])*y[2]+(solcur[2]*(x[2]*(-z[0]+z[1])+(x[0]-x[1])*z[2])));

    j[5]= normal[1];

    j[6]=0.125*((solcur[2]*(-(x[1]*y[0])+x[2]*y[0]+x[0]*y[1]-x[0]*y[2]))-2.*((-1.+solcur[1])*z[0]-(1.+solcur[1])*(z[1]-z[2])));

    j[7]=0.125*((solcur[2]*(x[2]*(y[0]-y[1])+(-x[0]+x[1])*y[2]))-2.*(1.+solcur[0])*z[0]+2.*(1.+solcur[0])*z[1]-2.*(-1.+solcur[0])*z[2]);

    j[8]= normal[2];

    jdet=-(j[2]*j[4]*j[6])+j[1]*j[5]*j[6]+j[2]*j[3]*j[7]-
     j[0]*j[5]*j[7]-j[1]*j[3]*j[8]+j[0]*j[4]*j[8];

    // Solve linear system (j*deltasol = -gn) for deltasol at step n+1

    deltasol[0] = (gn[2]*(j[2]*j[4]-j[1]*j[5])+gn[1]*(-(j[2]*j[7])+
                                                      j[1]*j[8])+gn[0]*(j[5]*j[7]-j[4]*j[8]))/jdet;
    deltasol[1] = (gn[2]*(-(j[2]*j[3])+j[0]*j[5])+gn[1]*(j[2]*j[6]-
                                                         j[0]*j[8])+gn[0]*(-(j[5]*j[6])+j[3]*j[8]))/jdet;
    deltasol[2] = (gn[2]*(j[1]*j[3]-j[0]*j[4])+gn[1]*(-(j[1]*j[6])+
                                                      j[0]*j[7])+gn[0]*(j[4]*j[6]-j[3]*j[7]))/jdet;

  } while ( !within_tolerance( vector_norm_sq(deltasol,3), isInElemConverged) &&
      ++i < MAX_NR_ITER );

  // Fill in solution vector; only include the distance (in the third
  // solution vector slot) if npar_coord = 3 (this is how the user
  // requests it)

  isoParCoord[0] = isoParCoord[1] = isoParCoord[2] = std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if ( i < MAX_NR_ITER ) {
    isoParCoord[0] = solcur[0] + deltasol[0];
    isoParCoord[1] = solcur[1] + deltasol[1];
    // Rescale the distance vector by the length of the (non-unit) normal vector,
    // which was used above in the NR iteration.
    const double area   = std::sqrt(vector_norm_sq(normal,3));
    const double length = std::sqrt(area);
    const double isoParCoord_2  = (solcur[2] + deltasol[2]) * length;
    isoParCoord[2] = isoParCoord_2;

    const std::array<double,3> xtmp = {{ isoParCoord[0], isoParCoord[1], isoParCoord_2 }};
    dist = parametric_distance(xtmp);
  }
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result )
{
  double one = 1.0;
  double half   = one / 2.0;
  double qtr    = one / 4.0;

  double s = isoParCoord[0];
  double t = isoParCoord[1];

  double one_m_s = one - s;
  double one_p_s = one + s;
  double one_m_t = one - t;
  double one_p_t = one + t;

  double one_m_ss = one - s * s;
  double one_m_tt = one - t * t;

  for ( int i = 0; i < nComp; i++ ) {
    int b = 9*i;       // Base 'field array' index for ith component

    result[i] =   qtr * s * t *  one_m_s * one_m_t * field[b+ 0]+
      -qtr * s * t *  one_p_s *  one_m_t * field[b+ 1]+
      qtr * s * t *  one_p_s *  one_p_t * field[b+ 2]+
      -qtr * s * t *  one_m_s *  one_p_t * field[b+ 3]+
      -half * t * one_p_s * one_m_s * one_m_t * field[b+ 4]+
      half * s * one_p_t * one_m_t * one_p_s * field[b+ 5]+
      half * t * one_p_s * one_m_s * one_p_t * field[b+ 6]+
      -half * s * one_p_t * one_m_t * one_m_s * field[b+ 7]+
      one_m_ss * one_m_tt * field[b+ 8];
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::general_shape_fcn(
  const int numIp,
  const double *isoParCoord,
  double *shpfc)
{
  quad9_shape_fcn(numIp, isoParCoord, shpfc);
}

//--------------------------------------------------------------------------
//-------- general_normal --------------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::general_normal(
  const double *isoParCoord,
  const double *coords,
  double *normal)
{
  // coords(3,9)
  const int nDim = 3;

  const double s = isoParCoord[0];
  const double t = isoParCoord[1];

  const double t2 = t*t;
  const double s2 = s*s;

  const double psi0Xi  =  0.25 * (2.0 * s  * t2 - 2.0*s*t-t2+t);
  const double psi1Xi  =  0.25 * (2.0 * s  * t2 - 2.0*s*t+t2-t);
  const double psi2Xi  =  0.25 * (2.0 * s  * t2 + 2.0*s*t+t2+t);
  const double psi3Xi  =  0.25 * (2.0 * s  * t2 + 2.0*s*t-t2-t);
  const double psi4Xi  =  -0.5 * (2.0 * s  * t2 - 2.0*s*t);
  const double psi5Xi  =  -0.5 * (2.0 * s  * t2 + t2 - 2.0*s-1.0);
  const double psi6Xi  =  -0.5 * (2.0 * s  * t2 + 2.0*s*t);
  const double psi7Xi  =  -0.5 * (2.0 * s  * t2 - t2 - 2.0*s+1.0);
  const double psi8Xi  =          2.0 * s  * t2      - 2.0*s;

  const double psi0Eta = 0.25 * (2.0 * s2 * t  - 2.0*s*t-s2+s);
  const double psi1Eta = 0.25 * (2.0 * s2 * t  + 2.0*s*t-s2-s);
  const double psi2Eta = 0.25 * (2.0 * s2 * t  + 2.0*s*t+s2+s);
  const double psi3Eta = 0.25 * (2.0 * s2 * t  - 2.0*s*t+s2-s);
  const double psi4Eta = -0.5 * (2.0 * s2 * t  - s2 - 2.0*t+1.0);
  const double psi5Eta = -0.5 * (2.0 * s2 * t  + 2.0*s*t);
  const double psi6Eta = -0.5 * (2.0 * s2 * t  + s2 - 2.0*t-1.0);
  const double psi7Eta = -0.5 * (2.0 * s2 * t  - 2.0*s*t);
  const double psi8Eta =         2.0 * s2 * t       - 2.0*t;

  const double DxDxi = coords[0*nDim+0]*psi0Xi +
    coords[1*nDim+0]*psi1Xi +
    coords[2*nDim+0]*psi2Xi +
    coords[3*nDim+0]*psi3Xi +
    coords[4*nDim+0]*psi4Xi +
    coords[5*nDim+0]*psi5Xi +
    coords[6*nDim+0]*psi6Xi +
    coords[7*nDim+0]*psi7Xi +
    coords[8*nDim+0]*psi8Xi;

  const double DyDxi = coords[0*nDim+1]*psi0Xi +
    coords[1*nDim+1]*psi1Xi +
    coords[2*nDim+1]*psi2Xi +
    coords[3*nDim+1]*psi3Xi +
    coords[4*nDim+1]*psi4Xi +
    coords[5*nDim+1]*psi5Xi +
    coords[6*nDim+1]*psi6Xi +
    coords[7*nDim+1]*psi7Xi +
    coords[8*nDim+1]*psi8Xi;

  const double DzDxi = coords[0*nDim+2]*psi0Xi +
    coords[1*nDim+2]*psi1Xi +
    coords[2*nDim+2]*psi2Xi +
    coords[3*nDim+2]*psi3Xi +
    coords[4*nDim+2]*psi4Xi +
    coords[5*nDim+2]*psi5Xi +
    coords[6*nDim+2]*psi6Xi +
    coords[7*nDim+2]*psi7Xi +
    coords[8*nDim+2]*psi8Xi;

  const double DxDeta = coords[0*nDim+0]*psi0Eta +
    coords[1*nDim+0]*psi1Eta +
    coords[2*nDim+0]*psi2Eta +
    coords[3*nDim+0]*psi3Eta +
    coords[4*nDim+0]*psi4Eta +
    coords[5*nDim+0]*psi5Eta +
    coords[6*nDim+0]*psi6Eta +
    coords[7*nDim+0]*psi7Eta +
    coords[8*nDim+0]*psi8Eta;

  const double DyDeta = coords[0*nDim+1]*psi0Eta +
    coords[1*nDim+1]*psi1Eta +
    coords[2*nDim+1]*psi2Eta +
    coords[3*nDim+1]*psi3Eta +
    coords[4*nDim+1]*psi4Eta +
    coords[5*nDim+1]*psi5Eta +
    coords[6*nDim+1]*psi6Eta +
    coords[7*nDim+1]*psi7Eta +
    coords[8*nDim+1]*psi8Eta;

  const double DzDeta = coords[0*nDim+2]*psi0Eta +
    coords[1*nDim+2]*psi1Eta +
    coords[2*nDim+2]*psi2Eta +
    coords[3*nDim+2]*psi3Eta +
    coords[4*nDim+2]*psi4Eta +
    coords[5*nDim+2]*psi5Eta +
    coords[6*nDim+2]*psi6Eta +
    coords[7*nDim+2]*psi7Eta +
    coords[8*nDim+2]*psi8Eta;

  const double detXY =  DxDxi*DyDeta - DxDeta*DyDxi;
  const double detYZ =  DyDxi*DzDeta - DyDeta*DzDxi;
  const double detXZ = -DxDxi*DzDeta + DxDeta*DzDxi;

  const double det = std::sqrt( detXY*detXY + detYZ*detYZ + detXZ*detXZ );

  normal[0] = detYZ / det;
  normal[1] = detXZ / det;
  normal[2] = detXY / det;
}

//--------------------------------------------------------------------------
//-------- non_unit_face_normal --------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::non_unit_face_normal(
  const double * isoParCoord,
  const double * elemNodalCoord,
  double * normalVector )
{
  double xi  = isoParCoord[0];
  double eta = isoParCoord[1];

  // Translate element so that node 0 is at (x,y,z) = (0,0,0)

  double x[3] = { elemNodalCoord[1] - elemNodalCoord[0],
                  elemNodalCoord[2] - elemNodalCoord[0],
                  elemNodalCoord[3] - elemNodalCoord[0] };

  double y[3] = { elemNodalCoord[10] - elemNodalCoord[9],
                  elemNodalCoord[11] - elemNodalCoord[9],
                  elemNodalCoord[12] - elemNodalCoord[9] };

  double z[3] = { elemNodalCoord[19] - elemNodalCoord[18],
                  elemNodalCoord[20] - elemNodalCoord[18],
                  elemNodalCoord[21] - elemNodalCoord[18] };

  // Mathematica-generated and simplified code for the normal vector

  const double n0 = 0.125*(xi*y[2]*z[0]+y[0]*z[1]+xi*y[0]*z[1]-y[2]*z[1]-
                                       xi*y[0]*z[2]+y[1]*(-((1.+xi)*z[0])+
                                                          (1.+eta)*z[2])+eta*(y[2]*z[0]-y[2]*z[1]-y[0]*z[2]));

  const double n1 = 0.125*(-(xi*x[2]*z[0])-x[0]*z[1]-xi*x[0]*z[1]+x[2]*z[1]+
                                       xi*x[0]*z[2]+x[1]*((1.+xi)*z[0]-
                                                          (1.+eta)*z[2])+eta*(-(x[2]*z[0])+x[2]*z[1]+x[0]*z[2]));

  const double n2 = 0.125*(xi*x[2]*y[0]+x[0]*y[1]+xi*x[0]*y[1]-x[2]*y[1]-
                                       xi*x[0]*y[2]+x[1]*(-((1.+xi)*y[0])+
                                                          (1.+eta)*y[2])+eta*(x[2]*y[0]-x[2]*y[1]-x[0]*y[2]));

  normalVector[0] = n0;
  normalVector[1] = n1;
  normalVector[2] = n2;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double Quad93DSCS::parametric_distance(const std::array<double,3> &x)
{
  const double ELEM_THICK  = 0.01;
  const double y[3] = { std::fabs(x[0]), std::fabs(x[1]), std::fabs(x[2]) };
  double d = y[0];
  if (d < y[1]) d = y[1];
  if (ELEM_THICK < y[2] && d < 1+y[2]) d = 1+y[2];
  return d;
}

//--------------------------------------------------------------------------
//-------- area_vector -----------------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::area_vector(
  const double *POINTER_RESTRICT elemNodalCoords,
  const double *POINTER_RESTRICT shapeDeriv,
  double *POINTER_RESTRICT areaVector) const
{
   // return the normal area vector given shape derivatives dnds OR dndt
   double dx_ds1 = 0.0; double dy_ds1 = 0.0; double dz_ds1 = 0.0;
   double dx_ds2 = 0.0; double dy_ds2 = 0.0; double dz_ds2 = 0.0;

   constexpr int nNodes2D = 9;
   for (int node = 0; node < nNodes2D; ++node) {
     const int vector_offset = nDim_ * node;
     const int surface_vector_offset = surfaceDimension_ * node;

     const double xCoord = elemNodalCoords[vector_offset+0];
     const double yCoord = elemNodalCoords[vector_offset+1];
     const double zCoord = elemNodalCoords[vector_offset+2];

     const double dn_ds1 = shapeDeriv[surface_vector_offset+0];
     const double dn_ds2 = shapeDeriv[surface_vector_offset+1];

     dx_ds1 += dn_ds1 * xCoord;
     dx_ds2 += dn_ds2 * xCoord;

     dy_ds1 += dn_ds1 * yCoord;
     dy_ds2 += dn_ds2 * yCoord;

     dz_ds1 += dn_ds1 * zCoord;
     dz_ds2 += dn_ds2 * zCoord;
   }

   //cross product
   areaVector[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
   areaVector[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
   areaVector[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}
}
}
