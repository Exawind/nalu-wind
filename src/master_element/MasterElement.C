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
//-------- constructor------------------------------------------------------
//--------------------------------------------------------------------------
QuadrilateralP2Element::QuadrilateralP2Element()
  : MasterElement(),
    scsDist_(std::sqrt(3.0)/3.0),
    nodes1D_(3),
    numQuad_(2)
{
  ndim(AlgTraits::nDim_);
  nodesPerElement_ = nodes1D_ * nodes1D_;

  // map the standard stk (refinement consistent) node numbering
  // to a tensor-product style node numbering (i.e. node (m,l,k) -> m+npe*l+npe^2*k)
  stkNodeMap_ = {
                  0, 4, 1, // bottom row of nodes
                  7, 8, 5, // middle row of nodes
                  3, 6, 2  // top row of nodes
                };

  sideNodeOrdinals_ = {
      0, 1, 4,
      1, 2, 5,
      2, 3, 6,
      3, 0, 7
  };

  // a padded list of scs locations
  scsEndLoc_ = { -1.0, -scsDist_, scsDist_, +1.0 };
}

//--------------------------------------------------------------------------
//-------- set_quadrature_rule ---------------------------------------------
//--------------------------------------------------------------------------
void
QuadrilateralP2Element::set_quadrature_rule()
{
  gaussAbscissaeShift_ = {-1.0,-1.0,0.0,0.0,+1.0,+1.0};
  std::tie(gaussAbscissae_, gaussWeight_) = gauss_legendre_rule(numQuad_);
  for (unsigned j = 0; j < gaussWeight_.size(); ++j) {
    gaussWeight_[j] *= 0.5;
  }
}

//--------------------------------------------------------------------------
//-------- tensor_product_node_map -----------------------------------------
//--------------------------------------------------------------------------
int
QuadrilateralP2Element::tensor_product_node_map(int i, int j) const
{
   return stkNodeMap_[i+nodes1D_*j];
}

//--------------------------------------------------------------------------
//-------- gauss_point_location --------------------------------------------
//--------------------------------------------------------------------------
double
QuadrilateralP2Element::gauss_point_location(
  int nodeOrdinal,
  int gaussPointOrdinal) const
{
   return isoparametric_mapping( scsEndLoc_[nodeOrdinal+1],
     scsEndLoc_[nodeOrdinal],
     gaussAbscissae_[gaussPointOrdinal] );
}
//--------------------------------------------------------------------------
//-------- shifted_gauss_point_location ------------------------------------
//--------------------------------------------------------------------------
double
QuadrilateralP2Element::shifted_gauss_point_location(
  int nodeOrdinal,
  int gaussPointOrdinal) const
{
  return gaussAbscissaeShift_[nodeOrdinal*numQuad_ + gaussPointOrdinal];
}
//--------------------------------------------------------------------------
//-------- tensor_product_weight -------------------------------------------
//--------------------------------------------------------------------------
double
QuadrilateralP2Element::tensor_product_weight(
  int s1Node, int s2Node,
  int s1Ip, int s2Ip) const
{
  //surface integration
  const double Ls1 = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
  const double Ls2 = scsEndLoc_[s2Node+1]-scsEndLoc_[s2Node];
  const double isoparametricArea = Ls1 * Ls2;

  const double weight = isoparametricArea * gaussWeight_[s1Ip] * gaussWeight_[s2Ip];

  return weight;
}

//--------------------------------------------------------------------------
//-------- tensor_product_weight -------------------------------------------
//--------------------------------------------------------------------------
double
QuadrilateralP2Element::tensor_product_weight(int s1Node, int s1Ip) const
{
  //line integration
  const double isoparametricLength = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
  const double weight = isoparametricLength * gaussWeight_[s1Ip];

  return weight;
}


//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
QuadrilateralP2Element::shape_fcn(double* shpfc)
{
  for (int ni = 0; ni < numIntPoints_ * nodesPerElement_; ++ni) {
    shpfc[ni] = shapeFunctions_[ni];
  }
}

void
QuadrilateralP2Element::shifted_shape_fcn(double* shpfc)
{
  for (int ip = 0; ip < numIntPoints_ * nodesPerElement_; ++ip) {
    shpfc[ip] = shapeFunctionsShift_[ip];
  }
}
//--------------------------------------------------------------------------
//-------- eval_shape_functions_at_ips -------------------------------------
//--------------------------------------------------------------------------
void
QuadrilateralP2Element::eval_shape_functions_at_ips()
{
  shapeFunctions_.resize(numIntPoints_*nodesPerElement_);
  quad9_shape_fcn(numIntPoints_, intgLoc_.data(), shapeFunctions_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
QuadrilateralP2Element::eval_shape_derivs_at_ips()
{
  shapeDerivs_.resize(numIntPoints_*nodesPerElement_*nDim_);
  quad9_shape_deriv(numIntPoints_, intgLoc_.data(), shapeDerivs_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_functions_at_shifted_ips -----------------------------
//--------------------------------------------------------------------------
void
QuadrilateralP2Element::eval_shape_functions_at_shifted_ips()
{
  shapeFunctionsShift_.resize(numIntPoints_*nodesPerElement_);
  quad9_shape_fcn(numIntPoints_, intgLocShift_.data(), shapeFunctionsShift_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
QuadrilateralP2Element::eval_shape_derivs_at_shifted_ips()
{
  shapeDerivsShift_.resize(numIntPoints_*nodesPerElement_*nDim_);
  quad9_shape_deriv(numIntPoints_, intgLocShift_.data(), shapeDerivsShift_.data());
}
//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_face_ips ----------------------------------------
//--------------------------------------------------------------------------
void
QuadrilateralP2Element::eval_shape_derivs_at_face_ips()
{
  expFaceShapeDerivs_.resize(numIntPoints_*nodesPerElement_*nDim_);
  quad9_shape_deriv(numIntPoints_, intgExpFace_.data(), expFaceShapeDerivs_.data());
}

//--------------------------------------------------------------------------
//-------- quad9_shape_fcn -------------------------------------------------
//--------------------------------------------------------------------------
void
QuadrilateralP2Element::quad9_shape_fcn(
  int  numIntPoints,
  const double *intgLoc,
  double *shpfc) const
{
  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    int nineIp = nodesPerElement_ * ip; // nodes per element is always 9
    int vector_offset = nDim_ * ip;
    const double s = intgLoc[vector_offset+0];
    const double t = intgLoc[vector_offset+1];

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
QuadrilateralP2Element::quad9_shape_deriv(
  int numIntPoints,
  const double *intgLoc,
  double *deriv) const
{
  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    const int grad_offset = nDim_ * nodesPerElement_ * ip; // nodes per element is always 9
    const int vector_offset = nDim_ * ip;
    int node; int offset;

    const double s = intgLoc[vector_offset+0];
    const double t = intgLoc[vector_offset+1];

    const double s2 = s*s;
    const double t2 = t*t;

    node = 0;
    offset = grad_offset + nDim_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 - 2.0 * s * t - t2 + t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t - 2.0 * s * t - s2 + s);

    node = 1;
    offset = grad_offset + nDim_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 - 2.0 * s * t + t2 - t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t + 2.0 * s * t - s2 - s);

    node = 2;
    offset = grad_offset + nDim_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 + 2.0 * s * t + t2 + t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t + 2.0 * s * t + s2 + s);

    node = 3;
    offset = grad_offset + nDim_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 + 2.0 * s * t - t2 - t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t - 2.0 * s * t + s2 - s);

    node = 4;
    offset = grad_offset + nDim_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 - 2.0 * s * t);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t - s2 - 2.0 * t + 1.0);

    node = 5;
    offset = grad_offset + nDim_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 + t2 - 2.0 * s - 1.0);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t + 2.0 * s * t);

    node = 6;
    offset = grad_offset + nDim_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 + 2.0 * s * t);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t + s2 - 2.0 * t - 1.0);

    node = 7;
    offset = grad_offset + nDim_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 - t2 - 2.0 * s + 1.0);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t - 2.0 * s * t);

    node = 8;
    offset = grad_offset + nDim_ * node;
    deriv[offset+0] = 2.0 * s * t2 - 2.0 * s;
    deriv[offset+1] = 2.0 * s2 * t - 2.0 * t;
  }
}
//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double QuadrilateralP2Element::parametric_distance(const std::array<double, 2>& x)
{
  double absXi  = std::abs(x[0]);
  double absEta = std::abs(x[1]);
  return (absXi > absEta) ? absXi : absEta;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
QuadrilateralP2Element::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result )
{
  constexpr int nNodes = 9;
  std::array<double, nNodes> shapefct;
  quad9_shape_fcn(1, isoParCoord, shapefct.data());

  for (int i = 0; i < nComp; i++) {
    result[i] = ddot(shapefct.data(), field + nNodes * i, nNodes);
  }
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double QuadrilateralP2Element::isInElement(
  const double *elemNodalCoord,
  const double *pointCoord,
  double *isoParCoord)
{
  // control the interation
  double isInElemConverged = 1.0e-16; // NOTE: the square of the tolerance on the distance
  int N_MAX_ITER = 100;

  constexpr int dim = 2;
  std::array<double, dim> guess = { { 0.0, 0.0 } };
  std::array<double, dim> delta;
  int iter = 0;

  do {
    // interpolate coordinate at guess
    constexpr int nNodes = 9;
    std::array<double, nNodes> weights;
    quad9_shape_fcn(1, guess.data(), weights.data());

    // compute difference between coordinates interpolated to the guessed isoParametric coordinates
    // and the actual point's coordinates
    std::array<double, dim> error_vec;
    error_vec[0] = pointCoord[0] - ddot(weights.data(), elemNodalCoord + 0 * nNodes, nNodes);
    error_vec[1] = pointCoord[1] - ddot(weights.data(), elemNodalCoord + 1 * nNodes, nNodes);

    // update guess along gradient of mapping from physical-to-reference coordinates
    // transpose of the jacobian of the forward mapping
    constexpr int deriv_size = nNodes * dim;
    std::array<double, deriv_size> deriv;
    quad9_shape_deriv(1, guess.data(), deriv.data());

    std::array<double, dim * dim> jact{};
    for(int j = 0; j < nNodes; ++j) {
      jact[0] += deriv[0 + j * dim] * elemNodalCoord[j + 0 * nNodes];
      jact[1] += deriv[1 + j * dim] * elemNodalCoord[j + 0 * nNodes];
      jact[2] += deriv[0 + j * dim] * elemNodalCoord[j + 1 * nNodes];
      jact[3] += deriv[1 + j * dim] * elemNodalCoord[j + 1 * nNodes];
    }

    // apply its inverse on the error vector
    solve22(jact.data(), error_vec.data(), delta.data());

    // update guess
    guess[0] += delta[0];
    guess[1] += delta[1];

    //continue to iterate if update was larger than the set tolerance until max iterations are reached
  } while(!within_tolerance(vector_norm_sq(delta.data(), 2), isInElemConverged) && (++iter < N_MAX_ITER));

  // output if failed:
  isoParCoord[0] = std::numeric_limits<double>::max();
  isoParCoord[1] = std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if (iter < N_MAX_ITER) {
    // output if succeeded:
    isoParCoord[0] = guess[0];
    isoParCoord[1] = guess[1];
    dist = parametric_distance(guess);
  }
  return dist;
}

void
QuadrilateralP2Element::sidePcoords_to_elemPcoords(
  const int & side_ordinal,
  const int & npoints,
  const double *side_pcoords,
  double *elem_pcoords)
{
  switch (side_ordinal) {
  case 0:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*2+0] = side_pcoords[i];
      elem_pcoords[i*2+1] = -1;
    }
    break;
  case 1:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*2+0] = 1;
      elem_pcoords[i*2+1] = side_pcoords[i];
    }
    break;
  case 2:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*2+0] = -side_pcoords[i];
      elem_pcoords[i*2+1] = 1;
    }
    break;
  case 3:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*2+0] = -1;
      elem_pcoords[i*2+1] = -side_pcoords[i];
    }
    break;
  default:
    throw std::runtime_error("QuadrilateralP2Element::sideMap invalid ordinal");
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Tri2DSCV::Tri2DSCV()
  : MasterElement()
{
  ndim(AlgTraits::nDim_);
  nodesPerElement_ = 3;
  numIntPoints_ = 3;

  // define ip node mappings
  ipNodeMap_.resize(3);
  ipNodeMap_[0] = 0; ipNodeMap_[1] = 1; ipNodeMap_[2] = 2;

  const double seven36ths = 7.0/36.0;
  const double eleven18ths = 11.0/18.0;
  intgLoc_ = {
    seven36ths, seven36ths,
    eleven18ths, seven36ths,
    seven36ths, eleven18ths
  };

  intgLocShift_ = {
      0.0,  0.0,
      1.0,  0.0,
      0.0,  1.0
  };
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Tri2DSCV::~Tri2DSCV()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Tri2DSCV::ipNodeMap(
  int /*ordinal*/)
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Tri2DSCV::shape_fcn(double *shpfc)
{
  tri_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Tri2DSCV::shifted_shape_fcn(double *shpfc)
{
  tri_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- tri_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
void
Tri2DSCV::tri_shape_fcn(
  const int  &npts,
  const double *isoParCoord,
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int threej = 3*j;
    const int k = 2*j;
    const double xi = isoParCoord[k];
    const double eta = isoParCoord[k+1];
    shape_fcn[threej] = 1.0 - xi - eta;
    shape_fcn[1 + threej] = xi;
    shape_fcn[2 + threej] = eta;
  }
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void Tri2DSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(tri_scv_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords,
      volume, error, &lerr );
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Tri3DSCS::Tri3DSCS()
  : MasterElement()
{
  ndim(AlgTraits::nDim_);
  nodesPerElement_ = 3;
  numIntPoints_ = 3;

  // define ip node mappings; ordinal size = 1
  ipNodeMap_.resize(3);
  ipNodeMap_[0] = 0;
  ipNodeMap_[1] = 1;
  ipNodeMap_[2] = 2;

  // standard integration location
  intgLoc_.resize(6);
  const double seven36ths = 7.0/36.0;
  const double eleven18ths = 11.0/18.0;
  intgLoc_[0]  = seven36ths;  intgLoc_[1] = seven36ths;  // surf 1
  intgLoc_[2]  = eleven18ths; intgLoc_[3] = seven36ths;  // surf 2
  intgLoc_[4]  = seven36ths;  intgLoc_[5] = eleven18ths; // surf 3

  // shifted
  intgLocShift_.resize(6);
  intgLocShift_[0]  =  0.00; intgLocShift_[1] =  0.00; // surf 1
  intgLocShift_[2]  =  1.00; intgLocShift_[3] =  0.00; // surf 2
  intgLocShift_[4]  =  0.00; intgLocShift_[5] =  1.00; // surf 3
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Tri3DSCS::~Tri3DSCS()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Tri3DSCS::ipNodeMap(
  int /*ordinal*/)
{
  // define ip->node mappings for each face (single ordinal); 
  return &ipNodeMap_[0];
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

  SIERRA_FORTRAN(tri3d_scs_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_,
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
  const int  &npts,
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

  // Translate position vector of point in same manner

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

  // find direction cosines transform position vector of
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

  std::vector<double> w = { par_coor[0], par_coor[1], zpnew/std::sqrt(Area2) };

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
  const std::vector<double> &x)
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

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Edge32DSCS::Edge32DSCS()
  : QuadrilateralP2Element()
{
  nodesPerElement_ = nodes1D_;

  // set up the one-dimensional quadrature rule
  set_quadrature_rule();

  numIntPoints_ = numQuad_ * nodes1D_;

  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_);

  intgLocShift_.resize(6);
  intgLocShift_[0] = -1.00; intgLocShift_[1]  = -1.00;
  intgLocShift_[2] =  0.00; intgLocShift_[3]  =  0.00;
  intgLocShift_[4] =  1.00; intgLocShift_[5]  =  1.00;

  ipWeight_.resize(numIntPoints_);

  std::vector<int> stk1DNodeMap = { 0, 2, 1 };

  int scalar_index = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    for (int i = 0; i < numQuad_; ++i) {
      ipNodeMap_[scalar_index] = stk1DNodeMap[k];
      intgLoc_[scalar_index] = gauss_point_location(k,i);
      ipWeight_[scalar_index] = tensor_product_weight(k,i);

      ++scalar_index;
    }
  }
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Edge32DSCS::ipNodeMap(
  int /*ordinal*/)
{
  // define ip->node mappings for each face (single ordinal); 
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void Edge32DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  std::array<double,2> areaVector;

  for (int k = 0; k < nelem; ++k) {
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;

    for (int ip = 0; ip < numIntPoints_; ++ip) {
      const int offset = nDim_ * ip + coord_elem_offset;

      // calculate the area vector
      area_vector( &coords[coord_elem_offset],
                   intgLoc_[ip],
                   areaVector.data() );

      // weight the area vector with the Gauss-quadrature weight for this IP
      areav[offset + 0] = ipWeight_[ip] * areaVector[0];
      areav[offset + 1] = ipWeight_[ip] * areaVector[1];
    }
  }

  // check
  *error = 0.0;
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Edge32DSCS::shape_fcn(double *shpfc)
{
  for ( int i =0; i< numIntPoints_; ++i ) {
    int j = 3*i;
    const double s = intgLoc_[i];
    shpfc[j  ] = -s*(1.0-s)*0.5;
    shpfc[j+1] = s*(1.0+s)*0.5;
    shpfc[j+2] = (1.0-s)*(1.0+s);
  }
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Edge32DSCS::shifted_shape_fcn(double *shpfc)
{
  for ( int i =0; i< numIntPoints_; ++i ) {
    int j = 3*i;
    const double s = intgLocShift_[i];
    shpfc[j  ] = -s*(1.0-s)*0.5;
    shpfc[j+1] = s*(1.0+s)*0.5;
    shpfc[j+2] = (1.0-s)*(1.0+s);
  }
}

//--------------------------------------------------------------------------
//-------- interpolate_point -----------------------------------------------
//--------------------------------------------------------------------------
void
Edge32DSCS::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result)
{
  constexpr int nNodes = 3;

  double s = isoParCoord[0];
  std::array<double, nNodes> shapefct = {{-0.5*s*(1-s), +0.5*s*(1+s), (1-s)*(1+s)}};

  for ( int i =0; i< nComp; ++i ) {
    result[i] = shapefct[0] * field[3*i+0] + shapefct[1] * field[3*i+1] + shapefct[2] * field[3*i+2];
  }
}

//--------------------------------------------------------------------------
//-------- area_vector -----------------------------------------------------
//--------------------------------------------------------------------------
void
Edge32DSCS::area_vector(
  const double *POINTER_RESTRICT coords,
  const double s,
  double *POINTER_RESTRICT areaVector) const
{
  // returns the normal area vector (dyds,-dxds) evaluated at s

  // create a parameterization of the curve
  // r(s) = (x(s),y(s)) s.t. r(-1) = (x0,y0); r(0) = (x2,y2); r(1) = (x1,y1);
  // x(s) = x2 + 0.5 (x1-x0) s + 0.5 (x1 - 2 x2 + x0) s^2,
  // y(s) = y2 + 0.5 (y1-y0) s + 0.5 (y1 - 2 y2 + y0) s^2
  // could equivalently use the shape function derivatives . . .

  // coordinate names
  const double x0 = coords[0]; const double y0 = coords[1];
  const double x1 = coords[2]; const double y1 = coords[3];
  const double x2 = coords[4]; const double y2 = coords[5];

  const double dxds = 0.5 * (x1 - x0) + (x1 - 2.0 * x2 + x0) * s;
  const double dyds = 0.5 * (y1 - y0) + (y1 - 2.0 * y2 + y0) * s;

  areaVector[0] =  dyds;
  areaVector[1] = -dxds;
}

} // namespace nalu
} // namespace sierra
