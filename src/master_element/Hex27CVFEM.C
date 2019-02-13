/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <master_element/Hex27CVFEM.h>

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
HexahedralP2Element::HexahedralP2Element()
  : MasterElement(),
    scsDist_(std::sqrt(3.0)/3.0),
    nodes1D_(3),
    numQuad_(2)
{
  nDim_ = 3;
  nodesPerElement_ = nodes1D_ * nodes1D_ * nodes1D_;

  // map the standard stk node numbering to a tensor-product style node numbering (i.e. node (m,l,k) -> m+npe*l+npe^2*k)
  stkNodeMap_ = {
                   0,  8,  1, // bottom front edge
                  11, 21,  9, // bottom mid-front edge
                   3, 10,  2, // bottom back edge
                  12, 25, 13, // mid-top front edge
                  23, 20, 24, // mid-top mid-front edge
                  15, 26, 14, // mid-top back edge
                   4, 16,  5, // top front edge
                  19, 22, 17, // top mid-front edge
                   7, 18,  6  // top back edge
                };

  sideNodeOrdinals_ = {
       0, 1, 5, 4, 8,13,16,12,25, //ordinal 0
       1, 2, 6, 5, 9,14,17,13,24, //ordinal 1
       2, 3, 7, 6,10,15,18,14,26, //ordinal 2
       0, 4, 7, 3,12,19,15,11,23, //ordinal 3
       0, 3, 2, 1,11,10, 9, 8,21, //ordinal 4
       4, 5, 6, 7,16,17,18,19,22  //ordinal 5
  };

  // a padded list of the scs locations
  scsEndLoc_ = { -1.0, -scsDist_, scsDist_, 1.0 };
}

//--------------------------------------------------------------------------
//-------- set_quadrature_rule ---------------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::set_quadrature_rule()
{
  gaussAbscissaeShift_ = {-1.0,-1.0,0.0,0.0,+1.0,+1.0};

  std::tie(gaussAbscissae_, gaussWeight_) = gauss_legendre_rule(numQuad_);
  for (unsigned j = 0; j < gaussWeight_.size(); ++j) {
    gaussWeight_[j] *= 0.5; // change from standard Gauss weights
  }
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double HexahedralP2Element::parametric_distance(const std::array<double, 3>& x)
{
  std::array<double,3> y;
  for (int i=0; i < 3; ++i) {
    y[i] = std::fabs(x[i]);
  }

  double d = 0;
  for (int i=0; i<3; ++i) {
    if (d < y[i]) {
      d = y[i];
    }
  }
  return d;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result )
{
  constexpr int nNodes = 27;
  std::array<double, nNodes> shapefct;
  hex27_shape_fcn(1, isoParCoord, shapefct.data());

  for (int i = 0; i < nComp; i++) {
    result[i] = ddot(shapefct.data(), field + nNodes * i, nNodes);
  }
}


//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double HexahedralP2Element::isInElement(
  const double *elemNodalCoord,
  const double *pointCoord,
  double *isoParCoord)
{
  // control the interation
  double isInElemConverged = 1.0e-16; // NOTE: the square of the tolerance on the distance
  int N_MAX_ITER = 100;

  constexpr int dim = 3;
  std::array<double, dim> guess = { { 0.0, 0.0, 0.0 } };
  std::array<double, dim> delta;
  int iter = 0;

  do {
    // interpolate coordinate at guess
    constexpr int nNodes = 27;
    std::array<double, nNodes> weights;
    hex27_shape_fcn(1, guess.data(), weights.data());

    // compute difference between coordinates interpolated to the guessed isoParametric coordinates
    // and the actual point's coordinates
    std::array<double, dim> error_vec;
    error_vec[0] = pointCoord[0] - ddot(weights.data(), elemNodalCoord + 0 * nNodes, nNodes);
    error_vec[1] = pointCoord[1] - ddot(weights.data(), elemNodalCoord + 1 * nNodes, nNodes);
    error_vec[2] = pointCoord[2] - ddot(weights.data(), elemNodalCoord + 2 * nNodes, nNodes);

    // update guess along gradient of mapping from physical-to-reference coordinates
    // transpose of the jacobian of the forward mapping
    constexpr int deriv_size = nNodes * dim;
    std::array<double, deriv_size> deriv;
    hex27_shape_deriv(1, guess.data(), deriv.data());

    std::array<double, dim * dim> jact{};
    for (int j = 0; j < nNodes; ++j) {
      jact[0] += deriv[0 + j * dim] * elemNodalCoord[j + 0 * nNodes];
      jact[1] += deriv[1 + j * dim] * elemNodalCoord[j + 0 * nNodes];
      jact[2] += deriv[2 + j * dim] * elemNodalCoord[j + 0 * nNodes];

      jact[3] += deriv[0 + j * dim] * elemNodalCoord[j + 1 * nNodes];
      jact[4] += deriv[1 + j * dim] * elemNodalCoord[j + 1 * nNodes];
      jact[5] += deriv[2 + j * dim] * elemNodalCoord[j + 1 * nNodes];

      jact[6] += deriv[0 + j * dim] * elemNodalCoord[j + 2 * nNodes];
      jact[7] += deriv[1 + j * dim] * elemNodalCoord[j + 2 * nNodes];
      jact[8] += deriv[2 + j * dim] * elemNodalCoord[j + 2 * nNodes];
    }

    // apply its inverse on the error vector
    solve33(jact.data(), error_vec.data(), delta.data());

    // update guess
    guess[0] += delta[0];
    guess[1] += delta[1];
    guess[2] += delta[2];

    //continue to iterate if update was larger than the set tolerance until max iterations are reached
  } while( !within_tolerance(vecnorm_sq3(delta.data()), isInElemConverged) && (++iter < N_MAX_ITER) );

  // output if failed:
  isoParCoord[0] = std::numeric_limits<double>::max();
  isoParCoord[1] = std::numeric_limits<double>::max();
  isoParCoord[2] = std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if (iter < N_MAX_ITER) {
    // output if succeeded:
    isoParCoord[0] = guess[0];
    isoParCoord[1] = guess[1];
    isoParCoord[2] = guess[2];
    dist = parametric_distance(guess);
  }
  return dist;
}
//--------------------------------------------------------------------------
//-------- tensor_product_node_map -----------------------------------------
//--------------------------------------------------------------------------
int
HexahedralP2Element::tensor_product_node_map(int i, int j, int k) const
{
   return stkNodeMap_[i + nodes1D_ * (j + nodes1D_ * k)];
}

//--------------------------------------------------------------------------
//-------- gauss_point_location --------------------------------------------
//--------------------------------------------------------------------------
double
HexahedralP2Element::gauss_point_location(
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
HexahedralP2Element::shifted_gauss_point_location(
  int nodeOrdinal,
  int gaussPointOrdinal) const
{
  return gaussAbscissaeShift_[nodeOrdinal*numQuad_ + gaussPointOrdinal];
}

//--------------------------------------------------------------------------
//-------- tensor_product_weight -------------------------------------------
//--------------------------------------------------------------------------

double
HexahedralP2Element::tensor_product_weight(
  int s1Node, int s2Node, int s3Node,
  int s1Ip, int s2Ip, int s3Ip) const
{
  // volume integration

  const double Ls1 = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
  const double Ls2 = scsEndLoc_[s2Node+1]-scsEndLoc_[s2Node];
  const double Ls3 = scsEndLoc_[s3Node+1]-scsEndLoc_[s3Node];
  const double isoparametricArea = Ls1 * Ls2 * Ls3;

  const double weight = isoparametricArea
                      * gaussWeight_[s1Ip]
                      * gaussWeight_[s2Ip]
                      * gaussWeight_[s3Ip];

  return weight;
}

//--------------------------------------------------------------------------
//-------- tensor_product_weight -------------------------------------------
//--------------------------------------------------------------------------
double
HexahedralP2Element::tensor_product_weight(
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
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::shape_fcn(double* shpfc)
{
  for (int ip = 0; ip < numIntPoints_ * nodesPerElement_; ++ip) {
    shpfc[ip] = shapeFunctions_[ip];
  }
}
//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::shifted_shape_fcn(double* shpfc)
{
  for (int ip = 0; ip < numIntPoints_ * nodesPerElement_; ++ip) {
    shpfc[ip] = shapeFunctionsShift_[ip];
  }
}


//--------------------------------------------------------------------------
//-------- eval_shape_functions_at_ips -------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_functions_at_ips()
{
  shapeFunctions_.resize(numIntPoints_*nodesPerElement_);
  hex27_shape_fcn(numIntPoints_, intgLoc_.data(), shapeFunctions_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_derivs_at_ips()
{
  shapeDerivs_.resize(numIntPoints_*nodesPerElement_*nDim_);
  hex27_shape_deriv(numIntPoints_, intgLoc_.data(), shapeDerivs_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_functions_at_shifted_ips -----------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_functions_at_shifted_ips()
{
  shapeFunctionsShift_.resize(numIntPoints_*nodesPerElement_);
  hex27_shape_fcn(numIntPoints_, intgLocShift_.data(), shapeFunctionsShift_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_derivs_at_shifted_ips()
{
  shapeDerivsShift_.resize(numIntPoints_*nodesPerElement_*nDim_);
  hex27_shape_deriv(numIntPoints_, intgLocShift_.data(), shapeDerivsShift_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_face_ips -----------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_derivs_at_face_ips()
{
  expFaceShapeDerivs_.resize(numIntPoints_*nodesPerElement_*nDim_);
  hex27_shape_deriv(
    numIntPoints_,
    intgExpFace_.data(),
    expFaceShapeDerivs_.data()
  );
}

//--------------------------------------------------------------------------
//-------- hex27_shape_fcn -------------------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::hex27_shape_fcn(
  int numIntPoints,
  const double *intgLoc,
  double *shpfc) const
{
  const double one = 1.0;
  const double half = 1.0/2.0;
  const double one4th = 1.0/4.0;
  const double one8th = 1.0/8.0;

  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    int ip_offset = nodesPerElement_*ip; // nodes per element is always 27
    int vector_offset = nDim_*ip;

    const double s = intgLoc[vector_offset+0];
    const double t = intgLoc[vector_offset+1];
    const double u = intgLoc[vector_offset+2];

    const double stu = s * t * u;
    const double  st  = s * t;
    const double  su  = s * u;
    const double  tu  = t * u;

    const double one_m_s = one - s;
    const double one_p_s = one + s;
    const double one_m_t = one - t;
    const double one_p_t = one + t;
    const double one_m_u = one - u;
    const double one_p_u = one + u;

    const double one_m_ss = one - s * s;
    const double one_m_tt = one - t * t;
    const double one_m_uu = one - u * u;

    shpfc[ip_offset+0]  = -one8th * stu * one_m_s  * one_m_t  * one_m_u;
    shpfc[ip_offset+1]  =  one8th * stu * one_p_s  * one_m_t  * one_m_u;
    shpfc[ip_offset+2]  = -one8th * stu * one_p_s  * one_p_t  * one_m_u;
    shpfc[ip_offset+3]  =  one8th * stu * one_m_s  * one_p_t  * one_m_u;
    shpfc[ip_offset+4]  =  one8th * stu * one_m_s  * one_m_t  * one_p_u;
    shpfc[ip_offset+5]  = -one8th * stu * one_p_s  * one_m_t  * one_p_u;
    shpfc[ip_offset+6]  =  one8th * stu * one_p_s  * one_p_t  * one_p_u;
    shpfc[ip_offset+7]  = -one8th * stu * one_m_s  * one_p_t  * one_p_u;
    shpfc[ip_offset+8]  =  one4th * tu  * one_m_ss * one_m_t  * one_m_u;
    shpfc[ip_offset+9]  = -one4th * su  * one_p_s  * one_m_tt * one_m_u;
    shpfc[ip_offset+10] = -one4th * tu  * one_m_ss * one_p_t  * one_m_u;
    shpfc[ip_offset+11] =  one4th * su  * one_m_s  * one_m_tt * one_m_u;
    shpfc[ip_offset+12] =  one4th * st  * one_m_s  * one_m_t  * one_m_uu;
    shpfc[ip_offset+13] = -one4th * st  * one_p_s  * one_m_t  * one_m_uu;
    shpfc[ip_offset+14] =  one4th * st  * one_p_s  * one_p_t  * one_m_uu;
    shpfc[ip_offset+15] = -one4th * st  * one_m_s  * one_p_t  * one_m_uu;
    shpfc[ip_offset+16] = -one4th * tu  * one_m_ss * one_m_t  * one_p_u;
    shpfc[ip_offset+17] =  one4th * su  * one_p_s  * one_m_tt * one_p_u;
    shpfc[ip_offset+18] =  one4th * tu  * one_m_ss * one_p_t  * one_p_u;
    shpfc[ip_offset+19] = -one4th * su  * one_m_s  * one_m_tt * one_p_u;
    shpfc[ip_offset+20] =                 one_m_ss * one_m_tt * one_m_uu;
    shpfc[ip_offset+21] =   -half * u   * one_m_ss * one_m_tt * one_m_u;
    shpfc[ip_offset+22] =    half * u   * one_m_ss * one_m_tt * one_p_u;
    shpfc[ip_offset+23] =   -half * s   * one_m_s  * one_m_tt * one_m_uu;
    shpfc[ip_offset+24] =    half * s   * one_p_s  * one_m_tt * one_m_uu;
    shpfc[ip_offset+25] =   -half * t   * one_m_ss * one_m_t  * one_m_uu;
    shpfc[ip_offset+26] =    half * t   * one_m_ss * one_p_t  * one_m_uu;
  }
}

//--------------------------------------------------------------------------
//-------- hex27_shape_deriv -----------------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::hex27_shape_deriv(
  int numIntPoints,
  const double *intgLoc,
  double *shapeDerivs) const
{
  const double half = 1.0/2.0;
  const double one4th = 1.0/4.0;
  const double one8th = 1.0/8.0;
  const double two = 2.0;

  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    const int vector_offset = nDim_ * ip;
    const int ip_offset  = nDim_ * nodesPerElement_ * ip;
    int node; int offset;

    const double s = intgLoc[vector_offset+0];
    const double t = intgLoc[vector_offset+1];
    const double u = intgLoc[vector_offset+2];

    const double stu = s * t * u;
    const double st  = s * t;
    const double su  = s * u;
    const double tu  = t * u;

    const double one_m_s = 1.0 - s;
    const double one_p_s = 1.0 + s;
    const double one_m_t = 1.0 - t;
    const double one_p_t = 1.0 + t;
    const double one_m_u = 1.0 - u;
    const double one_p_u = 1.0 + u;

    const double one_m_ss = 1.0 - s * s;
    const double one_m_tt = 1.0 - t * t;
    const double one_m_uu = 1.0 - u * u;

    const double one_m_2s = 1.0 - 2.0 * s;
    const double one_m_2t = 1.0 - 2.0 * t;
    const double one_m_2u = 1.0 - 2.0 * u;

    const double one_p_2s = 1.0 + 2.0 * s;
    const double one_p_2t = 1.0 + 2.0 * t;
    const double one_p_2u = 1.0 + 2.0 * u;

    node = 0;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one8th * tu * one_m_2s * one_m_t * one_m_u;
    shapeDerivs[offset + 1] = -one8th * su * one_m_s * one_m_2t * one_m_u;
    shapeDerivs[offset + 2] = -one8th * st * one_m_s * one_m_t * one_m_2u;

    node = 1;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one8th * tu * one_p_2s * one_m_t * one_m_u;
    shapeDerivs[offset + 1] = one8th * su * one_p_s * one_m_2t * one_m_u;
    shapeDerivs[offset + 2] = one8th * st * one_p_s * one_m_t * one_m_2u;

    node = 2;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one8th * tu * one_p_2s * one_p_t * one_m_u;
    shapeDerivs[offset + 1] = -one8th * su * one_p_s * one_p_2t * one_m_u;
    shapeDerivs[offset + 2] = -one8th * st * one_p_s * one_p_t * one_m_2u;

    node = 3;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one8th * tu * one_m_2s * one_p_t * one_m_u;
    shapeDerivs[offset + 1] = one8th * su * one_m_s * one_p_2t * one_m_u;
    shapeDerivs[offset + 2] = one8th * st * one_m_s * one_p_t * one_m_2u;

    node = 4;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one8th * tu * one_m_2s * one_m_t * one_p_u;
    shapeDerivs[offset + 1] = one8th * su * one_m_s * one_m_2t * one_p_u;
    shapeDerivs[offset + 2] = one8th * st * one_m_s * one_m_t * one_p_2u;

    node = 5;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one8th * tu * one_p_2s * one_m_t * one_p_u;
    shapeDerivs[offset + 1] = -one8th * su * one_p_s * one_m_2t * one_p_u;
    shapeDerivs[offset + 2] = -one8th * st * one_p_s * one_m_t * one_p_2u;

    node = 6;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one8th * tu * one_p_2s * one_p_t * one_p_u;
    shapeDerivs[offset + 1] = one8th * su * one_p_s * one_p_2t * one_p_u;
    shapeDerivs[offset + 2] = one8th * st * one_p_s * one_p_t * one_p_2u;

    node = 7;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one8th * tu * one_m_2s * one_p_t * one_p_u;
    shapeDerivs[offset + 1] = -one8th * su * one_m_s * one_p_2t * one_p_u;
    shapeDerivs[offset + 2] = -one8th * st * one_m_s * one_p_t * one_p_2u;

    node = 8;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -half * stu * one_m_t * one_m_u;
    shapeDerivs[offset + 1] = one4th * u * one_m_ss * one_m_2t * one_m_u;
    shapeDerivs[offset + 2] = one4th * t * one_m_ss * one_m_t * one_m_2u;

    node = 9;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one4th * u * one_p_2s * one_m_tt * one_m_u;
    shapeDerivs[offset + 1] = half * stu * one_p_s * one_m_u;
    shapeDerivs[offset + 2] = -one4th * s * one_p_s * one_m_tt * one_m_2u;

    node = 10;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = half * stu * one_p_t * one_m_u;
    shapeDerivs[offset + 1] = -one4th * u * one_m_ss * one_p_2t * one_m_u;
    shapeDerivs[offset + 2] = -one4th * t * one_m_ss * one_p_t * one_m_2u;

    node = 11;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one4th * u * one_m_2s * one_m_tt * one_m_u;
    shapeDerivs[offset + 1] = -half * stu * one_m_s * one_m_u;
    shapeDerivs[offset + 2] = one4th * s * one_m_s * one_m_tt * one_m_2u;

    node = 12;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one4th * t * one_m_2s * one_m_t * one_m_uu;
    shapeDerivs[offset + 1] = one4th * s * one_m_s * one_m_2t * one_m_uu;
    shapeDerivs[offset + 2] = -half * stu * one_m_s * one_m_t;

    node = 13;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one4th * t * one_p_2s * one_m_t * one_m_uu;
    shapeDerivs[offset + 1] = -one4th * s * one_p_s * one_m_2t * one_m_uu;
    shapeDerivs[offset + 2] = half * stu * one_p_s * one_m_t;

    node = 14;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one4th * t * one_p_2s * one_p_t * one_m_uu;
    shapeDerivs[offset + 1] = one4th * s * one_p_s * one_p_2t * one_m_uu;
    shapeDerivs[offset + 2] = -half * stu * one_p_s * one_p_t;

    node = 15;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one4th * t * one_m_2s * one_p_t * one_m_uu;
    shapeDerivs[offset + 1] = -one4th * s * one_m_s * one_p_2t * one_m_uu;
    shapeDerivs[offset + 2] = half * stu * one_m_s * one_p_t;

    node = 16;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = half * stu * one_m_t * one_p_u;
    shapeDerivs[offset + 1] = -one4th * u * one_m_ss * one_m_2t * one_p_u;
    shapeDerivs[offset + 2] = -one4th * t * one_m_ss * one_m_t * one_p_2u;

    node = 17;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one4th * u * one_p_2s * one_m_tt * one_p_u;
    shapeDerivs[offset + 1] = -half * stu * one_p_s * one_p_u;
    shapeDerivs[offset + 2] = one4th * s * one_p_s * one_m_tt * one_p_2u;

    node = 18;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -half * stu * one_p_t * one_p_u;
    shapeDerivs[offset + 1] = one4th * u * one_m_ss * one_p_2t * one_p_u;
    shapeDerivs[offset + 2] = one4th * t * one_m_ss * one_p_t * one_p_2u;

    node = 19;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one4th * u * one_m_2s * one_m_tt * one_p_u;
    shapeDerivs[offset + 1] = half * stu * one_m_s * one_p_u;
    shapeDerivs[offset + 2] = -one4th * s * one_m_s * one_m_tt * one_p_2u;

    node = 20;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -two * s * one_m_tt * one_m_uu;
    shapeDerivs[offset + 1] = -two * t * one_m_ss * one_m_uu;
    shapeDerivs[offset + 2] = -two * u * one_m_ss * one_m_tt;

    node = 21;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = su * one_m_tt * one_m_u;
    shapeDerivs[offset + 1] = tu * one_m_ss * one_m_u;
    shapeDerivs[offset + 2] = -half * one_m_ss * one_m_tt * one_m_2u;

    node = 22;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -su * one_m_tt * one_p_u;
    shapeDerivs[offset + 1] = -tu * one_m_ss * one_p_u;
    shapeDerivs[offset + 2] = half * one_m_ss * one_m_tt * one_p_2u;

    node = 23;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -half * one_m_2s * one_m_tt * one_m_uu;
    shapeDerivs[offset + 1] = st * one_m_s * one_m_uu;
    shapeDerivs[offset + 2] = su * one_m_s * one_m_tt;

    node = 24;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = half * one_p_2s * one_m_tt * one_m_uu;
    shapeDerivs[offset + 1] = -st * one_p_s * one_m_uu;
    shapeDerivs[offset + 2] = -su * one_p_s * one_m_tt;

    node = 25;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = st * one_m_t * one_m_uu;
    shapeDerivs[offset + 1] = -half * one_m_ss * one_m_2t * one_m_uu;
    shapeDerivs[offset + 2] = tu * one_m_ss * one_m_t;

    node = 26;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -st * one_p_t * one_m_uu;
    shapeDerivs[offset + 1] = half * one_m_ss * one_p_2t * one_m_uu;
    shapeDerivs[offset + 2] = -tu * one_m_ss * one_p_t;
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Hex27SCV::Hex27SCV()
  : HexahedralP2Element()
{
  // set up the one-dimensional quadrature rule
  set_quadrature_rule();

  // set up integration rule and relevant maps for scvs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  eval_shape_functions_at_ips();
  eval_shape_derivs_at_ips();
  eval_shape_functions_at_shifted_ips();
  eval_shape_derivs_at_shifted_ips();

  interpWeights_ = copy_interpolation_weights_to_view<InterpWeightType>(shapeFunctions_);
  shiftedInterpWeights_ = copy_interpolation_weights_to_view<InterpWeightType>(shapeFunctionsShift_);

  referenceGradWeights_ = copy_deriv_weights_to_view<GradWeightType>(shapeDerivs_);
  shiftedReferenceGradWeights_ = copy_deriv_weights_to_view<GradWeightType>(shapeDerivsShift_);
}

//--------------------------------------------------------------------------
//-------- set_interior_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCV::set_interior_info()
{
  //1D integration rule per sub-control volume
  numIntPoints_ = (nodes1D_ * nodes1D_  * nodes1D_) * ( numQuad_ * numQuad_ * numQuad_); // 216
  ThrowRequire(numIntPoints_ == AlgTraits::numScvIp_);

  // define ip node mappings
  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_*nDim_);
  intgLocShift_.resize(numIntPoints_*nDim_);
  ipWeight_.resize(numIntPoints_);

  // tensor product nodes (3x3x3) x tensor product quadrature (2 x 2 x 2)
  int vector_index = 0; int scalar_index = 0;
  for (int n = 0; n < nodes1D_; ++n) {
    for (int m = 0; m < nodes1D_; ++m) {
      for (int l = 0; l < nodes1D_; ++l) {

        // current node number
        const int nodeNumber = tensor_product_node_map(l,m,n);

        //tensor-product quadrature for a particular sub-cv
        for (int k = 0; k < numQuad_; ++k) {
          for (int j = 0; j < numQuad_; ++j) {
            for (int i = 0; i < numQuad_; ++i) {
              //integration point location
              intgLoc_[vector_index]     = gauss_point_location(l,i);
              intgLoc_[vector_index + 1] = gauss_point_location(m,j);
              intgLoc_[vector_index + 2] = gauss_point_location(n,k);

              intgLocShift_[vector_index]     = shifted_gauss_point_location(l,i);
              intgLocShift_[vector_index + 1] = shifted_gauss_point_location(m,j);
              intgLocShift_[vector_index + 2] = shifted_gauss_point_location(n,k);

              //weight
              ipWeight_[scalar_index] = tensor_product_weight(l,m,n,i,j,k);

              //sub-control volume association
              ipNodeMap_[scalar_index] = nodeNumber;

              // increment indices
              ++scalar_index;
              vector_index += nDim_;
            }
          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCV::ipNodeMap(
  int /*ordinal*/)
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
void Hex27SCV::shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      shpfc(ip,n) = interpWeights_(ip,n);
    }
  }
}
//--------------------------------------------------------------------------
void Hex27SCV::shifted_shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      shpfc(ip,n) = shiftedInterpWeights_(ip,n);
    }
  }
}
//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCV::determinant(
  const int  /* nelem */,
  const double *coords,
  double *volume,
  double *error)
{
  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
    const int grad_offset = nDim_ * nodesPerElement_ * ip;

    //weighted jacobian determinant
    const double det_j = jacobian_determinant(coords, &shapeDerivs_[grad_offset]);

    //apply weight and store to volume
    volume[ip] = ipWeight_[ip] * det_j;

    //flag error
    if (volume[ip] < tiny_positive_value()) {
      *error = 1.0;
    }
  }
}
//--------------------------------------------------------------------------
void Hex27SCV::determinant(SharedMemView<DoubleType**>& coords, SharedMemView<DoubleType*>& volume)
{
  weighted_volumes(referenceGradWeights_, coords, volume);
}

//--------------------------------------------------------------------------
void Hex27SCV::grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  generic_grad_op<AlgTraits>(referenceGradWeights_, coords, gradop);

  // copy derivs as well.  These aren't used, but are part of the interface
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      for (int d = 0; d < AlgTraits::nDim_; ++d) {
        deriv(ip,n,d) = referenceGradWeights_(ip,n,d);
      }
    }
  }
}

//--------------------------------------------------------------------------
void Hex27SCV::shifted_grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  generic_grad_op<AlgTraits>(shiftedReferenceGradWeights_, coords, gradop);

  // copy derivs as well.  These aren't used, but are part of the interface
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      for (int d = 0; d < AlgTraits::nDim_; ++d) {
        deriv(ip,n,d) = shiftedReferenceGradWeights_(ip,n,d);
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- jacobian_determinant---------------------------------------------
//--------------------------------------------------------------------------
double
Hex27SCV::jacobian_determinant(
  const double *elemNodalCoords,
  const double *shapeDerivs) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0; double dx_ds3 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0; double dy_ds3 = 0.0;
  double dz_ds1 = 0.0;  double dz_ds2 = 0.0; double dz_ds3 = 0.0;
  for (int node = 0; node <   AlgTraits::nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;

    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];
    const double zCoord = elemNodalCoords[vector_offset+2];

    const double dn_ds1 = shapeDerivs[vector_offset+0];
    const double dn_ds2 = shapeDerivs[vector_offset+1];
    const double dn_ds3 = shapeDerivs[vector_offset+2];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;
    dx_ds3 += dn_ds3 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
    dy_ds3 += dn_ds3 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
    dz_ds3 += dn_ds3 * zCoord;
  }

  const double det_j = dx_ds1 * ( dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3 )
                     + dy_ds1 * ( dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3 )
                     + dz_ds1 * ( dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3 );

  return det_j;
}

//--------------------------------------------------------------------------
//-------- Mij -------------------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCV::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraits>(numIntPoints_, deriv, coords, metric);
}
//--------------------------------------------------------------------------
void Hex27SCV::Mij(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& metric,
  SharedMemView<DoubleType***>&  /* deriv */)
{
  generic_Mij_3d<AlgTraits>(referenceGradWeights_, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Hex27SCS::Hex27SCS()
  : HexahedralP2Element()
{
  // set up the one-dimensional quadrature rule
  set_quadrature_rule();

  // set up integration rule and relevant maps on scs
  set_interior_info();

  // set up integration rule and relevant maps on faces
  set_boundary_info();

  // compute and save shape functions and derivatives at ips
  eval_shape_functions_at_ips();
  interpWeights_ = copy_interpolation_weights_to_view<InterpWeightType>(shapeFunctions_);

  eval_shape_derivs_at_ips();
  referenceGradWeights_ = copy_deriv_weights_to_view<GradWeightType>(shapeDerivs_);

  eval_shape_functions_at_shifted_ips();
  interpWeights_ = copy_interpolation_weights_to_view<InterpWeightType>(shapeFunctions_);

  eval_shape_derivs_at_shifted_ips();
  shiftedReferenceGradWeights_ = copy_deriv_weights_to_view<GradWeightType>(shapeDerivsShift_);

  eval_shape_derivs_at_face_ips();
  expReferenceGradWeights_ = copy_deriv_weights_to_view<ExpGradWeightType>(expFaceShapeDerivs_);
}

//--------------------------------------------------------------------------
//-------- set_interior_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::set_interior_info()
{
  const int surfacesPerDirection = nodes1D_ - 1; // 2
  const int ipsPerSurface = (numQuad_*numQuad_)*(nodes1D_*nodes1D_); // 36
  const int numSurfaces = surfacesPerDirection * nDim_; // 6

  numIntPoints_ = numSurfaces*ipsPerSurface; // 216
  ThrowRequire(numIntPoints_ == AlgTraits::numScsIp_);

  const int numVectorPoints = numIntPoints_*nDim_; // 648

  // define L/R mappings
  lrscv_.resize(2*numIntPoints_); // size = 432

  // standard integration location
  intgLoc_.resize(numVectorPoints);

  // shifted
  intgLocShift_.resize(numVectorPoints);

  // Save quadrature weight and directionality information
  ipInfo_.resize(numIntPoints_);

  // a list of the scs locations in 1D
  const std::vector<double> scsLoc = { -scsDist_, scsDist_ };

  // correct orientation of area vector
  const std::vector<double> orientation = {-1.0, +1.0};

  // specify integration point locations in a dimension-by-dimension manner
  //u direction: bottom-top (0-1)
  int vector_index = 0; int lrscv_index = 0; int scalar_index = 0;
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {
      for (int k = 0; k < nodes1D_; ++k) {

        int leftNode; int rightNode;
        if (m == 0) {
          leftNode = tensor_product_node_map(k,l,m);
          rightNode = tensor_product_node_map(k,l,m+1);
        }
        else {
          leftNode = tensor_product_node_map(k,l,m+1);
          rightNode = tensor_product_node_map(k,l,m);
        }

        for (int j = 0; j < numQuad_; ++j) {
          for (int i = 0; i < numQuad_; ++i) {
            lrscv_[lrscv_index]     = leftNode;
            lrscv_[lrscv_index + 1] = rightNode;

            intgLoc_[vector_index + 0] = gauss_point_location(k,i);
            intgLoc_[vector_index + 1] = gauss_point_location(l,j);
            intgLoc_[vector_index + 2] = scsLoc[m];

            intgLocShift_[vector_index + 0] = shifted_gauss_point_location(k,i);
            intgLocShift_[vector_index + 1] = shifted_gauss_point_location(l,j);
            intgLocShift_[vector_index + 2] = scsLoc[m];

            //compute the quadrature weight
            ipInfo_[scalar_index].weight = orientation[m] * tensor_product_weight(k,l,i,j);

            //direction
            ipInfo_[scalar_index].direction = Jacobian::U_DIRECTION;

            ++scalar_index;
            lrscv_index += 2;
            vector_index += nDim_;
          }
        }
      }
    }
  }

  //t direction: front-back (2-3)
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {
      for (int k = 0; k < nodes1D_; ++k) {

        int leftNode; int rightNode;
        if (m == 0) {
          leftNode = tensor_product_node_map(k,m,l);
          rightNode = tensor_product_node_map(k,m+1,l);
        }
        else {
          leftNode = tensor_product_node_map(k,m+1,l);
          rightNode = tensor_product_node_map(k,m,l);
        }

        for (int j = 0; j < numQuad_; ++j) {
          for (int i = 0; i < numQuad_; ++i) {
            lrscv_[lrscv_index]     = leftNode;
            lrscv_[lrscv_index + 1] = rightNode;

            intgLoc_[vector_index]     = gauss_point_location(k,i);
            intgLoc_[vector_index + 1] = scsLoc[m];
            intgLoc_[vector_index + 2] = gauss_point_location(l,j);

            intgLocShift_[vector_index]     = shifted_gauss_point_location(k,i);
            intgLocShift_[vector_index + 1] = scsLoc[m];
            intgLocShift_[vector_index + 2] = shifted_gauss_point_location(l,j);

            //compute the quadrature weight
            ipInfo_[scalar_index].weight = orientation[m] * tensor_product_weight(k,l,i,j);

            //direction
            ipInfo_[scalar_index].direction = Jacobian::T_DIRECTION;

            ++scalar_index;
            lrscv_index += 2;
            vector_index += nDim_;
          }
        }
      }
    }
  }

  //s direction: left-right (4-5)
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {
      for (int k = 0; k < nodes1D_; ++k) {

        int leftNode; int rightNode;
        if (m == 0) {
          leftNode = tensor_product_node_map(m,k,l);
          rightNode = tensor_product_node_map(m+1,k,l);
        }
        else {
          leftNode = tensor_product_node_map(m+1,k,l);
          rightNode = tensor_product_node_map(m,k,l);
        }

        for (int j = 0; j < numQuad_; ++j) {
          for (int i = 0; i < numQuad_; ++i) {
            lrscv_[lrscv_index]     = leftNode;
            lrscv_[lrscv_index + 1] = rightNode;

            intgLoc_[vector_index]     = scsLoc[m];
            intgLoc_[vector_index + 1] = gauss_point_location(k,i);
            intgLoc_[vector_index + 2] = gauss_point_location(l,j);

            intgLocShift_[vector_index]     = scsLoc[m];
            intgLocShift_[vector_index + 1] = shifted_gauss_point_location(k,i);
            intgLocShift_[vector_index + 2] = shifted_gauss_point_location(l,j);

            //compute the quadrature weight
            ipInfo_[scalar_index].weight = -orientation[m] * tensor_product_weight(k,l,i,j);

            //direction
            ipInfo_[scalar_index].direction = Jacobian::S_DIRECTION;

            ++scalar_index;
            lrscv_index += 2;
            vector_index += nDim_;
          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- set_boundary_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::set_boundary_info()
{
  const int numFaces = 2 * nDim_; // 6
  const int nodesPerFace = nodes1D_ * nodes1D_; // 9
  ipsPerFace_ = nodesPerFace * (numQuad_ * numQuad_); // 36
  const int numFaceIps = numFaces * ipsPerFace_; // 216 = numIntPoints_ for this element

  oppFace_.resize(numFaceIps);
  ipNodeMap_.resize(numFaceIps);
  oppNode_.resize(numFaceIps);
  intgExpFace_.resize(numFaceIps*nDim_); // size = 648

  // face ordinal to tensor-product style node ordering
  const std::vector<int> stkFaceNodeMap = {
                                            0,  8,  1, 12, 25, 13,  4, 16,  5, // face 0(2): front face (cclockwise)
                                            1,  9,  2, 13, 24, 14,  5, 17,  6, // face 1(5): right face (cclockwise)
                                            3, 10,  2, 15, 26, 14,  7, 18,  6, // face 2(3): back face  (clockwise)
                                            0, 11,  3, 12, 23, 15,  4, 19,  7, // face 3(4): left face  (clockwise)
                                            0,  8,  1, 11, 21, 9,   3, 10,  2, // face 4(0): bottom face (clockwise)
                                            4, 16,  5, 19, 22,  17, 7, 18,  6  // face 5(1): top face (cclockwise)
                                          };


  // tensor-product style access to the map
  auto face_node_number = [=] (int i, int j, int faceOrdinal)
  {
    return stkFaceNodeMap[i + nodes1D_ * j + nodesPerFace * faceOrdinal];
  };

  // map face ip ordinal to nearest sub-control surface ip ordinal
  // sub-control surface renumbering
  const std::vector<int> faceToSurface = { 2, 5, 3, 4, 0, 1 };
  auto opp_face_map = [=] ( int k, int l, int i, int j, int face_index)
  {
    int face_offset = faceToSurface[face_index] * ipsPerFace_;

    int node_index = k + nodes1D_ * l;
    int node_offset = node_index * (numQuad_ * numQuad_);

    int ip_index = face_offset+node_offset+i+numQuad_*j;

    return ip_index;
  };

  // location of the faces in the correct order
  const std::vector<double> faceLoc = {-1.0, +1.0, +1.0, -1.0, -1.0, +1.0};

  // Set points face-by-face
  int vector_index = 0; int scalar_index = 0; int faceOrdinal = 0;

  // front face: t = -1.0: counter-clockwise
  faceOrdinal = 0;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = tensor_product_node_map(k,1,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  // right face: s = +1.0: counter-clockwise
  faceOrdinal = 1;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = tensor_product_node_map(1,k,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  // back face: s = +1.0: s-direction reversed
  faceOrdinal = 2;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = nodes1D_-1; k >= 0; --k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = tensor_product_node_map(k,1,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = numQuad_-1; i >= 0; --i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  //left face: x = -1.0 swapped t and u
  faceOrdinal = 3;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(l,k,faceOrdinal);
      int oppNode = tensor_product_node_map(1,l,k);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index]   = oppNode;
          oppFace_[scalar_index]   = opp_face_map(l,k,j,i,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  //bottom face: u = -1.0: swapped s and t
  faceOrdinal = 4;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(l,k,faceOrdinal);
      int oppNode = tensor_product_node_map(l,k,1);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(l,k,j,i,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = faceLoc[faceOrdinal];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  //top face: u = +1.0: counter-clockwise
  faceOrdinal = 5;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = tensor_product_node_map(k,l,1);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = faceLoc[faceOrdinal];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCS::ipNodeMap(
  int ordinal)
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal*ipsPerFace_];
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCS::side_node_ordinals(
  int ordinal)
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return &sideNodeOrdinals_[ordinal*9];
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
int
Hex27SCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*ipsPerFace_+node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
int
Hex27SCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*ipsPerFace_+node];
}

//--------------------------------------------------------------------------
void Hex27SCS::shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      shpfc(ip,n) = interpWeights_(ip,n);
    }
  }
}
//--------------------------------------------------------------------------
void Hex27SCS::shifted_shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      shpfc(ip,n) = shiftedInterpWeights_(ip,n);
    }
  }
}
//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  ThrowRequireMsg(nelem == 1, "P2 elements are processed one-at-a-time");

  constexpr int dim = AlgTraits::nDim_;
  constexpr int ipsPerDirection = AlgTraits::numScsIp_ / dim;
  static_assert ( ipsPerDirection * dim == AlgTraits::numScsIp_, "Number of ips incorrect");

  constexpr int deriv_increment = dim * AlgTraits::nodesPerElement_;

  int index = 0;

  //returns the normal vector x_s x x_t for constant u surfaces
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    ThrowAssert(ipInfo_[index].direction == Jacobian::U_DIRECTION);
    area_vector<Jacobian::U_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav[index*dim]);
    ++index;
  }

  //returns the normal vector x_u x x_s for constant t surfaces
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    ThrowAssert(ipInfo_[index].direction == Jacobian::T_DIRECTION);
    area_vector<Jacobian::T_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav[index*dim]);
    ++index;
  }

  //returns the normal vector x_t x x_u for constant s curves
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    ThrowAssert(ipInfo_[index].direction == Jacobian::S_DIRECTION);
    area_vector<Jacobian::S_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav[index*dim]);
    ++index;
  }

  // Multiply with the integration point weighting
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    double weight = ipInfo_[ip].weight;
    areav[ip * dim + 0] *= weight;
    areav[ip * dim + 1] *= weight;
    areav[ip * dim + 2] *= weight;
  }

  *error = 0; // no error checking available
}
//--------------------------------------------------------------------------
void Hex27SCS::determinant(SharedMemView<DoubleType**>&coords,  SharedMemView<DoubleType**>&areav)
{
  weighted_area_vectors(referenceGradWeights_, coords, areav);
}

//--------------------------------------------------------------------------
//-------- area_vector -----------------------------------------------------
//--------------------------------------------------------------------------
template <Jacobian::Direction direction> void
Hex27SCS::area_vector(
  const double *POINTER_RESTRICT elemNodalCoords,
  double *POINTER_RESTRICT shapeDeriv,
  double *POINTER_RESTRICT areaVector) const
{

  constexpr int s1Component = (direction == Jacobian::T_DIRECTION) ?
      Jacobian::S_DIRECTION : Jacobian::T_DIRECTION;

  constexpr int s2Component = (direction == Jacobian::U_DIRECTION) ?
      Jacobian::S_DIRECTION : Jacobian::U_DIRECTION;

  // return the normal area vector given shape derivatives dnds OR dndt
  double dx_ds1 = 0.0; double dy_ds1 = 0.0; double dz_ds1 = 0.0;
  double dx_ds2 = 0.0; double dy_ds2 = 0.0; double dz_ds2 = 0.0;

  for (int node = 0; node < AlgTraits::nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];
    const double zCoord = elemNodalCoords[vector_offset+2];

    const double dn_ds1 = shapeDeriv[vector_offset+s1Component];
    const double dn_ds2 = shapeDeriv[vector_offset+s2Component];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
  }

  //cross product
  areaVector[0] = dy_ds1*dz_ds2 - dz_ds1*dy_ds2;
  areaVector[1] = dz_ds1*dx_ds2 - dx_ds1*dz_ds2;
  areaVector[2] = dx_ds1*dy_ds2 - dy_ds1*dx_ds2;
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  ThrowRequireMsg(nelem == 1, "P2 elements are processed one-at-a-time");

  *error = 0.0;

  // shape derivatives are stored: just copy
  constexpr int deriv_increment = AlgTraits::nDim_ * AlgTraits::nodesPerElement_;
  constexpr int numShapeDerivs = deriv_increment * AlgTraits::numScsIp_;
  for (int j = 0; j < numShapeDerivs; ++j) {
    deriv[j] = shapeDerivs_[j];
  }

  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    const int grad_offset = deriv_increment * ip;
    gradient(coords, &shapeDerivs_[grad_offset], &gradop[grad_offset], &det_j[ip]);

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }
  }
}
//--------------------------------------------------------------------------
void Hex27SCS::grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  generic_grad_op<AlgTraits>(referenceGradWeights_, coords, gradop);

  // copy derivs as well.  These aren't used, but are part of the interface
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      for (int d = 0; d < AlgTraits::nDim_; ++d) {
        deriv(ip,n,d) = referenceGradWeights_(ip,n,d);
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCS::shifted_grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  ThrowRequireMsg(nelem == 1, "P2 elements are processed one-at-a-time");

  *error = 0.0;

  // shape derivatives are stored: just copy
  constexpr int deriv_increment = AlgTraits::nDim_ * AlgTraits::nodesPerElement_;
  constexpr int numShapeDerivs = deriv_increment * AlgTraits::numScsIp_;
  for (int j = 0; j < numShapeDerivs; ++j) {
    deriv[j] = shapeDerivsShift_[j];
  }

  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    const int grad_offset = deriv_increment * ip;
    gradient(coords, &shapeDerivsShift_[grad_offset], &gradop[grad_offset], &det_j[ip]);

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }
  }
}
//--------------------------------------------------------------------------
void Hex27SCS::shifted_grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  generic_grad_op<AlgTraits>(shiftedReferenceGradWeights_, coords, gradop);

  // copy derivs as well.  These aren't used, but are part of the interface
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    for (unsigned n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      for (unsigned d = 0; d < AlgTraits::nDim_; ++d) {
        deriv(ip,n,d) = shiftedReferenceGradWeights_(ip,n,d);
      }
    }
  }
}
//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  ThrowRequireMsg(nelem == 1, "P2 elements are processed one-at-a-time");

  *error = 0.0;
  const int face_offset =  nDim_ * ipsPerFace_ * nodesPerElement_ * face_ordinal;
  const double* offsetFaceDerivs = &expFaceShapeDerivs_[face_offset];

  for (int ip = 0; ip < ipsPerFace_; ++ip) {
    const int grad_offset = nDim_ * nodesPerElement_ * ip;
    gradient(coords, &offsetFaceDerivs[grad_offset], &gradop[grad_offset], &det_j[ip]);

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }
  }
}

void Hex27SCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using traits = AlgTraitsQuad9Hex27;
  const int offset = traits::numFaceIp_ * face_ordinal;
  auto range = std::make_pair(offset, offset + traits::numFaceIp_);
  auto face_weights = Kokkos::subview(expReferenceGradWeights_, range, Kokkos::ALL(), Kokkos::ALL());
  generic_grad_op<AlgTraitsHex27>(face_weights, coords, gradop);
}


//--------------------------------------------------------------------------
//-------- gradient --------------------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::gradient(
  const double* elemNodalCoords,
  const double* shapeDeriv,
  double* grad,
  double* det_j) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0; double dx_ds3 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0; double dy_ds3 = 0.0;
  double dz_ds1 = 0.0;  double dz_ds2 = 0.0; double dz_ds3 = 0.0;

  //compute Jacobian
  for (int node = 0; node < AlgTraits::nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;

    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];
    const double zCoord = elemNodalCoords[vector_offset + 2];

    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];
    const double dn_ds3 = shapeDeriv[vector_offset + 2];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;
    dx_ds3 += dn_ds3 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
    dy_ds3 += dn_ds3 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
    dz_ds3 += dn_ds3 * zCoord;
  }

  *det_j = dx_ds1 * ( dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3 )
         + dy_ds1 * ( dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3 )
         + dz_ds1 * ( dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3 );

  const double inv_det_j = 1.0 / (*det_j);

  const double ds1_dx = inv_det_j*(dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3);
  const double ds2_dx = inv_det_j*(dz_ds1 * dy_ds3 - dy_ds1 * dz_ds3);
  const double ds3_dx = inv_det_j*(dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2);

  const double ds1_dy = inv_det_j*(dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3);
  const double ds2_dy = inv_det_j*(dx_ds1 * dz_ds3 - dz_ds1 * dx_ds3);
  const double ds3_dy = inv_det_j*(dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2);

  const double ds1_dz = inv_det_j*(dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3);
  const double ds2_dz = inv_det_j*(dy_ds1 * dx_ds3 - dx_ds1 * dy_ds3);
  const double ds3_dz = inv_det_j*(dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2);

  // metrics
  for (int node = 0; node < AlgTraits::nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;

    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];
    const double dn_ds3 = shapeDeriv[vector_offset + 2];

    grad[vector_offset + 0] = dn_ds1 * ds1_dx + dn_ds2 * ds2_dx + dn_ds3 * ds3_dx;
    grad[vector_offset + 1] = dn_ds1 * ds1_dy + dn_ds2 * ds2_dy + dn_ds3 * ds3_dy;
    grad[vector_offset + 2] = dn_ds1 * ds1_dz + dn_ds2 * ds2_dz + dn_ds3 * ds3_dz;
  }
}

//--------------------------------------------------------------------------
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCS::gij(
  const double *coords,
  double *gupperij,
  double *glowerij,
  double *deriv)
{
  SIERRA_FORTRAN(threed_gij)
    ( &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gupperij, glowerij);
}
//--------------------------------------------------------------------------
void Hex27SCS::gij(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gupper,
  SharedMemView<DoubleType***>& glower,
  SharedMemView<DoubleType***>& deriv)
{
  generic_gij_3d<AlgTraits>(referenceGradWeights_, coords, gupper, glower);

  for (unsigned ip = 0; ip < 216; ++ip) {
    for (unsigned n = 0; n < 27; ++n) {
      for (unsigned d = 0; d < 3; ++d) {
        deriv(ip,n,d) = referenceGradWeights_(ip,n,d);
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- Mij -------------------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCS::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraits>(numIntPoints_, deriv, coords, metric);
}
//--------------------------------------------------------------------------
void Hex27SCS::Mij(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& metric,
  SharedMemView<DoubleType***>&  /* deriv */)
{
  generic_Mij_3d<AlgTraits>(referenceGradWeights_, coords, metric);
}

//--------------------------------------------------------------------------
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::general_face_grad_op(
  const int  /* face_ordinal */,
  const double *isoParCoord,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  const int ipsPerFace = 1;
  std::array<double,AlgTraits::nodesPerElement_ * AlgTraits::nDim_> faceShapeFuncDerivs;

  hex27_shape_deriv(
    ipsPerFace,
    isoParCoord,
    faceShapeFuncDerivs.data());

  gradient( coords,
            faceShapeFuncDerivs.data(),
            gradop,
            det_j );

  if (det_j[0] < tiny_positive_value()) {
    *error = 1.0;
  }
}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::sidePcoords_to_elemPcoords(
  const int & side_ordinal,
  const int & npoints,
  const double *side_pcoords,
  double *elem_pcoords)
{
  // each ME are -1:1, e.g., hex27:quad93d
  switch (side_ordinal) {
  case 0:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = -1.0;
      elem_pcoords[i*3+2] = side_pcoords[2*i+1];
    }
    break;
  case 1:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 1.0;
      elem_pcoords[i*3+1] = side_pcoords[2*i+0];
      elem_pcoords[i*3+2] = side_pcoords[2*i+1];
    }
    break;
  case 2:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = -side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = 1.0;
      elem_pcoords[i*3+2] = side_pcoords[2*i+1];
    }
    break;
  case 3:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = -1.0;
      elem_pcoords[i*3+1] = side_pcoords[2*i+1];
      elem_pcoords[i*3+2] = side_pcoords[2*i+0];
    }
    break;
  case 4:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = side_pcoords[2*i+1];
      elem_pcoords[i*3+1] = side_pcoords[2*i+0];
      elem_pcoords[i*3+2] = -1.0;
    }
    break;
  case 5:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = side_pcoords[2*i+1];
      elem_pcoords[i*3+2] = 1.0;
    }
    break;
  default:
    throw std::runtime_error("Hex27SCS::sideMap invalid ordinal");
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Quad93DSCS::Quad93DSCS()
  : HexahedralP2Element(),
    surfaceDimension_(2)
{
  // set up the one-dimensional quadrature rule
  set_quadrature_rule();

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
  nodesPerElement_ = nodes1D_ * nodes1D_;

  std::vector<int> nodeMap = {
                               0, 4, 1,   // bottom row of nodes
                               7, 8, 5,   // middle row of nodes
                               3, 6, 2    // top row of nodes
                             };

  auto tensor_map_2D = [=] (int i, int j) { return nodeMap[i+nodes1D_*j]; };

  //1D integration rule per sub-control volume
   numIntPoints_ = (nodes1D_ * nodes1D_) * ( numQuad_ * numQuad_ ); // 36

   // define ip node mappings
   ipNodeMap_.resize(numIntPoints_);
   intgLoc_.resize(numIntPoints_*surfaceDimension_); // size = 72
   intgLocShift_.resize(numIntPoints_*surfaceDimension_); // size = 72
   ipWeight_.resize(numIntPoints_);

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
  shapeFunctions_.resize(numIntPoints_*nodesPerElement_);
  quad9_shape_fcn(numIntPoints_, intgLoc_.data(), shapeFunctions_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::eval_shape_derivs_at_ips()
{
  shapeDerivs_.resize(numIntPoints_*nodesPerElement_*surfaceDimension_);
  quad9_shape_deriv(numIntPoints_, intgLoc_.data(), shapeDerivs_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::eval_shape_functions_at_shifted_ips()
{
  shapeFunctionsShift_.resize(numIntPoints_*nodesPerElement_);
  quad9_shape_fcn(numIntPoints_, intgLocShift_.data(), shapeFunctionsShift_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::eval_shape_derivs_at_shifted_ips()
{
  shapeDerivsShift_.resize(numIntPoints_*nodesPerElement_*surfaceDimension_);
  quad9_shape_deriv(numIntPoints_, intgLocShift_.data(), shapeDerivsShift_.data());
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
  int /*ordinal*/)
{
  // define ip->node mappings for each face (single ordinal);
  return &ipNodeMap_[0];
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

    j[0]=0.125000000000000*(-2.00000000000000*(-1.00000000000000+solcur[1])*x[0]+(2.00000000000000*(1.00000000000000+solcur[1])*(x[1]-x[2])+solcur[2]*(-(y[1]*z[0])+y[2]*z[0]+y[0]*z[1]-y[0]*z[2])));

    j[1]=0.125000000000000*(-2.00000000000000*(1.00000000000000+solcur[0])*x[0]+2.00000000000000*(1.00000000000000+solcur[0])*x[1]-2.00000000000000*(-1.00000000000000+solcur[0])*x[2]+(solcur[2]*(y[2]*(z[0]-z[1])+(-y[0]+y[1])*z[2])));

    j[2]= normal[0];

    j[3]=0.125000000000000*(-2.00000000000000*(-1.00000000000000+solcur[1])*y[0]+(2.00000000000000*(1.00000000000000+solcur[1])*(y[1]-y[2])+solcur[2]*(x[1]*z[0]-x[2]*z[0]-x[0]*z[1]+x[0]*z[2])));

    j[4]=0.125000000000000*(-2.00000000000000*(1.00000000000000+solcur[0])*y[0]+2.00000000000000*(1.00000000000000+solcur[0])*y[1]-2.00000000000000*(-1.00000000000000+solcur[0])*y[2]+(solcur[2]*(x[2]*(-z[0]+z[1])+(x[0]-x[1])*z[2])));

    j[5]= normal[1];

    j[6]=0.125000000000000*((solcur[2]*(-(x[1]*y[0])+x[2]*y[0]+x[0]*y[1]-x[0]*y[2]))-2.00000000000000*((-1.00000000000000+solcur[1])*z[0]-(1.00000000000000+solcur[1])*(z[1]-z[2])));

    j[7]=0.125000000000000*((solcur[2]*(x[2]*(y[0]-y[1])+(-x[0]+x[1])*y[2]))-2.00000000000000*(1.00000000000000+solcur[0])*z[0]+2.00000000000000*(1.00000000000000+solcur[0])*z[1]-2.00000000000000*(-1.00000000000000+solcur[0])*z[2]);

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

    std::vector<double> xtmp = { isoParCoord[0], isoParCoord[1], isoParCoord_2 };
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

  const double n0 = 0.125000000000000*(xi*y[2]*z[0]+y[0]*z[1]+xi*y[0]*z[1]-y[2]*z[1]-
                                       xi*y[0]*z[2]+y[1]*(-((1.00000000000000+xi)*z[0])+
                                                          (1.00000000000000+eta)*z[2])+eta*(y[2]*z[0]-y[2]*z[1]-y[0]*z[2]));

  const double n1 = 0.125000000000000*(-(xi*x[2]*z[0])-x[0]*z[1]-xi*x[0]*z[1]+x[2]*z[1]+
                                       xi*x[0]*z[2]+x[1]*((1.00000000000000+xi)*z[0]-
                                                          (1.00000000000000+eta)*z[2])+eta*(-(x[2]*z[0])+x[2]*z[1]+x[0]*z[2]));

  const double n2 = 0.125000000000000*(xi*x[2]*y[0]+x[0]*y[1]+xi*x[0]*y[1]-x[2]*y[1]-
                                       xi*x[0]*y[2]+x[1]*(-((1.00000000000000+xi)*y[0])+
                                                          (1.00000000000000+eta)*y[2])+eta*(x[2]*y[0]-x[2]*y[1]-x[0]*y[2]));

  normalVector[0] = n0;
  normalVector[1] = n1;
  normalVector[2] = n2;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double Quad93DSCS::parametric_distance(const std::vector<double> &x)
{
  const double ELEM_THICK  = 0.01;
  std::vector<double> y = { std::fabs(x[0]), std::fabs(x[1]), std::fabs(x[2]) };
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
