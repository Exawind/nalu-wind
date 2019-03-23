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
  : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;
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
   return stkNodeMap_[k][j][i];
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
  hex27_shape_fcn(numIntPoints_, intgLoc_, shapeFunctions_);
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_derivs_at_ips()
{
  hex27_shape_deriv(numIntPoints_, intgLoc_, shapeDerivs_);
}

//--------------------------------------------------------------------------
//-------- eval_shape_functions_at_shifted_ips -----------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_functions_at_shifted_ips()
{
  hex27_shape_fcn(numIntPoints_, intgLocShift_, shapeFunctionsShift_);
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_derivs_at_shifted_ips()
{
  hex27_shape_deriv(numIntPoints_, intgLocShift_, shapeDerivsShift_);
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_face_ips -----------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_derivs_at_face_ips()
{
  hex27_shape_deriv(
    numFaceIps_,
    intgExpFace_,
    expFaceShapeDerivs_
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
  MasterElement::intgLocShift_.assign(intgLocShift_, numIntPoints_*nDim_+intgLocShift_);
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCV::ipNodeMap(
  int /*ordinal*/) const
{
  // define scv->node mappings
  return ipNodeMap_;
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
  shiftedInterpWeights_ = copy_interpolation_weights_to_view<InterpWeightType>(shapeFunctionsShift_);

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

  // a list of the scs locations in 1D
  const double scsLoc[2] = { -scsDist_, scsDist_ };

  // correct orientation of area vector
  const double orientation[2] = {-1.0, +1.0};

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
  MasterElement::intgLocShift_.assign(intgLocShift_, numIntPoints_*nDim_+intgLocShift_);
}

//--------------------------------------------------------------------------
//-------- set_boundary_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::set_boundary_info()
{
  // face ordinal to tensor-product style node ordering
  const int stkFaceNodeMap[54] = {
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
    return stkFaceNodeMap[i + nodes1D_ * j + nodesPerFace_ * faceOrdinal];
  };

  // map face ip ordinal to nearest sub-control surface ip ordinal
  // sub-control surface renumbering
  const int faceToSurface[6] = { 2, 5, 3, 4, 0, 1 };
  auto opp_face_map = [=] ( int k, int l, int i, int j, int face_index)
  {
    int face_offset = faceToSurface[face_index] * ipsPerFace_;

    int node_index = k + nodes1D_ * l;
    int node_offset = node_index * (numQuad_ * numQuad_);

    int ip_index = face_offset+node_offset+i+numQuad_*j;

    return ip_index;
  };

  // location of the faces in the correct order
  const double faceLoc[6] = {-1.0, +1.0, +1.0, -1.0, -1.0, +1.0};

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
  MasterElement::intgExpFace_.assign(intgExpFace_, numFaceIps_*nDim_+intgExpFace_);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_;
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCS::ipNodeMap(
  int ordinal) const
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
  return sideNodeOrdinals_[ordinal];
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
  const int numIntPoints = numIntPoints_;
  const int nodesPerElement = nodesPerElement_;
  SIERRA_FORTRAN(threed_gij)
    ( &nodesPerElement,
      &numIntPoints,
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

}
}
