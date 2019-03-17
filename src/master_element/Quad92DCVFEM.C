/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <master_element/Quad92DCVFEM.h>
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

//-------- quad_gradient_operator ---------------------------------------------------------
template <int nint, int npe>
void quad_gradient_operator(SharedMemView<DoubleType** >& coords,
                            SharedMemView<DoubleType***>& gradop,
                            SharedMemView<DoubleType***>& deriv) {
      
  DoubleType dx_ds1, dx_ds2;
  DoubleType dy_ds1, dy_ds2;

  for (int ki=0; ki<nint; ++ki) {
    dx_ds1 = 0.0;
    dx_ds2 = 0.0;
    dy_ds1 = 0.0;
    dy_ds2 = 0.0;
 
// calculate the jacobian at the integration station -
    for (int kn=0; kn<npe; ++kn) {
      dx_ds1 += deriv(ki,kn,0)*coords(kn,0);
      dx_ds2 += deriv(ki,kn,1)*coords(kn,0);
      dy_ds1 += deriv(ki,kn,0)*coords(kn,1);
      dy_ds2 += deriv(ki,kn,1)*coords(kn,1);
    }

// calculate the determinate of the jacobian at the integration station -
    const DoubleType det_j = dx_ds1*dy_ds2 - dy_ds1*dx_ds2;

// protect against a negative or small value for the determinate of the 
// jacobian. The value of real_min (set in precision.par) represents 
// the smallest Real value (based upon the precision set for this 
// compilation) which the machine can represent - 
    const DoubleType test = stk::math::if_then_else(det_j > 1.e+6*MEconstants::realmin, det_j, 1.0);
    const DoubleType denom = 1.0/test;

// compute the gradient operators at the integration station -
    const DoubleType ds1_dx =  denom*dy_ds2;
    const DoubleType ds2_dx = -denom*dy_ds1;
    const DoubleType ds1_dy = -denom*dx_ds2;
    const DoubleType ds2_dy =  denom*dx_ds1;

    for (int kn=0; kn<npe; ++kn) {
      gradop(ki,kn,0) = deriv(ki,kn,0)*ds1_dx + deriv(ki,kn,1)*ds2_dx;
      gradop(ki,kn,1) = deriv(ki,kn,0)*ds1_dy + deriv(ki,kn,1)*ds2_dy;
    }
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Quad92DSCV::Quad92DSCV()
: QuadrilateralP2Element()
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
}

//--------------------------------------------------------------------------
//-------- set_interior_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad92DSCV::set_interior_info()
{
  //1D integration rule per sub-control volume
  numIntPoints_ = (nodes1D_ * nodes1D_) * ( numQuad_ * numQuad_ ); // 36

  // define ip node mappings
  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_*nDim_); // size = 72
  intgLocShift_.resize(numIntPoints_*nDim_); // size = 72
  ipWeight_.resize(numIntPoints_);

  // tensor product nodes (3x3x3) x tensor product quadrature (2x2x2)
  int vector_index = 0; int scalar_index = 0;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nodeNumber = tensor_product_node_map(k,l);
      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          //integration point location
          intgLoc_[vector_index]     = gauss_point_location(k,i);
          intgLoc_[vector_index + 1] = gauss_point_location(l,j);

          intgLocShift_[vector_index]     = shifted_gauss_point_location(k,i);
          intgLocShift_[vector_index + 1] = shifted_gauss_point_location(l,j);

          //weight
          ipWeight_[scalar_index] = tensor_product_weight(k,l,i,j);

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

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Quad92DSCV::ipNodeMap(
  int /*ordinal*/)
{
 // define scv->node mappings
 return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
DoubleType
Quad92DSCV::jacobian_determinant(
  const SharedMemView<DoubleType**> &elemNodalCoords,      
  const double *POINTER_RESTRICT shapeDerivs) const
{
  DoubleType dx_ds1 = 0.0;  DoubleType dx_ds2 = 0.0;
  DoubleType dy_ds1 = 0.0;  DoubleType dy_ds2 = 0.0;

  for (int node = 0; node < AlgTraits::nodesPerElement_; ++node) {
    const int vector_offset = node * AlgTraits::nDim_;

    const DoubleType xCoord = elemNodalCoords(node,0);
    const DoubleType yCoord = elemNodalCoords(node,1);

    const double dn_ds1  = shapeDerivs[vector_offset + 0];
    const double dn_ds2  = shapeDerivs[vector_offset + 1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
  }

  const DoubleType det_j = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
  return det_j;
}

void Quad92DSCV::determinant(
  SharedMemView<DoubleType**> &coords,
  SharedMemView<DoubleType*>  &volume) 
{
    for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
      const int grad_offset = nDim_ * nodesPerElement_ * ip;

      //weighted jacobian determinant
      const DoubleType det_j = 
        jacobian_determinant(coords, &shapeDerivs_[grad_offset]);

      //apply weight and store to volume
      volume[ip] = ipWeight_[ip] * det_j;
    }
} 

void Quad92DSCV::grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) {
  for (int ki=0,j=0; ki<AlgTraits::numScsIp_; ++ki) {
    for (int kn=0; kn<AlgTraits::nodesPerElement_; ++kn) {
      for (int n=0; n<AlgTraits::nDim_; ++n,++j) {
        deriv(ki,kn,n) = shapeDerivs_[j];
      }
    }
  }
  quad_gradient_operator<AlgTraits::numScsIp_,AlgTraits::nodesPerElement_>(coords, gradop, deriv);
}

void Quad92DSCV::shifted_grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) {
  for (int ki=0,j=0; ki<AlgTraits::numScsIp_; ++ki) {
    for (int kn=0; kn<AlgTraits::nodesPerElement_; ++kn) {
      for (int n=0; n<AlgTraits::nDim_; ++n,++j) {
        deriv(ki,kn,n) = shapeDerivsShift_[j];
      }
    }
  }
  quad_gradient_operator<AlgTraits::numScsIp_,AlgTraits::nodesPerElement_>(coords, gradop, deriv);
}

void Quad92DSCV::determinant(
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
      if (det_j < tiny_positive_value()) {
        *error = 1.0;
      }
    }

}

//--------------------------------------------------------------------------
//-------- jacobian_determinant --------------------------------------------
//--------------------------------------------------------------------------
double
Quad92DSCV::jacobian_determinant(
  const double *POINTER_RESTRICT elemNodalCoords,
  const double *POINTER_RESTRICT shapeDerivs) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0;

  for (int node = 0; node < AlgTraits::nodesPerElement_; ++node) {
    const int vector_offset = node * nDim_;

    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];

    const double dn_ds1  = shapeDerivs[vector_offset + 0];
    const double dn_ds2  = shapeDerivs[vector_offset + 1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
  }

  const double det_j = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;

  return det_j;
}

//--------------------------------------------------------------------------
//-------- Metric Tensor Mij------------------------------------------------
//--------------------------------------------------------------------------
// This function computes the metric tensor Mij = (J J^T)^(1/2) where J is
// the Jacobian.  This is needed for the UT-A Hybrid LES model.  For
// reference please consult the Nalu theory manual description of the UT-A
// Hybrid LES model or S. Haering's PhD thesis: Anisotropic hybrid turbulence 
// modeling with specific application to the simulation of pulse-actuated 
// dynamic stall control.
//--------------------------------------------------------------------------
void Quad92DSCV::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_2d<AlgTraitsQuad9_2D>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void Quad92DSCV::Mij(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& metric,
  SharedMemView<DoubleType***>& deriv)
{
  generic_Mij_2d<AlgTraitsQuad9_2D>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Quad92DSCS::Quad92DSCS()
  : QuadrilateralP2Element()
{
  // set up the one-dimensional quadrature rule
  set_quadrature_rule();

  // set up integration rule and relevant maps for scs
  set_interior_info();

  // set up integration rule and relevant maps for faces
  set_boundary_info();

  // compute and save shape functions and derivatives at ips
  eval_shape_functions_at_ips();
  eval_shape_derivs_at_ips();
  eval_shape_derivs_at_face_ips();

  eval_shape_functions_at_shifted_ips();
  eval_shape_derivs_at_shifted_ips();
}

//--------------------------------------------------------------------------
//-------- set_interior_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad92DSCS::set_interior_info()
{
  const int linesPerDirection = nodes1D_ - 1; // 2
  const int ipsPerLine = numQuad_ * nodes1D_;
  const int numLines = linesPerDirection * nDim_;

  numIntPoints_ = numLines * ipsPerLine; // 24

  // define L/R mappings
  lrscv_.resize(2*numIntPoints_); // size = 48

  // standard integration location
  intgLoc_.resize(numIntPoints_*nDim_); // size = 48

  // shifted
  intgLocShift_.resize(numIntPoints_*nDim_);

  ipInfo_.resize(numIntPoints_);

  // a list of the scs locations in 1D
  const std::vector<double> scsLoc =  { -scsDist_, scsDist_ };

  // correct orientation for area vector
  const std::vector<double> orientation = { -1.0, +1.0 };

  // specify integration point locations in a dimension-by-dimension manner

  //u-direction
  int vector_index = 0;
  int lrscv_index = 0;
  int scalar_index = 0;
  for (int m = 0; m < linesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {

      int leftNode; int rightNode;
      if (m == 0) {
        leftNode  = tensor_product_node_map(l,m);
        rightNode = tensor_product_node_map(l,m + 1);
      }
      else {
        leftNode  = tensor_product_node_map(l,m + 1);
        rightNode = tensor_product_node_map(l,m);
      }

      for (int j = 0; j < numQuad_; ++j) {

        lrscv_[lrscv_index] = leftNode;
        lrscv_[lrscv_index + 1] = rightNode;

        intgLoc_[vector_index] = gauss_point_location(l,j);
        intgLoc_[vector_index + 1] = scsLoc[m];

        intgLocShift_[vector_index] = shifted_gauss_point_location(l,j);
        intgLocShift_[vector_index + 1] = scsLoc[m];

        //compute the quadrature weight
        ipInfo_[scalar_index].weight = orientation[m]*tensor_product_weight(l,j);

        //direction
        ipInfo_[scalar_index].direction = Jacobian::T_DIRECTION;

        ++scalar_index;
        lrscv_index += 2;
        vector_index += nDim_;
      }
    }
  }

  //t-direction
  for (int m = 0; m < linesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {

      int leftNode; int rightNode;
      if (m == 0) {
        leftNode  = tensor_product_node_map(m,l);
        rightNode = tensor_product_node_map(m+1,l);
      }
      else {
        leftNode  = tensor_product_node_map(m+1,l);
        rightNode = tensor_product_node_map(m,l);
      }

      for (int j = 0; j < numQuad_; ++j) {

        lrscv_[lrscv_index]   = leftNode;
        lrscv_[lrscv_index+1] = rightNode;

        intgLoc_[vector_index] = scsLoc[m];
        intgLoc_[vector_index+1] = gauss_point_location(l,j);

        intgLocShift_[vector_index] = scsLoc[m];
        intgLocShift_[vector_index+1] = shifted_gauss_point_location(l,j);

        //compute the quadrature weight
        ipInfo_[scalar_index].weight = -orientation[m]*tensor_product_weight(l,j);

        //direction
        ipInfo_[scalar_index].direction = Jacobian::S_DIRECTION;

        ++scalar_index;
        lrscv_index += 2;
        vector_index += nDim_;
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- set_boundary_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad92DSCS::set_boundary_info()
{
  const int numFaces = 2*nDim_;
  const int nodesPerFace = nodes1D_;
  ipsPerFace_ = nodesPerFace*numQuad_;

  const int numFaceIps = numFaces*ipsPerFace_; // 24 -- different from numIntPoints_ for p > 2 ?

  oppFace_.resize(numFaceIps);
  ipNodeMap_.resize(numFaceIps);
  oppNode_.resize(numFaceIps);
  intgExpFace_.resize(numFaceIps*nDim_);

  const std::vector<int> stkFaceNodeMap = {
                                            0, 4, 1, //face 0, bottom face
                                            1, 5, 2, //face 1, right face
                                            2, 6, 3, //face 2, top face  -- reversed order
                                            3, 7, 0  //face 3, left face -- reversed order
                                          };

  auto face_node_number = [=] (int number,int faceOrdinal)
  {
    return stkFaceNodeMap[number+nodes1D_*faceOrdinal];
  };

  const std::vector<int> faceToLine = { 0, 3, 1, 2 };
  const std::vector<double> faceLoc = {-1.0, +1.0, +1.0, -1.0};

  int scalar_index = 0; int vector_index = 0;
  int faceOrdinal = 0; //bottom face
  int oppFaceIndex = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = tensor_product_node_map(k,1);

    for (int j = 0; j < numQuad_; ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index]   = intgLoc_[oppFace_[scalar_index]*nDim_+0];
      intgExpFace_[vector_index+1] = faceLoc[faceOrdinal];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 1; //right face
  oppFaceIndex = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = tensor_product_node_map(1,k);

    for (int j = 0; j < numQuad_; ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index]   = faceLoc[faceOrdinal];
      intgExpFace_[vector_index+1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }


  faceOrdinal = 2; //top face
  oppFaceIndex = 0;
  //NOTE: this face is reversed
  for (int k = nodes1D_-1; k >= 0; --k) {
    const int nearNode = face_node_number(nodes1D_-k-1,faceOrdinal);
    int oppNode = tensor_product_node_map(k,1);
    for (int j = 0; j < numQuad_; ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index] = intgLoc_[oppFace_[scalar_index]*nDim_+0];
      intgExpFace_[vector_index+1] = faceLoc[faceOrdinal];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 3; //left face
  oppFaceIndex = 0;
  //NOTE: this faces is reversed
  for (int k = nodes1D_-1; k >= 0; --k) {
    const int nearNode = face_node_number(nodes1D_-k-1,faceOrdinal);
    int oppNode = tensor_product_node_map(1,k);
    for (int j = 0; j < numQuad_; ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index]   = faceLoc[faceOrdinal];
      intgExpFace_[vector_index+1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }
}


//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Quad92DSCS::ipNodeMap(
  int ordinal)
{
  // define ip->node mappings for each face (ordinal); 
  return &ipNodeMap_[ordinal*ipsPerFace_];
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
const int *
Quad92DSCS::side_node_ordinals(
  int ordinal)
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return &sideNodeOrdinals_[ordinal*3];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void 
Quad92DSCS::determinant(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType**>& areav) 
{
  //returns the normal vector (dyds,-dxds) for constant t curves
  //returns the normal vector (dydt,-dxdt) for constant s curves

  constexpr int dim = AlgTraits::nDim_;
  constexpr int ipsPerDirection = AlgTraits::numScsIp_ / dim;
  static_assert ( ipsPerDirection * dim == AlgTraits::numScsIp_, "Number of ips incorrect");

  constexpr int deriv_increment = dim * AlgTraits::nodesPerElement_;

  int index = 0;

   //returns the normal vector x_u x x_s for constant t surfaces
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    ThrowAssert(ipInfo_[index].direction == Jacobian::T_DIRECTION);
    area_vector<Jacobian::T_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav(index,0));
    ++index;
  }

  //returns the normal vector x_t x x_u for constant s curves
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    ThrowAssert(ipInfo_[index].direction == Jacobian::S_DIRECTION);
    area_vector<Jacobian::S_DIRECTION>(coords, &shapeDerivs_[deriv_increment * index], &areav(index,0));
    ++index;
  }

  // Multiply with the integration point weighting
  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
    double weight = ipInfo_[ip].weight;
    areav(ip,0) *= weight;
    areav(ip,1) *= weight;
  }
}

void
Quad92DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  //returns the normal vector (dyds,-dxds) for constant t curves
  //returns the normal vector (dydt,-dxdt) for constant s curves

  ThrowRequireMsg(nelem == 1, "P2 elements are processed one-at-a-time");

  constexpr int dim = AlgTraits::nDim_;
  constexpr int ipsPerDirection = AlgTraits::numScsIp_ / dim;
  static_assert ( ipsPerDirection * dim == AlgTraits::numScsIp_, "Number of ips incorrect");

  constexpr int deriv_increment = dim * AlgTraits::nodesPerElement_;

  int index = 0;

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
  }

  *error = 0; // no error checking available
}

void Quad92DSCS::grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) {
  for (int ki=0,j=0; ki<AlgTraits::numScsIp_; ++ki) {
    for (int kn=0; kn<AlgTraits::nodesPerElement_; ++kn) {
      for (int n=0; n<AlgTraits::nDim_; ++n,++j) {
        deriv(ki,kn,n) = shapeDerivs_[j];
      }
    }
  }
  quad_gradient_operator<AlgTraits::numScsIp_,AlgTraits::nodesPerElement_>(coords, gradop, deriv);
}

void Quad92DSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  constexpr int numShapeDerivs = AlgTraits::numScsIp_*AlgTraits::nodesPerElement_*AlgTraits::nDim_;
  for (int j = 0; j < numShapeDerivs; ++j) {
    deriv[j] = shapeDerivs_[j];
  }

  SIERRA_FORTRAN(quad_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative area.." << std::endl;

}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void Quad92DSCS::shifted_grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) {
  for (int ki=0,j=0; ki<AlgTraits::numScsIp_; ++ki) {
    for (int kn=0; kn<AlgTraits::nodesPerElement_; ++kn) {
      for (int n=0; n<AlgTraits::nDim_; ++n,++j) {
        deriv(ki,kn,n) = shapeDerivsShift_[j];
      }
    }
  }
  quad_gradient_operator<AlgTraits::numScsIp_,AlgTraits::nodesPerElement_>(coords, gradop, deriv);
}

void Quad92DSCS::shifted_grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  constexpr int numShapeDerivs = AlgTraits::numScsIp_*AlgTraits::nodesPerElement_*AlgTraits::nDim_;
  for (int j = 0; j < numShapeDerivs; ++j) {
    deriv[j] = shapeDerivsShift_[j];
  }

  SIERRA_FORTRAN(quad_gradient_operator)
  ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative area.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void Quad92DSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using traits = AlgTraitsEdge32DQuad92D;

  constexpr int derivSize = traits::numFaceIp_ * traits::nodesPerElement_ * traits::nDim_;
  DoubleType psi[derivSize];
  SharedMemView<DoubleType***> deriv(psi, traits::numFaceIp_, traits::nodesPerElement_, traits::nDim_);
  constexpr int offset = traits::nDim_*traits::numFaceIp_*traits::nodesPerElement_;
  const double* exp_face = &expFaceShapeDerivs_[offset*face_ordinal];
  for (int i=0,n=0; i<traits::numFaceIp_; ++i)
    for (int j=0; j<traits::nodesPerElement_; ++j)
      for (int k=0; k<traits::nDim_; ++k,++n)
          deriv(i,j,k) = exp_face[n];
  generic_grad_op<traits>(deriv, coords, gradop);
}

void Quad92DSCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  ThrowRequireMsg(nelem == 1, "P2 elements are processed one-at-a-time");

  int lerr = 0;

  const int nface = 1;
  const int face_offset =  nDim_ * ipsPerFace_ * nodesPerElement_ * face_ordinal;
  double* offsetFaceDerivs = &expFaceShapeDerivs_[face_offset];

  for (int ip = 0; ip < ipsPerFace_; ++ip) {
    const int grad_offset = nDim_ * nodesPerElement_ * ip;

    SIERRA_FORTRAN(quad_gradient_operator)
    ( & nface,
        &nodesPerElement_,
        &nface,
        &offsetFaceDerivs[grad_offset],
        coords,
        &gradop[grad_offset],
        &det_j[ip],
        error,
        &lerr
    );

    if (det_j[ip] < tiny_positive_value() || lerr != 0) {
      *error = 1.0;
    }
  }

}

//--------------------------------------------------------------------------
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
void Quad92DSCS::gij(
  SharedMemView<DoubleType** >& coords,
  SharedMemView<DoubleType***>& gupper,
  SharedMemView<DoubleType***>& glower,
  SharedMemView<DoubleType***>& deriv) {

  constexpr int npe  = AlgTraits::nodesPerElement_;
  constexpr int nint = AlgTraits::numScsIp_;

  DoubleType dx_ds[2][2], ds_dx[2][2];

  for (int ki=0; ki<nint; ++ki) {
    dx_ds[0][0] = 0.0; 
    dx_ds[0][1] = 0.0; 
    dx_ds[1][0] = 0.0; 
    dx_ds[1][1] = 0.0; 
 
// calculate the jacobian at the integration station -
    for (int kn=0; kn<npe; ++kn) {
      dx_ds[0][0] += deriv(ki,kn,0)*coords(kn,0);
      dx_ds[0][1] += deriv(ki,kn,1)*coords(kn,0);
      dx_ds[1][0] += deriv(ki,kn,0)*coords(kn,1);
      dx_ds[1][1] += deriv(ki,kn,1)*coords(kn,1);
    }    
// calculate the determinate of the jacobian at the integration station -
    const DoubleType det_j = dx_ds[0][0]*dx_ds[1][1] - dx_ds[1][0]*dx_ds[0][1];

// clip
    const DoubleType test = stk::math::if_then_else(det_j > 1.e+6*MEconstants::realmin, det_j, 1.0);
    const DoubleType denom = 1.0/test;

// compute the inverse jacobian
    ds_dx[0][0] =  dx_ds[1][1]*denom;
    ds_dx[0][1] = -dx_ds[0][1]*denom;
    ds_dx[1][0] = -dx_ds[1][0]*denom;
    ds_dx[1][1] =  dx_ds[0][0]*denom;
      
    for (int i=0; i<2; ++i) {
      for (int j=0; j<2; ++j) {
        gupper(ki,j,i) = dx_ds[i][0]*dx_ds[j][0]+dx_ds[i][1]*dx_ds[j][1];
        glower(ki,j,i) = ds_dx[0][i]*ds_dx[0][j]+ds_dx[1][i]*ds_dx[1][j];
      }    
    }    
  }
}

void Quad92DSCS::gij(
  const double *coords,
  double *gupperij,
  double *glowerij,
  double *deriv)
{
  SIERRA_FORTRAN(twod_gij)
    ( &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gupperij, glowerij);
}

//--------------------------------------------------------------------------
//-------- Metric Tensor Mij------------------------------------------------
//--------------------------------------------------------------------------
// This function computes the metric tensor Mij = (J J^T)^(1/2) where J is
// the Jacobian.  This is needed for the UT-A Hybrid LES model.  For
// reference please consult the Nalu theory manual description of the UT-A
// Hybrid LES model or S. Haering's PhD thesis: Anisotropic hybrid turbulence 
// modeling with specific application to the simulation of pulse-actuated 
// dynamic stall control.
//--------------------------------------------------------------------------
void Quad92DSCS::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_2d<AlgTraitsQuad9_2D>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void Quad92DSCS::Mij(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& metric,
  SharedMemView<DoubleType***>& deriv)
{
  generic_Mij_2d<AlgTraitsQuad9_2D>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
const int *
Quad92DSCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_.data();
}

//--------------------------------------------------------------------------
//-------- opposingNodes ---------------------------------------------------
//--------------------------------------------------------------------------
int
Quad92DSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*ipsPerFace_+node];
}

//--------------------------------------------------------------------------
//-------- opposingFace ----------------------------------------------------
//--------------------------------------------------------------------------
int
Quad92DSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*ipsPerFace_+node];
}

//--------------------------------------------------------------------------
//-------- area_vector -----------------------------------------------------
//--------------------------------------------------------------------------
template <Jacobian::Direction direction> void
Quad92DSCS::area_vector(
  const SharedMemView<DoubleType**>& elemNodalCoords,             
  double *POINTER_RESTRICT shapeDeriv,
  DoubleType *POINTER_RESTRICT normalVec ) const
{
  constexpr int s1Component = (direction == Jacobian::S_DIRECTION) ?
      Jacobian::T_DIRECTION : Jacobian::S_DIRECTION;

  DoubleType dxdr = 0.0;  DoubleType dydr = 0.0;
  for (int node = 0; node < AlgTraits::nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const DoubleType xCoord = elemNodalCoords(node,0);
    const DoubleType yCoord = elemNodalCoords(node,1);

    dxdr += shapeDeriv[vector_offset+s1Component] * xCoord;
    dydr += shapeDeriv[vector_offset+s1Component] * yCoord;
  }

  normalVec[0] =  dydr;
  normalVec[1] = -dxdr;
}
template <Jacobian::Direction direction> void
Quad92DSCS::area_vector(
  const double *POINTER_RESTRICT elemNodalCoords,
  double *POINTER_RESTRICT shapeDeriv,
  double *POINTER_RESTRICT normalVec ) const
{
  constexpr int s1Component = (direction == Jacobian::S_DIRECTION) ?
      Jacobian::T_DIRECTION : Jacobian::S_DIRECTION;

  double dxdr = 0.0;  double dydr = 0.0;
  for (int node = 0; node < AlgTraits::nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];

    dxdr += shapeDeriv[vector_offset+s1Component] * xCoord;
    dydr += shapeDeriv[vector_offset+s1Component] * yCoord;
  }

  normalVec[0] =  dydr;
  normalVec[1] = -dxdr;
}

} // namespace nalu
} // namespace sierra
