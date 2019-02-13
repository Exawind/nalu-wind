/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <master_element/Hex8FEM.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/TensorOps.h>

#include <FORTRAN_Proto.h>
#include <NaluEnv.h>

#include <cmath>
#include <iostream>

namespace sierra{
namespace nalu{

namespace {

  template<typename AlgTraits, typename GradViewType,  typename CoordViewType,
           typename DetjType, typename OutputViewType>
  void generic_grad_op_3d(
    GradViewType referenceGradWeights,
    CoordViewType coords,
    OutputViewType weights,
    DetjType detj)
  {
    /**
     * Given the reference gradient weights evaluated at the integration points and the cooordinates,
     * this method computes
     *     \nabla_\mathbf{x}|_{\alpha}  as \left( J^{-T} \nabla_{\mathbf{x}^\hat} \right)|_\alpha,
     *
     *  This operation can be specialized for efficiency on hex topologies (as tensor-contractions)
     *  or on tets (since the gradient is independent of \alpha).  But this can work as a fallback.
     *
     *  Also saves detj at the integration points---useful for FEM but not used at all in CVFEM
     */

    using ftype = typename CoordViewType::value_type;
    static_assert(std::is_same<ftype, typename GradViewType::value_type>::value, "Incompatiable value type for views");
    static_assert(std::is_same<ftype, typename OutputViewType::value_type>::value, "Incompatiable value type for views");
    static_assert(GradViewType::Rank == 3, "grad view assumed to be 3D");
    static_assert(CoordViewType::Rank == 2, "Coordinate view assumed to be 2D");
    static_assert(OutputViewType::Rank == 3, "Weight view assumed to be 3D");
    static_assert(AlgTraits::nDim_ == 3, "3D method");

    ThrowAssert(AlgTraits::nodesPerElement_ == referenceGradWeights.extent(1));
    ThrowAssert(AlgTraits::nDim_ == referenceGradWeights.extent(2));
    ThrowAssert(weights.extent(0) == referenceGradWeights.extent(0));
    ThrowAssert(weights.extent(1) == referenceGradWeights.extent(1));
    ThrowAssert(weights.extent(2) == referenceGradWeights.extent(2));

    for (unsigned ip = 0; ip < AlgTraits::numGp_; ++ip) {
      ftype jact[3][3] = { {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} };

      // compute Jacobian.  Stash away a local copy of the reference weights
      // since they're be used again soon
      ftype refGrad[AlgTraits::nodesPerElement_][3];
      for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
        refGrad[n][0] = referenceGradWeights(ip, n, 0);
        refGrad[n][1] = referenceGradWeights(ip, n, 1);
        refGrad[n][2] = referenceGradWeights(ip, n, 2);

        jact[0][0] += refGrad[n][0] * coords(n, 0);
        jact[0][1] += refGrad[n][1] * coords(n, 0);
        jact[0][2] += refGrad[n][2] * coords(n, 0);

        jact[1][0] += refGrad[n][0] * coords(n, 1);
        jact[1][1] += refGrad[n][1] * coords(n, 1);
        jact[1][2] += refGrad[n][2] * coords(n, 1);

        jact[2][0] += refGrad[n][0] * coords(n, 2);
        jact[2][1] += refGrad[n][1] * coords(n, 2);
        jact[2][2] += refGrad[n][2] * coords(n, 2);
      }

      ftype invJac[3][3];
      adjugate_matrix33(jact, invJac);

      detj(ip) = jact[0][0] * invJac[0][0] + jact[1][0] * invJac[1][0] + jact[2][0] * invJac[2][0];
      ThrowAssertMsg(stk::simd::are_any(detj(ip) > +tiny_positive_value()),"Problem with determinant");

      const ftype inv_detj = ftype(1.0) / detj(ip);
      for (int d_outer = 0; d_outer < 3; ++d_outer) {
        for (int d_inner = 0; d_inner < 3; ++d_inner) {
          invJac[d_outer][d_inner] *= inv_detj;
        }
      }

      for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
        weights(ip, n, 0) = invJac[0][0] * refGrad[n][0] + invJac[0][1] * refGrad[n][1] + invJac[0][2] * refGrad[n][2];
        weights(ip, n, 1) = invJac[1][0] * refGrad[n][0] + invJac[1][1] * refGrad[n][1] + invJac[1][2] * refGrad[n][2];
        weights(ip, n, 2) = invJac[2][0] * refGrad[n][0] + invJac[2][1] * refGrad[n][1] + invJac[2][2] * refGrad[n][2];
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Hex8FEM::Hex8FEM()
  : MasterElement()
{
  ndim(AlgTraits::nDim_);
  nodesPerElement_ = 8;
  numIntPoints_ = 8;

  // weights; -1:1
  weights_.resize(numIntPoints_);
  for ( int k = 0; k < numIntPoints_; ++k )
    weights_[k] = 1.0;

  // standard integration location +/ sqrt(3)/3
  intgLoc_.resize(24);
  const double gIP = std::sqrt(3.0)/3.0;
  intgLoc_[0]  = -gIP; intgLoc_[1]  = -gIP; intgLoc_[2]  = -gIP; 
  intgLoc_[3]  = +gIP; intgLoc_[4]  = -gIP; intgLoc_[5]  = -gIP; 
  intgLoc_[6]  = +gIP; intgLoc_[7]  = +gIP; intgLoc_[8]  = -gIP;
  intgLoc_[9]  = -gIP; intgLoc_[10] = +gIP; intgLoc_[11] = -gIP;
  intgLoc_[12] = -gIP; intgLoc_[13] = -gIP; intgLoc_[14] = +gIP;
  intgLoc_[15] = +gIP; intgLoc_[16] = -gIP; intgLoc_[17] = +gIP;
  intgLoc_[18] = +gIP; intgLoc_[19] = +gIP; intgLoc_[20] = +gIP;
  intgLoc_[21] = -gIP; intgLoc_[22] = +gIP; intgLoc_[23] = +gIP;

  // shifted to nodes (Gauss Lobatto)
  intgLocShift_.resize(24);
  const double glIP = 1.0;
  intgLocShift_[0]  = -glIP; intgLocShift_[1]  = -glIP; intgLocShift_[2]  = -glIP; 
  intgLocShift_[3]  = +glIP; intgLocShift_[4]  = -glIP; intgLocShift_[5]  = -glIP; 
  intgLocShift_[6]  = +glIP; intgLocShift_[7]  = +glIP; intgLocShift_[8]  = -glIP;
  intgLocShift_[9]  = -glIP; intgLocShift_[10] = +glIP; intgLocShift_[11] = -glIP;
  intgLocShift_[12] = -glIP; intgLocShift_[13] = -glIP; intgLocShift_[14] = +glIP;
  intgLocShift_[15] = +glIP; intgLocShift_[16] = -glIP; intgLocShift_[17] = +glIP;
  intgLocShift_[18] = +glIP; intgLocShift_[19] = +glIP; intgLocShift_[20] = +glIP;
  intgLocShift_[21] = -glIP; intgLocShift_[22] = +glIP; intgLocShift_[23] = +glIP;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Hex8FEM::~Hex8FEM()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- determinant ------------------ n/a ------------------------------
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void Hex8FEM::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  hex8_fem_derivative(numIntPoints_, &intgLoc_[0], deriv);
  
  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr ); 
}


//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void Hex8FEM::grad_op_fem(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv,
  SharedMemView<DoubleType*>&det_j)
{
  hex8_fem_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op_3d<AlgTraits>(deriv, coords, gradop, det_j);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void Hex8FEM::shifted_grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  hex8_fem_derivative(numIntPoints_, &intgLocShift_[0], deriv);
  
  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr ); 
}


//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void Hex8FEM::shifted_grad_op_fem(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv,
  SharedMemView<DoubleType*>&det_j)
{
  hex8_fem_derivative(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op_3d<AlgTraits>(deriv, coords, gradop, det_j);
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void Hex8FEM::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  int npf = 4;
  int ndim = 3;
  
  const int nface = 1;
  double dpsi[24];
  double grad[24];
  
  for ( int n=0; n<nelem; n++ ) {
    
    for ( int k=0; k<npf; k++ ) {

      const int row = 12*face_ordinal + k*ndim;
      hex8_fem_derivative
        ( nface,
          &intgExpFace_[row], dpsi );
      
      SIERRA_FORTRAN(hex_gradient_operator)
        ( &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[24*n], grad, &det_j[npf*n+k], error, &lerr );
      
      if ( lerr )
        NaluEnv::self().naluOutput() << "sorry, issue with face_grad_op.." << std::endl;
      
      for ( int j=0; j<24; j++) {
        gradop[k*nelem*24+n*24+j] = grad[j];
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::general_shape_fcn(
  const int numIp,
  const double *isoParCoord,
  double *shpfc)
{
  hex8_fem_shape_fcn(numIp,isoParCoord,shpfc);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::shape_fcn(double *shpfc)
{
  hex8_fem_shape_fcn(numIntPoints_,&intgLoc_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::shifted_shape_fcn(double *shpfc)
{
  hex8_fem_shape_fcn(numIntPoints_,&intgLocShift_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  hex8_fem_shape_fcn(numIntPoints_,&intgLoc_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::shifted_shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  hex8_fem_shape_fcn(numIntPoints_,&intgLocShift_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
void Hex8FEM::gij(
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
//-------- hex8_fem_shape_fcn ----------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::hex8_fem_shape_fcn(
  const int  &numIp,
  const double *isoParCoord, 
  double *shpfc)
{
  // -1:1 isoparametric range
  const int npe = nodesPerElement_;
  for ( int ip = 0; ip < numIp; ++ip ) {
    
    const int rowIpc = 3*ip;
    const int rowSfc = npe*ip;
    
    const double s1 = isoParCoord[rowIpc];
    const double s2 = isoParCoord[rowIpc+1];
    const double s3 = isoParCoord[rowIpc+2];
    shpfc[rowSfc  ] = 0.125*(1.0-s1)*(1.0-s2)*(1.0-s3);
    shpfc[rowSfc+1] = 0.125*(1.0+s1)*(1.0-s2)*(1.0-s3);
    shpfc[rowSfc+2] = 0.125*(1.0+s1)*(1.0+s2)*(1.0-s3);
    shpfc[rowSfc+3] = 0.125*(1.0-s1)*(1.0+s2)*(1.0-s3);
    shpfc[rowSfc+4] = 0.125*(1.0-s1)*(1.0-s2)*(1.0+s3);
    shpfc[rowSfc+5] = 0.125*(1.0+s1)*(1.0-s2)*(1.0+s3);
    shpfc[rowSfc+6] = 0.125*(1.0+s1)*(1.0+s2)*(1.0+s3);
    shpfc[rowSfc+7] = 0.125*(1.0-s1)*(1.0+s2)*(1.0+s3);
  }
}

//--------------------------------------------------------------------------
//-------- hex8_fem_shape_fcn ----------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::hex8_fem_shape_fcn(
  const int  &numIp,
  const double *isoParCoord,
  SharedMemView<DoubleType**> shpfc)
{
  // -1:1 isoparametric range
  for ( int ip = 0; ip < numIp; ++ip ) {
    const int rowIpc = 3*ip;
    const DoubleType s1 = isoParCoord[rowIpc+0];
    const DoubleType s2 = isoParCoord[rowIpc+1];
    const DoubleType s3 = isoParCoord[rowIpc+2];
    shpfc(ip,0) = 0.125*(1.0-s1)*(1.0-s2)*(1.0-s3);
    shpfc(ip,1) = 0.125*(1.0+s1)*(1.0-s2)*(1.0-s3);
    shpfc(ip,2) = 0.125*(1.0+s1)*(1.0+s2)*(1.0-s3);
    shpfc(ip,3) = 0.125*(1.0-s1)*(1.0+s2)*(1.0-s3);
    shpfc(ip,4) = 0.125*(1.0-s1)*(1.0-s2)*(1.0+s3);
    shpfc(ip,5) = 0.125*(1.0+s1)*(1.0-s2)*(1.0+s3);
    shpfc(ip,6) = 0.125*(1.0+s1)*(1.0+s2)*(1.0+s3);
    shpfc(ip,7) = 0.125*(1.0-s1)*(1.0+s2)*(1.0+s3);
  }
}

//--------------------------------------------------------------------------
//-------- hex8_fem_derivative ---------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::hex8_fem_derivative(
  const int npt, const double* par_coord,
  double* deriv)
{
  for (int i = 0; i < npt; ++i) {
    deriv[i*nodesPerElement_*3    ] = -0.125*(1.0-par_coord[i*3+1])*(1.0-par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 3] =  0.125*(1.0-par_coord[i*3+1])*(1.0-par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 6] =  0.125*(1.0+par_coord[i*3+1])*(1.0-par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 9] = -0.125*(1.0+par_coord[i*3+1])*(1.0-par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 12] = -0.125*(1.0-par_coord[i*3+1])*(1.0+par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 15] =  0.125*(1.0-par_coord[i*3+1])*(1.0+par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 18] =  0.125*(1.0+par_coord[i*3+1])*(1.0+par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 21] = -0.125*(1.0+par_coord[i*3+1])*(1.0+par_coord[i*3+2]);

    deriv[i*nodesPerElement_*3 + 1] = -0.125*(1.0-par_coord[i*3])*(1.0-par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 4] = -0.125*(1.0+par_coord[i*3])*(1.0-par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 7] =  0.125*(1.0+par_coord[i*3])*(1.0-par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 10] =  0.125*(1.0-par_coord[i*3])*(1.0-par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 13] = -0.125*(1.0-par_coord[i*3])*(1.0+par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 16] = -0.125*(1.0+par_coord[i*3])*(1.0+par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 19] =  0.125*(1.0+par_coord[i*3])*(1.0+par_coord[i*3+2]);
    deriv[i*nodesPerElement_*3 + 22] =  0.125*(1.0-par_coord[i*3])*(1.0+par_coord[i*3+2]);

    deriv[i*nodesPerElement_*3 + 2] = -0.125*(1.0-par_coord[i*3])*(1.0-par_coord[i*3+1]);
    deriv[i*nodesPerElement_*3 + 5] = -0.125*(1.0+par_coord[i*3])*(1.0-par_coord[i*3+1]);
    deriv[i*nodesPerElement_*3 + 8] =  -0.125*(1.0+par_coord[i*3])*(1.0+par_coord[i*3+1]);
    deriv[i*nodesPerElement_*3 + 11] =  -0.125*(1.0-par_coord[i*3])*(1.0+par_coord[i*3+1]);
    deriv[i*nodesPerElement_*3 + 14] = 0.125*(1.0-par_coord[i*3])*(1.0-par_coord[i*3+1]);
    deriv[i*nodesPerElement_*3 + 17] = 0.125*(1.0+par_coord[i*3])*(1.0-par_coord[i*3+1]);
    deriv[i*nodesPerElement_*3 + 20] =  0.125*(1.0+par_coord[i*3])*(1.0+par_coord[i*3+1]);
    deriv[i*nodesPerElement_*3 + 23] =  0.125*(1.0-par_coord[i*3])*(1.0+par_coord[i*3+1]);
  }                            
}

void
Hex8FEM::hex8_fem_derivative(
  const int npt, const double* par_coord,
  SharedMemView<DoubleType***> deriv)
{
  for (int ip = 0; ip < npt; ++ip) {
    DoubleType x = par_coord[ip*3+0];
    DoubleType y = par_coord[ip*3+1];
    DoubleType z = par_coord[ip*3+2];

    deriv(ip,0,0) = -0.125*(1.0-y)*(1.0-z);
    deriv(ip,1,0) =  0.125*(1.0-y)*(1.0-z);
    deriv(ip,2,0) =  0.125*(1.0+y)*(1.0-z);
    deriv(ip,3,0) = -0.125*(1.0+y)*(1.0-z);
    deriv(ip,4,0) = -0.125*(1.0-y)*(1.0+z);
    deriv(ip,5,0) =  0.125*(1.0-y)*(1.0+z);
    deriv(ip,6,0) =  0.125*(1.0+y)*(1.0+z);
    deriv(ip,7,0) = -0.125*(1.0+y)*(1.0+z);

    deriv(ip,0,1) = -0.125*(1.0-x)*(1.0-z);
    deriv(ip,1,1) = -0.125*(1.0+x)*(1.0-z);
    deriv(ip,2,1) =  0.125*(1.0+x)*(1.0-z);
    deriv(ip,3,1) =  0.125*(1.0-x)*(1.0-z);
    deriv(ip,4,1) = -0.125*(1.0-x)*(1.0+z);
    deriv(ip,5,1) = -0.125*(1.0+x)*(1.0+z);
    deriv(ip,6,1) =  0.125*(1.0+x)*(1.0+z);
    deriv(ip,7,1) =  0.125*(1.0-x)*(1.0+z);

    deriv(ip,0,2) = -0.125*(1.0-x)*(1.0-y);
    deriv(ip,1,2) = -0.125*(1.0+x)*(1.0-y);
    deriv(ip,2,2) = -0.125*(1.0+x)*(1.0+y);
    deriv(ip,3,2) = -0.125*(1.0-x)*(1.0+y);
    deriv(ip,4,2) =  0.125*(1.0-x)*(1.0-y);
    deriv(ip,5,2) =  0.125*(1.0+x)*(1.0-y);
    deriv(ip,6,2) =  0.125*(1.0+x)*(1.0+y);
    deriv(ip,7,2) =  0.125*(1.0-x)*(1.0+y);
  }
}

}
}
