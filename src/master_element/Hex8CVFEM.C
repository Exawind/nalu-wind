/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <master_element/Hex8CVFEM.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/TensorOps.h>
#include <master_element/Hex8GeometryFunctions.h>

#include <FORTRAN_Proto.h>
#include <NaluEnv.h>

#include <cmath>
#include <iostream>

namespace sierra{
namespace nalu{

//-------- hex8_derivative -------------------------------------------------
template <typename DerivType>
void hex8_derivative(
  const int npts,
  const double *intgLoc,
  DerivType &deriv)
{
  const DoubleType half = 0.50;
  const DoubleType one4th = 0.25;
  for (int  ip = 0; ip < npts; ++ip) {
    const DoubleType s1 = intgLoc[ip*3];
    const DoubleType s2 = intgLoc[ip*3+1];
    const DoubleType s3 = intgLoc[ip*3+2];
    const DoubleType s1s2 = s1*s2;
    const DoubleType s2s3 = s2*s3;
    const DoubleType s1s3 = s1*s3;

    // shape function derivative in the s1 direction -
    deriv(ip,0,0) = half*( s3 + s2 ) - s2s3 - one4th;
    deriv(ip,1,0) = half*(-s3 - s2 ) + s2s3 + one4th;
    deriv(ip,2,0) = half*(-s3 + s2 ) - s2s3 + one4th;
    deriv(ip,3,0) = half*( s3 - s2 ) + s2s3 - one4th;
    deriv(ip,4,0) = half*(-s3 + s2 ) + s2s3 - one4th;
    deriv(ip,5,0) = half*( s3 - s2 ) - s2s3 + one4th;
    deriv(ip,6,0) = half*( s3 + s2 ) + s2s3 + one4th;
    deriv(ip,7,0) = half*(-s3 - s2 ) - s2s3 - one4th;

    // shape function derivative in the s2 direction -
    deriv(ip,0,1) = half*( s3 + s1 ) - s1s3 - one4th;
    deriv(ip,1,1) = half*( s3 - s1 ) + s1s3 - one4th;
    deriv(ip,2,1) = half*(-s3 + s1 ) - s1s3 + one4th;
    deriv(ip,3,1) = half*(-s3 - s1 ) + s1s3 + one4th;
    deriv(ip,4,1) = half*(-s3 + s1 ) + s1s3 - one4th;
    deriv(ip,5,1) = half*(-s3 - s1 ) - s1s3 - one4th;
    deriv(ip,6,1) = half*( s3 + s1 ) + s1s3 + one4th;
    deriv(ip,7,1) = half*( s3 - s1 ) - s1s3 + one4th;

    // shape function derivative in the s3 direction -
    deriv(ip,0,2) = half*( s2 + s1 ) - s1s2 - one4th;
    deriv(ip,1,2) = half*( s2 - s1 ) + s1s2 - one4th;
    deriv(ip,2,2) = half*(-s2 - s1 ) - s1s2 - one4th;
    deriv(ip,3,2) = half*(-s2 + s1 ) + s1s2 - one4th;
    deriv(ip,4,2) = half*(-s2 - s1 ) + s1s2 + one4th;
    deriv(ip,5,2) = half*(-s2 + s1 ) - s1s2 + one4th;
    deriv(ip,6,2) = half*( s2 + s1 ) + s1s2 + one4th;
    deriv(ip,7,2) = half*( s2 - s1 ) - s1s2 + one4th;
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
HexSCV::HexSCV()
  : MasterElement()
{
  nDim_ = 3;
  nodesPerElement_ = 8;
  numIntPoints_ = 8;

  // define ip node mappings
  ipNodeMap_.resize(8);
  ipNodeMap_[0] = 0; ipNodeMap_[1] = 1; ipNodeMap_[2] = 2; ipNodeMap_[3] = 3;
  ipNodeMap_[4] = 4; ipNodeMap_[5] = 5; ipNodeMap_[6] = 6; ipNodeMap_[7] = 7;

  // standard integration location
  intgLoc_.resize(24);
  intgLoc_[0]  = -0.25; intgLoc_[1]  = -0.25; intgLoc_[2]  = -0.25;
  intgLoc_[3]  = +0.25; intgLoc_[4]  = -0.25; intgLoc_[5]  = -0.25;
  intgLoc_[6]  = +0.25; intgLoc_[7]  = +0.25; intgLoc_[8]  = -0.25;
  intgLoc_[9]  = -0.25; intgLoc_[10] = +0.25; intgLoc_[11] = -0.25;
  intgLoc_[12] = -0.25; intgLoc_[13] = -0.25; intgLoc_[14] = +0.25;
  intgLoc_[15] = +0.25; intgLoc_[16] = -0.25; intgLoc_[17] = +0.25;
  intgLoc_[18] = +0.25; intgLoc_[19] = +0.25; intgLoc_[20] = +0.25;
  intgLoc_[21] = -0.25; intgLoc_[22] = +0.25; intgLoc_[23] = +0.25;

  // shifted integration location
  intgLocShift_.resize(24);
  intgLocShift_[0]  = -0.5; intgLocShift_[1]  = -0.5; intgLocShift_[2]  = -0.5;
  intgLocShift_[3]  = +0.5; intgLocShift_[4]  = -0.5; intgLocShift_[5]  = -0.5;
  intgLocShift_[6]  = +0.5; intgLocShift_[7]  = +0.5; intgLocShift_[8]  = -0.5;
  intgLocShift_[9]  = -0.5; intgLocShift_[10] = +0.5; intgLocShift_[11] = -0.5;
  intgLocShift_[12] = -0.5; intgLocShift_[13] = -0.5; intgLocShift_[14] = +0.5;
  intgLocShift_[15] = +0.5; intgLocShift_[16] = -0.5; intgLocShift_[17] = +0.5;
  intgLocShift_[18] = +0.5; intgLocShift_[19] = +0.5; intgLocShift_[20] = +0.5;
  intgLocShift_[21] = -0.5; intgLocShift_[22] = +0.5; intgLocShift_[23] = +0.5;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
HexSCV::~HexSCV()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
HexSCV::ipNodeMap(
  int /*ordinal*/)
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(hex_scv_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords,
      volume, error, &lerr );

}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::determinant(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType*>& volume)
{
  constexpr int subDivisionTable[8][8] = {
      {  0,  8, 12, 11, 19, 20, 26, 25},
      {  8,  1,  9, 12, 20, 18, 24, 26},
      { 12,  9,  2, 10, 26, 24, 22, 23},
      { 11, 12, 10,  3, 25, 26, 23, 21},
      { 19, 20, 26, 25,  4, 13, 17, 16},
      { 20, 18, 24, 26, 13,  5, 14, 17},
      { 26, 24, 22, 23, 17, 14,  6, 15},
      { 25, 26, 23, 21, 16, 17, 15,  7}
  };

  DoubleType coordv[27][3];
  subdivide_hex_8(coords, coordv);

  constexpr int numSCV = 8;
  for (int ip = 0; ip < numSCV; ++ip) {
    DoubleType scvHex[8][3];
    for (int n = 0; n < 8; ++n) {
      const int subIndex = subDivisionTable[ip][n];
      for (int d = 0; d < 3; ++d) {
        scvHex[n][d] = coordv[subIndex][d];
      }
    }
    volume(ip) = hex_volume_grandy(scvHex);
  }
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  hex8_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(hex_derivative)
    ( &numIntPoints_,
      &intgLoc_[0], deriv );

  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative HexSCV volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::shifted_grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  hex8_derivative(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCV::shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(hex_shape_fcn)
    (&numIntPoints_,&intgLoc_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexSCV::shifted_shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(hex_shape_fcn)
    (&numIntPoints_,&intgLocShift_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
HexSCS::HexSCS()
  : MasterElement()
{
  nDim_ = 3;
  nodesPerElement_ = 8;
  numIntPoints_ = 12;
  scaleToStandardIsoFac_ = 2.0;

  // define L/R mappings
  lrscv_.resize(24);
  lrscv_[0]  = 0; lrscv_[1]  = 1;
  lrscv_[2]  = 1; lrscv_[3]  = 2;
  lrscv_[4]  = 2; lrscv_[5]  = 3;
  lrscv_[6]  = 0; lrscv_[7]  = 3;
  lrscv_[8]  = 4; lrscv_[9]  = 5;
  lrscv_[10] = 5; lrscv_[11] = 6;
  lrscv_[12] = 6; lrscv_[13] = 7;
  lrscv_[14] = 4; lrscv_[15] = 7;
  lrscv_[16] = 0; lrscv_[17] = 4;
  lrscv_[18] = 1; lrscv_[19] = 5;
  lrscv_[20] = 2; lrscv_[21] = 6;
  lrscv_[22] = 3; lrscv_[23] = 7;

  // define opposing node
  oppNode_.resize(24);
  // face 0
  oppNode_[0] = 3; oppNode_[1] = 2; oppNode_[2] = 6; oppNode_[3] = 7;
  // face 1
  oppNode_[4] = 0; oppNode_[5] = 3; oppNode_[6] = 7; oppNode_[7] = 4;
  // face 2
  oppNode_[8] = 1; oppNode_[9] = 0; oppNode_[10] = 4; oppNode_[11] = 5;
  // face 3
  oppNode_[12] = 1; oppNode_[13] = 5; oppNode_[14] = 6; oppNode_[15] = 2;
  // face 4
  oppNode_[16] = 4; oppNode_[17] = 7; oppNode_[18] = 6; oppNode_[19] = 5;
  // face 5
  oppNode_[20] = 0; oppNode_[21] = 1; oppNode_[22] = 2; oppNode_[23] = 3;

  // define opposing face
  oppFace_.resize(24);
  // face 0
  oppFace_[0]  = 3;  oppFace_[1] = 1;  oppFace_[2] = 5;   oppFace_[3] = 7;
  // face 1
  oppFace_[4]  = 0;  oppFace_[5] = 2;  oppFace_[6] = 6;   oppFace_[7] = 4;
  // face 2
  oppFace_[8]  = 1;  oppFace_[9] = 3;  oppFace_[10] = 7;  oppFace_[11] = 5;
  // face 3
  oppFace_[12] = 0; oppFace_[13] = 4;  oppFace_[14] = 6;  oppFace_[15] = 2;
  // face 4
  oppFace_[16] = 8; oppFace_[17] = 11; oppFace_[18] = 10; oppFace_[19] = 9;
  // face 5
  oppFace_[20] = 8; oppFace_[21] = 9;  oppFace_[22] = 10; oppFace_[23] = 11;

  // standard integration location
  intgLoc_.resize(36);
  intgLoc_[0]  =  0.00; intgLoc_[1]  = -0.25; intgLoc_[2]  = -0.25; // surf 1    1->2
  intgLoc_[3]  =  0.25; intgLoc_[4]  =  0.00; intgLoc_[5]  = -0.25; // surf 2    2->3
  intgLoc_[6]  =  0.00; intgLoc_[7]  =  0.25; intgLoc_[8]  = -0.25; // surf 3    3->4
  intgLoc_[9]  = -0.25; intgLoc_[10] =  0.00; intgLoc_[11] = -0.25; // surf 4    1->4
  intgLoc_[12] =  0.00; intgLoc_[13] = -0.25; intgLoc_[14] =  0.25; // surf 5    5->6
  intgLoc_[15] =  0.25; intgLoc_[16] =  0.00; intgLoc_[17] =  0.25; // surf 6    6->7
  intgLoc_[18] =  0.00; intgLoc_[19] =  0.25; intgLoc_[20] =  0.25; // surf 7    7->8
  intgLoc_[21] = -0.25; intgLoc_[22] =  0.00; intgLoc_[23] =  0.25; // surf 8    5->8
  intgLoc_[24] = -0.25; intgLoc_[25] = -0.25; intgLoc_[26] =  0.00; // surf 9    1->5
  intgLoc_[27] =  0.25; intgLoc_[28] = -0.25; intgLoc_[29] =  0.00; // surf 10   2->6
  intgLoc_[30] =  0.25; intgLoc_[31] =  0.25; intgLoc_[32] =  0.00; // surf 11   3->7
  intgLoc_[33] = -0.25; intgLoc_[34] =  0.25; intgLoc_[35] =  0.00; // surf 12   4->8

  // shifted
  intgLocShift_.resize(36);
  intgLocShift_[0]  =  0.00; intgLocShift_[1]  = -0.50; intgLocShift_[2]  = -0.50; // surf 1    1->2
  intgLocShift_[3]  =  0.50; intgLocShift_[4]  =  0.00; intgLocShift_[5]  = -0.50; // surf 2    2->3
  intgLocShift_[6]  =  0.00; intgLocShift_[7]  =  0.50; intgLocShift_[8]  = -0.50; // surf 3    3->4
  intgLocShift_[9]  = -0.50; intgLocShift_[10] =  0.00; intgLocShift_[11] = -0.50; // surf 4    1->4
  intgLocShift_[12] =  0.00; intgLocShift_[13] = -0.50; intgLocShift_[14] =  0.50; // surf 5    5->6
  intgLocShift_[15] =  0.50; intgLocShift_[16] =  0.00; intgLocShift_[17] =  0.50; // surf 6    6->7
  intgLocShift_[18] =  0.00; intgLocShift_[19] =  0.50; intgLocShift_[20] =  0.50; // surf 7    7->8
  intgLocShift_[21] = -0.50; intgLocShift_[22] =  0.00; intgLocShift_[23] =  0.50; // surf 8    5->8
  intgLocShift_[24] = -0.50; intgLocShift_[25] = -0.50; intgLocShift_[26] =  0.00; // surf 9    1->5
  intgLocShift_[27] =  0.50; intgLocShift_[28] = -0.50; intgLocShift_[29] =  0.00; // surf 10   2->6
  intgLocShift_[30] =  0.50; intgLocShift_[31] =  0.50; intgLocShift_[32] =  0.00; // surf 11   3->7
  intgLocShift_[33] = -0.50; intgLocShift_[34] =  0.50; intgLocShift_[35] =  0.00; // surf 12   4->8

  // exposed face
  intgExpFace_.resize(72);
  // face 0; scs 0, 1, 2, 3
  intgExpFace_[0]  = -0.25; intgExpFace_[1]  = -0.50; intgExpFace_[2]  = -0.25;
  intgExpFace_[3]  =  0.25; intgExpFace_[4]  = -0.50; intgExpFace_[5]  = -0.25;
  intgExpFace_[6]  =  0.25; intgExpFace_[7]  = -0.50; intgExpFace_[8]  =  0.25;
  intgExpFace_[9]  = -0.25; intgExpFace_[10] = -0.50; intgExpFace_[11] =  0.25;
  // face 1; scs 0, 1, 2, 3
  intgExpFace_[12] =  0.50; intgExpFace_[13] = -0.25; intgExpFace_[14] = -0.25;
  intgExpFace_[15] =  0.50; intgExpFace_[16] =  0.25; intgExpFace_[17] = -0.25;
  intgExpFace_[18] =  0.50; intgExpFace_[19] =  0.25; intgExpFace_[20] =  0.25;
  intgExpFace_[21] =  0.50; intgExpFace_[22] = -0.25; intgExpFace_[23] =  0.25;
  // face 2; scs 0, 1, 2, 3
  intgExpFace_[24] =  0.25; intgExpFace_[25] =  0.50; intgExpFace_[26] = -0.25;
  intgExpFace_[27] = -0.25; intgExpFace_[28] =  0.50; intgExpFace_[29] = -0.25;
  intgExpFace_[30] = -0.25; intgExpFace_[31] =  0.50; intgExpFace_[32] =  0.25;
  intgExpFace_[33] =  0.25; intgExpFace_[34] =  0.50; intgExpFace_[35] =  0.25;
  // face 3; scs 0, 1, 2, 3
  intgExpFace_[36] = -0.50; intgExpFace_[37] = -0.25; intgExpFace_[38] = -0.25;
  intgExpFace_[39] = -0.50; intgExpFace_[40] = -0.25; intgExpFace_[41] =  0.25;
  intgExpFace_[42] = -0.50; intgExpFace_[43] =  0.25; intgExpFace_[44] =  0.25;
  intgExpFace_[45] = -0.50; intgExpFace_[46] =  0.25; intgExpFace_[47] = -0.25;
  // face 4; scs 0, 1, 2, 3
  intgExpFace_[48] = -0.25; intgExpFace_[49] = -0.25; intgExpFace_[50] = -0.50;
  intgExpFace_[51] = -0.25; intgExpFace_[52] =  0.25; intgExpFace_[53] = -0.50;
  intgExpFace_[54] =  0.25; intgExpFace_[55] =  0.25; intgExpFace_[56] = -0.50;
  intgExpFace_[57] =  0.25; intgExpFace_[58] = -0.25; intgExpFace_[59] = -0.50;
  // face 5; scs 0, 1, 2, 3
  intgExpFace_[60] = -0.25; intgExpFace_[61] = -0.25; intgExpFace_[62] =  0.50;
  intgExpFace_[63] =  0.25; intgExpFace_[64] = -0.25; intgExpFace_[65] =  0.50;
  intgExpFace_[66] =  0.25; intgExpFace_[67] =  0.25; intgExpFace_[68] =  0.50;
  intgExpFace_[69] = -0.25; intgExpFace_[70] =  0.25; intgExpFace_[71] =  0.50;

  // boundary integration point ip node mapping (ip on an ordinal to local node number)
  ipNodeMap_.resize(24); // 4 ips * 6 faces
  // face 0;
  ipNodeMap_[0] = 0;  ipNodeMap_[1] = 1;  ipNodeMap_[2] = 5;  ipNodeMap_[3] = 4;
  // face 1;
  ipNodeMap_[4] = 1;  ipNodeMap_[5] = 2;  ipNodeMap_[6] = 6;  ipNodeMap_[7] = 5;
  // face 2;
  ipNodeMap_[8] = 2;  ipNodeMap_[9] = 3;  ipNodeMap_[10] = 7; ipNodeMap_[11] = 6;
  // face 3;
  ipNodeMap_[12] = 0; ipNodeMap_[13] = 4; ipNodeMap_[14] = 7; ipNodeMap_[15] = 3;
  // face 4;
  ipNodeMap_[16] = 0; ipNodeMap_[17] = 3; ipNodeMap_[18] = 2; ipNodeMap_[19] = 1;
  // face 5;
  ipNodeMap_[20] = 4; ipNodeMap_[21] = 5; ipNodeMap_[22] = 6; ipNodeMap_[23] = 7;

  // nodes for collocation calculations
  nodeLoc_.resize(24);
  // node 0
  nodeLoc_[0] = -0.5; nodeLoc_[1] = -0.5; nodeLoc_[2] = -0.5;
  // node 1
  nodeLoc_[3] =  0.5; nodeLoc_[4] = -0.5; nodeLoc_[5] = -0.5;
  // node 2
  nodeLoc_[6] =  0.5; nodeLoc_[7] =  0.5; nodeLoc_[8] = -0.5;
  // node 3
  nodeLoc_[9] = -0.5; nodeLoc_[10] =  0.5; nodeLoc_[11] = -0.5;
  // node 4
  nodeLoc_[12] = -0.5; nodeLoc_[13] = -0.5; nodeLoc_[14] =  0.5;
  // node 5
  nodeLoc_[15] =  0.5; nodeLoc_[16] = -0.5; nodeLoc_[17] =  0.5;
  // node 6
  nodeLoc_[18] =  0.5; nodeLoc_[19] =  0.5; nodeLoc_[20] =  0.5;
  // node 7
  nodeLoc_[21] = -0.5; nodeLoc_[22] =  0.5; nodeLoc_[23] =  0.5;

  // mapping from a side ordinal to the node ordinals on that side
  sideNodeOrdinals_ = {
      0, 1, 5, 4, // ordinal 0
      1, 2, 6, 5, // ordinal 1
      2, 3, 7, 6, // ordinal 2
      0, 4, 7, 3, // ordinal 3
      0, 3, 2, 1, // ordinal 4
      4, 5, 6, 7  // ordinal 5
  };

  std::vector<std::vector<double>> nodeLocations =
  {
      {-0.5,-0.5,-0.5}, {+0.5,-0.5,-0.5}, {+0.5,+0.5,-0.5}, {-0.5,+0.5,-0.5},
      {-0.5,-0.5,+0.5}, {+0.5,-0.5,+0.5}, {+0.5,+0.5,+0.5}, {-0.5,+0.5,+0.5}
  };

  intgExpFaceShift_.resize(72);
  int index = 0;
  stk::topology topo = stk::topology::HEX_8;
  for (unsigned k = 0; k < topo.num_sides(); ++k) {
    stk::topology side_topo = topo.side_topology(k);
    const int* ordinals = side_node_ordinals(k);
    for (unsigned n = 0; n < side_topo.num_nodes(); ++n) {
      intgExpFaceShift_[3*index + 0] = nodeLocations[ordinals[n]][0];
      intgExpFaceShift_[3*index + 1] = nodeLocations[ordinals[n]][1];
      intgExpFaceShift_[3*index + 2] = nodeLocations[ordinals[n]][2];
      ++index;
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
HexSCS::~HexSCS()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
HexSCS::ipNodeMap(
  int ordinal)
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal*4];
}


//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::shifted_shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- hex8_shape_fcn --------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::hex8_shape_fcn(
  const int  &npts,
  const double *isoParCoord,
  SharedMemView<DoubleType**> &shape_fcn)
{
  const DoubleType half = 0.50;
  const DoubleType one4th = 0.25;
  const DoubleType one8th = 0.125;
  for ( int j = 0; j < numIntPoints_; ++j ) {

    const DoubleType s1 = isoParCoord[j*3];
    const DoubleType s2 = isoParCoord[j*3+1];
    const DoubleType s3 = isoParCoord[j*3+2];

    shape_fcn(j,0) = one8th + one4th*(-s1 - s2 - s3)
      + half*( s2*s3 + s3*s1 + s1*s2 ) - s1*s2*s3;
    shape_fcn(j,1) = one8th + one4th*( s1 - s2 - s3)
      + half*( s2*s3 - s3*s1 - s1*s2 ) + s1*s2*s3;
    shape_fcn(j,2) = one8th + one4th*( s1 + s2 - s3)
      + half*(-s2*s3 - s3*s1 + s1*s2 ) - s1*s2*s3;
    shape_fcn(j,3) = one8th + one4th*(-s1 + s2 - s3)
      + half*(-s2*s3 + s3*s1 - s1*s2 ) + s1*s2*s3;
    shape_fcn(j,4) = one8th + one4th*(-s1 - s2 + s3)
      + half*(-s2*s3 - s3*s1 + s1*s2 ) + s1*s2*s3;
    shape_fcn(j,5) = one8th + one4th*( s1 - s2 + s3)
      + half*(-s2*s3 + s3*s1 - s1*s2 ) - s1*s2*s3;
    shape_fcn(j,6) = one8th + one4th*( s1 + s2 + s3)
      + half*( s2*s3 + s3*s1 + s1*s2 ) + s1*s2*s3;
    shape_fcn(j,7) = one8th + one4th*(-s1 + s2 + s3)
      + half*( s2*s3 - s3*s1 - s1*s2 ) - s1*s2*s3;
  }
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  hex8_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
 }

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------c------------------------------------
void HexSCS::shifted_grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  hex8_derivative(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::determinant(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType**>&areav)
{
  constexpr int hex_edge_facet_table[12][4] = {
      { 20,  8, 12, 26 },
      { 24,  9, 12, 26 },
      { 10, 12, 26, 23 },
      { 11, 25, 26, 12 },
      { 13, 20, 26, 17 },
      { 17, 14, 24, 26 },
      { 17, 15, 23, 26 },
      { 16, 17, 26, 25 },
      { 19, 20, 26, 25 },
      { 20, 18, 24, 26 },
      { 22, 23, 26, 24 },
      { 21, 25, 26, 23 }
  };

  DoubleType coordv[27][3];
  subdivide_hex_8(coords, coordv);

  constexpr int npf = 4;
  constexpr int nscs = 12;
  for (int ics=0; ics < nscs; ++ics) {
    DoubleType scscoords[4][3];
    for (int inode = 0; inode < npf; ++inode) {
      const int itrianglenode = hex_edge_facet_table[ics][inode];
      for (int d=0; d < 3; ++d) {
        scscoords[inode][d] = coordv[itrianglenode][d];
      }
    }
    quad_area_by_triangulation(ics, scscoords, areav);
  }
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
const int *
HexSCS::side_node_ordinals(int ordinal)
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return &sideNodeOrdinals_[ordinal*4];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  SIERRA_FORTRAN(hex_scs_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords, areav );

  // all is always well; no error checking
  *error = 0;
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(hex_derivative)
    ( &numIntPoints_,
      &intgLoc_[0], deriv );

  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative HexSCS volume.." << std::endl;
 }

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::shifted_grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(hex_derivative)
    ( &numIntPoints_,
      &intgLocShift_[0], deriv );

  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative HexSCS volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::face_grad_op(
  const int face_ordinal,
  const bool shifted,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using traits = AlgTraitsQuad4Hex8;
  const std::vector<double> &exp_face = shifted ? intgExpFaceShift_ : intgExpFace_;

  constexpr int derivSize = traits::numFaceIp_ * traits::nodesPerElement_ * traits::nDim_;
  DoubleType psi[derivSize];
  SharedMemView<DoubleType***> deriv(psi, traits::numFaceIp_, traits::nodesPerElement_, traits::nDim_);

  const int offset = traits::numFaceIp_ * traits::nDim_ * face_ordinal;
  hex8_derivative(traits::numFaceIp_, &exp_face[offset], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

void HexSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  constexpr bool shifted = false;
  face_grad_op(face_ordinal, shifted, coords, gradop);
}

void HexSCS::face_grad_op(
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

  for ( int n=0; n<nelem; n++ ) {

    for ( int k=0; k<npf; k++ ) {

      const int row = 12*face_ordinal + k*ndim;
      SIERRA_FORTRAN(hex_derivative)
        ( &nface,
          &intgExpFace_[row], dpsi );

      SIERRA_FORTRAN(hex_gradient_operator)
        ( &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[24*n], &gradop[k*nelem*24+n*24], &det_j[npf*n+k], error, &lerr );

      if ( lerr )
        NaluEnv::self().naluOutput() << "sorry, issue with face_grad_op.." << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- shifted_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  constexpr bool shifted = true;
  face_grad_op(face_ordinal, shifted, coords, gradop);
}

void HexSCS::shifted_face_grad_op(
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

  for ( int n=0; n<nelem; n++ ) {

    for ( int k=0; k<npf; k++ ) {

      const int row = 12*face_ordinal + k*ndim;
      SIERRA_FORTRAN(hex_derivative)
        ( &nface,
          &intgExpFaceShift_[row], dpsi );

      SIERRA_FORTRAN(hex_gradient_operator)
        ( &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[24*n], &gradop[k*nelem*24+n*24], &det_j[npf*n+k], error, &lerr );

      if ( lerr )
        NaluEnv::self().naluOutput() << "sorry, issue with face_grad_op.." << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::gij(
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
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::gij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv)
{
  hex8_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_gij_3d<AlgTraitsHex8>(deriv, coords, gupper, glower);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
const int *
HexSCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(hex_shape_fcn)
    (&numIntPoints_,&intgLoc_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::shifted_shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(hex_shape_fcn)
    (&numIntPoints_,&intgLocShift_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
int
HexSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*4+node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
int
HexSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*4+node];
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
HexSCS::isInElement(
    const double * elem_nodal_coor,     // (8,3)
    const double * point_coor,          // (3)
    double * par_coor )
{
  const int maxNonlinearIter = 20;
  const double isInElemConverged = 1.0e-16;
  // Translate element so that (x,y,z) coordinates of the first node are (0,0,0)

  double x[] = {0.,
        0.125*(elem_nodal_coor[1] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[2] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[3] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[4] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[5] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[6] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[7] - elem_nodal_coor[0]) };
  double y[] = {0.,
        0.125*(elem_nodal_coor[9 ] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[10] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[11] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[12] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[13] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[14] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[15] - elem_nodal_coor[8]) };
  double z[] = {0.,
        0.125*(elem_nodal_coor[17] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[18] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[19] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[20] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[21] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[22] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[23] - elem_nodal_coor[16]) };

  // (xp,yp,zp) is the point at which we're searching for (xi,eta,zeta)
  // (must translate this also)

  double xp = point_coor[0] - elem_nodal_coor[0];
  double yp = point_coor[1] - elem_nodal_coor[8];
  double zp = point_coor[2] - elem_nodal_coor[16];

  // Newton-Raphson iteration for (xi,eta,zeta)
  double j[9];
  double f[3];
  double shapefct[8];
  double xinew = 0.5;     // initial guess
  double etanew = 0.5;
  double zetanew = 0.5;
  double xicur = 0.5;
  double etacur = 0.5;
  double zetacur = 0.5;
  double xidiff[] = { 1.0, 1.0, 1.0 };
  int i = 0;

  do
  {
    j[0]=
      -(1.0-etacur)*(1.0-zetacur)*x[1]
      -(1.0+etacur)*(1.0-zetacur)*x[2]
      +(1.0+etacur)*(1.0-zetacur)*x[3]
      +(1.0-etacur)*(1.0+zetacur)*x[4]
      -(1.0-etacur)*(1.0+zetacur)*x[5]
      -(1.0+etacur)*(1.0+zetacur)*x[6]
      +(1.0+etacur)*(1.0+zetacur)*x[7];

    j[1]=
       (1.0+xicur)*(1.0-zetacur)*x[1]
      -(1.0+xicur)*(1.0-zetacur)*x[2]
      -(1.0-xicur)*(1.0-zetacur)*x[3]
      +(1.0-xicur)*(1.0+zetacur)*x[4]
      +(1.0+xicur)*(1.0+zetacur)*x[5]
      -(1.0+xicur)*(1.0+zetacur)*x[6]
      -(1.0-xicur)*(1.0+zetacur)*x[7];

    j[2]=
       (1.0-etacur)*(1.0+xicur)*x[1]
      +(1.0+etacur)*(1.0+xicur)*x[2]
      +(1.0+etacur)*(1.0-xicur)*x[3]
      -(1.0-etacur)*(1.0-xicur)*x[4]
      -(1.0-etacur)*(1.0+xicur)*x[5]
      -(1.0+etacur)*(1.0+xicur)*x[6]
      -(1.0+etacur)*(1.0-xicur)*x[7];

    j[3]=
      -(1.0-etacur)*(1.0-zetacur)*y[1]
      -(1.0+etacur)*(1.0-zetacur)*y[2]
      +(1.0+etacur)*(1.0-zetacur)*y[3]
      +(1.0-etacur)*(1.0+zetacur)*y[4]
      -(1.0-etacur)*(1.0+zetacur)*y[5]
      -(1.0+etacur)*(1.0+zetacur)*y[6]
      +(1.0+etacur)*(1.0+zetacur)*y[7];

    j[4]=
       (1.0+xicur)*(1.0-zetacur)*y[1]
      -(1.0+xicur)*(1.0-zetacur)*y[2]
      -(1.0-xicur)*(1.0-zetacur)*y[3]
      +(1.0-xicur)*(1.0+zetacur)*y[4]
      +(1.0+xicur)*(1.0+zetacur)*y[5]
      -(1.0+xicur)*(1.0+zetacur)*y[6]
      -(1.0-xicur)*(1.0+zetacur)*y[7];

    j[5]=
       (1.0-etacur)*(1.0+xicur)*y[1]
      +(1.0+etacur)*(1.0+xicur)*y[2]
      +(1.0+etacur)*(1.0-xicur)*y[3]
      -(1.0-etacur)*(1.0-xicur)*y[4]
      -(1.0-etacur)*(1.0+xicur)*y[5]
      -(1.0+etacur)*(1.0+xicur)*y[6]
      -(1.0+etacur)*(1.0-xicur)*y[7];

    j[6]=
      -(1.0-etacur)*(1.0-zetacur)*z[1]
      -(1.0+etacur)*(1.0-zetacur)*z[2]
      +(1.0+etacur)*(1.0-zetacur)*z[3]
      +(1.0-etacur)*(1.0+zetacur)*z[4]
      -(1.0-etacur)*(1.0+zetacur)*z[5]
      -(1.0+etacur)*(1.0+zetacur)*z[6]
      +(1.0+etacur)*(1.0+zetacur)*z[7];

    j[7]=
       (1.0+xicur)*(1.0-zetacur)*z[1]
      -(1.0+xicur)*(1.0-zetacur)*z[2]
      -(1.0-xicur)*(1.0-zetacur)*z[3]
      +(1.0-xicur)*(1.0+zetacur)*z[4]
      +(1.0+xicur)*(1.0+zetacur)*z[5]
      -(1.0+xicur)*(1.0+zetacur)*z[6]
      -(1.0-xicur)*(1.0+zetacur)*z[7];

    j[8]=
       (1.0-etacur)*(1.0+xicur)*z[1]
      +(1.0+etacur)*(1.0+xicur)*z[2]
      +(1.0+etacur)*(1.0-xicur)*z[3]
      -(1.0-etacur)*(1.0-xicur)*z[4]
      -(1.0-etacur)*(1.0+xicur)*z[5]
      -(1.0+etacur)*(1.0+xicur)*z[6]
      -(1.0+etacur)*(1.0-xicur)*z[7];

    double jdet=-(j[2]*j[4]*j[6])+j[1]*j[5]*j[6]+j[2]*j[3]*j[7]-
      j[0]*j[5]*j[7]-j[1]*j[3]*j[8]+j[0]*j[4]*j[8];

    if (!jdet) {
      i = maxNonlinearIter;
      break;
    }
    shapefct[0]=(1.0-etacur)*(1.0-xicur)*(1.0-zetacur);

    shapefct[1]=(1.0-etacur)*(1.0+xicur)*(1.0-zetacur);

    shapefct[2]=(1.0+etacur)*(1.0+xicur)*(1.0-zetacur);

    shapefct[3]=(1.0+etacur)*(1.0-xicur)*(1.0-zetacur);

    shapefct[4]=(1.0-etacur)*(1.0-xicur)*(1.0+zetacur);

    shapefct[5]=(1.0-etacur)*(1.0+xicur)*(1.0+zetacur);

    shapefct[6]=(1.0+etacur)*(1.0+xicur)*(1.0+zetacur);

    shapefct[7]=(1.0+etacur)*(1.0-xicur)*(1.0+zetacur);

    f[0]=xp-shapefct[1]*x[1]-shapefct[2]*x[2]-shapefct[3]*x[3]-shapefct[4]*x[4]-\
      shapefct[5]*x[5]-shapefct[6]*x[6]-shapefct[7]*x[7];

    f[1]=yp-shapefct[1]*y[1]-shapefct[2]*y[2]-shapefct[3]*y[3]-shapefct[4]*y[4]-\
      shapefct[5]*y[5]-shapefct[6]*y[6]-shapefct[7]*y[7];

    f[2]=zp-shapefct[1]*z[1]-shapefct[2]*z[2]-shapefct[3]*z[3]-shapefct[4]*z[4]-\
      shapefct[5]*z[5]-shapefct[6]*z[6]-shapefct[7]*z[7];

    xinew = (jdet*xicur+f[2]*(j[2]*j[4]-j[1]*j[5])-f[1]*j[2]*j[7]+f[0]*j[5]*j[7]+
       f[1]*j[1]*j[8]-f[0]*j[4]*j[8])/jdet;

    etanew = (etacur*jdet+f[2]*(-(j[2]*j[3])+j[0]*j[5])+f[1]*j[2]*j[6]-f[0]*j[5]*j[6]-
        f[1]*j[0]*j[8]+f[0]*j[3]*j[8])/jdet;

    zetanew = (jdet*zetacur+f[2]*(j[1]*j[3]-j[0]*j[4])-f[1]*j[1]*j[6]+
         f[0]*j[4]*j[6]+f[1]*j[0]*j[7]-f[0]*j[3]*j[7])/jdet;

    xidiff[0] = xinew - xicur;
    xidiff[1] = etanew - etacur;
    xidiff[2] = zetanew - zetacur;
    xicur = xinew;
    etacur = etanew;
    zetacur = zetanew;

  }
  while ( !within_tolerance( vector_norm_sq(xidiff,3), isInElemConverged) && ++i < maxNonlinearIter);

  par_coor[0] = par_coor[1] = par_coor[2] = std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if (i <maxNonlinearIter) {
    par_coor[0] = xinew;
    par_coor[1] = etanew;
    par_coor[2] = zetanew;

    std::vector<double> xtmp(3);
    xtmp[0] = par_coor[0];
    xtmp[1] = par_coor[1];
    xtmp[2] = par_coor[2];
    dist = parametric_distance(xtmp);
  }
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::interpolatePoint(
    const int  & ncomp_field,
    const double * par_coord,           // (3)
    const double * field,               // (8,ncomp_field)
    double * result ) // (ncomp_field)
{
  // 'field' is a flat array of dimension (8,ncomp_field) (Fortran ordering);
  double xi   = par_coord[0];
  double eta  = par_coord[1];
  double zeta = par_coord[2];

  // NOTE: this uses a [-1,1] definition of the reference element,
  // contrary to the rest of the code

  for ( int i = 0; i < ncomp_field; i++ )
  {
    // Base 'field array' index for ith component
    int b = 8*i;

    result[i] = 0.125 * (
        (1 - xi) * (1 - eta) * (1 - zeta) * field[b + 0]
      + (1 + xi) * (1 - eta) * (1 - zeta) * field[b + 1]
      + (1 + xi) * (1 + eta) * (1 - zeta) * field[b + 2]
      + (1 - xi) * (1 + eta) * (1 - zeta) * field[b + 3]
      + (1 - xi) * (1 - eta) * (1 + zeta) * field[b + 4]
      + (1 + xi) * (1 - eta) * (1 + zeta) * field[b + 5]
      + (1 + xi) * (1 + eta) * (1 + zeta) * field[b + 6]
      + (1 - xi) * (1 + eta) * (1 + zeta) * field[b + 7]
    );
  }

}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::general_shape_fcn(
  const int numIp,
  const double *isoParCoord,
  double *shpfc)
{
  // -1:1 isoparametric range
  const double npe = nodesPerElement_;
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
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::general_face_grad_op(
  const int face_ordinal,
  const double *isoParCoord,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  const int nface = 1;

  double dpsi[24];

  SIERRA_FORTRAN(hex_derivative)
    ( &nface, &isoParCoord[0], dpsi );

  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nface,
      &nodesPerElement_,
      &nface,
      dpsi,
      &coords[0], &gradop[0], &det_j[0], error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "HexSCS::general_face_grad_op: issue.." << std::endl;

}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::sidePcoords_to_elemPcoords(
  const int & side_ordinal,
  const int & npoints,
  const double *side_pcoords,
  double *elem_pcoords)
{
  switch (side_ordinal) {
  case 0:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = -0.5;
      elem_pcoords[i*3+2] = 0.5*side_pcoords[2*i+1];
    }
    break;
  case 1:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 0.5;
      elem_pcoords[i*3+1] = 0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+2] = 0.5*side_pcoords[2*i+1];
    }
    break;
  case 2:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = -0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = 0.5;
      elem_pcoords[i*3+2] = 0.5*side_pcoords[2*i+1];
    }
    break;
  case 3:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = -0.5;
      elem_pcoords[i*3+1] = 0.5*side_pcoords[2*i+1];
      elem_pcoords[i*3+2] = 0.5*side_pcoords[2*i+0];
    }
    break;
  case 4:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 0.5*side_pcoords[2*i+1];
      elem_pcoords[i*3+1] = 0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+2] = -0.5;
    }
    break;
  case 5:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = 0.5*side_pcoords[2*i+1];
      elem_pcoords[i*3+2] = 0.5;
    }
    break;
  default:
    throw std::runtime_error("HexSCS::sideMap invalid ordinal");
  }
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double HexSCS::parametric_distance(const std::vector<double> &x)
{
  std::vector<double> y(3);
  for (int i=0; i<3; ++i) {
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

}
}
