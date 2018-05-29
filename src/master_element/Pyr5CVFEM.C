/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <master_element/MasterElement.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/Pyr5CVFEM.h>
#include <master_element/Hex8GeometryFunctions.h>
#include <master_element/MasterElementFunctions.h>

#include <master_element/MasterElementUtils.h>

#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/QuadratureRule.h>
#include <AlgTraits.h>

#include <NaluEnv.h>
#include <FORTRAN_Proto.h>

#include <stk_util/environment/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <iostream>

#include <cmath>
#include <limits>
#include <array>
#include <map>
#include <memory>

namespace sierra{
namespace nalu{

//-------- pyr_deriv -------------------------------------------------------
template <typename DerivType>
void pyr_deriv(const int npts,
  const double *intgLoc,
  DerivType& deriv)
{
  // d3d(c,s,j) = deriv[c + 3*(s + 5*j)] = deriv[c+3s+15j]

  for ( int j = 0; j < npts; ++j) {
    const int k = j*3;

    const double r = intgLoc[k+0];
    const double s = intgLoc[k+1];
    const double t = intgLoc[k+2];

    deriv(j,0,0) =-0.25*(1.0-s)*(1.0-t);  // d(N_1)/ d(r) = deriv[0]
    deriv(j,0,1) =-0.25*(1.0-r)*(1.0-t);  // d(N_1)/ d(s) = deriv[1]
    deriv(j,0,2) =-0.25*(1.0-r)*(1.0-s);  // d(N_1)/ d(t) = deriv[2]
              
    deriv(j,1,0) = 0.25*(1.0-s)*(1.0-t);  // d(N_2)/ d(r) = deriv[0+3]
    deriv(j,1,1) =-0.25*(1.0+r)*(1.0-t);  // d(N_2)/ d(s) = deriv[1+3]
    deriv(j,1,2) =-0.25*(1.0+r)*(1.0-s);  // d(N_2)/ d(t) = deriv[2+3]
              
    deriv(j,2,0) = 0.25*(1.0+s)*(1.0-t);  // d(N_3)/ d(r) = deriv[0+6]
    deriv(j,2,1) = 0.25*(1.0+r)*(1.0-t);  // d(N_3)/ d(s) = deriv[1+6]
    deriv(j,2,2) =-0.25*(1.0+r)*(1.0+s);  // d(N_3)/ d(t) = deriv[2+6]
              
    deriv(j,3,0) =-0.25*(1.0+s)*(1.0-t);  // d(N_4)/ d(r) = deriv[0+9]
    deriv(j,3,1) = 0.25*(1.0-r)*(1.0-t);  // d(N_4)/ d(s) = deriv[1+9]
    deriv(j,3,2) =-0.25*(1.0-r)*(1.0+s);  // d(N_4)/ d(t) = deriv[2+9]
              
    deriv(j,4,0) = 0.0;                   // d(N_5)/ d(r) = deriv[0+12]
    deriv(j,4,1) = 0.0;                   // d(N_5)/ d(s) = deriv[1+12]
    deriv(j,4,2) = 1.0;                   // d(N_5)/ d(t) = deriv[2+12]
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
PyrSCV::PyrSCV()
  : MasterElement()
{
  nDim_ = 3;
  nodesPerElement_ = 5;
  numIntPoints_ = 5; 

  // define ip node mappings
  ipNodeMap_.resize(5);
  ipNodeMap_[0] = 0; ipNodeMap_[1] = 1; ipNodeMap_[2] = 2; ipNodeMap_[3] = 3;
  ipNodeMap_[4] = 4;

  // standard integration location
  intgLoc_.resize(15);
  intgLoc_[0]  = -19.0/48.0; intgLoc_[1]  = -19.0/48.0; intgLoc_[2]  = 41.0/240.0;  // vol 0
  intgLoc_[3]  =  19.0/48.0; intgLoc_[4]  = -19.0/48.0; intgLoc_[5]  = 41.0/240.0;  // vol 1
  intgLoc_[6]  =  19.0/48.0; intgLoc_[7]  =  19.0/48.0; intgLoc_[8]  = 41.0/240.0;  // vol 2
  intgLoc_[9]  = -19.0/48.0; intgLoc_[10] =  19.0/48.0; intgLoc_[11] = 41.0/240.0;  // vol 3
  intgLoc_[12] =   0.0;      intgLoc_[13] =   0.0;      intgLoc_[14] = 0.6 ;        // vol 4

  // shifted
  intgLocShift_.resize(15);
  intgLocShift_[0]  = -1.0; intgLocShift_[1]  = -1.0; intgLocShift_[2]  = 0.0;  // vol 0
  intgLocShift_[3]  =  1.0; intgLocShift_[4]  = -1.0; intgLocShift_[5]  = 0.0;  // vol 1
  intgLocShift_[6]  =  1.0; intgLocShift_[7]  =  1.0; intgLocShift_[8]  = 0.0;  // vol 2
  intgLocShift_[9]  = -1.0; intgLocShift_[10] =  1.0; intgLocShift_[11] = 0.0;  // vol 3
  intgLocShift_[12] =  0.0; intgLocShift_[13] =  0.0; intgLocShift_[14] = 1.0;  // vol 4
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
PyrSCV::~PyrSCV()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
PyrSCV::ipNodeMap(
  int /*ordinal*/)
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}

DoubleType polyhedral_volume_by_faces(int ncoords, const DoubleType volcoords[][3],
                                      int ntriangles, const int triangleFaceTable[][3])
{
  DoubleType xface[3];

  DoubleType volume = 0.0;

  // loop over each triangular facet
  for(int itriangle=0; itriangle<ntriangles; ++itriangle) {
    // c-index ordering is used in the table, so change to fortran
    int ip = triangleFaceTable[itriangle][0];
    int iq = triangleFaceTable[itriangle][1];
    int ir = triangleFaceTable[itriangle][2];
    // set spatial coordinate of integration point
    for(int k=0; k<3; ++k) {
      xface[k] = volcoords[ip][k] + volcoords[iq][k] + volcoords[ir][k];
    }
    // calculate contribution of triangular face to volume
    volume = volume
      + xface[0]*( ( volcoords[iq][1]-volcoords[ip][1] )*
                   ( volcoords[ir][2]-volcoords[ip][2] )
                 - ( volcoords[ir][1]-volcoords[ip][1] )*
                   ( volcoords[iq][2]-volcoords[ip][2] ) )
      - xface[1]*( ( volcoords[iq][0]-volcoords[ip][0] )*
                   ( volcoords[ir][2]-volcoords[ip][2] )
                 - ( volcoords[ir][0]-volcoords[ip][0] )*
                   ( volcoords[iq][2]-volcoords[ip][2] ) )
      + xface[2]*( ( volcoords[iq][0]-volcoords[ip][0] )*
                   ( volcoords[ir][1]-volcoords[ip][1] )
                 - ( volcoords[ir][0]-volcoords[ip][0] )*
                   ( volcoords[iq][1]-volcoords[ip][1] ) );
  }

  // apply constants that were factored out for calculation of
  // the integration point, the area, and the gauss divergence
  // theorem.
  volume = volume/18.0;
  return volume;
}

DoubleType octohedron_volume_by_triangle_facets(const DoubleType volcoords[10][3])
{
  DoubleType coords[14][3];
  const int triangularFacetTable[24][3] = {
    {1, 3, 10}, 
    {2, 10, 3},
    {2, 9, 10}, 
    {10, 9, 1},
    {4, 3, 11}, 
    {3, 1, 11}, 
    {11, 1, 5},
    {4, 11, 5},
    {1, 12, 5},
    {1, 7, 12}, 
    {12, 7, 6},
    {5, 12, 6},
    {9, 8, 13}, 
    {13, 8, 7},
    {13, 7, 1},
    {9, 13, 1},
    {4, 5, 0},
    {5, 6, 0},
    {6, 7, 0},
    {7, 8, 0},
    {0, 8, 9},
    {0, 9, 2},
    {0, 2, 3},
    {0, 3, 4}
  };

  // the first ten coordinates are the vertices of the octohedron
  for(int j=0; j<10; ++j) {
    for(int k=0; k<3; ++k) {
      coords[j][k] = volcoords[j][k];
    }
  }
  // we now add face midpoints only for the four faces that are
  // not planar
  for(int k=0; k<3; ++k) {
    coords[10][k] = 0.25*( volcoords[1][k] + volcoords[3][k] + volcoords[2][k] + volcoords[9][k] );
  }
  for(int k=0; k<3; ++k) {
    coords[11][k] = 0.25*( volcoords[1][k] + volcoords[5][k] + volcoords[4][k] + volcoords[3][k] );
  }
  for(int k=0; k<3; ++k) {
    coords[12][k] = 0.25*( volcoords[1][k] + volcoords[7][k] + volcoords[6][k] + volcoords[5][k] );
  }
  for(int k=0; k<3; ++k) {
    coords[13][k] = 0.25*( volcoords[1][k] + volcoords[9][k] + volcoords[8][k] + volcoords[7][k] );
  }

  int ncoords = 14;
  int ntriangles = 24;

  // compute the volume using the new equivalent polyhedron
  return polyhedral_volume_by_faces(ncoords, coords, ntriangles, triangularFacetTable);
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCV::determinant(
    SharedMemView<DoubleType**>& cordel,
    SharedMemView<DoubleType*>& vol)
{
  int npe = nodesPerElement_;
  int nscv = numIntPoints_;
  DoubleType coords[19][3];
  DoubleType ehexcoords[8][3];
  DoubleType epyrcoords[10][3];

  const int pyramidSubcontrolNodeTable[5][10] = {
     {0,  5,  9,  8, 11, 12, 18, 17, -1, -1},
     {5,  1,  6,  9, 12, 10, 14, 18, -1, -1},
     {6,  2,  7,  9, 14, 13, 16, 18, -1, -1},
     {8,  9,  7,  3, 17, 18, 16, 15, -1, -1},
     {4, 18, 15, 17, 11, 12, 10, 14, 13, 16}
  };
  const double one3rd = 1.0/3.0;

  for(int j=0; j<5; ++j) {
    for(int k=0; k<3; ++k) {
      coords[j][k] = cordel(j,k);
    }
  }

  // face 1 (quad)
  // 4++++8+++3
  // +         +
  // +         +
  // 9   10    7
  // +         +
  // +         +
  // 1++++6++++2

  // edge midpoints
  for(int k=0; k<3; ++k) {
    coords[5][k] = 0.5*(cordel(0,k) + cordel(1,k));
  }
  for(int k=0; k<3; ++k) {
    coords[6][k] = 0.5*(cordel(1,k) + cordel(2,k));
  }
  for(int k=0; k<3; ++k) {
    coords[7][k] = 0.5*(cordel(2,k) + cordel(3,k));
  }
  for(int k=0; k<3; ++k) {
    coords[8][k] = 0.5*(cordel(3,k) + cordel(0,k));
  }

  //face midpoint
  for(int k=0; k<3; ++k) {
    coords[9][k] = 0.25*(cordel(0,k) + cordel(1,k) + cordel(2,k) + cordel(3,k));
  }

  // face 2 (tri)
  //
  // edge midpoints
  for(int k=0; k<3; ++k) {
    coords[10][k] = 0.5*(cordel(1,k) + cordel(4,k));
  }
  for(int k=0; k<3; ++k) {
    coords[11][k] = 0.5*(cordel(4,k) + cordel(0,k));
  }

  // face midpoint
  for(int k=0; k<3; ++k) {
    coords[12][k] = one3rd*(cordel(0,k) + cordel(1,k) + cordel(4,k));
  }

  // face 3 (tri)

  // edge midpoint
  for(int k=0; k<3; ++k) {
    coords[13][k] = 0.5*(cordel(2,k) + cordel(4,k));
  }

  // face midpoint
  for(int k=0; k<3; ++k) {
    coords[14][k] = one3rd*(cordel(1,k) + cordel(2,k) + cordel(4,k));
  }

  // face 4 (tri)

  // edge midpoint
  for(int k=0; k<3; ++k) {
    coords[15][k] = 0.5*(cordel(3,k) + cordel(4,k));
  }

  // face midpoint
  for(int k=0; k<3; ++k) {
    coords[16][k] = one3rd*(cordel(3,k) + cordel(4,k) + cordel(2,k));
  }

  // face 5 (tri)

  // face midpoint
  for(int k=0; k<3; ++k) {
    coords[17][k] = one3rd*(cordel(0,k) + cordel(4,k) + cordel(3,k));
  }

  // element centroid
  for(int k=0; k<3; ++k) {
    coords[18][k] = 0.0;
  }
  for(int j=0; j<npe; ++j) {
    for(int k=0; k<3; ++k) {
      coords[18][k] += 0.2*cordel(j,k);
    }
  }

  // loop over hexahedral volumes first
  for(int icv=0; icv<nscv-1; ++icv) {
    // loop over vertices of hexahedral scv
    for(int inode=0; inode<8; ++inode) {
      // set coordinates of scv from node table
      for(int k=0; k<3; ++k) {
        ehexcoords[inode][k] = coords[pyramidSubcontrolNodeTable[icv][inode]][k];
      }
    }
    // compute volume use an equivalent polyhedron
    vol(icv) = hex_volume_grandy(ehexcoords);
  }

  // now do octohedron on pyramid tip
  int icv = nscv-1;
  // loop over vertices of octohedral scv
  for(int inode=0; inode<10; ++inode) {
    // set coordinates based on node table
    for(int k=0; k<3; ++k) {
      epyrcoords[inode][k] = coords[pyramidSubcontrolNodeTable[icv][inode]][k];
    }
  }
  // compute volume using an equivalent polyhedron
  vol(icv) = octohedron_volume_by_triangle_facets(epyrcoords);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCV::grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv)
{
  pyr_deriv(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsPyr5>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCV::shifted_grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv)
{
  pyr_deriv(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op<AlgTraitsPyr5>(deriv, coords, gradop);
}

void PyrSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{

  int lerr = 0;

  SIERRA_FORTRAN(pyr_scv_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords,
      volume, error, &lerr );
}


//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
PyrSCV::shape_fcn(double *shpfc)
{
  pyr_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
PyrSCV::shifted_shape_fcn(double *shpfc)
{
  pyr_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- pyr_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
void
PyrSCV::pyr_shape_fcn(
  const int  &npts,
  const double *par_coord, 
  double *shape_fcn)
{
  const double one  = 1.0;
  for ( int j = 0; j < npts; ++j ) {
    const int fivej = 5*j;
    const int k     = 3*j;
    const double r    = par_coord[k+0];
    const double s    = par_coord[k+1];
    const double t    = par_coord[k+2];

    shape_fcn[0 + fivej] = 0.25*(1.0-r)*(1.0-s)*(one-t);
    shape_fcn[1 + fivej] = 0.25*(1.0+r)*(1.0-s)*(one-t);
    shape_fcn[2 + fivej] = 0.25*(1.0+r)*(1.0+s)*(one-t);
    shape_fcn[3 + fivej] = 0.25*(1.0-r)*(1.0+s)*(one-t);
    shape_fcn[4 + fivej] = t;
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
PyrSCS::PyrSCS()
  : MasterElement()
{
  nDim_ = 3;
  nodesPerElement_ = 5;
  numIntPoints_ = 8;

  // define L/R mappings
  lrscv_.resize(16);
  lrscv_[0]  = 0; lrscv_[1]  = 1;
  lrscv_[2]  = 1; lrscv_[3]  = 2;
  lrscv_[4]  = 2; lrscv_[5]  = 3;
  lrscv_[6]  = 0; lrscv_[7]  = 3;
  lrscv_[8]  = 0; lrscv_[9]  = 4;
  lrscv_[10] = 1; lrscv_[11] = 4;
  lrscv_[12] = 2; lrscv_[13] = 4;
  lrscv_[14] = 3; lrscv_[15] = 4;
  
  // define opposing node
  // opposing node for node 4 is never uniquely defined: pick one
  oppNode_.resize(20);
  // face 0; nodes 0,1,4
  oppNode_[0] = 3; oppNode_[1] = 2; oppNode_[2] = 2; oppNode_[3] = -1;
  // face 1; nodes 1,2,4
  oppNode_[4] = 0; oppNode_[5] = 3; oppNode_[6] = 3; oppNode_[7] = -1;
  // face 2; nodes 2,3,4
  oppNode_[8] = 1; oppNode_[9] = 0; oppNode_[10] = 0; oppNode_[11] = -1;
  // face 3; nodes 0,4,3
  oppNode_[12] = 1; oppNode_[13] = 1; oppNode_[14] = 2; oppNode_[15] = -1;
  // face 4; nodes 0,3,2,1
  oppNode_[16] = 4; oppNode_[17] = 4; oppNode_[18] = 4; oppNode_[19] = 4;

  // define opposing face
  // the 5th node maps to two opposing sub-faces, we pick one
  oppFace_.resize(20);
  // face 0
  oppFace_[0] = 3;  oppFace_[1] = 1;  oppFace_[2] = 6;  oppFace_[3] = -1;
  // face 1
  oppFace_[4] = 0;  oppFace_[5] = 2;  oppFace_[6] = 7;  oppFace_[7] = -1;
  // face 2
  oppFace_[8] = 1;  oppFace_[9] = 3;  oppFace_[10] = 4; oppFace_[11] = -1;
  // face 3
  oppFace_[12] = 0; oppFace_[13] = 5; oppFace_[14] = 2; oppFace_[15] = -1;
  // face 4
  oppFace_[16] = 4; oppFace_[17] = 7; oppFace_[18] = 6; oppFace_[19] = 5;

  // standard integration location
  intgLoc_.resize(24);
  const double fortyFiveHundredFourths = 45.0/104.0;
  const double fortyOneHundredTwentyths = 41.0/120.0;
  const double sevenFiftyTwoths = 7.0/52.0;
  const double sevenTwentyFourths = 7.0/24.0;
  intgLoc_[0]  =  0.00;                     intgLoc_[1]  = -fortyFiveHundredFourths; intgLoc_[2]  = sevenFiftyTwoths; // surf 1    1->2
  intgLoc_[3]  =  fortyFiveHundredFourths;  intgLoc_[4]  = 0.00;                     intgLoc_[5]  = sevenFiftyTwoths; // surf 2    2->3
  intgLoc_[6]  =  0.00;                     intgLoc_[7]  = fortyFiveHundredFourths;  intgLoc_[8]  = sevenFiftyTwoths; // surf 3    3->4
  intgLoc_[9]  =  -fortyFiveHundredFourths; intgLoc_[10] = 0.0;                      intgLoc_[11] = sevenFiftyTwoths; // surf 4    1->4
  intgLoc_[12] =  -sevenTwentyFourths;      intgLoc_[13] = -sevenTwentyFourths;      intgLoc_[14] = fortyOneHundredTwentyths; // surf 5    1->5
  intgLoc_[15] =  sevenTwentyFourths;       intgLoc_[16] = -sevenTwentyFourths;      intgLoc_[17] = fortyOneHundredTwentyths; // surf 6    2->5
  intgLoc_[18] =  sevenTwentyFourths;       intgLoc_[19] = sevenTwentyFourths;       intgLoc_[20] = fortyOneHundredTwentyths; // surf 7    3->5
  intgLoc_[21] =  -sevenTwentyFourths;      intgLoc_[22] = sevenTwentyFourths;       intgLoc_[23] = fortyOneHundredTwentyths; // surf 8    4->5

  // shifted
  intgLocShift_.resize(24);
  intgLocShift_[0]  =  0.00; intgLocShift_[1]  = -1.00; intgLocShift_[2]  =  0.00; // surf 1    1->2
  intgLocShift_[3]  =  1.00; intgLocShift_[4]  =  0.00; intgLocShift_[5]  =  0.00; // surf 2    2->3
  intgLocShift_[6]  =  0.00; intgLocShift_[7]  =  1.00; intgLocShift_[8]  =  0.00; // surf 3    3->4
  intgLocShift_[9]  = -1.00; intgLocShift_[10] =  0.00; intgLocShift_[11] =  0.00; // surf 4    1->4
  intgLocShift_[12] = -0.50; intgLocShift_[13] = -0.50; intgLocShift_[14] =  0.50; // surf 5    1->5
  intgLocShift_[15] =  0.50; intgLocShift_[16] = -0.50; intgLocShift_[17] =  0.50; // surf 6    2->5
  intgLocShift_[18] =  0.50; intgLocShift_[19] =  0.50; intgLocShift_[20] =  0.50; // surf 7    3->5
  intgLocShift_[21] = -0.50; intgLocShift_[22] =  0.50; intgLocShift_[23] =  0.50; // surf 8    4->5

  // exposed face
  intgExpFace_.resize(48);
  const double five24ths = 5.0/24.0;
  const double nineteen24ths = 19.0/24.0;
  const double three8ths = 3.0/8.0;
  const double one3rd = 1.0/3.0;
  const double two3rds = 2.0/3.0;
  // face 0; nodes 0,1,4: scs 0, 1, 2
  intgExpFace_[0]  = -three8ths;     intgExpFace_[1]  = -nineteen24ths; intgExpFace_[2]  = five24ths;
  intgExpFace_[3]  =  three8ths;     intgExpFace_[4]  = -nineteen24ths; intgExpFace_[5]  = five24ths;
  intgExpFace_[6]  =  0.0;           intgExpFace_[7]  = -one3rd;        intgExpFace_[8]  = two3rds;
  // face 1; nodes 1,2,4; scs 0, 1, 2
  intgExpFace_[9]  = nineteen24ths;  intgExpFace_[10] = -three8ths;     intgExpFace_[11] = five24ths;
  intgExpFace_[12] = nineteen24ths;  intgExpFace_[13] =  three8ths;     intgExpFace_[14] = five24ths;
  intgExpFace_[15] = one3rd;         intgExpFace_[16] =  0.0;           intgExpFace_[17] = two3rds;
  // face 2; nodes 2,3,4; scs 0, 1, 2
  intgExpFace_[18] =  three8ths;     intgExpFace_[19] = nineteen24ths;  intgExpFace_[20] = five24ths;
  intgExpFace_[21] = -three8ths;     intgExpFace_[22] = nineteen24ths;  intgExpFace_[23] = five24ths;
  intgExpFace_[24] =  0.00;          intgExpFace_[25] = one3rd;         intgExpFace_[26] = two3rds;
  //face 3; nodes 0,4,3; scs 0, 1, 2
  intgExpFace_[27] = -nineteen24ths; intgExpFace_[28] = -three8ths;     intgExpFace_[29] = five24ths;
  intgExpFace_[30] = -one3rd;        intgExpFace_[31] = 0.0;            intgExpFace_[32] = two3rds;
  intgExpFace_[33] = -nineteen24ths; intgExpFace_[34] =  three8ths;     intgExpFace_[35] = five24ths;
  // face 4; nodes 0,3,2,1; scs 0, 1, 2
  intgExpFace_[36] = -0.5;           intgExpFace_[37] = -0.5;           intgExpFace_[38] = 0.0;
  intgExpFace_[39] = -0.5;           intgExpFace_[40] =  0.5;           intgExpFace_[41] = 0.0;
  intgExpFace_[42] =  0.5;           intgExpFace_[43] =  0.5;           intgExpFace_[44] = 0.0;
  intgExpFace_[45] =  0.5;           intgExpFace_[46] = -0.5;           intgExpFace_[47] = 0.0;

  sideNodeOrdinals_ = {
      0, 1, 4,    // ordinal 0
      1, 2, 4,    // ordinal 1
      2, 3, 4,    // ordinal 2
      0, 4, 3,    // ordinal 3
      0, 3, 2, 1  // ordinal 4
  };

  ipNodeMap_.resize(16);
  // Face 0
  ipNodeMap_[0]  = 0; ipNodeMap_[1]  = 1; ipNodeMap_[2]  = 4;
  // Face 1
  ipNodeMap_[3]  = 1; ipNodeMap_[4]  = 2; ipNodeMap_[5]  = 4;
  // Face 2
  ipNodeMap_[6]  = 2; ipNodeMap_[7]  = 3; ipNodeMap_[8]  = 4;
  // Face 3
  ipNodeMap_[9]  = 0; ipNodeMap_[10] = 4; ipNodeMap_[11] = 3;
  // Face 4 (quad face)
  ipNodeMap_[12] = 0; ipNodeMap_[13] = 3; ipNodeMap_[14] = 2; ipNodeMap_[15] = 1;

  std::vector<std::vector<double>> nodeLocations =
  {
      {-1.0, -1.0, +0.0}, {+1.0, -1.0, +0.0}, {+1.0, +1.0, +0.0}, {-1.0, +1.0, +0.0},
      {0.0, 0.0, +1.0}
  };

  intgExpFaceShift_.resize(48);
  int index = 0;
  stk::topology topo = stk::topology::PYRAMID_5;
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
PyrSCS::~PyrSCS()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
const int *
PyrSCS::side_node_ordinals(
  int ordinal)
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return &sideNodeOrdinals_[ordinal*3];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::determinant(
    SharedMemView<DoubleType**>& cordel,
    SharedMemView<DoubleType**>& areav)
{
  const int pyramidEdgeFacetTable[8][4] = {
    { 5,  9, 18, 12},  // sc face 1 -- points from 1 -> 2
    { 6,  9, 18, 14},  // sc face 2 -- points from 2 -> 3
    { 7,  9, 18, 16},  // sc face 3 -- points from 3 -> 4
    { 8, 17, 18,  9},  // sc face 4 -- points from 1 -> 4
    {11, 12, 18, 17},  // sc face 5 -- points from 1 -> 5
    {10, 14, 18, 12},  // sc face 6 -- points from 2 -> 5
    {13, 16, 18, 14},  // sc face 7 -- points from 3 -> 5
    {15, 17, 18, 16}   // sc face 8 -- points from 4 -> 5
  };
  DoubleType coords[19][3];
  DoubleType scscoords[4][3];
  const double half = 0.5;
  const double one3rd = 1.0/3.0;
  const double one4th = 1.0/4.0;

  // element vertices
  for(int j=0; j<5; ++j) {
    for(int k=0; k<3; ++k) {
      coords[j][k] = cordel(j,k);
    }
  }

  // face 1 (quad)
  // 4++++8+++3
  // +         +
  // +         +
  // 9   10    7
  // +         +
  // +         +
  // 1++++6++++2

  // edge midpoints
  for(int k=0; k<3; ++k) {
    coords[5][k] = half*(cordel(0,k) + cordel(1,k));
  }
  for(int k=0; k<3; ++k) {
    coords[6][k] = half*(cordel(1,k) + cordel(2,k));
  }
  for(int k=0; k<3; ++k) {
    coords[7][k] = half*(cordel(2,k) + cordel(3,k));
  }
  for(int k=0; k<3; ++k) {
    coords[8][k] = half*(cordel(3,k) + cordel(0,k));
  }

  // face midpoint
  for(int k=0; k<3; ++k) {
    coords[9][k] = one4th*(cordel(0,k) + cordel(1,k) + cordel(2,k) + cordel(3,k));
  }

  // face 2 (tri)
  //
  // edge midpoints
  for(int k=0; k<3; ++k) {
    coords[10][k] = half*(cordel(1,k) + cordel(4,k));
  }
  for(int k=0; k<3; ++k) {
    coords[11][k] = half*(cordel(4,k) + cordel(0,k));
  }

  // face midpoint
  for(int k=0; k<3; ++k) {
    coords[12][k] = one3rd*(cordel(0,k) + cordel(1,k) + cordel(4,k));
  }
  // face 3 (tri)

  // edge midpoint
  for(int k=0; k<3; ++k) {
    coords[13][k] = half*(cordel(2,k) + cordel(4,k));
  }

  // face midpoint
  for(int k=0; k<3; ++k) {
    coords[14][k] = one3rd*(cordel(1,k) + cordel(2,k) + cordel(4,k));
  }

  // face 4 (tri)

  // edge midpoint
  for(int k=0; k<3; ++k) {
    coords[15][k] = half*(cordel(3,k) + cordel(4,k));
  }

  // face midpoint
  for(int k=0; k<3; ++k) {
    coords[16][k] = one3rd*(cordel(3,k) + cordel(4,k) + cordel(2,k));
  }

  // face 5 (tri)

  // face midpoint
  for(int k=0; k<3; ++k) {
    coords[17][k] = one3rd*(cordel(0,k) + cordel(4,k) + cordel(3,k));
  }

  // element centroid
  for(int k=0; k<3; ++k) {
    coords[18][k] = 0.0;
  }
  for(int j=0; j<nodesPerElement_; ++j) {
    for(int k=0; k<3; ++k) {
      coords[18][k] += 0.2*cordel(j,k);
    }
  }

  // loop over subcontrol surfaces
  for(int ics=0; ics<numIntPoints_; ++ics) {
    // loop over vertices of scs
    for(int inode=0; inode<4; ++inode) {
      // set coordinates of vertices using node table
      int itrianglenode = pyramidEdgeFacetTable[ics][inode];
      for(int k=0; k<3; ++k) {
        scscoords[inode][k] = coords[itrianglenode][k];
      }
    }
    // compute area vector using triangle decomposition
    quad_area_by_triangulation( ics, scscoords, areav );
  }
}

void PyrSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  SIERRA_FORTRAN(pyr_scs_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords, areav );

  // all is always well; no error checking
  *error = 0;
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv)
{
  pyr_deriv(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsPyr5>(deriv, coords, gradop);
}

void PyrSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  pyr_derivative(numIntPoints_, &intgLoc_[0], deriv);
  
  SIERRA_FORTRAN(pyr_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative PyrSCS volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::shifted_grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv)
{
  pyr_deriv(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op<AlgTraitsPyr5>(deriv, coords, gradop);
}

void PyrSCS::shifted_grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  pyr_derivative(numIntPoints_, &intgLocShift_[0], deriv);

  SIERRA_FORTRAN(pyr_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative PyrSCS volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  const int ndim = 3;  
  const int nface = 1;
  double dpsi[15];

  // ordinal four is a quad4
  const int npf = (face_ordinal < 4 ) ? 3 : 4;

  for ( int n=0; n<nelem; n++ ) {
    
    for ( int k=0; k<npf; k++ ) {

      const int row = 9*face_ordinal + k*ndim;
      pyr_derivative(nface, &intgExpFace_[row], dpsi);
      
      SIERRA_FORTRAN(pyr_gradient_operator)
        ( &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[15*n], &gradop[k*nelem*15+n*15], &det_j[npf*n+k], error, &lerr );
      
      if ( lerr )
        NaluEnv::self().naluOutput() << "problem with PyrSCS::face_grad_op." << std::endl;
    }
  }
}

void PyrSCS::face_grad_op_quad(const int face_ordinal, const bool shifted, 
                               SharedMemView<DoubleType**>& coords, QuadFaceGradType& gradop)
{
  using tri_traits = AlgTraitsTri3Pyr5;
  using quad_traits = AlgTraitsQuad4Pyr5;

  // quad4 is the only face that can be safely shifted
  const double *p_intgExp = (!shifted || face_ordinal < 4 ) ? &intgExpFace_[0] : &intgExpFaceShift_[0];

  constexpr int derivSize = quad_traits::numFaceIp_ *  quad_traits::nodesPerElement_ * quad_traits::nDim_;

  NALU_ALIGNED DoubleType psi[derivSize];
  QuadFaceGradType deriv(psi,quad_traits::numFaceIp_,quad_traits::nodesPerElement_,quad_traits::nDim_);

  const int offset = (face_ordinal != 4) ? 0 : tri_traits::nDim_ * tri_traits::numFaceIp_ * face_ordinal;
  pyr_deriv(quad_traits::numFaceIp_, &p_intgExp[offset], deriv);
  generic_grad_op<AlgTraitsPyr5>(deriv, coords, gradop);
}

void PyrSCS::face_grad_op_tri(const int face_ordinal, const bool shifted, 
                               SharedMemView<DoubleType**>& coords, TriFaceGradType& gradop)
{
  using tri_traits = AlgTraitsTri3Pyr5;
  // quad4 is the only face that can be safely shifted
  const double *p_intgExp = (!shifted || face_ordinal < 4 ) ? &intgExpFace_[0] : &intgExpFaceShift_[0];

  constexpr int derivSize = tri_traits::numFaceIp_ *  tri_traits::nodesPerElement_ * tri_traits::nDim_;
  NALU_ALIGNED DoubleType psi[derivSize];
  TriFaceGradType deriv(psi,tri_traits::numFaceIp_,tri_traits::nodesPerElement_,tri_traits::nDim_ );

  const int offset = (face_ordinal != 4) ? tri_traits::nDim_ * tri_traits::numFaceIp_ * face_ordinal : 0;
  pyr_deriv(tri_traits::numFaceIp_, &p_intgExp[offset], deriv);
  generic_grad_op<AlgTraitsPyr5>(deriv, coords, gradop);
}

void PyrSCS::face_grad_op(
  const int face_ordinal,
  const bool shifted,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using tri_traits = AlgTraitsTri3Pyr5;
  using quad_traits = AlgTraitsQuad4Pyr5;

  constexpr int quad_derivSize = quad_traits::numFaceIp_ *  quad_traits::nodesPerElement_ * quad_traits::nDim_;
  NALU_ALIGNED DoubleType quad_grad_temp[quad_derivSize];
  QuadFaceGradType quad_gradop(quad_grad_temp,quad_traits::numFaceIp_,quad_traits::nodesPerElement_,quad_traits::nDim_);
  face_grad_op_quad(face_ordinal, shifted, coords, quad_gradop);

  constexpr int tri_derivSize = tri_traits::numFaceIp_ *  tri_traits::nodesPerElement_ * tri_traits::nDim_;
  NALU_ALIGNED DoubleType tri_grad_temp[tri_derivSize];
  TriFaceGradType tri_gradop(tri_grad_temp,tri_traits::numFaceIp_,tri_traits::nodesPerElement_,tri_traits::nDim_ );
  face_grad_op_tri(face_ordinal, shifted, coords, tri_gradop);

  const int length = (face_ordinal == 4) ? quad_derivSize : tri_derivSize;
  DoubleType triMask = (face_ordinal == 4) ? 0: 1;
  DoubleType* gradop_ptr = gradop.ptr_on_device();
  for (int k = 0; k < length; ++k) {
    gradop_ptr[k] = (1 - triMask) * quad_grad_temp[k] + triMask * tri_grad_temp[k];
  }
}

void PyrSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  constexpr bool shifted = false;
  face_grad_op(face_ordinal, shifted, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_face_grad_op -------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  constexpr bool shifted = true;
  face_grad_op(face_ordinal, shifted, coords, gradop);
}

void PyrSCS::shifted_face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  const int ndim = 3;
  const int nface = 1;
  double dpsi[15];

  // ordinal four is a quad4
  const int npf = (face_ordinal < 4 ) ? 3 : 4;

  // quad4 is the only face that can be safely shifted
  const double *p_intgExp = (face_ordinal < 4 ) ? &intgExpFace_[0] : &intgExpFaceShift_[0];
  for ( int n=0; n<nelem; n++ ) {

    for ( int k=0; k<npf; k++ ) {

      const int row = 9*face_ordinal + k*ndim;
      pyr_derivative(nface, &p_intgExp[row], dpsi);

      SIERRA_FORTRAN(pyr_gradient_operator)
        ( &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[15*n], &gradop[k*nelem*15+n*15], &det_j[npf*n+k], error, &lerr );

      if ( lerr )
        NaluEnv::self().naluOutput() << "problem with PyrSCS::shifted_face_grad_op." << std::endl;

    }
  }
}

double PyrSCS::parametric_distance(const std::array<double, 3>& x)
{
  const double X = x[0];
  const double Y = x[1];
  const double Z = x[2] - 1. / 3.;
  const double dist0 = (3. / 2.) * (Z + std::max(std::fabs(X), std::fabs(Y)));
  const double dist1 = -3 * Z;
  const double dist = std::max(dist0, dist1);
  return dist;
}

double dot5(const double* u, const double* v)
{
  return (u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + u[4] * v[4]);
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
PyrSCS::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result )
{
  double shapefct[5];
  pyr_shape_fcn(1, isoParCoord, shapefct);

  for (int i = 0; i < nComp; i++) {
    result[i] = dot5(shapefct, field + 5 * i);
  }
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double PyrSCS::isInElement(
  const double *elemNodalCoord,
  const double *pointCoord,
  double *isoParCoord)
{
  // control the interation
  double isInElemConverged = 1.0e-16; // NOTE: the square of the tolerance on the distance
  int N_MAX_ITER = 100;

  constexpr int dim = 3;
  std::array<double, dim> guess = { { 0.0, 0.0, 1.0 / 3.0 } };
  std::array<double, dim> delta;
  int iter = 0;

  do {
    // interpolate coordinate at guess
    constexpr int nNodes = 5;
    std::array<double, nNodes> weights;
    pyr_shape_fcn(1, guess.data(), weights.data());

    // compute difference between coordinates interpolated to the guessed isoParametric coordinates
    // and the actual point's coordinates
    std::array<double, dim> error_vec;
    error_vec[0] = pointCoord[0] - dot5(weights.data(), elemNodalCoord + 0 * nNodes);
    error_vec[1] = pointCoord[1] - dot5(weights.data(), elemNodalCoord + 1 * nNodes);
    error_vec[2] = pointCoord[2] - dot5(weights.data(), elemNodalCoord + 2 * nNodes);

    // update guess along gradient of mapping from physical-to-reference coordinates
    // transpose of the jacobian of the forward mapping
    constexpr int deriv_size = nNodes * dim;
    std::array<double, deriv_size> deriv;
    pyr_derivative(1, guess.data(), deriv.data());

    std::array<double, dim * dim> jact{};
    for(int j = 0; j < nNodes; ++j) {
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
  } while(!within_tolerance(vector_norm_sq(delta.data(), 3), isInElemConverged) && (++iter < N_MAX_ITER));

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
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void 
PyrSCS::general_face_grad_op(
  const int face_ordinal,
  const double *isoParCoord,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  const int nface = 1;

  double dpsi[15];

  pyr_derivative(nface, &isoParCoord[0], dpsi);
      
  SIERRA_FORTRAN(pyr_gradient_operator)
    ( &nface,
      &nodesPerElement_,
      &nface,
      dpsi,
      &coords[0], &gradop[0], &det_j[0], error, &lerr );
  
  if ( lerr )
    NaluEnv::self().naluOutput() << "PyrSCS::general_face_grad_op: issue.." << std::endl;
  
}

//--------------------------------------------------------------------------
//-------- pyr_derivative --------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::pyr_derivative(
  const int npts,
  const double *intgLoc,
  double *deriv)
{
  // d3d(c,s,j) = deriv[c + 3*(s + 5*j)] = deriv[c+3s+15j]

  for ( int j = 0; j < npts; ++j) {
    const int k = j*3;
    const int p = 15*j;

    double r = intgLoc[k+0];
    double s = intgLoc[k+1];
    double t = intgLoc[k+2];

    deriv[0+3*0+p] =-0.25*(1.0-s)*(1.0-t);  // d(N_1)/ d(r) = deriv[0]
    deriv[1+3*0+p] =-0.25*(1.0-r)*(1.0-t);  // d(N_1)/ d(s) = deriv[1]
    deriv[2+3*0+p] =-0.25*(1.0-r)*(1.0-s);  // d(N_1)/ d(t) = deriv[2]

    deriv[0+3*1+p] = 0.25*(1.0-s)*(1.0-t);  // d(N_2)/ d(r) = deriv[0+3]
    deriv[1+3*1+p] =-0.25*(1.0+r)*(1.0-t);  // d(N_2)/ d(s) = deriv[1+3]
    deriv[2+3*1+p] =-0.25*(1.0+r)*(1.0-s);  // d(N_2)/ d(t) = deriv[2+3]

    deriv[0+3*2+p] = 0.25*(1.0+s)*(1.0-t);  // d(N_3)/ d(r) = deriv[0+6]
    deriv[1+3*2+p] = 0.25*(1.0+r)*(1.0-t);  // d(N_3)/ d(s) = deriv[1+6]
    deriv[2+3*2+p] =-0.25*(1.0+r)*(1.0+s);  // d(N_3)/ d(t) = deriv[2+6]

    deriv[0+3*3+p] =-0.25*(1.0+s)*(1.0-t);  // d(N_4)/ d(r) = deriv[0+9]
    deriv[1+3*3+p] = 0.25*(1.0-r)*(1.0-t);  // d(N_4)/ d(s) = deriv[1+9]
    deriv[2+3*3+p] =-0.25*(1.0-r)*(1.0+s);  // d(N_4)/ d(t) = deriv[2+9]

    deriv[0+3*4+p] = 0.0;                   // d(N_5)/ d(r) = deriv[0+12]
    deriv[1+3*4+p] = 0.0;                   // d(N_5)/ d(s) = deriv[1+12]
    deriv[2+3*4+p] = 1.0;                   // d(N_5)/ d(t) = deriv[2+12]
  }
}

//--------------------------------------------------------------------------
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::gij( 
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv)
{
  generic_gij_3d<AlgTraitsPyr5>(deriv, coords, gupper, glower);
}

void PyrSCS::gij(
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
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
const int *
PyrSCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
PyrSCS::shape_fcn(double *shpfc)
{
  pyr_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
PyrSCS::shifted_shape_fcn(double *shpfc)
{
  pyr_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- pyr_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
void
PyrSCS::pyr_shape_fcn(
  const int  &npts,
  const double *par_coord, 
  double *shape_fcn)
{
  const double one  = 1.0;
  for ( int j = 0; j < npts; ++j ) {
    const int fivej = 5*j;
    const int k     = 3*j;
    const double r    = par_coord[k+0];
    const double s    = par_coord[k+1];
    const double t    = par_coord[k+2];

    shape_fcn[0 + fivej] = 0.25*(1.0-r)*(1.0-s)*(one-t);
    shape_fcn[1 + fivej] = 0.25*(1.0+r)*(1.0-s)*(one-t);
    shape_fcn[2 + fivej] = 0.25*(1.0+r)*(1.0+s)*(one-t);
    shape_fcn[3 + fivej] = 0.25*(1.0-r)*(1.0+s)*(one-t);
    shape_fcn[4 + fivej] = t;
  }
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
int
PyrSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*4+node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
int
PyrSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*4+node];
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
PyrSCS::ipNodeMap(
  int ordinal)
{
  // define ip->node mappings for each face (ordinal); 
  return &ipNodeMap_[ordinal*3];
}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void 
PyrSCS::sidePcoords_to_elemPcoords(
  const int & side_ordinal,
  const int & npoints,
  const double *side_pcoords,
  double *elem_pcoords)
{
  ThrowRequireMsg(side_ordinal >= 0 && side_ordinal <= 4,
    "Invalid pyramid side ordinal " + std::to_string(side_ordinal));

  for (int i = 0; i < npoints; i++) {
    const double x = side_pcoords[2 * i + 0];
    const double y = side_pcoords[2 * i + 1];
    switch (side_ordinal)
    {
      case 0:
      {
        elem_pcoords[i * 3 + 0] = -1 + 2 * x + y;
        elem_pcoords[i * 3 + 1] = -1 + y;
        elem_pcoords[i * 3 + 2] = y;
        break;
      }
      case 1:
      {
        elem_pcoords[i * 3 + 0] = 1 - y;
        elem_pcoords[i * 3 + 1] = -1 + 2 * x + y;
        elem_pcoords[i * 3 + 2] = y;
        break;
      }
      case 2:
      {
        elem_pcoords[i * 3 + 0] = 1 - 2 * x - y;
        elem_pcoords[i * 3 + 1] = 1 - y;
        elem_pcoords[i * 3 + 2] = y;
        break;
      }
      case 3:
      {
        elem_pcoords[i * 3 + 0] = -1 + x;
        elem_pcoords[i * 3 + 1] = -1 + x + 2 * y;
        elem_pcoords[i * 3 + 2] = x;
        break;
      }
      case 4:
      {
        elem_pcoords[i * 3 + 0] = y;
        elem_pcoords[i * 3 + 1] = x;
        elem_pcoords[i * 3 + 2] = 0;
        break;
      }
      default:
        break;
    }
  }
}

} // namespace nalu
} // namespace sierra
