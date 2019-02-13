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

//-------- pyr_deriv -------------------------------------------------------
template <typename DerivType>
void pyr_deriv(const int npts,
  const double *intgLoc,
  DerivType& deriv)
{
  // d3d(c,s,j) = deriv[c + 3*(s + 5*j)] = deriv[c+3s+15j]

  const double eps = std::numeric_limits<double>::epsilon();
  
  for ( int j = 0; j < npts; ++j) {
    const int k = j*3;
    
    const double r = intgLoc[k+0];
    const double s = intgLoc[k+1];
    const double t_tmp = intgLoc[k+2];
    
    const double one_minus_t = 1.0 - t_tmp;
    const double t = (std::fabs(one_minus_t) > eps) ? t_tmp : 1.0 + std::copysign(eps, one_minus_t);
    const double quarter_inv_tm1 = 0.25 / (1.0 - t);
    const double t_term = 4.0 * r * s * quarter_inv_tm1 * quarter_inv_tm1;
    
    deriv(j,0,0) = -(1.0 - s - t) * quarter_inv_tm1;
    deriv(j,0,1) = -(1.0 - r - t) * quarter_inv_tm1;
    deriv(j,0,2) = (+t_term - 0.25);
    
    // node 1
    deriv(j,1,0) = +(1.0 - s - t) * quarter_inv_tm1;
    deriv(j,1,1) = -(1.0 + r - t) * quarter_inv_tm1;
    deriv(j,1,2) = (-t_term - 0.25);
    
    // node 2
    deriv(j,2,0) = +(1.0 + s - t) * quarter_inv_tm1;
    deriv(j,2,1) = +(1.0 + r - t) * quarter_inv_tm1;
    deriv(j,2,2) = (+t_term - 0.25);
    
    // node 3
    deriv(j,3,0) = -(1.0 + s - t) * quarter_inv_tm1;
    deriv(j,3,1) = +(1.0 - r - t) * quarter_inv_tm1;
    deriv(j,3,2) = (-t_term - 0.25);
    
    // node 4
    deriv(j,4,0) = 0.0;
    deriv(j,4,1) = 0.0;
    deriv(j,4,2) = 1.0;
  }
}

//-------- shifted_pyr_deriv -------------------------------------------------------
template <typename DerivType>
void shifted_pyr_deriv(const int npts,
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
  ndim(AlgTraits::nDim_);
  nodesPerElement_ = 5;
  numIntPoints_ = 5; 

  // define ip node mappings
  ipNodeMap_.resize(5);
  ipNodeMap_[0] = 0; ipNodeMap_[1] = 1; ipNodeMap_[2] = 2; ipNodeMap_[3] = 3;
  ipNodeMap_[4] = 4;

  // standard integration location
  intgLoc_.resize(15);
  const double one69r384 = 169.0/384.0;
  const double five77r3840 = 577.0/3840.0;
  const double seven73r1560 = 773.0/1560.0;
  intgLoc_[0]  = -one69r384; intgLoc_[1]  = -one69r384; intgLoc_[2]  = five77r3840;  // vol 0
  intgLoc_[3]  =  one69r384; intgLoc_[4]  = -one69r384; intgLoc_[5]  = five77r3840;  // vol 1
  intgLoc_[6]  =  one69r384; intgLoc_[7]  =  one69r384; intgLoc_[8]  = five77r3840;  // vol 2
  intgLoc_[9]  = -one69r384; intgLoc_[10] =  one69r384; intgLoc_[11] = five77r3840;  // vol 3
  intgLoc_[12] =   0.0;      intgLoc_[13] =   0.0;      intgLoc_[14] = seven73r1560; // vol 4

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

DoubleType polyhedral_volume_by_faces(int  /* ncoords */, const DoubleType volcoords[][3],
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
    coords[10][k] = 0.50*( volcoords[3][k] + volcoords[9][k] );
  }
  for(int k=0; k<3; ++k) {
    coords[11][k] = 0.50*( volcoords[3][k] + volcoords[5][k] );
  }
  for(int k=0; k<3; ++k) {
    coords[12][k] = 0.50*( volcoords[5][k] + volcoords[7][k] );
  }
  for(int k=0; k<3; ++k) {
    coords[13][k] = 0.50*( volcoords[7][k] + volcoords[9][k] );
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
     {1,  6,  9,  5, 10, 14, 18, 12, -1, -1},
     {2,  7,  9,  6, 13, 16, 18, 14, -1, -1},
     {3,  8,  9,  7, 15, 17, 18, 16, -1, -1},
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
    vol(icv) = bhex_volume_grandy(ehexcoords);
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
  shifted_pyr_deriv(numIntPoints_, &intgLocShift_[0], deriv);
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
  shifted_pyr_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
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
  const double eps = std::numeric_limits<double>::epsilon();
  
  for ( int j = 0; j < npts; ++j ) {
    const int fivej = 5*j;
    const int k     = 3*j;
    const double r    = par_coord[k+0];
    const double s    = par_coord[k+1];
    const double t_tmp    = par_coord[k+2];
    
    const double one_minus_t = 1.0 - t_tmp;
    const double t = (std::fabs(one_minus_t) > eps) ? t_tmp : 1.0 + std::copysign(eps, one_minus_t);
    const double quarter_inv_tm1 = 0.25 / (1.0 - t);
    
    shape_fcn[0 + fivej] = (1.0 - r - t) * (1.0 - s - t) * quarter_inv_tm1;
    shape_fcn[1 + fivej] = (1.0 + r - t) * (1.0 - s - t) * quarter_inv_tm1;
    shape_fcn[2 + fivej] = (1.0 + r - t) * (1.0 + s - t) * quarter_inv_tm1;
    shape_fcn[3 + fivej] = (1.0 - r - t) * (1.0 + s - t) * quarter_inv_tm1;
    shape_fcn[4 + fivej] = t;
    
  }
}

//--------------------------------------------------------------------------
//-------- shifted_pyr_shape_fcn -------------------------------------------
//--------------------------------------------------------------------------
void
PyrSCV::shifted_pyr_shape_fcn(
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
//-------- Mij -------------------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraitsPyr5>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void PyrSCS::Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv)
{
  generic_Mij_3d<AlgTraitsPyr5>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
PyrSCS::PyrSCS()
  : MasterElement()
{
  ndim(AlgTraits::nDim_);
  nodesPerElement_ = 5;
  numIntPoints_ = 12;

  // define L/R mappings
  lrscv_.resize(24);
  lrscv_[0]  = 0; lrscv_[1]  = 1;
  lrscv_[2]  = 1; lrscv_[3]  = 2;
  lrscv_[4]  = 2; lrscv_[5]  = 3;
  lrscv_[6]  = 0; lrscv_[7]  = 3;
  lrscv_[8]  = 0; lrscv_[9]  = 4;
  lrscv_[10] = 0; lrscv_[11] = 4;
  lrscv_[12] = 1; lrscv_[13] = 4;
  lrscv_[14] = 1; lrscv_[15] = 4;
  lrscv_[16] = 2; lrscv_[17] = 4;
  lrscv_[18] = 2; lrscv_[19] = 4;
  lrscv_[20] = 3; lrscv_[21] = 4;
  lrscv_[22] = 3; lrscv_[23] = 4;

  // elem-edge map from ip
  scsIpEdgeOrd_.resize(numIntPoints_);
  scsIpEdgeOrd_[0]  = 0; scsIpEdgeOrd_[1]  = 1; 
  scsIpEdgeOrd_[2]  = 2; scsIpEdgeOrd_[3]  = 3; 
  scsIpEdgeOrd_[4]  = 4; scsIpEdgeOrd_[5]  = 4; 
  scsIpEdgeOrd_[6]  = 5; scsIpEdgeOrd_[7]  = 5;
  scsIpEdgeOrd_[8]  = 6; scsIpEdgeOrd_[9]  = 6;
  scsIpEdgeOrd_[10] = 7; scsIpEdgeOrd_[11] = 7;

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
  oppFace_[0] = 3;  oppFace_[1] = 1;  oppFace_[2] = 8;  oppFace_[3] = -1;
  // face 1
  oppFace_[4] = 0;  oppFace_[5] = 2;  oppFace_[6] = 10; oppFace_[7] = -1;
  // face 2
  oppFace_[8] = 1;  oppFace_[9] = 3;  oppFace_[10] = 4; oppFace_[11] = -1;
  // face 3
  oppFace_[12] = 0; oppFace_[13] = 6; oppFace_[14] = 2; oppFace_[15] = -1;
  // face 4
  oppFace_[16] = 4; oppFace_[17] = 10; oppFace_[18] = 8; oppFace_[19] = 6;

  // standard integration location
  intgLoc_.resize(36);
  const double twentynine63rd = 29.0/63.0;
  const double fortyone315th = 41.0/315.0;
  const double two9th = 2.0/9.0;
  const double thirteen45th = 13.0/45.0;
  const double seven18th = 7.0/18.0;
  intgLoc_[0]  = 0.0;             intgLoc_[1]  = -twentynine63rd; intgLoc_[2]  = fortyone315th; // surf 0  1->2
  intgLoc_[3]  = twentynine63rd;  intgLoc_[4]  = 0.0;             intgLoc_[5]  = fortyone315th; // surf 1  2->3
  intgLoc_[6]  = 0.0;             intgLoc_[7]  = twentynine63rd;  intgLoc_[8]  = fortyone315th; // surf 2  3->4
  intgLoc_[9]  = -twentynine63rd; intgLoc_[10] = 0.0;             intgLoc_[11] = fortyone315th; // surf 3  1->4
  intgLoc_[12] = -two9th;         intgLoc_[13] = -two9th;         intgLoc_[14] = thirteen45th;  // surf 4  1->5 inner
  intgLoc_[15] = -seven18th;      intgLoc_[16] = -seven18th;      intgLoc_[17] = seven18th;     // surf 5  1->5 outer
  intgLoc_[18] = two9th;          intgLoc_[19] = -two9th;         intgLoc_[20] = thirteen45th;  // surf 6  2->5 inner
  intgLoc_[21] = seven18th;       intgLoc_[22] = -seven18th;      intgLoc_[23] = seven18th;     // surf 7  2->5 outer
  intgLoc_[24] = two9th;          intgLoc_[25] = two9th;          intgLoc_[26] = thirteen45th;  // surf 8  3->5 inner
  intgLoc_[27] = seven18th;       intgLoc_[28] = seven18th;       intgLoc_[29] = seven18th;     // surf 9  3->5 outer
  intgLoc_[30] = -two9th;         intgLoc_[31] = two9th;          intgLoc_[32] = thirteen45th;  // surf 10  4->5 inner
  intgLoc_[33] = -seven18th;      intgLoc_[34] = seven18th;       intgLoc_[35] = seven18th;     // surf 11  4->5 outer

  // shifted
  intgLocShift_.resize(36);
  intgLocShift_[0]  =  0.00; intgLocShift_[1]  = -1.00; intgLocShift_[2]  =  0.00; // surf 1    1->2
  intgLocShift_[3]  =  1.00; intgLocShift_[4]  =  0.00; intgLocShift_[5]  =  0.00; // surf 2    2->3
  intgLocShift_[6]  =  0.00; intgLocShift_[7]  =  1.00; intgLocShift_[8]  =  0.00; // surf 3    3->4
  intgLocShift_[9]  = -1.00; intgLocShift_[10] =  0.00; intgLocShift_[11] =  0.00; // surf 4    1->4
  intgLocShift_[12] = -0.50; intgLocShift_[13] = -0.50; intgLocShift_[14] =  0.50; // surf 5    1->5 I
  intgLocShift_[15] = -0.50; intgLocShift_[16] = -0.50; intgLocShift_[17] =  0.50; // surf 6    1->5 O
  intgLocShift_[18] =  0.50; intgLocShift_[19] = -0.50; intgLocShift_[20] =  0.50; // surf 7    2->5 I
  intgLocShift_[21] =  0.50; intgLocShift_[22] = -0.50; intgLocShift_[23] =  0.50; // surf 8    2->5 O
  intgLocShift_[24] =  0.50; intgLocShift_[25] =  0.50; intgLocShift_[26] =  0.50; // surf 9    3->5 I 
  intgLocShift_[27] =  0.50; intgLocShift_[28] =  0.50; intgLocShift_[29] =  0.50; // surf 10   3->5 O
  intgLocShift_[30] = -0.50; intgLocShift_[31] =  0.50; intgLocShift_[32] =  0.50; // surf 11   4->5 I 
  intgLocShift_[33] = -0.50; intgLocShift_[34] =  0.50; intgLocShift_[35] =  0.50; // surf 12   4->5 O

  // exposed face
  intgExpFace_.resize(48);
  const double seven36th = 7.0/36.0;
  const double twentynine36th = 29.0/36.0;
  const double five12th = 5.0/12.0;
  const double eleven18th = 11.0/18.0;
  // face 0; nodes 0,1,4: scs 0, 1, 2
  intgExpFace_[0]  = -five12th;       intgExpFace_[1]  = -twentynine36th; intgExpFace_[2]  = seven36th;
  intgExpFace_[3]  =  five12th;       intgExpFace_[4]  = -twentynine36th; intgExpFace_[5]  = seven36th;
  intgExpFace_[6]  =  0.0;            intgExpFace_[7]  = -seven18th;      intgExpFace_[8]  = eleven18th;
  // face 1; nodes 1,2,4; scs 0, 1, 2
  intgExpFace_[9]  = twentynine36th;  intgExpFace_[10] = -five12th;       intgExpFace_[11] = seven36th;
  intgExpFace_[12] = twentynine36th;  intgExpFace_[13] =  five12th;       intgExpFace_[14] = seven36th;
  intgExpFace_[15] = seven18th;       intgExpFace_[16] =  0.0;            intgExpFace_[17] = eleven18th;
  // face 2; nodes 2,3,4; scs 0, 1, 2
  intgExpFace_[18] =  five12th;       intgExpFace_[19] = twentynine36th;  intgExpFace_[20] = seven36th;
  intgExpFace_[21] = -five12th;       intgExpFace_[22] = twentynine36th;  intgExpFace_[23] = seven36th;
  intgExpFace_[24] =  0.00;           intgExpFace_[25] = seven18th;       intgExpFace_[26] = eleven18th;
  //face 3; nodes 0,4,3; scs 0, 1, 2
  intgExpFace_[27] = -twentynine36th; intgExpFace_[28] = -five12th;       intgExpFace_[29] = seven36th;
  intgExpFace_[30] = -seven18th;      intgExpFace_[31] = 0.0;             intgExpFace_[32] = eleven18th;
  intgExpFace_[33] = -twentynine36th; intgExpFace_[34] =  five12th;       intgExpFace_[35] = seven36th;
  // face 4; nodes 0,3,2,1; scs 0, 1, 2
  intgExpFace_[36] = -0.5;            intgExpFace_[37] = -0.5;            intgExpFace_[38] = 0.0;
  intgExpFace_[39] = -0.5;            intgExpFace_[40] =  0.5;            intgExpFace_[41] = 0.0;
  intgExpFace_[42] =  0.5;            intgExpFace_[43] =  0.5;            intgExpFace_[44] = 0.0;
  intgExpFace_[45] =  0.5;            intgExpFace_[46] = -0.5;            intgExpFace_[47] = 0.0;

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
  const int pyramidEdgeFacetTable[12][4] = {
    { 5,  9, 18, 12},  // sc face 1  -- points from 1 -> 2
    { 6,  9, 18, 14},  // sc face 2  -- points from 2 -> 3
    { 7,  9, 18, 16},  // sc face 3  -- points from 3 -> 4
    { 8, 17, 18,  9},  // sc face 4  -- points from 1 -> 4
    {12, 12, 18, 17},  // sc face 5  -- points from 1 -> 5 I
    {11, 12, 12, 17},  // sc face 6  -- points from 1 -> 5 O
    {14, 14, 18, 12},  // sc face 7  -- points from 2 -> 5 I
    {10, 14, 14, 12},  // sc face 8  -- points from 2 -> 5 O
    {16, 16, 18, 14},  // sc face 9  -- points from 3 -> 5 I
    {13, 16, 16, 14},  // sc face 10 -- points from 3 -> 5 O
    {17, 17, 18, 16},  // sc face 11 -- points from 4 -> 5 I
    {15, 17, 17, 16}   // sc face 12 -- points from 4 -> 5 O
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
  shifted_pyr_deriv(numIntPoints_, &intgLocShift_[0], deriv);
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

  shifted_pyr_derivative(numIntPoints_, &intgLocShift_[0], deriv);

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

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using tri_traits = AlgTraitsTri3Wed6;
  using quad_traits = AlgTraitsQuad4Wed6;
  constexpr int dim = 3;

  constexpr int maxDerivSize = quad_traits::numFaceIp_ *  quad_traits::nodesPerElement_ * dim;
  NALU_ALIGNED DoubleType psi[maxDerivSize];

  const int numFaceIps = (face_ordinal == 4) ? quad_traits::numFaceIp_ : tri_traits::numFaceIp_;
  SharedMemView<DoubleType***> deriv(psi, numFaceIps, AlgTraitsPyr5::nodesPerElement_, dim);

  const int offset = tri_traits::numFaceIp_ * face_ordinal;
  pyr_deriv(numFaceIps, &intgExpFace_[dim * offset], deriv);
  generic_grad_op<AlgTraitsPyr5>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using tri_traits = AlgTraitsTri3Wed6;
  using quad_traits = AlgTraitsQuad4Wed6;
  constexpr int dim = 3;

  constexpr int maxDerivSize = quad_traits::numFaceIp_ *  quad_traits::nodesPerElement_ * dim;
  NALU_ALIGNED DoubleType psi[maxDerivSize];

  const int numFaceIps = (face_ordinal == 4) ? quad_traits::numFaceIp_ : tri_traits::numFaceIp_;
  SharedMemView<DoubleType***> deriv(psi, numFaceIps, AlgTraitsPyr5::nodesPerElement_, dim);

  const int offset = tri_traits::numFaceIp_ * face_ordinal;
  shifted_pyr_deriv(numFaceIps, &intgExpFaceShift_[dim * offset], deriv);
  generic_grad_op<AlgTraitsPyr5>(deriv, coords, gradop);
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
      shifted_pyr_derivative(nface, &p_intgExp[row], dpsi);

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
  const int  /* face_ordinal */,
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
  const double eps = std::numeric_limits<double>::epsilon();
  
  for ( int j = 0; j < npts; ++j) {
    const int k = j*3;
    const int p = 15*j;
    
    const double r = intgLoc[k+0];
    const double s = intgLoc[k+1];
    const double t_tmp = intgLoc[k+2];
    
    const double one_minus_t = 1.0 - t_tmp;
    const double t = (std::fabs(one_minus_t) > eps) ? t_tmp : 1.0 + std::copysign(eps, one_minus_t);
    const double quarter_inv_tm1 = 0.25 / (1.0 - t);
    const double t_term = 4.0 * r * s * quarter_inv_tm1 * quarter_inv_tm1;
    
    // node 0
    deriv[0+3*0+p] = -(1.0 - s - t) * quarter_inv_tm1;
    deriv[1+3*0+p] = -(1.0 - r - t) * quarter_inv_tm1;
    deriv[2+3*0+p] = (+t_term - 0.25);
    
    // node 1
    deriv[0+3*1+p] = +(1.0 - s - t) * quarter_inv_tm1;
    deriv[1+3*1+p] = -(1.0 + r - t) * quarter_inv_tm1;
    deriv[2+3*1+p] = (-t_term - 0.25);
    
    // node 2
    deriv[0+3*2+p] = +(1.0 + s - t) * quarter_inv_tm1;
    deriv[1+3*2+p] = +(1.0 + r - t) * quarter_inv_tm1;
    deriv[2+3*2+p] = (+t_term - 0.25);
    
    // node 3
    deriv[0+3*3+p] = -(1.0 + s - t) * quarter_inv_tm1;
    deriv[1+3*3+p] = +(1.0 - r - t) * quarter_inv_tm1;
    deriv[2+3*3+p] = (-t_term - 0.25);
    
    // node 4
    deriv[0+3*4+p] = 0.0;
    deriv[1+3*4+p] = 0.0;
    deriv[2+3*4+p] = 1.0;
  } 
}

//--------------------------------------------------------------------------
//-------- shifted_pyr_derivative ------------------------------------------
//--------------------------------------------------------------------------
void PyrSCS::shifted_pyr_derivative(
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
//-------- Mij -------------------------------------------------------------
//--------------------------------------------------------------------------
void PyrSCV::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraitsPyr5>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void PyrSCV::Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv)
{
  generic_Mij_3d<AlgTraitsPyr5>(deriv, coords, metric);
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
//-------- scsIpEdgeOrd ----------------------------------------------------
//--------------------------------------------------------------------------
const int *
PyrSCS::scsIpEdgeOrd()
{
  return &scsIpEdgeOrd_[0];
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
  shifted_pyr_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
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
  const double eps = std::numeric_limits<double>::epsilon();
  
  for ( int j = 0; j < npts; ++j ) {
    const int fivej = 5*j;
    const int k     = 3*j;
    const double r    = par_coord[k+0];
    const double s    = par_coord[k+1];
    const double t_tmp    = par_coord[k+2];
    
    const double one_minus_t = 1.0 - t_tmp;
    const double t = (std::fabs(one_minus_t) > eps) ? t_tmp : 1.0 + std::copysign(eps, one_minus_t);
    const double quarter_inv_tm1 = 0.25 / (1.0 - t);
    
    shape_fcn[0 + fivej] = (1.0 - r - t) * (1.0 - s - t) * quarter_inv_tm1;
    shape_fcn[1 + fivej] = (1.0 + r - t) * (1.0 - s - t) * quarter_inv_tm1;
    shape_fcn[2 + fivej] = (1.0 + r - t) * (1.0 + s - t) * quarter_inv_tm1;
    shape_fcn[3 + fivej] = (1.0 - r - t) * (1.0 + s - t) * quarter_inv_tm1;
    shape_fcn[4 + fivej] = t;
    
  }
}

//--------------------------------------------------------------------------
//-------- shifted_pyr_shape_fcn -------------------------------------------
//--------------------------------------------------------------------------
void
PyrSCS::shifted_pyr_shape_fcn(
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
