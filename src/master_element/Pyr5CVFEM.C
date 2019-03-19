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
KOKKOS_FUNCTION
PyrSCV::PyrSCV()
  : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_; 

  MasterElement::ipNodeMap_.assign(ipNodeMap_, ipNodeMap_+5);

  MasterElement::intgLoc_.assign(intgLoc_, intgLoc_+15);
  MasterElement::intgLocShift_.assign(intgLocShift_, intgLocShift_+15);
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
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

void fill_intg_exp_face_shift(double* intgExpFaceShift, const int* sideNodeOrdinals)
{
  const double nodeLocations[5][3] = {
    {-1.0, -1.0, +0.0}, {+1.0, -1.0, +0.0}, {+1.0, +1.0, +0.0}, {-1.0, +1.0, +0.0},
    {0.0, 0.0, +1.0}
  };

  int index = 0;
  stk::topology topo = stk::topology::PYRAMID_5;
  for (unsigned k = 0; k < topo.num_sides(); ++k) {
    stk::topology side_topo = topo.side_topology(k);
    const int* ordinals = &sideNodeOrdinals[k*3];
    for (unsigned n = 0; n < side_topo.num_nodes(); ++n) {
      intgExpFaceShift[3*index + 0] = nodeLocations[ordinals[n]][0];
      intgExpFaceShift[3*index + 1] = nodeLocations[ordinals[n]][1];
      intgExpFaceShift[3*index + 2] = nodeLocations[ordinals[n]][2];
      ++index;
    }
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
PyrSCS::PyrSCS()
  : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;

  MasterElement::scsIpEdgeOrd_.assign(scsIpEdgeOrd_, scsIpEdgeOrd_+AlgTraits::numScsIp_);
  MasterElement::oppNode_.assign(oppNode_, oppNode_+20);
  MasterElement::oppFace_.assign(oppFace_, oppFace_+20);

  MasterElement::intgLoc_.assign(intgLoc_, intgLoc_+36);
  MasterElement::intgLocShift_.assign(intgLocShift_, intgLocShift_+36);

  MasterElement::intgExpFace_.assign(intgExpFace_, intgExpFace_+48);

  MasterElement::ipNodeMap_.assign(ipNodeMap_, ipNodeMap_+16);

  fill_intg_exp_face_shift(intgExpFaceShift_, sideNodeOrdinals_);
  MasterElement::intgExpFaceShift_.assign(intgExpFaceShift_,intgExpFaceShift_+48);
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
  return lrscv_;
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
