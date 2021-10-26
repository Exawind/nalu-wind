// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/MasterElementWork.h>
#include <master_element/CalculateQuadArea.h>

namespace sierra{
namespace nalu{

/* *********************************************************************
 ***********************************************************************
 
  description:
      This  routine returns the area vectors for each of the 12
      subcontrol surfaces of the hex.
 
  formal parameters - input:
      nelem         int   number of elements in the workset
      npe           int   number of nodes per element
      nscs          int   number of sub control surfaces
      cordel        real  element local coordinates [3][npe][nelem]  
 
  formal parameters - output:
      area_vec      real  area vector: A = Ax i + Ay j + Az k
                                         = Area*(i + j + k)
                          [3][nelem][nscs]
 
 ******************************************************************** */
KOKKOS_FUNCTION
void hex_scs_det(
  const int nelem, 
  const double *cordel, 
  double *area_vec)
{ 
  const int npe = AlgTraitsHex8::nodesPerElement_;
  const int nscs = AlgTraitsHex8::numScsIp_;

  // coordinates of the vertices needed to define the scs
  double coords[3][27];

  // coordinates of the vertices of the scs
  double scscoords[3][4];

  // this table defines the scs
  const int HexEdgeFacetTable[12][4] =
    {{  20,  8, 12, 26}, // sc face 1 -- points from 1 -> 2
     {  24,  9, 12, 26}, // sc face 2 -- points from 2 -> 3
     {  10, 12, 26, 23}, // sc face 3 -- points from 3 -> 4
     {  11, 25, 26, 12}, // sc face 4 -- points from 1 -> 4
     {  13, 20, 26, 17}, // sc face 5 -- points from 5 -> 6
     {  17, 14, 24, 26}, // sc face 6 -- points from 6 -> 7
     {  17, 15, 23, 26}, // sc face 7 -- points from 7 -> 8
     {  16, 17, 26, 25}, // sc face 8 -- points from 5 -> 8
     {  19, 20, 26, 25}, // sc face 9 -- points from 1 -> 5
     {  20, 18, 24, 26}, // sc face 10 -- points from 2 -> 6
     {  22, 23, 26, 24}, // sc face 11 -- points from 3 -> 7
     {  21, 25, 26, 23}};// sc face 12 -- points from 4 -> 8

  const double half   = 0.5; 
  const double one4th = 0.25; 
  const double one8th = 0.125; 

  auto idx = [&] (const int k, const int j, const int i)
  {
    return k + j*3 + i*3*npe;
  };
  // loop over elements
  for (int ielem = 0; ielem < nelem; ++ielem) 
  {
    for (int k=0; k<3; ++k)
    {
      // element vertices
      for (int j=0; j<npe; ++j)
        coords[k][j] = cordel[idx(k,j,ielem)];

      // face 1 (front)
      // 4++++11+++3
      // +         +
      // +         +
      // 12   13   10
      // +         +
      // +         +
      // 1++++9++++2

      // edge midpoints
      coords[k][ 8] = half*(cordel[idx(k,0,ielem)] + cordel[idx(k,1,ielem)]);
      coords[k][ 9] = half*(cordel[idx(k,1,ielem)] + cordel[idx(k,2,ielem)]);
      coords[k][10] = half*(cordel[idx(k,2,ielem)] + cordel[idx(k,3,ielem)]);
      coords[k][11] = half*(cordel[idx(k,3,ielem)] + cordel[idx(k,0,ielem)]);
      // face midpoint
      coords[k][12] = 
        one4th*(cordel[idx(k,0,ielem)] + cordel[idx(k,1,ielem)]
              + cordel[idx(k,2,ielem)] + cordel[idx(k,3,ielem)]);

      // face 2 (back)
      // 8++++16+++7
      // +         +
      // +         +
      // 17   18   15
      // +         +
      // +         +
      // 5++++14+++6

      // edge midpoints
      coords[k][13] = half*(cordel[idx(k,4,ielem)] + cordel[idx(k,5,ielem)]);
      coords[k][14] = half*(cordel[idx(k,5,ielem)] + cordel[idx(k,6,ielem)]);
      coords[k][15] = half*(cordel[idx(k,6,ielem)] + cordel[idx(k,7,ielem)]);
      coords[k][16] = half*(cordel[idx(k,7,ielem)] + cordel[idx(k,4,ielem)]);
      // face midpoint
      coords[k][17] = 
        one4th*(cordel[idx(k,4,ielem)] + cordel[idx(k,5,ielem)]
              + cordel[idx(k,6,ielem)] + cordel[idx(k,7,ielem)]);
      // face 3 (bottom)
      // 5++++14+++6
      // +         +
      // +         +
      // 20   21   19
      // +         +
      // +         +
      // 1++++9++++//
      // edge midpoints
      coords[k][18] = half*(cordel[idx(k,1,ielem)] + cordel[idx(k,5,ielem)]);
      coords[k][19] = half*(cordel[idx(k,0,ielem)] + cordel[idx(k,4,ielem)]);
      // face midpoint
      coords[k][20] = 
        one4th*(cordel[idx(k,0,ielem)] + cordel[idx(k,1,ielem)]
              + cordel[idx(k,4,ielem)] + cordel[idx(k,5,ielem)]);
      // face 4 (top)
      // 8++++16+++7
      // +         +
      // +         +
      // 22   24   23
      // +         +
      // +         +
      // 4++++11+++3
      // edge mipdoints
      coords[k][21] = half*(cordel[idx(k,3,ielem)] + cordel[idx(k,7,ielem)]);
      coords[k][22] = half*(cordel[idx(k,2,ielem)] + cordel[idx(k,6,ielem)]);
      // face midpoint
      coords[k][23] = 
        one4th*(cordel[idx(k,2,ielem)] + cordel[idx(k,3,ielem)]
              + cordel[idx(k,6,ielem)] + cordel[idx(k,7,ielem)]);
      // face 5 (right)
      // 3++++23+++7
      // +         +
      // +         +
      // 10   25   15
      // +         +
      // +         +
      // 2++++19+++6
      // face midpoint
      coords[k][24] = 
        one4th*(cordel[idx(k,1,ielem)] + cordel[idx(k,2,ielem)]
              + cordel[idx(k,5,ielem)] + cordel[idx(k,6,ielem)]);
      // face 6 (left)
      // 4++++22+++8
      // +         +
      // +         +
      // 12   26   18
      // +         +
      // +         +
      // 1++++20+++5
      // face midpoint
      coords[k][25] =
        one4th*(cordel[idx(k,0,ielem)] + cordel[idx(k,3,ielem)]
              + cordel[idx(k,4,ielem)] + cordel[idx(k,7,ielem)]);
      // volume centroid
      coords[k][26] = one8th*(
             cordel[idx(k,0,ielem)] + cordel[idx(k,1,ielem)]
           + cordel[idx(k,2,ielem)] + cordel[idx(k,3,ielem)]
           + cordel[idx(k,4,ielem)] + cordel[idx(k,5,ielem)]
           + cordel[idx(k,6,ielem)] + cordel[idx(k,7,ielem)] );
    }
    const int npf = 4;
    // loop over subcontrol surfaces
    for (int ics=0; ics<nscs; ++ics)
    {
      // loop over vertices of scs
      for (int inode=0; inode<npf; ++inode)
      {
        const int itrianglenode = HexEdgeFacetTable[ics][inode];
        // set scs coordinates using node table
        for (int k=0; k<3; ++k) {
          scscoords[k][inode] = coords[k][itrianglenode];
        } 
      }
      // compute quad area vector using triangle decomposition
      quadAreaByTriangleFacets(&scscoords[0][0], &area_vec[3*ielem + 3*nelem*ics]);
    }
  }
}
}
}