/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef Hex8GeometryFunctions_h
#define Hex8GeometryFunctions_h

#include <AlgTraits.h>

#include <master_element/TensorOps.h>
#include <CVFEMTypeDefs.h>

#include <SimdInterface.h>
#include <Kokkos_Core.hpp>

#include <stk_util/util/ReportHandler.hpp>

#include <master_element/DirectionMacros.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>
#include <type_traits>

namespace sierra {
namespace nalu {

  template <typename ViewType>
  void quad_area_by_triangulation(
    int ics,
    const typename ViewType::value_type areacoords[4][3],
    const ViewType& area)
  {
    /**
     * Form up the area vec consistently with the triangulation used
     * in the Grandy algorithm, on each subcontrol volume hex
     *
     * "Efficient computation of volume of
     * Hexahedral Cells", Jeffrey Grandy, LLNL, UCRL-ID-128886,
     *  October 30, 1997.
     */
    using ftype = typename ViewType::value_type;

    area(ics,0) = 0.0;
    area(ics,1) = 0.0;
    area(ics,2) = 0.0;

    const ftype xmid[3] = {
        0.25 * (areacoords[0][0] + areacoords[1][0] + areacoords[2][0] + areacoords[3][0]),
        0.25 * (areacoords[0][1] + areacoords[1][1] + areacoords[2][1] + areacoords[3][1]),
        0.25 * (areacoords[0][2] + areacoords[1][2] + areacoords[2][2] + areacoords[3][2])
    };

    ftype r1[3] = { areacoords[0][0] - xmid[0], areacoords[0][1] - xmid[1], areacoords[0][2] - xmid[2] };
    for (int itriangle = 0; itriangle < 4; ++itriangle) {
      const int t_index = (itriangle + 1) % 4;
      const ftype r2[3] = {
          areacoords[t_index][0] - xmid[0],
          areacoords[t_index][1] - xmid[1],
          areacoords[t_index][2] - xmid[2]
      };

      area(ics, 0) += r1[1] * r2[2] - r2[1] * r1[2];
      area(ics, 1) += r1[2] * r2[0] - r2[2] * r1[0];
      area(ics, 2) += r1[0] * r2[1] - r2[0] * r1[1];

      r1[0] = r2[0];
      r1[1] = r2[1];
      r1[2] = r2[2];
    }
    area(ics,0) *= 0.5;
    area(ics,1) *= 0.5;
    area(ics,2) *= 0.5;
  }

  template <typename RealType>
  RealType hex_volume_grandy(RealType scvcoords[8][3])
  {
    /**
     * The Grandy algorithm for computing the volume of a multilinear box
     *
     * "Efficient computation of volume ofl
     * Hexahedral Cells", Jeffrey Grandy, LLNL, UCRL-ID-128886,
     *  October 30, 1997.
     */
    constexpr int nTri = 24;
    constexpr int dim = 3;

    constexpr int nNodes = 8;
    constexpr int nFaces = 6;
    constexpr int npv = nNodes + nFaces;

    RealType coordv[npv][dim];

    // copy coordinates
    for (int n = 0; n < nNodes; ++n) {
      coordv[n][0] = scvcoords[n][0];
      coordv[n][1] = scvcoords[n][1];
      coordv[n][2] = scvcoords[n][2];
    }

    constexpr int nodesPerFace = 4;
    constexpr int face_nodes[nFaces][nodesPerFace] = {
        { 0, 3, 2, 1 }, { 4, 5, 6, 7 },
        { 0, 1, 5, 4 }, { 2, 3, 7, 6 },
        { 1, 2, 6, 5 }, { 0, 4, 3, 7 }
    };

    // append face midpoint coordinates
    for (int k = 0; k < nFaces; ++k) {
      const int coordIndex = k + nNodes;
      for (int d = 0; d < dim; ++d) {
        coordv[coordIndex][d] = 0.25 * (
             coordv[face_nodes[k][0]][d]
           + coordv[face_nodes[k][1]][d]
           + coordv[face_nodes[k][2]][d]
           + coordv[face_nodes[k][3]][d]
        );
      }
    }

    constexpr int triangular_facets[nTri][3] = {
        { 0,  8,  1}, { 8,  2,  1}, { 3,  2,  8},
        { 3,  8,  0}, { 6,  9,  5}, { 7,  9,  6},
        { 4,  9,  7}, { 4,  5,  9}, {10,  0,  1},
        { 5, 10,  1}, { 4, 10,  5}, { 4,  0, 10},
        { 7,  6, 11}, { 6,  2, 11}, { 2,  3, 11},
        { 3,  7, 11}, { 6, 12,  2}, { 5, 12,  6},
        { 5,  1, 12}, { 1,  2, 12}, { 0,  4, 13},
        { 4,  7, 13}, { 7,  3, 13}, { 3,  0, 13}
    };

    RealType volume = 0.0;
    for (int k = 0; k < nTri; ++k) {
      const int p = triangular_facets[k][0];
      const int q = triangular_facets[k][1];
      const int r = triangular_facets[k][2];

      const RealType triFaceMid[3] = {
          coordv[p][0] + coordv[q][0] + coordv[r][0],
          coordv[p][1] + coordv[q][1] + coordv[r][1],
          coordv[p][2] + coordv[q][2] + coordv[r][2]
      };

      enum {XC = 0, YC = 1, ZC = 2};
      RealType dxv[3];

      dxv[0] = ( coordv[q][YC] - coordv[p][YC] ) * ( coordv[r][ZC] - coordv[p][ZC] )
             - ( coordv[r][YC] - coordv[p][YC] ) * ( coordv[q][ZC] - coordv[p][ZC] );

      dxv[1] = ( coordv[r][XC] - coordv[p][XC] ) * ( coordv[q][ZC] - coordv[p][ZC] )
             - ( coordv[q][XC] - coordv[p][XC] ) * ( coordv[r][ZC] - coordv[p][ZC] );

      dxv[2] = ( coordv[q][XC] - coordv[p][XC] ) * ( coordv[r][YC] - coordv[p][YC] )
             - ( coordv[r][XC] - coordv[p][XC] ) * ( coordv[q][YC] - coordv[p][YC] );

      volume += triFaceMid[0] * dxv[0] + triFaceMid[1] * dxv[1] + triFaceMid[2] * dxv[2];
    }
    volume /= RealType(18.0);
    return volume;
  }

  template <typename RealType>
  RealType bhex_volume_grandy(RealType scvcoords[8][3])
  {
    /**
     * The Grandy algorithm for computing the volume of a multilinear box
     *
     * "Efficient computation of volume ofl
     * Hexahedral Cells", Jeffrey Grandy, LLNL, UCRL-ID-128886,
     *  October 30, 1997.
     * modified for non-planar bent top face which we have broken into two triangles
     */
    constexpr int nTri = 24;
    constexpr int dim = 3;

    constexpr int nNodes = 8;
    constexpr int nFaces = 6;
    constexpr int npv = nNodes + nFaces;

    RealType coordv[npv][dim];

    // copy coordinates
    for (int n = 0; n < nNodes; ++n) {
      coordv[n][0] = scvcoords[n][0];
      coordv[n][1] = scvcoords[n][1];
      coordv[n][2] = scvcoords[n][2];
    }

    // now we add the face midpoints
    for (int k = 0; k < 3; ++k ) {
      coordv[8][k] = 0.25*( scvcoords[0][k] + scvcoords[1][k]
                            + scvcoords[2][k] + scvcoords[3][k] );
    }
    
    for (int k = 0; k < 3; ++k ) {
      coordv[9][k] = 0.5*( scvcoords[5][k] + scvcoords[7][k] );
    }

    for (int k = 0; k < 3; ++k ) {
      coordv[10][k] = 0.25*( scvcoords[0][k] + scvcoords[1][k]
                             + scvcoords[5][k] + scvcoords[4][k] );
    }   
    
    for (int k = 0; k < 3; ++k ) {
      coordv[11][k] = 0.25*( scvcoords[3][k] + scvcoords[2][k]
                             + scvcoords[6][k] + scvcoords[7][k] );
    }
    
    for (int k = 0; k < 3; ++k ) {
      coordv[12][k] = 0.25*( scvcoords[1][k] + scvcoords[2][k]
                             + scvcoords[6][k] + scvcoords[5][k] );
    }
    
    for (int k = 0; k < 3; ++k ) {
      coordv[13][k] = 0.25*( scvcoords[0][k] + scvcoords[3][k]
                             + scvcoords[7][k] + scvcoords[4][k] );
    }
    
    constexpr int triangular_facets[nTri][3] = {
      { 0,  8,  1}, { 8,  2,  1}, { 3,  2,  8},
      { 3,  8,  0}, { 6,  9,  5}, { 7,  9,  6},
      { 4,  9,  7}, { 4,  5,  9}, {10,  0,  1},
      { 5, 10,  1}, { 4, 10,  5}, { 4,  0, 10},
      { 7,  6, 11}, { 6,  2, 11}, { 2,  3, 11},
      { 3,  7, 11}, { 6, 12,  2}, { 5, 12,  6},
      { 5,  1, 12}, { 1,  2, 12}, { 0,  4, 13},
      { 4,  7, 13}, { 7,  3, 13}, { 3,  0, 13}
    };

    RealType volume = 0.0;
    for (int k = 0; k < nTri; ++k) {
      const int p = triangular_facets[k][0];
      const int q = triangular_facets[k][1];
      const int r = triangular_facets[k][2];

      const RealType triFaceMid[3] = {
          coordv[p][0] + coordv[q][0] + coordv[r][0],
          coordv[p][1] + coordv[q][1] + coordv[r][1],
          coordv[p][2] + coordv[q][2] + coordv[r][2]
      };

      enum {XC = 0, YC = 1, ZC = 2};
      RealType dxv[3];

      dxv[0] = ( coordv[q][YC] - coordv[p][YC] ) * ( coordv[r][ZC] - coordv[p][ZC] )
             - ( coordv[r][YC] - coordv[p][YC] ) * ( coordv[q][ZC] - coordv[p][ZC] );

      dxv[1] = ( coordv[r][XC] - coordv[p][XC] ) * ( coordv[q][ZC] - coordv[p][ZC] )
             - ( coordv[q][XC] - coordv[p][XC] ) * ( coordv[r][ZC] - coordv[p][ZC] );

      dxv[2] = ( coordv[q][XC] - coordv[p][XC] ) * ( coordv[r][YC] - coordv[p][YC] )
             - ( coordv[r][XC] - coordv[p][XC] ) * ( coordv[q][YC] - coordv[p][YC] );

      volume += triFaceMid[0] * dxv[0] + triFaceMid[1] * dxv[1] + triFaceMid[2] * dxv[2];
    }
    volume /= RealType(18.0);
    return volume;
  }

  template <typename CoordViewType>
  void subdivide_hex_8(CoordViewType coords, typename CoordViewType::value_type coordv[27][3])
  {
    /**
     * Subdivide the coordinates of a hex8 element into 8 hexs along edge, face, and volume midpoints
     */
    constexpr int numBaseNodes = 8;

    for (int n = 0; n < numBaseNodes; ++n) {
      coordv[n][0] = coords(n,0);
      coordv[n][1] = coords(n,1);
      coordv[n][2] = coords(n,2);
    }

    // Face-by-face ordering for the subdivided hex.  This is different than what is done
    // for a multilinear Hex27 element, which has equivalent nodal locations.
    for (int d = 0; d < 3; ++d) {
      // Face 0
      coordv[8][d]  = 0.5 * (coords(0,d) + coords(1,d)); // edge 1
      coordv[9][d]  = 0.5 * (coords(1,d) + coords(2,d)); // edge 2
      coordv[10][d] = 0.5 * (coords(2,d) + coords(3,d)); // edge 3
      coordv[11][d] = 0.5 * (coords(3,d) + coords(0,d)); // edge 4

      coordv[12][d] = 0.25 * (coords(0,d) + coords(1,d) + coords(2,d) + coords(3,d));

      // Face 1
      coordv[13][d] = 0.5 * (coords(4,d) + coords(5,d)); // edge 5
      coordv[14][d] = 0.5 * (coords(5,d) + coords(6,d)); // edge 6
      coordv[15][d] = 0.5 * (coords(6,d) + coords(7,d)); // edge 7
      coordv[16][d] = 0.5 * (coords(7,d) + coords(4,d)); // edge 8

      coordv[17][d] = 0.25 * (coords(4,d) + coords(5,d) + coords(6,d) + coords(7,d));

      // Face 2
      coordv[18][d] = 0.5 * (coords(1,d) + coords(5,d)); // edge 9
      coordv[19][d] = 0.5 * (coords(0,d) + coords(4,d)); // edge 10

      coordv[20][d] = 0.25 * (coords(0,d) + coords(1,d) + coords(4,d) + coords(5,d));

      // Face 3
      coordv[21][d] = 0.5 * (coords(3,d) + coords(7,d)); // edge 11
      coordv[22][d] = 0.5 * (coords(2,d) + coords(6,d)); // edge 12

      coordv[23][d] = 0.25 * (coords(2,d) + coords(3,d) + coords(6,d) + coords(7,d));

      // Face 4
      coordv[24][d] = 0.25 * (coords(1,d) + coords(2,d) + coords(5,d) + coords(6,d));

      // Face 5
      coordv[25][d] = 0.25 * (coords(0,d) + coords(3,d) + coords(4,d) + coords(7,d));

      // Volume centroid
      coordv[26][d] = 0.;
      for (int n = 0; n < 8; ++n) {
        coordv[26][d] += coords(n,d);
      }
      coordv[26][d] *= 0.125;
    }
  }

  template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
Scalar jacobian_component_xh(int d, const Scalar base_box[3][8], const Scalar interpj[2], const Scalar interpk[2])
{
  return( -interpj[0] * interpk[0] * base_box[d][0]
        +  interpj[0] * interpk[0] * base_box[d][1]
        +  interpj[1] * interpk[0] * base_box[d][2]
        -  interpj[1] * interpk[0] * base_box[d][3]
        -  interpj[0] * interpk[1] * base_box[d][4]
        +  interpj[0] * interpk[1] * base_box[d][5]
        +  interpj[1] * interpk[1] * base_box[d][6]
        -  interpj[1] * interpk[1] * base_box[d][7]
  ) * 0.5;
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
Scalar jacobian_component_yh(int d, const Scalar base_box[3][8],  const Scalar interpi[2], const Scalar interpk[2])
{
  return ( -interpi[0] * interpk[0] * base_box[d][0]
         -  interpi[1] * interpk[0] * base_box[d][1]
         +  interpi[1] * interpk[0] * base_box[d][2]
         +  interpi[0] * interpk[0] * base_box[d][3]
         -  interpi[0] * interpk[1] * base_box[d][4]
         -  interpi[1] * interpk[1] * base_box[d][5]
         +  interpi[1] * interpk[1] * base_box[d][6]
         +  interpi[0] * interpk[1] * base_box[d][7]
  ) * 0.5;
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
Scalar jacobian_component_zh(int d, const Scalar base_box[3][8], const Scalar interpi[2], const Scalar interpj[2])
{
  return ( -interpi[0] * interpj[0] * base_box[d][0]
         -  interpi[1] * interpj[0] * base_box[d][1]
         -  interpi[1] * interpj[1] * base_box[d][2]
         -  interpi[0] * interpj[1] * base_box[d][3]
         +  interpi[0] * interpj[0] * base_box[d][4]
         +  interpi[1] * interpj[0] * base_box[d][5]
         +  interpi[1] * interpj[1] * base_box[d][6]
         +  interpi[0] * interpj[1] * base_box[d][7]
  ) * 0.5;
}

template <int di, int dj, typename Scalar>
Scalar hex_jacobian_component(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2])
{

  return (dj == XH) ?
  ( -interpi[0] * interpk[0] * base_box[di][0]
  -  interpi[1] * interpk[0] * base_box[di][1]
  +  interpi[1] * interpk[0] * base_box[di][2]
  +  interpi[0] * interpk[0] * base_box[di][3]
  -  interpi[0] * interpk[1] * base_box[di][4]
  -  interpi[1] * interpk[1] * base_box[di][5]
  +  interpi[1] * interpk[1] * base_box[di][6]
  +  interpi[0] * interpk[1] * base_box[di][7]
  ) * 0.5
  : (dj == YH) ?
  ( -interpi[0] * interpk[0] * base_box[di][0]
  -  interpi[1] * interpk[0] * base_box[di][1]
  +  interpi[1] * interpk[0] * base_box[di][2]
  +  interpi[0] * interpk[0] * base_box[di][3]
  -  interpi[0] * interpk[1] * base_box[di][4]
  -  interpi[1] * interpk[1] * base_box[di][5]
  +  interpi[1] * interpk[1] * base_box[di][6]
  +  interpi[0] * interpk[1] * base_box[di][7]
  ) * 0.5
  :
  ( -interpi[0] * interpj[0] * base_box[di][0]
  -  interpi[1] * interpj[0] * base_box[di][1]
  -  interpi[1] * interpj[1] * base_box[di][2]
  -  interpi[0] * interpj[1] * base_box[di][3]
  +  interpi[0] * interpj[0] * base_box[di][4]
  +  interpi[1] * interpj[0] * base_box[di][5]
  +  interpi[1] * interpj[1] * base_box[di][6]
  +  interpi[0] * interpj[1] * base_box[di][7]
  ) * 0.5;
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_jacobian(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar jac[3][3])
{
  for (int d = 0; d < 3; ++d) {
    jac[0][d] = jacobian_component_xh(d, base_box, interpj, interpk);
  }

  for (int d = 0; d < 3; ++d) {
    jac[1][d] = jacobian_component_yh(d, base_box, interpi, interpk);
  }

  for (int d = 0; d < 3; ++d) {
    jac[2][d] = jacobian_component_zh(d, base_box, interpi, interpj);
  }
}


template <typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_jacobian_t(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar jac[3][3])
{
  for (int d = 0; d < 3; ++d) {
    jac[d][0] = jacobian_component_xh(d, base_box, interpj, interpk);
    jac[d][1] = jacobian_component_yh(d, base_box, interpi, interpk);
    jac[d][2] = jacobian_component_zh(d, base_box, interpi, interpj);
  }
}

template<int dir, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void areav_from_jacobian_t(
  const Scalar jact[3][3],
  Scalar area[3])
{
  constexpr int orth_comp_1 = (dir == XH) ? ZH : (dir == YH) ? XH : YH;
  constexpr int orth_comp_2 = (dir == XH) ? YH : (dir == YH) ? ZH : XH;
  area[XH] = jact[YH][orth_comp_1] * jact[ZH][orth_comp_2] - jact[ZH][orth_comp_1] * jact[YH][orth_comp_2];
  area[YH] = jact[ZH][orth_comp_1] * jact[XH][orth_comp_2] - jact[XH][orth_comp_1] * jact[ZH][orth_comp_2];
  area[ZH] = jact[XH][orth_comp_1] * jact[YH][orth_comp_2] - jact[YH][orth_comp_1] * jact[XH][orth_comp_2];
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_areav_x(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar area[3])
{
  const auto dx_ds1 = jacobian_component_zh(0, base_box, interpi, interpj);
  const auto dx_ds2 = jacobian_component_yh(0, base_box, interpi, interpk);

  const auto dy_ds1 = jacobian_component_zh(1, base_box, interpi, interpj);
  const auto dy_ds2 = jacobian_component_yh(1, base_box, interpi, interpk);

  const auto dz_ds1 = jacobian_component_zh(2, base_box, interpi, interpj);
  const auto dz_ds2 = jacobian_component_yh(2, base_box, interpi, interpk);

  area[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
  area[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
  area[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}


template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_areav_y(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar area[3])
{
  const auto dx_ds1 = jacobian_component_xh(0, base_box, interpj, interpk);
  const auto dx_ds2 = jacobian_component_zh(0, base_box, interpi, interpj);

  const auto dy_ds1 = jacobian_component_xh(1, base_box, interpj, interpk);
  const auto dy_ds2 = jacobian_component_zh(1, base_box, interpi, interpj);

  const auto dz_ds1 = jacobian_component_xh(2, base_box, interpj, interpk);
  const auto dz_ds2 = jacobian_component_zh(2, base_box, interpi, interpj);

  area[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
  area[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
  area[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_areav_z(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar area[3])
{
  const auto dx_ds1 = jacobian_component_yh(0, base_box, interpi, interpk);
  const auto dx_ds2 = jacobian_component_xh(0, base_box, interpj, interpk);

  const auto dy_ds1 = jacobian_component_yh(1, base_box, interpi, interpk);
  const auto dy_ds2 = jacobian_component_xh(1, base_box, interpj, interpk);

  const auto dz_ds1 = jacobian_component_yh(2, base_box, interpi, interpk);
  const auto dz_ds2 = jacobian_component_xh(2, base_box, interpj, interpk);

  area[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
  area[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
  area[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}

template<int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_vertex_coordinates(
  const nodal_vector_view<p, Scalar>& xc,
  Scalar base_box[3][8])
{
  for (int d = 0; d < 3; ++d) {
    base_box[d][0] = xc(0, 0, 0, d);
    base_box[d][1] = xc(0, 0, p, d);
    base_box[d][2] = xc(0, p, p, d);
    base_box[d][3] = xc(0, p, 0, d);

    base_box[d][4] = xc(p, 0, 0, d);
    base_box[d][5] = xc(p, 0, p, d);
    base_box[d][6] = xc(p, p, p, d);
    base_box[d][7] = xc(p, p, 0, d);
  }
}


} // namespace nalu
} // namespace Sierra

#endif
