// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/Tet4CVFEM.h>
#include <master_element/Hex8GeometryFunctions.h>

#include <AlgTraits.h>

#include <NaluEnv.h>

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <iostream>

#include <cmath>
#include <limits>
#include <array>
#include <map>
#include <memory>

namespace sierra {
namespace nalu {

//-------- tet_deriv -------------------------------------------------------
template <typename DerivType>
KOKKOS_FUNCTION void
tet_deriv(DerivType& deriv)
{
  for (size_t j = 0; j < deriv.extent(0); ++j) {
    deriv(j, 0, 0) = -1.0;
    deriv(j, 0, 1) = -1.0;
    deriv(j, 0, 2) = -1.0;

    deriv(j, 1, 0) = 1.0;
    deriv(j, 1, 1) = 0.0;
    deriv(j, 1, 2) = 0.0;

    deriv(j, 2, 0) = 0.0;
    deriv(j, 2, 1) = 1.0;
    deriv(j, 2, 2) = 0.0;

    deriv(j, 3, 0) = 0.0;
    deriv(j, 3, 1) = 0.0;
    deriv(j, 3, 2) = 1.0;
  }
}

int
tet_gradient_operator(
  const SharedMemView<const double***, HostShmem>& cordel,
  const SharedMemView<const double***, HostShmem>& deriv,
  SharedMemView<double****, HostShmem>& gradop,
  SharedMemView<double**, HostShmem>& det_j,
  SharedMemView<double*, HostShmem>& err)
{

  //**********************************************************************
  //**********************************************************************
  //
  // description:
  //    This  routine returns the gradient operator, determinate of
  //    the Jacobian, and error count for an element workset of 3D
  //    subcontrol surface elements The gradient operator and the
  //    determinate of the jacobians are computed at the center of
  //    each control surface (the locations for the integration rule
  //    are at the center of each control surface).
  //
  // formal parameters - input:
  //    deriv         real  shape function derivatives evaluated at the
  //                        integration stations
  //    cordel        real  element local coordinates
  //
  // formal parameters - output:
  //    gradop        real  element gradient operator at each integration
  //                        station
  //    det_j         real  determinate of the jacobian at each integration
  //                        station
  //    err           real  positive volume check (0 = no error, 1 = error))
  //**********************************************************************
  //
  const unsigned nint = deriv.extent(0);
  const unsigned npe = deriv.extent(1);
  STK_ThrowRequireMsg(
    3 == deriv.extent(2), "tet_gradient_operator: Error in derivative array");

  const unsigned nelem = cordel.extent(0);
  STK_ThrowRequireMsg(
    npe == cordel.extent(1),
    "tet_gradient_operator: Error in coorindate array");
  STK_ThrowRequireMsg(
    3 == cordel.extent(2), "tet_gradient_operator: Error in coorindate array");

  STK_ThrowRequireMsg(
    nint == gradop.extent(0), "tet_gradient_operator: Error in gradient array");
  STK_ThrowRequireMsg(
    nelem == gradop.extent(1),
    "tet_gradient_operator: Error in gradient array");
  STK_ThrowRequireMsg(
    npe == gradop.extent(2), "tet_gradient_operator: Error in gradient array");
  STK_ThrowRequireMsg(
    3 == gradop.extent(3), "tet_gradient_operator: Error in gradient array");

  STK_ThrowRequireMsg(
    nint == det_j.extent(0),
    "tet_gradient_operator: Error in determinent array");
  STK_ThrowRequireMsg(
    nelem == det_j.extent(1),
    "tet_gradient_operator: Error in determinent array");

  STK_ThrowRequireMsg(
    nelem == err.extent(0), "tet_gradient_operator: Error in error array");

  const double realmin = std::numeric_limits<double>::min();

  for (unsigned ke = 0; ke < nelem; ++ke)
    err(ke) = 0;

  for (unsigned ki = 0; ki < nint; ++ki) {
    for (unsigned ke = 0; ke < nelem; ++ke) {
      double dx_ds0 = 0;
      double dx_ds1 = 0;
      double dx_ds2 = 0;
      double dy_ds0 = 0;
      double dy_ds1 = 0;
      double dy_ds2 = 0;
      double dz_ds0 = 0;
      double dz_ds1 = 0;
      double dz_ds2 = 0;

      // calculate the jacobian at the integration station -
      for (unsigned kn = 0; kn < npe; ++kn) {

        dx_ds0 += deriv(ki, kn, 0) * cordel(ke, kn, 0);
        dx_ds1 += deriv(ki, kn, 1) * cordel(ke, kn, 0);
        dx_ds2 += deriv(ki, kn, 2) * cordel(ke, kn, 0);

        dy_ds0 += deriv(ki, kn, 0) * cordel(ke, kn, 1);
        dy_ds1 += deriv(ki, kn, 1) * cordel(ke, kn, 1);
        dy_ds2 += deriv(ki, kn, 2) * cordel(ke, kn, 1);

        dz_ds0 += deriv(ki, kn, 0) * cordel(ke, kn, 2);
        dz_ds1 += deriv(ki, kn, 1) * cordel(ke, kn, 2);
        dz_ds2 += deriv(ki, kn, 2) * cordel(ke, kn, 2);
      }

      // calculate the determinate of the jacobian at the integration station -
      det_j(ki, ke) = dx_ds0 * (dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2) +
                      dy_ds0 * (dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2) +
                      dz_ds0 * (dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2);

      // protect against a negative or small value for the determinate of the
      // jacobian. The value of real_min (set in precision.par) represents
      // the smallest Real value (based upon the precision set for this
      // compilation) which the machine can represent -
      double test = det_j(ke, ki);
      if (test <= 1.e6 * realmin) {
        test = 1;
        err(ke) = 1;
      }
      const double denom = 1.0 / test;

      // compute the gradient operators at the integration station -

      const double ds0_dx = denom * (dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2);
      const double ds1_dx = denom * (dz_ds0 * dy_ds2 - dy_ds0 * dz_ds2);
      const double ds2_dx = denom * (dy_ds0 * dz_ds1 - dz_ds0 * dy_ds1);

      const double ds0_dy = denom * (dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2);
      const double ds1_dy = denom * (dx_ds0 * dz_ds2 - dz_ds0 * dx_ds2);
      const double ds2_dy = denom * (dz_ds0 * dx_ds1 - dx_ds0 * dz_ds1);

      const double ds0_dz = denom * (dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2);
      const double ds1_dz = denom * (dy_ds0 * dx_ds2 - dx_ds0 * dy_ds2);
      const double ds2_dz = denom * (dx_ds0 * dy_ds1 - dy_ds0 * dx_ds1);

      for (unsigned kn = 0; kn < npe; ++kn) {

        gradop(ki, ke, kn, 0) = deriv(ki, kn, 0) * ds0_dx +
                                deriv(ki, kn, 1) * ds1_dx +
                                deriv(ki, kn, 2) * ds2_dx;

        gradop(ki, ke, kn, 1) = deriv(ki, kn, 0) * ds0_dy +
                                deriv(ki, kn, 1) * ds1_dy +
                                deriv(ki, kn, 2) * ds2_dy;

        gradop(ki, ke, kn, 2) = deriv(ki, kn, 0) * ds0_dz +
                                deriv(ki, kn, 1) * ds1_dz +
                                deriv(ki, kn, 2) * ds2_dz;
      }
    }
  }

  // summarize volume error checks -
  double sum = 0;
  for (unsigned ke = 0; ke < nelem; ++ke)
    sum += err(ke);
  int nerr = 0;
  if (sum)
    // flag error -
    for (unsigned ke = 0; ke < nelem; ++ke)
      if (err(ke))
        nerr = ke;
  return nerr;
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
TetSCV::TetSCV() : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
TetSCV::ipNodeMap(int /*ordinal*/) const
{
  // define scv->node mappings
  return ipNodeMap_;
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
TetSCV::determinant_scv(
  const SharedMemView<DBLTYPE**, SHMEM>& coordel,
  SharedMemView<DBLTYPE*, SHMEM>& volume) const
{
  const int tetSubcontrolNodeTable[4][8] = {
    {0, 4, 7, 6, 11, 13, 14, 12},
    {1, 5, 7, 4, 9, 10, 14, 13},
    {2, 6, 7, 5, 8, 12, 14, 10},
    {3, 9, 13, 11, 8, 10, 14, 12}};

  const double half = 0.5;
  const double one3rd = 1.0 / 3.0;
  DBLTYPE coords[15][3];
  DBLTYPE ehexcoords[8][3];
  const int dim[3] = {0, 1, 2};

  // element vertices
  for (int j = 0; j < 4; ++j) {
    for (int k : dim) {
      coords[j][k] = coordel(j, k);
    }
  }

  // face 1 (tri)

  // edge midpoints
  for (int k : dim) {
    coords[4][k] = half * (coordel(0, k) + coordel(1, k));
  }
  for (int k : dim) {
    coords[5][k] = half * (coordel(1, k) + coordel(2, k));
  }
  for (int k : dim) {
    coords[6][k] = half * (coordel(2, k) + coordel(0, k));
  }

  // face mipdoint
  for (int k : dim) {
    coords[7][k] = one3rd * (coordel(0, k) + coordel(1, k) + coordel(2, k));
  }

  // face 2 (tri)

  // edge midpoints
  for (int k : dim) {
    coords[8][k] = half * (coordel(2, k) + coordel(3, k));
  }
  for (int k : dim) {
    coords[9][k] = half * (coordel(3, k) + coordel(1, k));
  }

  // face midpoint
  for (int k : dim) {
    coords[10][k] = one3rd * (coordel(1, k) + coordel(2, k) + coordel(3, k));
  }

  // face 3 (tri)

  // edge midpoint
  for (int k : dim) {
    coords[11][k] = half * (coordel(0, k) + coordel(3, k));
  }

  // face midpoint
  for (int k : dim) {
    coords[12][k] = one3rd * (coordel(0, k) + coordel(2, k) + coordel(3, k));
  }

  // face 4 (tri)

  // face midpoint
  for (int k : dim) {
    coords[13][k] = one3rd * (coordel(0, k) + coordel(1, k) + coordel(3, k));
  }

  // element centroid
  for (int k : dim) {
    coords[14][k] = 0.0;
  }
  for (int j = 0; j < nodesPerElement_; ++j) {
    for (int k : dim) {
      coords[14][k] = coords[14][k] + 0.25 * coordel(j, k);
    }
  }

  // loop over subcontrol volumes
  for (int icv = 0; icv < numIntPoints_; ++icv) {
    // loop over nodes of scv
    for (int inode = 0; inode < 8; ++inode) {
      // define scv coordinates using node table
      for (int k : dim) {
        ehexcoords[inode][k] = coords[tetSubcontrolNodeTable[icv][inode]][k];
      }
    }
    // compute volume using an equivalent polyhedron
    volume(icv) = hex_volume_grandy(ehexcoords);
    // check for negative volume
    // STK_ThrowAssertMsg( volume(icv) < 0.0, "ERROR in TetSCV::determinant,
    // negative volume.");
  }
}
KOKKOS_FUNCTION void
TetSCV::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coordel,
  SharedMemView<DoubleType*, DeviceShmem>& volume)
{
  determinant_scv(coordel, volume);
}

void
TetSCV::determinant(
  const SharedMemView<double**>& coordel, SharedMemView<double*>& volume)
{
  determinant_scv(coordel, volume);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
TetSCV::grad_op(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tet_deriv(deriv);
  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
TetSCV::shifted_grad_op(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tet_deriv(deriv);
  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
TetSCV::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tet_shape_fcn(numIntPoints_, intgLoc_[0], shpfc);
}
KOKKOS_FUNCTION void
TetSCV::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
TetSCV::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
TetSCV::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tet_shape_fcn(numIntPoints_, intgLocShift_[0], shpfc);
}
KOKKOS_FUNCTION void
TetSCV::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
TetSCV::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- tet_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
TetSCV::tet_shape_fcn(
  const int npts,
  const double* par_coord,
  SharedMemView<SCALAR**, SHMEM>& shpfc) const
{
  for (int j = 0; j < npts; ++j) {
    const int k = 3 * j;
    const double xi = par_coord[k];
    const double eta = par_coord[k + 1];
    const double zeta = par_coord[k + 2];
    shpfc(j, 0) = 1.0 - xi - eta - zeta;
    shpfc(j, 1) = xi;
    shpfc(j, 2) = eta;
    shpfc(j, 3) = zeta;
  }
}

//--------------------------------------------------------------------------
//-------- Mij ------------------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCV::Mij(const double* coords, double* metric, double* deriv)
{
  generic_Mij_3d<AlgTraitsTet4>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
KOKKOS_FUNCTION
void
TetSCV::Mij(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& metric,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tet_deriv(deriv);
  generic_Mij_3d<AlgTraitsTet4>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
TetSCS::TetSCS() : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;

#if !defined(KOKKOS_ENABLE_GPU)
  const double nodeLocations[4][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  stk::topology topo = stk::topology::TET_4;
  for (unsigned k = 0; k < topo.num_sides(); ++k) {
    stk::topology side_topo = topo.side_topology(k);
    const int* ordinals = side_node_ordinals(k);
    for (unsigned n = 0; n < side_topo.num_nodes(); ++n) {
      intgExpFaceShift_[k][n][0] = nodeLocations[ordinals[n]][0];
      intgExpFaceShift_[k][n][1] = nodeLocations[ordinals[n]][1];
      intgExpFaceShift_[k][n][2] = nodeLocations[ordinals[n]][2];
    }
  }
#endif
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
TetSCS::ipNodeMap(int ordinal) const
{
  // define ip->node mappings for each face (ordinal);
  return ipNodeMap_[ordinal];
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
TetSCS::side_node_ordinals(int ordinal) const
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return sideNodeOrdinals_[ordinal];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
TetSCS::determinant_scs(
  const SharedMemView<DBLTYPE**, SHMEM>& coordel,
  SharedMemView<DBLTYPE**, SHMEM>& areav) const
{
  int tetEdgeFacetTable[6][4] = {{4, 7, 14, 13},  {7, 14, 10, 5},
                                 {6, 12, 14, 7},  {11, 13, 14, 12},
                                 {13, 9, 10, 14}, {10, 8, 12, 14}};

  const int npe = nodesPerElement_;
  const int nscs = numIntPoints_;
  const double half = 0.5;
  const double one3rd = 1.0 / 3.0;
  const double one4th = 1.0 / 4.0;
  const int dim[] = {0, 1, 2};
  DBLTYPE coords[15][3];
  DBLTYPE scscoords[4][3];

  // element vertices
  for (int j = 0; j < 4; ++j) {
    for (int k : dim) {
      coords[j][k] = coordel(j, k);
    }
  }

  // face 1 (tri)
  //
  // edge midpoints
  for (int k : dim) {
    coords[4][k] = half * (coordel(0, k) + coordel(1, k));
  }
  for (int k : dim) {
    coords[5][k] = half * (coordel(1, k) + coordel(2, k));
  }
  for (int k : dim) {
    coords[6][k] = half * (coordel(2, k) + coordel(0, k));
  }

  // face midpoint
  for (int k : dim) {
    coords[7][k] = one3rd * (coordel(0, k) + coordel(1, k) + coordel(2, k));
  }

  // face 2 (tri)
  //
  // edge midpoints
  for (int k : dim) {
    coords[8][k] = half * (coordel(2, k) + coordel(3, k));
  }
  for (int k : dim) {
    coords[9][k] = half * (coordel(3, k) + coordel(1, k));
  }

  // face midpoint
  for (int k : dim) {
    coords[10][k] = one3rd * (coordel(1, k) + coordel(2, k) + coordel(3, k));
  }

  // face 3 (tri)
  //
  // edge midpoint
  for (int k : dim) {
    coords[11][k] = half * (coordel(0, k) + coordel(3, k));
  }

  // face midpoint
  for (int k : dim) {
    coords[12][k] = one3rd * (coordel(0, k) + coordel(2, k) + coordel(3, k));
  }

  // face 4 (tri)
  //
  // face midpoint
  for (int k : dim) {
    coords[13][k] = one3rd * (coordel(0, k) + coordel(1, k) + coordel(3, k));
  }

  // element centroid
  for (int k : dim) {
    coords[14][k] = 0.0;
  }
  for (int j = 0; j < npe; ++j) {
    for (int k : dim) {
      coords[14][k] += one4th * coordel(j, k);
    }
  }

  // loop over subcontrol surface
  for (int ics = 0; ics < nscs; ++ics) {
    // loop over nodes of scs
    for (int inode = 0; inode < 4; ++inode) {
      int itrianglenode = tetEdgeFacetTable[ics][inode];
      for (int k : dim) {
        scscoords[inode][k] = coords[itrianglenode][k];
      }
    }
    quad_area_by_triangulation(ics, scscoords, areav);
  }
}
KOKKOS_FUNCTION void
TetSCS::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coordel,
  SharedMemView<DoubleType**, DeviceShmem>& areav)
{
  determinant_scs(coordel, areav);
}
void
TetSCS::determinant(
  const SharedMemView<double**>& coordel, SharedMemView<double**>& areav)
{
  determinant_scs(coordel, areav);
}
//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
TetSCS::grad_op(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tet_deriv(deriv);
  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

void
TetSCS::grad_op(
  const SharedMemView<double**>& coords,
  SharedMemView<double***>& gradop,
  SharedMemView<double***>& deriv)
{
  tet_deriv(deriv);
  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
TetSCS::shifted_grad_op(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tet_deriv(deriv);

  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------

KOKKOS_FUNCTION
void
TetSCS::face_grad_op(
  int /*face_ordinal*/,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tet_deriv(deriv);
  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
TetSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  // no difference for regular face_grad_op
  face_grad_op(face_ordinal, coords, gradop, deriv);
}
//--------------------------------------------------------------------------
//-------- gij ------------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
TetSCS::gij(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gupper,
  SharedMemView<DoubleType***, DeviceShmem>& glower,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  generic_gij_3d<AlgTraitsTet4>(deriv, coords, gupper, glower);
}

//--------------------------------------------------------------------------
//-------- Mij ------------------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::Mij(const double* coords, double* metric, double* deriv)
{
  generic_Mij_3d<AlgTraitsTet4>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
KOKKOS_FUNCTION
void
TetSCS::Mij(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& metric,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tet_deriv(deriv);
  generic_Mij_3d<AlgTraitsTet4>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
TetSCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_;
}

//--------------------------------------------------------------------------
//-------- scsIpEdgeOrd ----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
TetSCS::scsIpEdgeOrd()
{
  return &scsIpEdgeOrd_[0];
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
TetSCS::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tet_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}
KOKKOS_FUNCTION void
TetSCS::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
TetSCS::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
TetSCS::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tet_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}
KOKKOS_FUNCTION void
TetSCS::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
TetSCS::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- tet_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
TetSCS::tet_shape_fcn(
  const int npts,
  const double* par_coord,
  SharedMemView<SCALAR**, SHMEM>& shpfc) const
{
  for (int j = 0; j < npts; ++j) {
    const int k = 3 * j;
    const double xi = par_coord[k];
    const double eta = par_coord[k + 1];
    const double zeta = par_coord[k + 2];
    shpfc(j, 0) = 1.0 - xi - eta - zeta;
    shpfc(j, 1) = xi;
    shpfc(j, 2) = eta;
    shpfc(j, 3) = zeta;
  }
}

void
TetSCS::tet_shape_fcn(
  const int npts, const double* par_coord, double* shape_fcn)
{
  for (int j = 0; j < npts; ++j) {
    const int fourj = 4 * j;
    const int k = 3 * j;
    const double xi = par_coord[k];
    const double eta = par_coord[k + 1];
    const double zeta = par_coord[k + 2];
    shape_fcn[fourj] = 1.0 - xi - eta - zeta;
    shape_fcn[1 + fourj] = xi;
    shape_fcn[2 + fourj] = eta;
    shape_fcn[3 + fourj] = zeta;
  }
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
int
TetSCS::opposingNodes(const int ordinal, const int node)
{
  return oppNode_[ordinal][node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
int
TetSCS::opposingFace(const int ordinal, const int node)
{
  return oppFace_[ordinal][node];
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
TetSCS::isInElement(
  const double* elem_nodal_coor, const double* point_coor, double* par_coor)
{
  // load up the element coordinates
  const double x1 = elem_nodal_coor[0];
  const double x2 = elem_nodal_coor[1];
  const double x3 = elem_nodal_coor[2];
  const double x4 = elem_nodal_coor[3];

  const double y1 = elem_nodal_coor[4];
  const double y2 = elem_nodal_coor[5];
  const double y3 = elem_nodal_coor[6];
  const double y4 = elem_nodal_coor[7];

  const double z1 = elem_nodal_coor[8];
  const double z2 = elem_nodal_coor[9];
  const double z3 = elem_nodal_coor[10];
  const double z4 = elem_nodal_coor[11];

  // determinant of matrix M in eqn x-x1 = M*xi
  const double det =
    (x2 - x1) * ((y3 - y1) * (z4 - z1) - (y4 - y1) * (z3 - z1)) -
    (x3 - x1) * ((y2 - y1) * (z4 - z1) - (y4 - y1) * (z2 - z1)) +
    (x4 - x1) * ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1));

  const double invDet = 1.0 / det;

  // matrix entries in inverse of M

  const double m11 = y3 * z4 - y1 * z4 - y4 * z3 + y1 * z3 + y4 * z1 - y3 * z1;
  const double m12 =
    -(x3 * z4 - x1 * z4 - x4 * z3 + x1 * z3 + x4 * z1 - x3 * z1);
  const double m13 = x3 * y4 - x1 * y4 - x4 * y3 + x1 * y3 + x4 * y1 - x3 * y1;
  const double m21 =
    -(y2 * z4 - y1 * z4 - y4 * z2 + y1 * z2 + y4 * z1 - y2 * z1);
  const double m22 = x2 * z4 - x1 * z4 - x4 * z2 + x1 * z2 + x4 * z1 - x2 * z1;
  const double m23 =
    -(x2 * y4 - x1 * y4 - x4 * y2 + x1 * y2 + x4 * y1 - x2 * y1);
  const double m31 = y2 * z3 - y1 * z3 - y3 * z2 + y1 * z2 + y3 * z1 - y2 * z1;
  const double m32 =
    -(x2 * z3 - x1 * z3 - x3 * z2 + x1 * z2 + x3 * z1 - x2 * z1);
  const double m33 = x2 * y3 - x1 * y3 - x3 * y2 + x1 * y2 + x3 * y1 - x2 * y1;

  const double xx1 = point_coor[0] - x1;
  const double yy1 = point_coor[1] - y1;
  const double zz1 = point_coor[2] - z1;

  // solve for parametric coordinates

  const double xi = invDet * (m11 * xx1 + m12 * yy1 + m13 * zz1);
  const double eta = invDet * (m21 * xx1 + m22 * yy1 + m23 * zz1);
  const double zeta = invDet * (m31 * xx1 + m32 * yy1 + m33 * zz1);

  // if volume coordinates are negative, point is outside the tet

  par_coor[0] = xi;
  par_coor[1] = eta;
  par_coor[2] = zeta;

  const double dist = parametric_distance(par_coor);

  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::interpolatePoint(
  const int& ncomp_field,
  const double* par_coord, // (3)
  const double* field,     // (4,ncomp_field)
  double* result)          // (ncomp_field)
{
  const double xi = par_coord[0];
  const double eta = par_coord[1];
  const double zeta = par_coord[2];

  const double psi1 = 1.0 - xi - eta - zeta;

  for (int i = 0; i < ncomp_field; ++i) {
    const int fourI = 4 * i;

    const double f1 = field[fourI];
    const double f2 = field[1 + fourI];
    const double f3 = field[2 + fourI];
    const double f4 = field[3 + fourI];

    result[i] = f1 * psi1 + f2 * xi + f3 * eta + f4 * zeta;
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::general_shape_fcn(
  const int numIp, const double* isoParCoord, double* shpfc)
{
  tet_shape_fcn(numIp, &isoParCoord[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::general_face_grad_op(
  const int /*face_ordinal*/,
  const double* /*isoParCoord*/,
  const double* coords,
  double* gradop,
  double* det_j,
  double* error)
{
  int lerr = 0;
  const int nface = 1;
  const int npe = nodesPerElement_;
  double dpsi[12];

  SharedMemView<double***, HostShmem> deriv(
    dpsi, nface, nodesPerElement_, nDim_);
  tet_deriv(deriv);

  const SharedMemView<const double***, HostShmem> cordel(coords, nface, npe, 3);
  SharedMemView<double****, HostShmem> grad(gradop, nface, nface, npe, 3);
  SharedMemView<double**, HostShmem> det(det_j, nface, nface);
  SharedMemView<double*, HostShmem> err(error, nface);
  lerr = tet_gradient_operator(cordel, deriv, grad, det, err);

  if (lerr)
    throw std::runtime_error("TetSCS::general_face_grad_op issue");
}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::sidePcoords_to_elemPcoords(
  const int& side_ordinal,
  const int& npoints,
  const double* side_pcoords,
  double* elem_pcoords)
{
  switch (side_ordinal) {
  case 0:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] = side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 1] = 0.0;
      elem_pcoords[i * 3 + 2] = side_pcoords[2 * i + 1];
    }
    break;
  case 1:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] =
        1.0 - side_pcoords[2 * i + 0] - side_pcoords[2 * i + 1];
      elem_pcoords[i * 3 + 1] = side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 2] = side_pcoords[2 * i + 1];
    }
    break;
  case 2:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] = 0.0;
      elem_pcoords[i * 3 + 1] = side_pcoords[2 * i + 1];
      elem_pcoords[i * 3 + 2] = side_pcoords[2 * i + 0];
    }
    break;
  case 3:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] = side_pcoords[2 * i + 1];
      elem_pcoords[i * 3 + 1] = side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 2] = 0.0;
    }
    break;
  default:
    throw std::runtime_error("TetSCS::sideMap invalid ordinal");
  }
  return;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
TetSCS::parametric_distance(const double* x)
{
  const double X = x[0] - 1. / 4.;
  const double Y = x[1] - 1. / 4.;
  const double Z = x[2] - 1. / 4.;
  const double dist0 = -4 * X;
  const double dist1 = -4 * Y;
  const double dist2 = -4 * Z;
  const double dist3 = 4 * (X + Y + Z);
  const double dist = std::max(std::max(dist0, dist1), std::max(dist2, dist3));
  return dist;
}

} // namespace nalu
} // namespace sierra
