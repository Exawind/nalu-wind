// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "master_element/Wed6CVFEM.h"
#include "master_element/MasterElementFunctions.h"
#include "master_element/Hex8GeometryFunctions.h"
#include "NaluEnv.h"

#include <array>

namespace sierra {
namespace nalu {

int
wed_gradient_operator(
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
  ThrowRequireMsg(
    3 == deriv.extent(2), "wed_gradient_operator: Error in derivative array");

  const unsigned nelem = cordel.extent(0);
  ThrowRequireMsg(
    npe == cordel.extent(1),
    "wed_gradient_operator: Error in coorindate array");
  ThrowRequireMsg(
    3 == cordel.extent(2), "wed_gradient_operator: Error in coorindate array");

  ThrowRequireMsg(
    nint == gradop.extent(0), "wed_gradient_operator: Error in gradient array");
  ThrowRequireMsg(
    nelem == gradop.extent(1),
    "wed_gradient_operator: Error in gradient array");
  ThrowRequireMsg(
    npe == gradop.extent(2), "wed_gradient_operator: Error in gradient array");
  ThrowRequireMsg(
    3 == gradop.extent(3), "wed_gradient_operator: Error in gradient array");

  ThrowRequireMsg(
    nint == det_j.extent(0),
    "wed_gradient_operator: Error in determinent array");
  ThrowRequireMsg(
    nelem == det_j.extent(1),
    "wed_gradient_operator: Error in determinent array");

  ThrowRequireMsg(
    nelem == err.extent(0), "wed_gradient_operator: Error in error array");

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

template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
wed_shape_fcn(
  const int npts,
  const double* isoParCoord,
  SharedMemView<SCALAR**, SHMEM>& shape_fcn)
{
  for (int j = 0; j < npts; ++j) {
    int k = 3 * j;
    double r = isoParCoord[k];
    double s = isoParCoord[k + 1];
    double t = 1.0 - r - s;
    double xi = isoParCoord[k + 2];
    shape_fcn(j, 0) = 0.5 * t * (1.0 - xi);
    shape_fcn(j, 1) = 0.5 * r * (1.0 - xi);
    shape_fcn(j, 2) = 0.5 * s * (1.0 - xi);
    shape_fcn(j, 3) = 0.5 * t * (1.0 + xi);
    shape_fcn(j, 4) = 0.5 * r * (1.0 + xi);
    shape_fcn(j, 5) = 0.5 * s * (1.0 + xi);
  }
}

//-------- wed_deriv -------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_FUNCTION void
wed_deriv(
  const int npts,
  const double* intgLoc,
  SharedMemView<DBLTYPE***, SHMEM>& deriv)
{
  for (int j = 0; j < npts; ++j) {
    int k = j * 3;

    const DBLTYPE r = intgLoc[k];
    const DBLTYPE s = intgLoc[k + 1];
    const DBLTYPE t = 1.0 - r - s;
    const DBLTYPE xi = intgLoc[k + 2];

    deriv(j, 0, 0) = -0.5 * (1.0 - xi); // d(N_1)/ d(r)  = deriv[0]
    deriv(j, 0, 1) = -0.5 * (1.0 - xi); // d(N_1)/ d(s)  = deriv[1]
    deriv(j, 0, 2) = -0.5 * t;          // d(N_1)/ d(xi) = deriv[2]

    deriv(j, 1, 0) = 0.5 * (1.0 - xi); // d(N_2)/ d(r)  = deriv[0 + 3]
    deriv(j, 1, 1) = 0.0;              // d(N_2)/ d(s)  = deriv[1 + 3]
    deriv(j, 1, 2) = -0.5 * r;         // d(N_2)/ d(xi) = deriv[2 + 3]

    deriv(j, 2, 0) = 0.0;              // d(N_3)/ d(r)  = deriv[0 + 6]
    deriv(j, 2, 1) = 0.5 * (1.0 - xi); // d(N_3)/ d(s)  = deriv[1 + 6]
    deriv(j, 2, 2) = -0.5 * s;         // d(N_3)/ d(xi) = deriv[2 + 6]

    deriv(j, 3, 0) = -0.5 * (1.0 + xi); // d(N_4)/ d(r)  = deriv[0 + 9]
    deriv(j, 3, 1) = -0.5 * (1.0 + xi); // d(N_4)/ d(s)  = deriv[1 + 9]
    deriv(j, 3, 2) = 0.5 * t;           // d(N_4)/ d(xi) = deriv[2 + 9]

    deriv(j, 4, 0) = 0.5 * (1.0 + xi); // d(N_5)/ d(r)  = deriv[0 + 12]
    deriv(j, 4, 1) = 0.0;              // d(N_5)/ d(s)  = deriv[1 + 12]
    deriv(j, 4, 2) = 0.5 * r;          // d(N_5)/ d(xi) = deriv[2 + 12]

    deriv(j, 5, 0) = 0.0;              // d(N_6)/ d(r)  = deriv[0 + 15]
    deriv(j, 5, 1) = 0.5 * (1.0 + xi); // d(N_6)/ d(s)  = deriv[1 + 15]
    deriv(j, 5, 2) = 0.5 * s;          // d(N_6)/ d(xi) = deriv[2 + 15]
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
WedSCV::WedSCV() : MasterElement()
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
WedSCV::ipNodeMap(int /*ordinal*/) const
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}

template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
WedSCV::determinant_scv(
  const SharedMemView<DBLTYPE**, SHMEM>& coordel,
  SharedMemView<DBLTYPE*, SHMEM>& volume) const
{
  const int wedSubControlNodeTable[6][8] = {
    {0, 15, 16, 6, 8, 19, 20, 9},    {9, 6, 1, 7, 20, 16, 14, 18},
    {8, 9, 7, 2, 19, 20, 18, 17},    {19, 15, 16, 20, 12, 3, 10, 13},
    {20, 16, 14, 18, 13, 10, 4, 11}, {19, 20, 18, 17, 12, 13, 11, 5},
  };

  const double half = 0.5;
  const double one3rd = 1.0 / 3.0;
  const double one6th = 1.0 / 6.0;
  DBLTYPE coords[21][3];
  DBLTYPE ehexcoords[8][3];
  const int dim[3] = {0, 1, 2};

  // element vertices
  for (int j = 0; j < 6; j++)
    for (int k : dim)
      coords[j][k] = coordel(j, k);

  // face 1 (tri)

  // edge midpoints
  for (int k : dim)
    coords[6][k] = half * (coordel(0, k) + coordel(1, k));

  for (int k : dim)
    coords[7][k] = half * (coordel(1, k) + coordel(2, k));

  for (int k : dim)
    coords[8][k] = half * (coordel(2, k) + coordel(0, k));

  // face midpoint
  for (int k : dim)
    coords[9][k] = one3rd * (coordel(0, k) + coordel(1, k) + coordel(2, k));

  // face 2 (tri)

  // edge midpoints
  for (int k : dim)
    coords[10][k] = half * (coordel(3, k) + coordel(4, k));

  for (int k : dim)
    coords[11][k] = half * (coordel(4, k) + coordel(5, k));

  for (int k : dim)
    coords[12][k] = half * (coordel(5, k) + coordel(3, k));

  // face midpoint
  for (int k : dim)
    coords[13][k] = one3rd * (coordel(3, k) + coordel(4, k) + coordel(5, k));

  // face 3 (quad)

  // edge midpoints
  for (int k : dim)
    coords[14][k] = half * (coordel(1, k) + coordel(4, k));

  for (int k : dim)
    coords[15][k] = half * (coordel(0, k) + coordel(3, k));

  // face midpoint
  for (int k : dim)
    coords[16][k] =
      0.25 * (coordel(0, k) + coordel(1, k) + coordel(4, k) + coordel(3, k));

  // face 4 (quad)

  // edge midpoint
  for (int k : dim)
    coords[17][k] = half * (coordel(2, k) + coordel(5, k));

  // face midpoint
  for (int k : dim)
    coords[18][k] =
      0.25 * (coordel(1, k) + coordel(4, k) + coordel(5, k) + coordel(2, k));

  // face 5 (quad)

  // face midpoint
  for (int k : dim)
    coords[19][k] =
      0.25 * (coordel(5, k) + coordel(3, k) + coordel(0, k) + coordel(2, k));

  // element centroid
  for (int k : dim)
    coords[20][k] = 0.0;
  for (int j = 0; j < nodesPerElement_; j++)
    for (int k : dim)
      coords[20][k] += one6th * coordel(j, k);

  // loop over SCVs
  for (int icv = 0; icv < numIntPoints_; icv++) {
    for (int inode = 0; inode < 8; inode++)
      for (int k : dim)
        ehexcoords[inode][k] = coords[wedSubControlNodeTable[icv][inode]][k];

    // compute volume using an equivalent polyhedron
    volume(icv) = hex_volume_grandy(ehexcoords);
  }
}
KOKKOS_FUNCTION void
WedSCV::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coordel,
  SharedMemView<DoubleType*, DeviceShmem>& volume)
{
  determinant_scv(coordel, volume);
}

void
WedSCV::determinant(
  const SharedMemView<double**>& coordel, SharedMemView<double*>& volume)
{
  determinant_scv(coordel, volume);
}
//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
WedSCV::grad_op(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  wed_deriv(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
WedSCV::shifted_grad_op(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  wed_deriv(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}

template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
WedSCV::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  wed_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}
KOKKOS_FUNCTION void
WedSCV::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
WedSCV::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
WedSCV::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  wed_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}
KOKKOS_FUNCTION void
WedSCV::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
WedSCV::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- wedge_shape_fcn -------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCV::wedge_shape_fcn(
  const int npts, const double* isoParCoord, double* shape_fcn)
{
  for (int j = 0; j < npts; ++j) {
    int sixj = 6 * j;
    int k = 3 * j;
    double r = isoParCoord[k];
    double s = isoParCoord[k + 1];
    double t = 1.0 - r - s;
    double xi = isoParCoord[k + 2];
    shape_fcn[sixj] = 0.5 * t * (1.0 - xi);
    shape_fcn[1 + sixj] = 0.5 * r * (1.0 - xi);
    shape_fcn[2 + sixj] = 0.5 * s * (1.0 - xi);
    shape_fcn[3 + sixj] = 0.5 * t * (1.0 + xi);
    shape_fcn[4 + sixj] = 0.5 * r * (1.0 + xi);
    shape_fcn[5 + sixj] = 0.5 * s * (1.0 + xi);
  }
}

//--------------------------------------------------------------------------
//-------- Mij ------------------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCV::Mij(const double* coords, double* metric, double* deriv)
{
  generic_Mij_3d<AlgTraitsWed6>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
KOKKOS_FUNCTION
void
WedSCV::Mij(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& metric,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  wed_deriv(numIntPoints_, &intgLoc_[0], deriv);
  generic_Mij_3d<AlgTraitsWed6>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
WedSCS::WedSCS() : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;

#if !defined(KOKKOS_ENABLE_GPU)
  const double nodeLocations[6][3] = {{0.0, 0.0, -1.0},  {+1.0, 0.0, -1.0},
                                      {0.0, +1.0, -1.0}, {0.0, 0.0, +1.0},
                                      {+1.0, 0.0, +1.0}, {0.0, +1.0, +1.0}};
  int index = 0;
  stk::topology topo = stk::topology::WEDGE_6;
  for (unsigned k = 0; k < topo.num_sides(); ++k) {
    stk::topology side_topo = topo.side_topology(k);
    const int* ordinals = side_node_ordinals(k);
    for (unsigned n = 0; n < side_topo.num_nodes(); ++n) {
      intgExpFaceShift_[3 * index + 0] = nodeLocations[ordinals[n]][0];
      intgExpFaceShift_[3 * index + 1] = nodeLocations[ordinals[n]][1];
      intgExpFaceShift_[3 * index + 2] = nodeLocations[ordinals[n]][2];
      ++index;
    }
  }
#endif
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
WedSCS::ipNodeMap(int ordinal) const
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal * 4];
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
WedSCS::side_node_ordinals(int ordinal) const
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return &sideNodeOrdinals_[sideOffset_[ordinal]];
}

template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
WedSCS::determinant_scs(
  const SharedMemView<DBLTYPE**, SHMEM>& coordel,
  SharedMemView<DBLTYPE**, SHMEM>& areav) const
{
  const int wedEdgeFacetTable[9][4] = {
    {6, 9, 20, 16},   // sc face 1 -- points from 1 -> 2
    {7, 9, 20, 18},   // sc face 2 -- points from 2 -> 3
    {9, 8, 19, 20},   // sc face 3 -- points from 1 -> 3
    {10, 16, 20, 13}, // sc face 4 -- points from 4 -> 5
    {13, 11, 18, 20}, // sc face 5 -- points from 5 -> 6
    {12, 13, 20, 19}, // sc face 6 -- points from 4 -> 6
    {15, 16, 20, 19}, // sc face 7 -- points from 1 -> 4
    {16, 14, 18, 20}, // sc face 8 -- points from 2 -> 5
    {19, 20, 18, 17}  // sc face 9 -- points from 3 -> 6
  };

  const double one3rd = 1.0 / 3.0;
  const double one6th = 1.0 / 6.0;
  const double half = 0.5;
  const int dim[3] = {0, 1, 2};
  DBLTYPE coords[21][3];
  DBLTYPE scscoords[4][3];

  // element vertices
  for (int j = 0; j < 6; j++)
    for (int k : dim)
      coords[j][k] = coordel(j, k);

  // face 1 (tri)

  // edge midpoints
  for (int k : dim)
    coords[6][k] = half * (coordel(0, k) + coordel(1, k));

  for (int k : dim)
    coords[7][k] = half * (coordel(1, k) + coordel(2, k));

  for (int k : dim)
    coords[8][k] = half * (coordel(2, k) + coordel(0, k));

  // face midpoint
  for (int k : dim)
    coords[9][k] = one3rd * (coordel(0, k) + coordel(1, k) + coordel(2, k));

  // face 2 (tri)

  // edge midpoints
  for (int k : dim)
    coords[10][k] = half * (coordel(3, k) + coordel(4, k));

  for (int k : dim)
    coords[11][k] = half * (coordel(4, k) + coordel(5, k));

  for (int k : dim)
    coords[12][k] = half * (coordel(5, k) + coordel(3, k));

  // face midpoint
  for (int k : dim)
    coords[13][k] = one3rd * (coordel(3, k) + coordel(4, k) + coordel(5, k));

  // face 3 (quad)

  // edge midpoints
  for (int k : dim)
    coords[14][k] = half * (coordel(1, k) + coordel(4, k));

  for (int k : dim)
    coords[15][k] = half * (coordel(0, k) + coordel(3, k));

  // face midpoint
  for (int k : dim)
    coords[16][k] =
      0.25 * (coordel(0, k) + coordel(1, k) + coordel(4, k) + coordel(3, k));

  // face 4 (quad)

  // edge midpoint
  for (int k : dim)
    coords[17][k] = half * (coordel(2, k) + coordel(5, k));

  // face midpoint
  for (int k : dim)
    coords[18][k] =
      0.25 * (coordel(1, k) + coordel(4, k) + coordel(5, k) + coordel(2, k));

  // face 5 (quad)

  // face midpoint
  for (int k : dim)
    coords[19][k] =
      0.25 * (coordel(5, k) + coordel(3, k) + coordel(0, k) + coordel(2, k));

  // element centroid
  for (int k : dim)
    coords[20][k] = 0.0;
  for (int j = 0; j < nodesPerElement_; j++)
    for (int k : dim)
      coords[20][k] += one6th * coordel(j, k);

  // loop over SCSs
  for (int ics = 0; ics < numIntPoints_; ics++) {
    for (int inode = 0; inode < 4; inode++)
      for (int k : dim)
        scscoords[inode][k] = coords[wedEdgeFacetTable[ics][inode]][k];
    quad_area_by_triangulation(ics, scscoords, areav);
  }
}
KOKKOS_FUNCTION void
WedSCS::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coordel,
  SharedMemView<DoubleType**, DeviceShmem>& areav)
{
  determinant_scs(coordel, areav);
}

void
WedSCS::determinant(
  const SharedMemView<double**>& coordel, SharedMemView<double**>& areav)
{
  determinant_scs(coordel, areav);
}
//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
WedSCS::grad_op(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  wed_deriv(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}

void
WedSCS::grad_op(
  const SharedMemView<double**>& coords,
  SharedMemView<double***>& gradop,
  SharedMemView<double***>& deriv)
{
  wed_deriv(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}

KOKKOS_FUNCTION
void
WedSCS::shifted_grad_op(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  wed_deriv(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- wedge_derivative --------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::wedge_derivative(const int npts, const double* intgLoc, double* deriv)
{
  // d3d(c,s,j) = deriv[c + 3*(s + 6*j)] = deriv[c+3s+18j]

  for (int j = 0; j < npts; ++j) {

    int k = j * 3;
    const int p = 18 * j;

    const double r = intgLoc[k];
    const double s = intgLoc[k + 1];
    const double t = 1.0 - r - s;
    const double xi = intgLoc[k + 2];

    deriv[0 + 3 * 0 + p] = -0.5 * (1.0 - xi); // d(N_1)/ d(r)  = deriv[0]
    deriv[1 + 3 * 0 + p] = -0.5 * (1.0 - xi); // d(N_1)/ d(s)  = deriv[1]
    deriv[2 + 3 * 0 + p] = -0.5 * t;          // d(N_1)/ d(xi) = deriv[2]

    deriv[0 + 3 * 1 + p] = 0.5 * (1.0 - xi); // d(N_2)/ d(r)  = deriv[0 + 3]
    deriv[1 + 3 * 1 + p] = 0.0;              // d(N_2)/ d(s)  = deriv[1 + 3]
    deriv[2 + 3 * 1 + p] = -0.5 * r;         // d(N_2)/ d(xi) = deriv[2 + 3]

    deriv[0 + 3 * 2 + p] = 0.0;              // d(N_3)/ d(r)  = deriv[0 + 6]
    deriv[1 + 3 * 2 + p] = 0.5 * (1.0 - xi); // d(N_3)/ d(s)  = deriv[1 + 6]
    deriv[2 + 3 * 2 + p] = -0.5 * s;         // d(N_3)/ d(xi) = deriv[2 + 6]

    deriv[0 + 3 * 3 + p] = -0.5 * (1.0 + xi); // d(N_4)/ d(r)  = deriv[0 + 9]
    deriv[1 + 3 * 3 + p] = -0.5 * (1.0 + xi); // d(N_4)/ d(s)  = deriv[1 + 9]
    deriv[2 + 3 * 3 + p] = 0.5 * t;           // d(N_4)/ d(xi) = deriv[2 + 9]

    deriv[0 + 3 * 4 + p] = 0.5 * (1.0 + xi); // d(N_5)/ d(r)  = deriv[0 + 12]
    deriv[1 + 3 * 4 + p] = 0.0;              // d(N_5)/ d(s)  = deriv[1 + 12]
    deriv[2 + 3 * 4 + p] = 0.5 * r;          // d(N_5)/ d(xi) = deriv[2 + 12]

    deriv[0 + 3 * 5 + p] = 0.0;              // d(N_6)/ d(r)  = deriv[0 + 15]
    deriv[1 + 3 * 5 + p] = 0.5 * (1.0 + xi); // d(N_6)/ d(s)  = deriv[1 + 15]
    deriv[2 + 3 * 5 + p] = 0.5 * s;          // d(N_6)/ d(xi) = deriv[2 + 15]
  }
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
WedSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  using tri_traits = AlgTraitsTri3Wed6;
  using quad_traits = AlgTraitsQuad4Wed6;
  constexpr int dim = 3;
  const int numFaceIps =
    (face_ordinal < 3) ? quad_traits::numFaceIp_ : tri_traits::numFaceIp_;
  const int offset = quad_traits::numFaceIp_ * face_ordinal;
  wed_deriv(numFaceIps, &intgExpFace_[dim * offset], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}
//--------------------------------------------------------------------------
//-------- shifted_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
WedSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  using tri_traits = AlgTraitsTri3Wed6;
  using quad_traits = AlgTraitsQuad4Wed6;
  constexpr int dim = 3;
  const int numFaceIps =
    (face_ordinal < 3) ? quad_traits::numFaceIp_ : tri_traits::numFaceIp_;
  const int offset = sideOffset_[face_ordinal];
  wed_deriv(numFaceIps, &intgExpFaceShift_[dim * offset], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}
//--------------------------------------------------------------------------
//-------- gij ------------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
WedSCS::gij(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gupper,
  SharedMemView<DoubleType***, DeviceShmem>& glower,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  generic_gij_3d<AlgTraitsWed6>(deriv, coords, gupper, glower);
}

//--------------------------------------------------------------------------
//-------- Mij ------------------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::Mij(const double* coords, double* metric, double* deriv)
{
  generic_Mij_3d<AlgTraitsWed6>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
KOKKOS_FUNCTION
void
WedSCS::Mij(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& metric,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  wed_deriv(numIntPoints_, &intgLoc_[0], deriv);
  generic_Mij_3d<AlgTraitsWed6>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
WedSCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_;
}

//--------------------------------------------------------------------------
//-------- scsIpEdgeOrd ----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
WedSCS::scsIpEdgeOrd()
{
  return &scsIpEdgeOrd_[0];
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
int
WedSCS::opposingNodes(const int ordinal, const int node)
{
  return oppNode_[ordinal * 4 + node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
int
WedSCS::opposingFace(const int ordinal, const int node)
{
  return oppFace_[ordinal * 4 + node];
}

template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
WedSCS::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  wed_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}
KOKKOS_FUNCTION void
WedSCS::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
WedSCS::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
WedSCS::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  wed_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}
KOKKOS_FUNCTION void
WedSCS::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
WedSCS::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
WedSCS::isInElement(
  const double* elemNodalCoord, const double* pointCoord, double* isoParCoord)
{
  const double isInElemConverged = 1.0e-16;

  // ------------------------------------------------------------------
  // Pentahedron master element space is (r,s,xi):
  // r=([0,1]), s=([0,1]), xi=([-1,+1])
  // Use natural coordinates to determine if point is in pentahedron.
  // ------------------------------------------------------------------

  // Translate element so that (x,y,z) coordinates of first node are (0,0,0)

  double x[] = {
    0.0,
    elemNodalCoord[1] - elemNodalCoord[0],
    elemNodalCoord[2] - elemNodalCoord[0],
    elemNodalCoord[3] - elemNodalCoord[0],
    elemNodalCoord[4] - elemNodalCoord[0],
    elemNodalCoord[5] - elemNodalCoord[0]};
  double y[] = {
    0.0,
    elemNodalCoord[7] - elemNodalCoord[6],
    elemNodalCoord[8] - elemNodalCoord[6],
    elemNodalCoord[9] - elemNodalCoord[6],
    elemNodalCoord[10] - elemNodalCoord[6],
    elemNodalCoord[11] - elemNodalCoord[6]};
  double z[] = {
    0.0,
    elemNodalCoord[13] - elemNodalCoord[12],
    elemNodalCoord[14] - elemNodalCoord[12],
    elemNodalCoord[15] - elemNodalCoord[12],
    elemNodalCoord[16] - elemNodalCoord[12],
    elemNodalCoord[17] - elemNodalCoord[12]};

  // (xp,yp,zp) is the point to be mapped into (r,s,xi) coordinate system.
  // This point must also be translated as above.

  double xp = pointCoord[0] - elemNodalCoord[0];
  double yp = pointCoord[1] - elemNodalCoord[6];
  double zp = pointCoord[2] - elemNodalCoord[12];

  // Newton-Raphson iteration for (r,s,xi)
  double j[3][3];
  double jinv[3][3];
  double f[3];
  double shapefct[6];
  double rnew = 1.0 / 3.0; // initial guess (centroid)
  double snew = 1.0 / 3.0;
  double xinew = 0.0;
  double rcur = rnew;
  double scur = snew;
  double xicur = xinew;
  double xidiff[] = {1.0, 1.0, 1.0};

  double shp_func_deriv[18];
  double current_pc[3];

  const int MAX_NR_ITER = 20;
  int i = 0;
  do {
    current_pc[0] = rcur = rnew;
    current_pc[1] = scur = snew;
    current_pc[2] = xicur = xinew;

    // Build Jacobian and Invert

    // aj(1,1)=( dN/dr  ) * x[]
    // aj(1,2)=( dN/ds  ) * x[]
    // aj(1,3)=( dN/dxi ) * x[]
    // aj(2,1)=( dN/dr  ) * y[]
    // aj(2,2)=( dN/ds  ) * y[]
    // aj(2,3)=( dN/dxi ) * y[]
    // aj(3,1)=( dN/dr  ) * z[]
    // aj(3,2)=( dN/ds  ) * z[]
    // aj(3,3)=( dN/dxi ) * z[]

    wedge_derivative(1, current_pc, shp_func_deriv);

    for (int row = 0; row != 3; ++row)
      for (int col = 0; col != 3; ++col)
        j[row][col] = 0.0;

    for (int k = 1; k != 6; ++k) {
      j[0][0] -= shp_func_deriv[k * 3 + 0] * x[k];
      j[0][1] -= shp_func_deriv[k * 3 + 1] * x[k];
      j[0][2] -= shp_func_deriv[k * 3 + 2] * x[k];

      j[1][0] -= shp_func_deriv[k * 3 + 0] * y[k];
      j[1][1] -= shp_func_deriv[k * 3 + 1] * y[k];
      j[1][2] -= shp_func_deriv[k * 3 + 2] * y[k];

      j[2][0] -= shp_func_deriv[k * 3 + 0] * z[k];
      j[2][1] -= shp_func_deriv[k * 3 + 1] * z[k];
      j[2][2] -= shp_func_deriv[k * 3 + 2] * z[k];
    }

    const double jdet = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) -
                        j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) +
                        j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);

    jinv[0][0] = (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / jdet;
    jinv[0][1] = -(j[0][1] * j[2][2] - j[2][1] * j[0][2]) / jdet;
    jinv[0][2] = (j[1][2] * j[0][1] - j[0][2] * j[1][1]) / jdet;
    jinv[1][0] = -(j[1][0] * j[2][2] - j[2][0] * j[1][2]) / jdet;
    jinv[1][1] = (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / jdet;
    jinv[1][2] = -(j[0][0] * j[1][2] - j[1][0] * j[0][2]) / jdet;
    jinv[2][0] = (j[1][0] * j[2][1] - j[2][0] * j[1][1]) / jdet;
    jinv[2][1] = -(j[0][0] * j[2][1] - j[2][0] * j[0][1]) / jdet;
    jinv[2][2] = (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / jdet;

    wedge_shape_fcn(1, current_pc, shapefct);

    // x[0] = y[0] = z[0] = 0 by construction
    f[0] = xp - (shapefct[1] * x[1] + shapefct[2] * x[2] + shapefct[3] * x[3] +
                 shapefct[4] * x[4] + shapefct[5] * x[5]);
    f[1] = yp - (shapefct[1] * y[1] + shapefct[2] * y[2] + shapefct[3] * y[3] +
                 shapefct[4] * y[4] + shapefct[5] * y[5]);
    f[2] = zp - (shapefct[1] * z[1] + shapefct[2] * z[2] + shapefct[3] * z[3] +
                 shapefct[4] * z[4] + shapefct[5] * z[5]);

    rnew = rcur - (f[0] * jinv[0][0] + f[1] * jinv[0][1] + f[2] * jinv[0][2]);
    snew = scur - (f[0] * jinv[1][0] + f[1] * jinv[1][1] + f[2] * jinv[1][2]);
    xinew = xicur - (f[0] * jinv[2][0] + f[1] * jinv[2][1] + f[2] * jinv[2][2]);

    xidiff[0] = rnew - rcur;
    xidiff[1] = snew - scur;
    xidiff[2] = xinew - xicur;
  } while (!within_tolerance(vector_norm_sq(xidiff, 3), isInElemConverged) &&
           ++i != MAX_NR_ITER);

  isoParCoord[0] = isoParCoord[1] = isoParCoord[2] =
    std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if (i < MAX_NR_ITER) {
    isoParCoord[0] = rnew;
    isoParCoord[1] = snew;
    isoParCoord[2] = xinew;
    std::array<double, 3> xx = {
      {isoParCoord[0], isoParCoord[1], isoParCoord[2]}};

    dist = parametric_distance(xx);
  }
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::interpolatePoint(
  const int& nComp,
  const double* isoParCoord,
  const double* field,
  double* result)
{
  double shapefct[6];

  wedge_shape_fcn(1, isoParCoord, shapefct);

  for (int i = 0; i < nComp; i++) {
    // Base 'field array' index for i_th component
    int b = 6 * i;

    result[i] = 0.0;

    for (int j = 0; j != 6; ++j)
      result[i] += shapefct[j] * field[b + j];
  }
}

//--------------------------------------------------------------------------
//-------- wedge_shape_fcn -------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::wedge_shape_fcn(
  const int npts, const double* isoParCoord, double* shape_fcn)
{
  for (int j = 0; j < npts; ++j) {
    int sixj = 6 * j;
    int k = 3 * j;
    double r = isoParCoord[k];
    double s = isoParCoord[k + 1];
    double t = 1.0 - r - s;
    double xi = isoParCoord[k + 2];
    shape_fcn[sixj] = 0.5 * t * (1.0 - xi);
    shape_fcn[1 + sixj] = 0.5 * r * (1.0 - xi);
    shape_fcn[2 + sixj] = 0.5 * s * (1.0 - xi);
    shape_fcn[3 + sixj] = 0.5 * t * (1.0 + xi);
    shape_fcn[4 + sixj] = 0.5 * r * (1.0 + xi);
    shape_fcn[5 + sixj] = 0.5 * s * (1.0 + xi);
  }
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
WedSCS::parametric_distance(const double X, const double Y)
{
  const double dist0 = -3 * X;
  const double dist1 = -3 * Y;
  const double dist2 = 3 * (X + Y);
  const double dist = std::max(std::max(dist0, dist1), dist2);
  return dist;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
WedSCS::parametric_distance(const std::array<double, 3>& x)
{
  const double X = x[0] - 1. / 3.;
  const double Y = x[1] - 1. / 3.;
  const double Z = x[2];
  const double dist_t = parametric_distance(X, Y);
  const double dist_z = std::fabs(Z);
  const double dist = std::max(dist_z, dist_t);
  return dist;
}

//--------------------------------------------------------------------------
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::general_face_grad_op(
  const int /* face_ordinal */,
  const double* isoParCoord,
  const double* coords,
  double* gradop,
  double* det_j,
  double* error)
{
  const int npe = nodesPerElement_;
  const int nface = 1;
  double dpsi[18];

  wedge_derivative(nface, &isoParCoord[0], dpsi);

  const SharedMemView<double***, HostShmem> deriv(
    dpsi, nface, nodesPerElement_, nDim_);
  const SharedMemView<const double***, HostShmem> cordel(coords, nface, npe, 3);
  SharedMemView<double****, HostShmem> grad(gradop, nface, nface, npe, 3);
  SharedMemView<double**, HostShmem> det(det_j, nface, nface);
  SharedMemView<double*, HostShmem> err(error, nface);
  const int lerr = wed_gradient_operator(cordel, deriv, grad, det, err);

  if (lerr)
    NaluEnv::self().naluOutput()
      << "problem with EwedSCS::general_face_grad" << std::endl;
}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::sidePcoords_to_elemPcoords(
  const int& side_ordinal,
  const int& npoints,
  const double* side_pcoords,
  double* elem_pcoords)
{
  switch (side_ordinal) {
  case 0:
    for (int i = 0; i < npoints; i++) { // face0:quad: (x,y) -> (0.5*(1 +
                                        // x),0,y)
      elem_pcoords[i * 3 + 0] = 0.5 * (1.0 + side_pcoords[2 * i + 0]);
      elem_pcoords[i * 3 + 1] = 0.0;
      elem_pcoords[i * 3 + 2] = side_pcoords[2 * i + 1];
    }
    break;
  case 1:
    for (int i = 0; i < npoints;
         i++) { // face1:quad: (x,y) -> (0.5*(1-y),0.5*(1 + y),x)
      elem_pcoords[i * 3 + 0] = 0.5 * (1.0 - side_pcoords[2 * i + 0]);
      elem_pcoords[i * 3 + 1] = 0.5 * (1.0 + side_pcoords[2 * i + 0]);
      elem_pcoords[i * 3 + 2] = side_pcoords[2 * i + 1];
    }
    break;
  case 2:
    for (int i = 0; i < npoints; i++) { // face2:quad: (x,y) -> (0,0.5*(1 +
                                        // x),y)
      elem_pcoords[i * 3 + 0] = 0.0;
      elem_pcoords[i * 3 + 1] = 0.5 * (1.0 + side_pcoords[2 * i + 1]);
      elem_pcoords[i * 3 + 2] = side_pcoords[2 * i + 0];
    }
    break;
  case 3:
    for (int i = 0; i < npoints; i++) { // face3:tri: (x,y) -> (x,y,-1)
      elem_pcoords[i * 3 + 0] = side_pcoords[2 * i + 1];
      elem_pcoords[i * 3 + 1] = side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 2] = -1.0;
    }
    break;
  case 4:
    for (int i = 0; i < npoints; i++) { // face4:tri: (x,y) -> (x,y,+1 )
      elem_pcoords[i * 3 + 0] = side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 1] = side_pcoords[2 * i + 1];
      elem_pcoords[i * 3 + 2] = 1.0;
    }
    break;
  default:
    throw std::runtime_error(
      "WedSCS::sidePcoords_to_elemPcoords invalid ordinal");
  }
}

} // namespace nalu
} // namespace sierra
