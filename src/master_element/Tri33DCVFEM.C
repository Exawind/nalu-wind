// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/Tri33DCVFEM.h>

#include <AlgTraits.h>

#include <NaluEnv.h>

#include <stk_topology/topology.hpp>

#include <iostream>

#include <cmath>
#include <limits>
#include <array>
#include <map>
#include <memory>

namespace sierra {
namespace nalu {

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
Tri3DSCS::Tri3DSCS() : MasterElement()
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
Tri3DSCS::ipNodeMap(int /*ordinal*/) const
{
  // define ip->node mappings for each face (single ordinal);
  return ipNodeMap_;
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
Tri3DSCS::determinant_scs(
  const SharedMemView<DBLTYPE**, SHMEM>& coords,
  SharedMemView<DBLTYPE**, SHMEM>& areav)
{
  constexpr int dim = nDim_;
  constexpr int nnodes = nodesPerElement_;
  constexpr int nint = numIntPoints_;

  DBLTYPE dx13, dx24, dy13, dy24, dz13, dz24;

  DBLTYPE area[dim][nint];
  DBLTYPE p[dim][nnodes], e[dim][nnodes], c[dim];

  const DBLTYPE half = 0.5;
  const DBLTYPE one3rd = 1.0 / 3.0;

  for (int n = 0; n < nnodes; ++n) {
    for (int d = 0; d < dim; ++d) {
      p[d][n] = coords(n, d);
    }
  }
  for (int d = 0; d < dim; ++d) {
    e[d][0] = (p[d][0] + p[d][1]) * half;
    e[d][1] = (p[d][1] + p[d][2]) * half;
    e[d][2] = (p[d][2] + p[d][0]) * half;
    c[d] = (p[d][0] + p[d][1] + p[d][2]) * one3rd;
  }

  //... CALCULATE SUBCONTROL VOLUME FACE AREAS ...
  //    ... subcontrol volume face 1

  dx13 = c[0] - p[0][0];
  dx24 = e[0][2] - e[0][0];
  dy13 = c[1] - p[1][0];
  dy24 = e[1][2] - e[1][0];
  dz13 = c[2] - p[2][0];
  dz24 = e[2][2] - e[2][0];

  area[0][0] = half * (dz24 * dy13 - dz13 * dy24);
  area[1][0] = half * (dx24 * dz13 - dx13 * dz24);
  area[2][0] = half * (dy24 * dx13 - dy13 * dx24);

  // ... subcontrol volume face 2

  dx13 = c[0] - p[0][1];
  dx24 = e[0][0] - e[0][1];
  dy13 = c[1] - p[1][1];
  dy24 = e[1][0] - e[1][1];
  dz13 = c[2] - p[2][1];
  dz24 = e[2][0] - e[2][1];

  area[0][1] = half * (dz24 * dy13 - dz13 * dy24);
  area[1][1] = half * (dx24 * dz13 - dx13 * dz24);
  area[2][1] = half * (dy24 * dx13 - dy13 * dx24);

  // ... subcontrol volume face 3

  dx13 = c[0] - p[0][2];
  dx24 = e[0][1] - e[0][2];
  dy13 = c[1] - p[1][2];
  dy24 = e[1][1] - e[1][2];
  dz13 = c[2] - p[2][2];
  dz24 = e[2][1] - e[2][2];

  area[0][2] = half * (dz24 * dy13 - dz13 * dy24);
  area[1][2] = half * (dx24 * dz13 - dx13 * dz24);
  area[2][2] = half * (dy24 * dx13 - dy13 * dx24);

  for (int f = 0; f < nint; ++f) {
    for (int d = 0; d < dim; ++d) {
      areav(f, d) = area[d][f];
    }
  }
}
KOKKOS_FUNCTION void
Tri3DSCS::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType**, DeviceShmem>& areav)
{
  determinant_scs(coords, areav);
}
void
Tri3DSCS::determinant(
  const SharedMemView<double**>& coords, SharedMemView<double**>& areav)
{
  determinant_scs(coords, areav);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Tri3DSCS::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tri_shape_fcn(intgLoc_, shpfc);
}
KOKKOS_FUNCTION void
Tri3DSCS::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
Tri3DSCS::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Tri3DSCS::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tri_shape_fcn(intgLocShift_, shpfc);
}
KOKKOS_FUNCTION void
Tri3DSCS::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
Tri3DSCS::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- tri_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Tri3DSCS::tri_shape_fcn(
  const double* isoParCoord, SharedMemView<SCALAR**, SHMEM>& shape_fcn)
{
  for (int j = 0; j < numIntPoints_; ++j) {
    const int k = 2 * j;
    const double xi = isoParCoord[k];
    const double eta = isoParCoord[k + 1];
    shape_fcn(j, 0) = 1.0 - xi - eta;
    shape_fcn(j, 1) = xi;
    shape_fcn(j, 2) = eta;
  }
}

void
Tri3DSCS::tri_shape_fcn(
  const int npts, const double* isoParCoord, double* shape_fcn)
{
  for (int j = 0; j < npts; ++j) {
    const int threej = 3 * j;
    const int k = 2 * j;
    const double xi = isoParCoord[k];
    const double eta = isoParCoord[k + 1];
    shape_fcn[threej] = 1.0 - xi - eta;
    shape_fcn[1 + threej] = xi;
    shape_fcn[2 + threej] = eta;
  }
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
Tri3DSCS::isInElement(
  const double* elem_nodal_coor, const double* point_coor, double* par_coor)
{
  // always intended for 3D...
  const int npar_coord = 3;
  // Translate element so that (x,y,z) coordinates of the
  // first node
  double x[2] = {elem_nodal_coor[1] - elem_nodal_coor[0],
                 elem_nodal_coor[2] - elem_nodal_coor[0]};
  double y[2] = {elem_nodal_coor[4] - elem_nodal_coor[3],
                 elem_nodal_coor[5] - elem_nodal_coor[3]};
  double z[2] = {elem_nodal_coor[7] - elem_nodal_coor[6],
                 elem_nodal_coor[8] - elem_nodal_coor[6]};

  // Translate position of point in same manner

  double xp = point_coor[0] - elem_nodal_coor[0];
  double yp = point_coor[1] - elem_nodal_coor[3];
  double zp = point_coor[2] - elem_nodal_coor[6];

  // Set new nodal coordinates with Node 1 at origin and with new
  // x and y axes lying in the plane of the element
  double len12 = std::sqrt(x[0] * x[0] + y[0] * y[0] + z[0] * z[0]);
  double len13 = std::sqrt(x[1] * x[1] + y[1] * y[1] + z[1] * z[1]);

  double xnew[3];
  double ynew[3];
  double znew[3];

  // Use cross-product of 12 and 13 to find enclosed angle and
  // direction of new z-axis

  znew[0] = y[0] * z[1] - y[1] * z[0];
  znew[1] = x[1] * z[0] - x[0] * z[1];
  znew[2] = x[0] * y[1] - x[1] * y[0];

  double Area2 =
    std::sqrt(znew[0] * znew[0] + znew[1] * znew[1] + znew[2] * znew[2]);

  // find sin of angle
  double sin_theta = Area2 / (len12 * len13);

  // find cosine of angle
  double cos_theta =
    (x[0] * x[1] + y[0] * y[1] + z[0] * z[1]) / (len12 * len13);

  // nodal coordinates of nodes 2 and 3 in new system
  // (coordinates of node 1 are identically 0.0)
  double x_nod_new[2] = {len12, len13 * cos_theta};
  double y_nod_new[2] = {0.0, len13 * sin_theta};

  // find direction cosines transform position of
  // point to be checked into new coordinate system

  // direction cosines of new x axis along side 12

  xnew[0] = x[0] / len12;
  xnew[1] = y[0] / len12;
  xnew[2] = z[0] / len12;

  // direction cosines of new z axis
  znew[0] = znew[0] / Area2;
  znew[1] = znew[1] / Area2;
  znew[2] = znew[2] / Area2;

  // direction cosines of new y-axis (cross-product of znew and xnew)
  ynew[0] = znew[1] * xnew[2] - xnew[1] * znew[2];
  ynew[1] = xnew[0] * znew[2] - znew[0] * xnew[2];
  ynew[2] = znew[0] * xnew[1] - xnew[0] * znew[1];

  // compute transformed coordinates of point
  // (coordinates in xnew,ynew,znew)
  double xpnew = xnew[0] * xp + xnew[1] * yp + xnew[2] * zp;
  double ypnew = ynew[0] * xp + ynew[1] * yp + ynew[2] * zp;
  double zpnew = znew[0] * xp + znew[1] * yp + znew[2] * zp;

  // Find parametric coordinates of point and check that
  // it lies in the element
  par_coor[0] =
    1. - xpnew / x_nod_new[0] + ypnew * (x_nod_new[1] - x_nod_new[0]) / Area2;
  par_coor[1] = (xpnew * y_nod_new[1] - ypnew * x_nod_new[1]) / Area2;

  if (3 == npar_coord)
    par_coor[2] = zpnew / std::sqrt(Area2);

  std::array<double, 3> w = {
    {par_coor[0], par_coor[1], zpnew / std::sqrt(Area2)}};

  par_coor[0] = w[1];
  par_coor[1] = 1.0 - w[0] - w[1];

  const double dist = parametric_distance(w);

  return dist;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
Tri3DSCS::parametric_distance(const std::array<double, 3>& x)
{
  const double ELEM_THICK = 0.01;
  const double X = x[0] - 1. / 3.;
  const double Y = x[1] - 1. / 3.;
  const double dist0 = -3 * X;
  const double dist1 = -3 * Y;
  const double dist2 = 3 * (X + Y);
  double dist = std::max(std::max(dist0, dist1), dist2);
  const double y = std::fabs(x[2]);
  if (ELEM_THICK < y && dist < 1 + y)
    dist = 1 + y;
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
Tri3DSCS::interpolatePoint(
  const int& ncomp_field,
  const double* isoParCoord,
  const double* field,
  double* result)
{
  const double r = isoParCoord[0];
  const double s = isoParCoord[1];
  const double t = 1.0 - r - s;

  for (int i = 0; i < ncomp_field; i++) {
    int b = 3 * i; // Base 'field array' index for ith component
    result[i] = t * field[b] + r * field[b + 1] + s * field[b + 2];
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Tri3DSCS::general_shape_fcn(
  const int numIp, const double* isoParCoord, double* shpfc)
{
  tri_shape_fcn(numIp, isoParCoord, shpfc);
}

//--------------------------------------------------------------------------
//-------- general_normal --------------------------------------------------
//--------------------------------------------------------------------------
void
Tri3DSCS::general_normal(
  const double* /*isoParCoord*/, const double* coords, double* normal)
{
  // can be only linear
  const double ax = coords[3] - coords[0];
  const double ay = coords[4] - coords[1];
  const double az = coords[5] - coords[2];
  const double bx = coords[6] - coords[0];
  const double by = coords[7] - coords[1];
  const double bz = coords[8] - coords[2];

  normal[0] = (ay * bz - az * by);
  normal[1] = (az * bx - ax * bz);
  normal[2] = (ax * by - ay * bx);

  const double mag = std::sqrt(
    normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  normal[0] /= mag;
  normal[1] /= mag;
  normal[2] /= mag;
}
} // namespace nalu
} // namespace sierra
