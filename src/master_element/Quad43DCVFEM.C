// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "master_element/MasterElement.h"
#include "master_element/Quad43DCVFEM.h"
#include "master_element/Hex8GeometryFunctions.h"

#include <array>

namespace sierra {
namespace nalu {

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
Quad3DSCS::Quad3DSCS()
  : MasterElement(Quad3DSCS::scaleToStandardIsoFac_), elemThickness_(0.1)
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
Quad3DSCS::ipNodeMap(int /*ordinal*/) const
{
  // define ip->node mappings for each face (single ordinal);
  return ipNodeMap_;
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Quad3DSCS::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  quad4_shape_fcn(intgLoc_, shpfc);
}
KOKKOS_FUNCTION void
Quad3DSCS::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
Quad3DSCS::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Quad3DSCS::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  quad4_shape_fcn(intgLocShift_, shpfc);
}
KOKKOS_FUNCTION void
Quad3DSCS::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
Quad3DSCS::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
Quad3DSCS::determinant_scs(
  const SharedMemView<DBLTYPE**, SHMEM>& coords,
  SharedMemView<DBLTYPE**, SHMEM>& areav) const
{
  constexpr int npf = 4;  // Nodes per face
  constexpr int nscs = 4; // Number of sub-control surfaces per face
  // Coordinates of nodes for SCS
  DBLTYPE coordv[9][3];

  // Index map of nodes (in coordv) for the nodes
  constexpr int quad_edge_facet_table[4][4] = {
    {0, 4, 8, 7}, {4, 1, 5, 8}, {8, 5, 2, 6}, {7, 8, 6, 3}};

  for (int i = 0; i < npf; ++i) {
    coordv[i][0] = coords(i, 0);
    coordv[i][1] = coords(i, 1);
    coordv[i][2] = coords(i, 2);
  }

  for (int d = 0; d < 3; ++d) {
    coordv[4][d] = 0.5 * (coords(0, d) + coords(1, d)); // edge 1
    coordv[5][d] = 0.5 * (coords(1, d) + coords(2, d)); // edge 2
    coordv[6][d] = 0.5 * (coords(2, d) + coords(3, d)); // edge 3
    coordv[7][d] = 0.5 * (coords(3, d) + coords(0, d)); // edge 4

    // centroid
    coordv[8][d] =
      0.25 * (coords(0, d) + coords(1, d) + coords(2, d) + coords(3, d));
  }

  for (int ics = 0; ics < nscs; ++ics) {
    DBLTYPE scscoords[4][3];
    for (int inode = 0; inode < npf; ++inode) {
      const int itrianglenode = quad_edge_facet_table[ics][inode];
      scscoords[inode][0] = coordv[itrianglenode][0];
      scscoords[inode][1] = coordv[itrianglenode][1];
      scscoords[inode][2] = coordv[itrianglenode][2];
    }
    quad_area_by_triangulation(ics, scscoords, areav);
  }
}

KOKKOS_FUNCTION void
Quad3DSCS::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType**, DeviceShmem>& areav)
{
  determinant_scs(coords, areav);
}

void
Quad3DSCS::determinant(
  const SharedMemView<double**>& coords, SharedMemView<double**>& areav)
{
  determinant_scs(coords, areav);
}
//--------------------------------------------------------------------------
//-------- quad4_shape_fcn --------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Quad3DSCS::quad4_shape_fcn(
  const double* isoParCoord, SharedMemView<SCALAR**, SHMEM>& shape_fcn)
{
  // -1/2:1/2 isoparametric range
  const SCALAR half = 0.50;
  const SCALAR one4th = 0.25;
  for (int j = 0; j < numIntPoints_; ++j) {

    const SCALAR s1 = isoParCoord[j * 2];
    const SCALAR s2 = isoParCoord[j * 2 + 1];

    shape_fcn(j, 0) = one4th + half * (-s1 - s2) + s1 * s2;
    shape_fcn(j, 1) = one4th + half * (s1 - s2) - s1 * s2;
    shape_fcn(j, 2) = one4th + half * (s1 + s2) + s1 * s2;
    shape_fcn(j, 3) = one4th + half * (-s1 + s2) - s1 * s2;
  }
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
Quad3DSCS::isInElement(
  const double* elemNodalCoord, const double* pointCoord, double* isoParCoord)
{
  // square of the desired norm, 1.0e-8
  const double isInElemConverged = 1.0e-16;
  const int maxNonlinearIter = 20;

  // Translate element so that (x,y,z) coordinates of the first node are (0,0,0)

  double x[3] = {
    elemNodalCoord[1] - elemNodalCoord[0],
    elemNodalCoord[2] - elemNodalCoord[0],
    elemNodalCoord[3] - elemNodalCoord[0]};

  double y[3] = {
    elemNodalCoord[5] - elemNodalCoord[4],
    elemNodalCoord[6] - elemNodalCoord[4],
    elemNodalCoord[7] - elemNodalCoord[4]};

  double z[3] = {
    elemNodalCoord[9] - elemNodalCoord[8],
    elemNodalCoord[10] - elemNodalCoord[8],
    elemNodalCoord[11] - elemNodalCoord[8]};

  // (xp,yp,zp) is the point at which we're searching for (xi,eta,d)
  // (must translate this also)
  // d = (scaled) distance in (x,y,z) space from point (xp,yp,zp) to the
  //     surface defined by the face element (the distance is scaled by
  //     the length of the non-unit normal; rescaling of d is done
  //     following the NR iteration below).

  double xp = pointCoord[0] - elemNodalCoord[0];
  double yp = pointCoord[1] - elemNodalCoord[4];
  double zp = pointCoord[2] - elemNodalCoord[8];

  // Newton-Raphson iteration for (xi,eta,d)

  double jdet;
  double j[9];
  double gn[3];
  double xcur[3];   // current (x,y,z) point on element surface
  double normal[3]; // (non-unit) normal computed at xcur

  // Solution solcur[3] = {xi,eta,d}
  double solcur[3] = {-0.5, -0.5, -0.5}; // initial guess
  double deltasol[] = {1.0, 1.0, 1.0};

  int i = 0;
  do {
    // Update guess point
    solcur[0] += deltasol[0];
    solcur[1] += deltasol[1];
    solcur[2] += deltasol[2];

    interpolatePoint(3, solcur, elemNodalCoord, xcur);

    // Translate xcur ((x,y,z) point corresponding
    // to current (xi,eta) guess)

    xcur[0] -= elemNodalCoord[0];
    xcur[1] -= elemNodalCoord[4];
    xcur[2] -= elemNodalCoord[8];

    non_unit_face_normal(solcur, elemNodalCoord, normal);

    gn[0] = xcur[0] - xp + solcur[2] * normal[0];
    gn[1] = xcur[1] - yp + solcur[2] * normal[1];
    gn[2] = xcur[2] - zp + solcur[2] * normal[2];

    // Mathematica-generated code for the jacobian

    j[0] = 0.125 * (-2.00 * (-1.00 + solcur[1]) * x[0] +
                    (2.00 * (1.00 + solcur[1]) * (x[1] - x[2]) +
                     solcur[2] * (-(y[1] * z[0]) + y[2] * z[0] + y[0] * z[1] -
                                  y[0] * z[2])));

    j[1] =
      0.125 *
      (-2.00 * (1.00 + solcur[0]) * x[0] + 2.00 * (1.00 + solcur[0]) * x[1] -
       2.00 * (-1.00 + solcur[0]) * x[2] +
       (solcur[2] * (y[2] * (z[0] - z[1]) + (-y[0] + y[1]) * z[2])));

    j[2] = normal[0];

    j[3] =
      0.125 *
      (-2.00 * (-1.00 + solcur[1]) * y[0] +
       (2.00 * (1.00 + solcur[1]) * (y[1] - y[2]) +
        solcur[2] * (x[1] * z[0] - x[2] * z[0] - x[0] * z[1] + x[0] * z[2])));

    j[4] =
      0.125 *
      (-2.00 * (1.00 + solcur[0]) * y[0] + 2.00 * (1.00 + solcur[0]) * y[1] -
       2.00 * (-1.00 + solcur[0]) * y[2] +
       (solcur[2] * (x[2] * (-z[0] + z[1]) + (x[0] - x[1]) * z[2])));

    j[5] = normal[1];

    j[6] = 0.125 * ((solcur[2] * (-(x[1] * y[0]) + x[2] * y[0] + x[0] * y[1] -
                                  x[0] * y[2])) -
                    2.00 * ((-1.00 + solcur[1]) * z[0] -
                            (1.00 + solcur[1]) * (z[1] - z[2])));

    j[7] =
      0.125 *
      ((solcur[2] * (x[2] * (y[0] - y[1]) + (-x[0] + x[1]) * y[2])) -
       2.00 * (1.00 + solcur[0]) * z[0] + 2.00 * (1.00 + solcur[0]) * z[1] -
       2.00 * (-1.00 + solcur[0]) * z[2]);

    j[8] = normal[2];

    jdet = -(j[2] * j[4] * j[6]) + j[1] * j[5] * j[6] + j[2] * j[3] * j[7] -
           j[0] * j[5] * j[7] - j[1] * j[3] * j[8] + j[0] * j[4] * j[8];

    // Solve linear system (j*deltasol = -gn) for deltasol at step n+1

    deltasol[0] = (gn[2] * (j[2] * j[4] - j[1] * j[5]) +
                   gn[1] * (-(j[2] * j[7]) + j[1] * j[8]) +
                   gn[0] * (j[5] * j[7] - j[4] * j[8])) /
                  jdet;
    deltasol[1] = (gn[2] * (-(j[2] * j[3]) + j[0] * j[5]) +
                   gn[1] * (j[2] * j[6] - j[0] * j[8]) +
                   gn[0] * (-(j[5] * j[6]) + j[3] * j[8])) /
                  jdet;
    deltasol[2] = (gn[2] * (j[1] * j[3] - j[0] * j[4]) +
                   gn[1] * (-(j[1] * j[6]) + j[0] * j[7]) +
                   gn[0] * (j[4] * j[6] - j[3] * j[7])) /
                  jdet;

  } while (!within_tolerance(vector_norm_sq(deltasol, 3), isInElemConverged) &&
           ++i < maxNonlinearIter);

  // Fill in solution; only include the distance (in the third
  // solution slot) if npar_coord = 3 (this is how the user
  // requests it)

  isoParCoord[0] = std::numeric_limits<double>::max();
  isoParCoord[1] = std::numeric_limits<double>::max();
  isoParCoord[2] = std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if (i < maxNonlinearIter) {
    isoParCoord[0] = solcur[0] + deltasol[0];
    isoParCoord[1] = solcur[1] + deltasol[1];
    // Rescale the distance by the length of the (non-unit) normal,
    // which was used above in the NR iteration.
    const double area = std::sqrt(vector_norm_sq(normal, 3));
    const double length = std::sqrt(area);

    const double par_coor_2 = (solcur[2] + deltasol[2]) * length;
    // if ( npar_coord == 3 ) par_coor[2] = par_coor_2;
    isoParCoord[2] = par_coor_2;

    std::array<double, 3> xtmp;
    xtmp[0] = isoParCoord[0];
    xtmp[1] = isoParCoord[1];
    xtmp[2] = isoParCoord[2];
    dist = parametric_distance(xtmp);
  }
  return dist;
}

void
Quad3DSCS::non_unit_face_normal(
  const double* isoParCoord,     // (2)
  const double* elem_nodal_coor, // (4,3)
  double* normal_vector)         // (3)
{
  double xi = isoParCoord[0];
  double eta = isoParCoord[1];

  // Translate element so that node 0 is at (x,y,z) = (0,0,0)

  double x[3] = {
    elem_nodal_coor[1] - elem_nodal_coor[0],
    elem_nodal_coor[2] - elem_nodal_coor[0],
    elem_nodal_coor[3] - elem_nodal_coor[0]};

  double y[3] = {
    elem_nodal_coor[5] - elem_nodal_coor[4],
    elem_nodal_coor[6] - elem_nodal_coor[4],
    elem_nodal_coor[7] - elem_nodal_coor[4]};

  double z[3] = {
    elem_nodal_coor[9] - elem_nodal_coor[8],
    elem_nodal_coor[10] - elem_nodal_coor[8],
    elem_nodal_coor[11] - elem_nodal_coor[8]};

  // Mathematica-generated and simplified code for the normal vector

  double n0 = 0.125 * (xi * y[2] * z[0] + y[0] * z[1] + xi * y[0] * z[1] -
                       y[2] * z[1] - xi * y[0] * z[2] +
                       y[1] * (-((1.00 + xi) * z[0]) + (1.00 + eta) * z[2]) +
                       eta * (y[2] * z[0] - y[2] * z[1] - y[0] * z[2]));

  double n1 = 0.125 * (-(xi * x[2] * z[0]) - x[0] * z[1] - xi * x[0] * z[1] +
                       x[2] * z[1] + xi * x[0] * z[2] +
                       x[1] * ((1.00 + xi) * z[0] - (1.00 + eta) * z[2]) +
                       eta * (-(x[2] * z[0]) + x[2] * z[1] + x[0] * z[2]));

  double n2 = 0.125 * (xi * x[2] * y[0] + x[0] * y[1] + xi * x[0] * y[1] -
                       x[2] * y[1] - xi * x[0] * y[2] +
                       x[1] * (-((1.00 + xi) * y[0]) + (1.00 + eta) * y[2]) +
                       eta * (x[2] * y[0] - x[2] * y[1] - x[0] * y[2]));

  normal_vector[0] = n0;
  normal_vector[1] = n1;
  normal_vector[2] = n2;
}

double
Quad3DSCS::parametric_distance(const std::array<double, 3>& x)
{
  const int NCOORD = 3;
  std::array<double, NCOORD> y;

  for (int i = 0; i < NCOORD; ++i) {
    y[i] = std::abs(x[i]);
  }

  double d = y[0];
  if (d < y[1])
    d = y[1];
  if (elemThickness_ < y[2] && d < 1 + y[2])
    d = 1 + y[2];
  return d;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::interpolatePoint(
  const int& nComp,
  const double* isoParCoord,
  const double* field,
  double* result)
{
  // this is the same as the 2D implementation... Consider consolidation
  const double xi = isoParCoord[0];
  const double eta = isoParCoord[1];

  for (int i = 0; i < nComp; i++) {
    // Base 'field array' index for ith component
    int b = 4 * i;

    result[i] = 0.250 * ((1.00 - eta) * (1.00 - xi) * field[b + 0] +
                         (1.00 - eta) * (1.00 + xi) * field[b + 1] +
                         (1.00 + eta) * (1.00 + xi) * field[b + 2] +
                         (1.00 + eta) * (1.00 - xi) * field[b + 3]);
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::general_shape_fcn(
  const int numIp, const double* isoParCoord, double* shpfc)
{
  // -1:1 isoparametric range
  const double npe = nodesPerElement_;
  for (int ip = 0; ip < numIp; ++ip) {

    const int rowIpc = 2 * ip;
    const int rowSfc = npe * ip;

    const double s1 = isoParCoord[rowIpc];
    const double s2 = isoParCoord[rowIpc + 1];
    shpfc[rowSfc] = 0.25 * (1.0 - s1) * (1.0 - s2);
    shpfc[rowSfc + 1] = 0.25 * (1.0 + s1) * (1.0 - s2);
    shpfc[rowSfc + 2] = 0.25 * (1.0 + s1) * (1.0 + s2);
    shpfc[rowSfc + 3] = 0.25 * (1.0 - s1) * (1.0 + s2);
  }
}

//--------------------------------------------------------------------------
//-------- general_normal --------------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::general_normal(
  const double* isoParCoord, const double* coords, double* normal)
{
  const int nDim = 3;

  const double psi0Xi = -0.25 * (1.0 - isoParCoord[1]);
  const double psi1Xi = 0.25 * (1.0 - isoParCoord[1]);
  const double psi2Xi = 0.25 * (1.0 + isoParCoord[1]);
  const double psi3Xi = -0.25 * (1.0 + isoParCoord[1]);

  const double psi0Eta = -0.25 * (1.0 - isoParCoord[0]);
  const double psi1Eta = -0.25 * (1.0 + isoParCoord[0]);
  const double psi2Eta = 0.25 * (1.0 + isoParCoord[0]);
  const double psi3Eta = 0.25 * (1.0 - isoParCoord[0]);

  const double DxDxi =
    coords[0 * nDim + 0] * psi0Xi + coords[1 * nDim + 0] * psi1Xi +
    coords[2 * nDim + 0] * psi2Xi + coords[3 * nDim + 0] * psi3Xi;

  const double DyDxi =
    coords[0 * nDim + 1] * psi0Xi + coords[1 * nDim + 1] * psi1Xi +
    coords[2 * nDim + 1] * psi2Xi + coords[3 * nDim + 1] * psi3Xi;

  const double DzDxi =
    coords[0 * nDim + 2] * psi0Xi + coords[1 * nDim + 2] * psi1Xi +
    coords[2 * nDim + 2] * psi2Xi + coords[3 * nDim + 2] * psi3Xi;

  const double DxDeta =
    coords[0 * nDim + 0] * psi0Eta + coords[1 * nDim + 0] * psi1Eta +
    coords[2 * nDim + 0] * psi2Eta + coords[3 * nDim + 0] * psi3Eta;

  const double DyDeta =
    coords[0 * nDim + 1] * psi0Eta + coords[1 * nDim + 1] * psi1Eta +
    coords[2 * nDim + 1] * psi2Eta + coords[3 * nDim + 1] * psi3Eta;

  const double DzDeta =
    coords[0 * nDim + 2] * psi0Eta + coords[1 * nDim + 2] * psi1Eta +
    coords[2 * nDim + 2] * psi2Eta + coords[3 * nDim + 2] * psi3Eta;

  const double detXY = DxDxi * DyDeta - DxDeta * DyDxi;
  const double detYZ = DyDxi * DzDeta - DyDeta * DzDxi;
  const double detXZ = -DxDxi * DzDeta + DxDeta * DzDxi;

  const double det = std::sqrt(detXY * detXY + detYZ * detYZ + detXZ * detXZ);

  normal[0] = detYZ / det;
  normal[1] = detXZ / det;
  normal[2] = detXY / det;
}

} // namespace nalu
} // namespace sierra
