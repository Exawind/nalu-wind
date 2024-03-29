// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/Edge22DCVFEM.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>

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

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
Edge2DSCS::Edge2DSCS()
  : MasterElement(Edge2DSCS::scaleToStandardIsoFac_), elemThickness_(0.01)
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
Edge2DSCS::ipNodeMap(int /*ordinal*/) const
{
  // define ip->node mappings for each face (single ordinal);
  return ipNodeMap_;
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
Edge2DSCS::determinant_scs(
  const SharedMemView<DBLTYPE**, SHMEM>& coords,
  SharedMemView<DBLTYPE**, SHMEM>& area) const
{
  constexpr int npe = nodesPerElement_;
  constexpr int dim = nDim_;
  DBLTYPE p[dim][npe], c[dim];

  const DBLTYPE half = 0.5;

  for (int i = 0; i < npe; ++i) {
    for (int idim = 0; idim < dim; ++idim) {
      p[idim][i] = coords(i, idim);
    }
  }
  for (int idim = 0; idim < dim; ++idim)
    c[idim] = (p[idim][0] + p[idim][1]) * half;

  DBLTYPE dx13 = coords(0, 0) - c[0];
  DBLTYPE dy13 = coords(0, 1) - c[1];

  area(0, 0) = -dy13;
  area(0, 1) = dx13;

  dx13 = coords(1, 0) - c[0];
  dy13 = coords(1, 1) - c[1];

  area(1, 0) = dy13;
  area(1, 1) = -dx13;
}

KOKKOS_FUNCTION void
Edge2DSCS::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType**, DeviceShmem>& area)
{
  determinant_scs(coords, area);
}

void
Edge2DSCS::determinant(
  const SharedMemView<double**>& coords, SharedMemView<double**>& area)
{
  determinant_scs(coords, area);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Edge2DSCS::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  for (int i = 0; i < numIntPoints_; ++i) {
    shpfc(i, 0) = 0.5 - intgLoc_[i];
    shpfc(i, 1) = 0.5 + intgLoc_[i];
  }
}

KOKKOS_FUNCTION void
Edge2DSCS::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
Edge2DSCS::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Edge2DSCS::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  for (int i = 0; i < numIntPoints_; ++i) {
    shpfc(i, 0) = 0.5 - intgLocShift_[i];
    shpfc(i, 1) = 0.5 + intgLocShift_[i];
  }
}

KOKKOS_FUNCTION void
Edge2DSCS::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
Edge2DSCS::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
Edge2DSCS::isInElement(
  const double* elem_nodal_coor, // (2,2)
  const double* point_coor,      // (2)
  double* par_coor)
{
  // elem_nodal_coor has the endpoints of the line
  // segment defining this element.  Set the first
  // endpoint to zero.  This means subtrace the
  // first endpoint from the second.
  const double X1 = elem_nodal_coor[1] - elem_nodal_coor[0];
  const double X2 = elem_nodal_coor[3] - elem_nodal_coor[2];

  // Now subtract the first endpoint from the target point
  const double P1 = point_coor[0] - elem_nodal_coor[0];
  const double P2 = point_coor[1] - elem_nodal_coor[2];

  // Now find the projection along the line of the point
  // This is the parametric coordinate in range (0,1)
  const double norm2 = X1 * X1 + X2 * X2;

  const double xi = (P1 * X1 + P2 * X2) / norm2;
  // rescale to (-1,1)
  par_coor[0] = 2 * xi - 1;

  // Now find the projection from the point to a perpenducular
  // line.  This gives the distance from the point to the element.
  const double alpha = std::abs(P1 * X2 - P2 * X1) / norm2;
  if (2 == nDim_)
    par_coor[1] = alpha;

  std::array<double, 2> x;
  x[0] = par_coor[0];
  x[1] = alpha;
  const double dist = parametric_distance(x);

  return dist;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
Edge2DSCS::parametric_distance(const std::array<double, 2>& x)
{
  double dist = std::fabs(x[0]);
  if (elemThickness_ < x[1] && dist < 1.0 + x[1])
    dist = 1 + x[1];
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
Edge2DSCS::interpolatePoint(
  const int& nComp,
  const double* isoParCoord,
  const double* field,
  double* result)
{
  double xi = isoParCoord[0];
  for (int i = 0; i < nComp; i++) {
    // Base 'field array' index for ith component
    int b = 2 * i;
    result[i] =
      0.5 * (1.0 - xi) * field[b + 0] + 0.5 * (1.0 + xi) * field[b + 1];
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Edge2DSCS::general_shape_fcn(
  const int numIp, const double* isoParCoord, double* shpfc)
{
  const double npe = nodesPerElement_;
  for (int ip = 0; ip < numIp; ++ip) {
    int j = npe * ip;
    shpfc[j] = 0.5 * (1.0 - isoParCoord[ip]);
    shpfc[j + 1] = 0.5 * (1.0 + isoParCoord[ip]);
  }
}

//--------------------------------------------------------------------------
//-------- general_normal --------------------------------------------------
//--------------------------------------------------------------------------
void
Edge2DSCS::general_normal(
  const double* /*isoParCoord*/, const double* coords, double* normal)
{
  // can be only linear
  const double dx = coords[2] - coords[0];
  const double dy = coords[3] - coords[1];
  const double mag = std::sqrt(dx * dx + dy * dy);

  normal[0] = dy / mag;
  normal[1] = -dx / mag;
}
} // namespace nalu
} // namespace sierra
