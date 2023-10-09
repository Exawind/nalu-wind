// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/Tri32DCVFEM.h>
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

//-------- tri_derivative -----------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_FUNCTION void
tri_derivative(SharedMemView<DBLTYPE***, SHMEM>& deriv)
{
  const int npts = deriv.extent(0);
  for (int j = 0; j < npts; ++j) {
    deriv(j, 0, 0) = -1.0;
    deriv(j, 1, 0) = 1.0;
    deriv(j, 2, 0) = 0.0;
    deriv(j, 0, 1) = -1.0;
    deriv(j, 1, 1) = 0.0;
    deriv(j, 2, 1) = 1.0;
  }
}

//-------- tri_gradient_operator
//-----------------------------------------------------
template <typename DBLTYPE, typename CONST_DBLTYPE, typename SHMEM>
KOKKOS_FUNCTION int
tri_gradient_operator(
  const SharedMemView<CONST_DBLTYPE**, SHMEM>& coords,
  const SharedMemView<DBLTYPE***, SHMEM>& deriv,
  SharedMemView<DBLTYPE***, SHMEM>& gradop,
  SharedMemView<DBLTYPE*, SHMEM>& det_j)
{
  int nerr = 0;
  const double realmin = std::numeric_limits<double>::min();
  const int nint = deriv.extent(0);
  const int npe = deriv.extent(1);

  DBLTYPE dx_ds1, dx_ds2;
  DBLTYPE dy_ds1, dy_ds2;

  for (int ki = 0; ki < nint; ++ki) {
    dx_ds1 = 0.0;
    dx_ds2 = 0.0;
    dy_ds1 = 0.0;
    dy_ds2 = 0.0;

    // calculate the jacobian at the integration station -
    for (int kn = 0; kn < npe; ++kn) {
      dx_ds1 += deriv(ki, kn, 0) * coords(kn, 0);
      dx_ds2 += deriv(ki, kn, 1) * coords(kn, 0);
      dy_ds1 += deriv(ki, kn, 0) * coords(kn, 1);
      dy_ds2 += deriv(ki, kn, 1) * coords(kn, 1);
    }

    // calculate the determinate of the jacobian at the integration station -
    det_j(ki) = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;

    // protect against a negative or small value for the determinate of the
    // jacobian. The value of real_min (set in precision.par) represents
    // the smallest Real value (based upon the precision set for this
    // compilation) which the machine can represent -
    const DBLTYPE denom = stk::math::if_then_else(
      det_j(ki) < 1.e6 * MEconstants::realmin, 1.0, 1.0 / det_j(ki));
    if (stk::simd::get_data(det_j(ki), 0) <= 1.e6 * realmin)
      nerr = ki;

    // compute the gradient operators at the integration station -
    const DBLTYPE ds1_dx = denom * dy_ds2;
    const DBLTYPE ds2_dx = -denom * dy_ds1;
    const DBLTYPE ds1_dy = -denom * dx_ds2;
    const DBLTYPE ds2_dy = denom * dx_ds1;

    for (int kn = 0; kn < npe; ++kn) {
      gradop(ki, kn, 0) = deriv(ki, kn, 0) * ds1_dx + deriv(ki, kn, 1) * ds2_dx;
      gradop(ki, kn, 1) = deriv(ki, kn, 0) * ds1_dy + deriv(ki, kn, 1) * ds2_dy;
    }
  }
  return nerr;
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
Tri32DSCV::Tri32DSCV() : MasterElement()
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
Tri32DSCV::ipNodeMap(int /*ordinal*/) const
{
  // define scv->node mappings
  return ipNodeMap_;
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Tri32DSCV::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tri_shape_fcn(intgLoc_, shpfc);
}
KOKKOS_FUNCTION void
Tri32DSCV::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
Tri32DSCV::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Tri32DSCV::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tri_shape_fcn(intgLocShift_, shpfc);
}
KOKKOS_FUNCTION void
Tri32DSCV::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
Tri32DSCV::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- tri_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Tri32DSCV::tri_shape_fcn(
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

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
Tri32DSCV::determinant_scv(
  const SharedMemView<DBLTYPE**, SHMEM>& coords,
  SharedMemView<DBLTYPE*, SHMEM>& vol) const
{

  const int nint = numIntPoints_;

  DBLTYPE deriv[2][4];
  DBLTYPE xyval[2][4][3];

  // Gaussian quadrature points within an interval [-.5,+.5]
  const double gpp = 0.288675134;
  const double gpm = -0.288675134;

  const double zero = 0.0;
  const double half = 0.5;
  const double one4th = 0.25;

  // store sub-volume centroids
  const double xigp[2][4] = {{gpm, gpp, gpp, gpm}, {gpm, gpm, gpp, gpp}};

  const double one3rd = 1.0 / 3.0;
  const int kx = 0;
  const int ky = 1;

  // 2d cartesian, no cross-section area
  const DBLTYPE xc = one3rd * (coords(0, kx) + coords(1, kx) + coords(2, kx));
  const DBLTYPE yc = one3rd * (coords(0, ky) + coords(1, ky) + coords(2, ky));

  // sub-volume 1
  xyval[kx][0][0] = coords(0, kx);
  xyval[kx][1][0] = half * (coords(0, kx) + coords(1, kx));
  xyval[kx][2][0] = xc;
  xyval[kx][3][0] = half * (coords(2, kx) + coords(0, kx));

  xyval[ky][0][0] = coords(0, ky);
  xyval[ky][1][0] = half * (coords(0, ky) + coords(1, ky));
  xyval[ky][2][0] = yc;
  xyval[ky][3][0] = half * (coords(2, ky) + coords(0, ky));

  // sub-volume 2
  xyval[kx][0][1] = coords(1, kx);
  xyval[kx][1][1] = half * (coords(1, kx) + coords(2, kx));
  xyval[kx][2][1] = xc;
  xyval[kx][3][1] = half * (coords(0, kx) + coords(1, kx));

  xyval[ky][0][1] = coords(1, ky);
  xyval[ky][1][1] = half * (coords(1, ky) + coords(2, ky));
  xyval[ky][2][1] = yc;
  xyval[ky][3][1] = half * (coords(0, ky) + coords(1, ky));

  // sub-volume 3
  xyval[kx][0][2] = coords(2, kx);
  xyval[kx][1][2] = half * (coords(2, kx) + coords(0, kx));
  xyval[kx][2][2] = xc;
  xyval[kx][3][2] = half * (coords(1, kx) + coords(2, kx));

  xyval[ky][0][2] = coords(2, ky);
  xyval[ky][1][2] = half * (coords(2, ky) + coords(0, ky));
  xyval[ky][2][2] = yc;
  xyval[ky][3][2] = half * (coords(1, ky) + coords(2, ky));

  DBLTYPE dx_ds1, dx_ds2, dy_ds1, dy_ds2;
  double etamod, ximod;
  for (int ki = 0; ki < nint; ++ki) {
    vol[ki] = zero;

    for (int kq = 0; kq < 4; ++kq) {
      dx_ds1 = zero;
      dx_ds2 = zero;
      dy_ds1 = zero;
      dy_ds2 = zero;

      ximod = xigp[0][kq];
      etamod = xigp[1][kq];

      deriv[0][0] = -(half - etamod);
      deriv[0][1] = (half - etamod);
      deriv[0][2] = (half + etamod);
      deriv[0][3] = -(half + etamod);

      deriv[1][0] = -(half - ximod);
      deriv[1][1] = -(half + ximod);
      deriv[1][2] = (half + ximod);
      deriv[1][3] = (half - ximod);

      //  calculate the jacobian at the integration station -
      for (int kn = 0; kn < 4; ++kn) {
        dx_ds1 += deriv[0][kn] * xyval[kx][kn][ki];
        dx_ds2 += deriv[1][kn] * xyval[kx][kn][ki];

        dy_ds1 += deriv[0][kn] * xyval[ky][kn][ki];
        dy_ds2 += deriv[1][kn] * xyval[ky][kn][ki];
      }

      // calculate the determinate of the jacobian at the integration station -
      const DBLTYPE det_j = (dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2);
      vol[ki] += det_j * one4th;
    }
  }
}
KOKKOS_FUNCTION void
Tri32DSCV::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType*, DeviceShmem>& vol)
{
  determinant_scv(coords, vol);
}
void
Tri32DSCV::determinant(
  const SharedMemView<double**>& coords, SharedMemView<double*>& vol)
{
  determinant_scv(coords, vol);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Tri32DSCV::grad_op(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tri_derivative(deriv);
  DoubleType det[numIntPoints_];
  SharedMemView<DoubleType*, DeviceShmem> det_j(det, numIntPoints_);
  tri_gradient_operator(coords, deriv, gradop, det_j);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Tri32DSCV::shifted_grad_op(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tri_derivative(deriv);
  DoubleType det[numIntPoints_];
  SharedMemView<DoubleType*, DeviceShmem> det_j(det, numIntPoints_);
  tri_gradient_operator(coords, deriv, gradop, det_j);
}

//--------------------------------------------------------------------------
//-------- Metric Tensor Mij------------------------------------------------
//--------------------------------------------------------------------------
// This function computes the metric tensor Mij = (J J^T)^(1/2) where J is
// the Jacobian.  This is needed for the UT-A Hybrid LES model.  For
// reference please consult the Nalu theory manual description of the UT-A
// Hybrid LES model or S. Haering's PhD thesis: Anisotropic hybrid turbulence
// modeling with specific application to the simulation of pulse-actuated
// dynamic stall control.
//--------------------------------------------------------------------------
void
Tri32DSCV::Mij(const double* coords, double* metric, double* deriv)
{
  generic_Mij_2d<AlgTraitsTri3_2D>(numIntPoints_, deriv, coords, metric);
}

//-------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Tri32DSCV::Mij(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& metric,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  generic_Mij_2d<AlgTraitsTri3_2D>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
Tri32DSCS::Tri32DSCS() : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;

#if !defined(KOKKOS_ENABLE_GPU)
  const std::array<std::array<double, 2>, 3> nodeLocations = {
    {{{0.0, 0.0}}, {{1.0, 0}}, {{0.0, 1.0}}}};
  stk::topology topo = stk::topology::TRIANGLE_3_2D;
  for (unsigned k = 0; k < topo.num_sides(); ++k) {
    stk::topology side_topo = topo.side_topology(k);
    const int* ordinals = side_node_ordinals(k);
    for (unsigned n = 0; n < side_topo.num_nodes(); ++n) {
      intgExpFaceShift_[k][n][0] = nodeLocations[ordinals[n]][0];
      intgExpFaceShift_[k][n][1] = nodeLocations[ordinals[n]][1];
    }
  }
#endif
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
Tri32DSCS::ipNodeMap(int ordinal) const
{
  // define ip->node mappings for each face (ordinal);
  return ipNodeMap_[ordinal];
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
Tri32DSCS::side_node_ordinals(int ordinal) const
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return sideNodeOrdinals_[ordinal];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
Tri32DSCS::determinant_scs(
  const SharedMemView<DBLTYPE**, SHMEM>& coords,
  SharedMemView<DBLTYPE**, SHMEM>& areav) const
{
  DBLTYPE coord_mid_face[2][3];

  const double one = 1.0;
  const double zero = 0.0;
  const double half = 0.5;

  const double one3rd = 1.0 / 3.0;

  const int kx = 0;
  const int ky = 1;

  // Cartesian
  const double a1 = one;
  const double a2 = zero;
  const double a3 = zero;

  // calculate element mid-point coordinates
  const DBLTYPE x1 = (coords(0, kx) + coords(1, kx) + coords(2, kx)) * one3rd;
  const DBLTYPE y1 = (coords(0, ky) + coords(1, ky) + coords(2, ky)) * one3rd;

  // calculate element mid-face coordinates
  coord_mid_face[kx][0] = (coords(0, kx) + coords(1, kx)) * half;
  coord_mid_face[kx][1] = (coords(1, kx) + coords(2, kx)) * half;
  coord_mid_face[kx][2] = (coords(2, kx) + coords(0, kx)) * half;
  coord_mid_face[ky][0] = (coords(0, ky) + coords(1, ky)) * half;
  coord_mid_face[ky][1] = (coords(1, ky) + coords(2, ky)) * half;
  coord_mid_face[ky][2] = (coords(2, ky) + coords(0, ky)) * half;

  DBLTYPE x2, y2, rr;
  // Control surface 1
  x2 = coord_mid_face[kx][0];
  y2 = coord_mid_face[ky][0];

  rr = a1 + a2 * (x1 + x2) + a3 * (y1 + y2);

  areav(0, kx) = -(y2 - y1) * rr;
  areav(0, ky) = (x2 - x1) * rr;

  // Control surface 2
  x2 = coord_mid_face[kx][1];
  y2 = coord_mid_face[ky][1];

  rr = a1 + a2 * (x1 + x2) + a3 * (y1 + y2);

  areav(1, kx) = -(y2 - y1) * rr;
  areav(1, ky) = (x2 - x1) * rr;

  // Control surface 3
  x2 = coord_mid_face[kx][2];
  y2 = coord_mid_face[ky][2];

  rr = a1 + a2 * (x1 + x2) + a3 * (y1 + y2);

  areav(2, kx) = (y2 - y1) * rr;
  areav(2, ky) = -(x2 - x1) * rr;
}
KOKKOS_FUNCTION void
Tri32DSCS::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType**, DeviceShmem>& areav)
{
  determinant_scs(coords, areav);
}
void
Tri32DSCS::determinant(
  const SharedMemView<double**>& coords, SharedMemView<double**>& areav)
{
  determinant_scs(coords, areav);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Tri32DSCS::grad_op(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tri_derivative(deriv);
  DoubleType det[numIntPoints_];
  SharedMemView<DoubleType*, DeviceShmem> det_j(det, numIntPoints_);
  tri_gradient_operator(coords, deriv, gradop, det_j);
}

void
Tri32DSCS::grad_op(
  const SharedMemView<double**>& coords,
  SharedMemView<double***>& gradop,
  SharedMemView<double***>& deriv)
{
  tri_derivative(deriv);
  double det[numIntPoints_];
  SharedMemView<double*> det_j(det, numIntPoints_);
  tri_gradient_operator(coords, deriv, gradop, det_j);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Tri32DSCS::shifted_grad_op(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tri_derivative(deriv);
  DoubleType det[numIntPoints_];
  SharedMemView<DoubleType*, DeviceShmem> det_j(det, numIntPoints_);
  tri_gradient_operator(coords, deriv, gradop, det_j);
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Tri32DSCS::face_grad_op(
  int /*face_ordinal*/,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  tri_derivative(deriv);
  generic_grad_op<AlgTraitsEdge2DTri32D>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Tri32DSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  // same as regular face_grad_op
  face_grad_op(face_ordinal, coords, gradop, deriv);
}
//--------------------------------------------------------------------------
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Tri32DSCS::gij(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gupper,
  SharedMemView<DoubleType***, DeviceShmem>& glower,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{

  const int npe = nodesPerElement_;
  const int nint = numIntPoints_;

  DoubleType dx_ds[2][2], ds_dx[2][2];

  for (int ki = 0; ki < nint; ++ki) {
    // zero out
    dx_ds[0][0] = 0.0;
    dx_ds[0][1] = 0.0;
    dx_ds[1][0] = 0.0;
    dx_ds[1][1] = 0.0;

    // calculate the jacobian at the integration station -
    for (int kn = 0; kn < npe; ++kn) {
      dx_ds[0][0] += deriv(ki, kn, 0) * coords(kn, 0);
      dx_ds[0][1] += deriv(ki, kn, 1) * coords(kn, 0);
      dx_ds[1][0] += deriv(ki, kn, 0) * coords(kn, 1);
      dx_ds[1][1] += deriv(ki, kn, 1) * coords(kn, 1);
    }

    // calculate the determinate of the jacobian at the integration station -
    const DoubleType det_j =
      dx_ds[0][0] * dx_ds[1][1] - dx_ds[1][0] * dx_ds[0][1];

    // clip
    const DoubleType denom = stk::math::if_then_else(
      det_j < 1.e6 * MEconstants::realmin, 1.0, 1.0 / det_j);

    // compute the inverse jacobian
    ds_dx[0][0] = dx_ds[1][1] * denom;
    ds_dx[0][1] = -dx_ds[0][1] * denom;
    ds_dx[1][0] = -dx_ds[1][0] * denom;
    ds_dx[1][1] = dx_ds[0][0] * denom;

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        gupper(ki, j, i) =
          dx_ds[i][0] * dx_ds[j][0] + dx_ds[i][1] * dx_ds[j][1];
        glower(ki, j, i) =
          ds_dx[0][i] * ds_dx[0][j] + ds_dx[1][i] * ds_dx[1][j];
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- Metric Tensor Mij------------------------------------------------
//--------------------------------------------------------------------------
// This function computes the metric tensor Mij = (J J^T)^(1/2) where J is
// the Jacobian.  This is needed for the UT-A Hybrid LES model.  For
// reference please consult the Nalu theory manual description of the UT-A
// Hybrid LES model or S. Haering's PhD thesis: Anisotropic hybrid turbulence
// modeling with specific application to the simulation of pulse-actuated
// dynamic stall control.
//--------------------------------------------------------------------------
void
Tri32DSCS::Mij(const double* coords, double* metric, double* deriv)
{
  generic_Mij_2d<AlgTraitsTri3_2D>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Tri32DSCS::Mij(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& metric,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  generic_Mij_2d<AlgTraitsTri3_2D>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
Tri32DSCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_;
}

//--------------------------------------------------------------------------
//-------- scsIpEdgeOrd ----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
Tri32DSCS::scsIpEdgeOrd()
{
  return scsIpEdgeOrd_;
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Tri32DSCS::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tri_shape_fcn(intgLoc_, shpfc);
}
KOKKOS_FUNCTION void
Tri32DSCS::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
Tri32DSCS::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Tri32DSCS::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  tri_shape_fcn(intgLocShift_, shpfc);
}
KOKKOS_FUNCTION void
Tri32DSCS::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
Tri32DSCS::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- tri_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Tri32DSCS::tri_shape_fcn(
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
Tri32DSCS::tri_shape_fcn(
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
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
int
Tri32DSCS::opposingNodes(const int ordinal, const int node)
{
  return oppNode_[ordinal][node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
int
Tri32DSCS::opposingFace(const int ordinal, const int node)
{
  return oppFace_[ordinal][node];
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
Tri32DSCS::isInElement(
  const double* elemNodalCoord, const double* pointCoord, double* isoParCoord)
{
  // Translate element so that (x,y) coordinates of the
  // first node

  double x[2] = {
    elemNodalCoord[1] - elemNodalCoord[0],
    elemNodalCoord[2] - elemNodalCoord[0]};
  double y[2] = {
    elemNodalCoord[4] - elemNodalCoord[3],
    elemNodalCoord[5] - elemNodalCoord[3]};

  // Translate position of point in same manner

  double xp = pointCoord[0] - elemNodalCoord[0];
  double yp = pointCoord[1] - elemNodalCoord[3];

  // Set new nodal coordinates with Node 1 at origin and with new
  // x and y axes lying in the plane of the element
  double len12 = std::sqrt(x[0] * x[0] + y[0] * y[0]);
  double len13 = std::sqrt(x[1] * x[1] + y[1] * y[1]);

  double xnew[2];
  double ynew[2];

  // Use cross-product of find enclosed angle

  const double cross = x[0] * y[1] - x[1] * y[0];

  double Area2 = std::sqrt(cross * cross);

  // find sin of angle
  double sin_theta = Area2 / (len12 * len13);

  // find cosine of angle
  double cos_theta = (x[0] * x[1] + y[0] * y[1]) / (len12 * len13);

  // nodal coordinates of nodes 2 and 3 in new system
  // (coordinates of node 1 are identically 0.0)
  double x_nod_new[2] = {len12, len13 * cos_theta};
  double y_nod_new[2] = {0.0, len13 * sin_theta};

  // find direction cosines transform position of
  // point to be checked into new coordinate system
  // direction cosines of new x axis along side 12

  xnew[0] = x[0] / len12;
  xnew[1] = y[0] / len12;

  // direction cosines of new y-axis
  ynew[0] = -xnew[1];
  ynew[1] = xnew[0];

  // compute transformed coordinates of point
  // (coordinates in xnew,ynew)
  double xpnew = xnew[0] * xp + xnew[1] * yp;
  double ypnew = ynew[0] * xp + ynew[1] * yp;

  // Find parametric coordinates of point and check that
  // it lies in the element
  isoParCoord[0] =
    1. - xpnew / x_nod_new[0] + ypnew * (x_nod_new[1] - x_nod_new[0]) / Area2;
  isoParCoord[1] = (xpnew * y_nod_new[1] - ypnew * x_nod_new[1]) / Area2;

  std::array<double, 2> w;
  w[0] = isoParCoord[0];
  w[1] = isoParCoord[1];

  isoParCoord[0] = w[1];
  isoParCoord[1] = 1.0 - w[0] - w[1];

  const double dist = tri_parametric_distance(w);

  return dist;
}

//--------------------------------------------------------------------------
//-------- tri_parametric_distance -----------------------------------------
//--------------------------------------------------------------------------
double
Tri32DSCS::tri_parametric_distance(const std::array<double, 2>& x)
{
  const double X = x[0] - 1. / 3.;
  const double Y = x[1] - 1. / 3.;
  const double dist0 = -3 * X;
  const double dist1 = -3 * Y;
  const double dist2 = 3 * (X + Y);
  const double dist = std::max(std::max(dist0, dist1), dist2);
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
Tri32DSCS::interpolatePoint(
  const int& nComp,
  const double* isoParCoord,
  const double* field,
  double* result)
{
  const double s = isoParCoord[0];
  const double t = isoParCoord[1];
  const double oneMinusST = 1.0 - s - t;
  for (int i = 0; i < nComp; i++) {
    const int b = 3 * i;
    result[i] = oneMinusST * field[b + 0] + s * field[b + 1] + t * field[b + 2];
  }
}

//--------------------------------------------------------------------------
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void
Tri32DSCS::general_face_grad_op(
  const int /*face_ordinal*/,
  const double* /*isoParCoord*/,
  const double* coords,
  double* gradop,
  double* det_j,
  double*)
{
  int lerr = 0;

  const int nface = 1;
  const int npe = nodesPerElement_;
  double dpsi[6];

  SharedMemView<double***, HostShmem> deriv(
    dpsi, nface, nodesPerElement_, nDim_);
  tri_derivative(deriv);

  const SharedMemView<const double**, HostShmem> cordel(coords, npe, 3);
  SharedMemView<double***, HostShmem> grad(gradop, nface, npe, 3);
  SharedMemView<double*, HostShmem> det(det_j, nface);
  lerr = tri_gradient_operator(cordel, deriv, grad, det);

  if (lerr)
    NaluEnv::self().naluOutput()
      << "sorry, issue with face_grad_op.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void
Tri32DSCS::sidePcoords_to_elemPcoords(
  const int& side_ordinal,
  const int& npoints,
  const double* side_pcoords,
  double* elem_pcoords)
{
  switch (side_ordinal) {
  case 0:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 2 + 0] = 0.5 * (1.0 + side_pcoords[i]);
      elem_pcoords[i * 2 + 1] = 0.0;
    }
    break;
  case 1:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 2 + 0] = 1. - 0.5 * (side_pcoords[i] + 1.);
      elem_pcoords[i * 2 + 1] = 0.5 * (side_pcoords[i] + 1.);
    }
    break;
  case 2:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 2 + 0] = 0.0;
      elem_pcoords[i * 2 + 1] = 1. - 0.5 * (side_pcoords[i] + 1.);
    }
    break;
  default:
    throw std::runtime_error("Tri32DSCS::sideMap invalid ordinal");
  }
}

} // namespace nalu
} // namespace sierra
