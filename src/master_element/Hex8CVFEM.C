// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/Hex8CVFEM.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/TensorOps.h>
#include <master_element/Hex8GeometryFunctions.h>
#include <master_element/CompileTimeElements.h>

#include <NaluEnv.h>

#include <cmath>
#include <iostream>
#include <array>
#include <limits>

namespace sierra {
namespace nalu {

template <typename DBLTYPE, typename SHMEM>
KOKKOS_FUNCTION void
hex8_derivative(
  const SharedMemView<const double**, SHMEM>& par_coord,
  SharedMemView<DBLTYPE***, SHMEM>& deriv)
{
  // formal parameters - input:
  //     par_coord     real  parametric coordinates of the points to be
  //                         evaluated (typically, the gauss pts)
  //
  // formal parameters - output:
  //     deriv         real  shape function derivatives evaluated at
  //                         evaluation points.
  //
  //**********************************************************************
#if defined(KOKKOS_ENABLE_GPU)

  if (8 != deriv.extent(1))
    ThrowErrorMsgDevice("hex8_derivative: Error in derivative array index 1");
  if (3 != deriv.extent(2))
    ThrowErrorMsgDevice("hex8_derivative: Error in derivative array index 2");
  if (3 != par_coord.extent(1))
    ThrowErrorMsgDevice("hex8_derivative: Error in par_coord array index 1");
  if (deriv.extent(0) != par_coord.extent(0))
    ThrowErrorMsgDevice(
      "hex8_derivative: Error in deriv or par_coord array index 0");
#else

  STK_ThrowRequireMsg(
    8 == deriv.extent(1), "hex8_derivative: Error in derivative array");
  STK_ThrowRequireMsg(
    3 == deriv.extent(2), "hex8_derivative: Error in derivative array");
  STK_ThrowRequireMsg(
    3 == par_coord.extent(1), "hex8_derivative: Error in derivative array");
  STK_ThrowRequireMsg(
    deriv.extent(0) == par_coord.extent(0),
    "hex8_derivative: Error in derivative array");

#endif

  const int npts = deriv.extent(0);

  const double half = 1.0 / 2.0;
  const double one4th = 1.0 / 4.0;

  for (int j = 0; j < npts; ++j) {
    const double s1 = par_coord(j, 0);
    const double s2 = par_coord(j, 1);
    const double s3 = par_coord(j, 2);

    const double s1s2 = s1 * s2;
    const double s2s3 = s2 * s3;
    const double s1s3 = s1 * s3;

    deriv(j, 0, 0) = half * (s3 + s2) - s2s3 - one4th;
    deriv(j, 1, 0) = half * (-s3 - s2) + s2s3 + one4th;
    deriv(j, 2, 0) = half * (-s3 + s2) - s2s3 + one4th;
    deriv(j, 3, 0) = half * (+s3 - s2) + s2s3 - one4th;
    deriv(j, 4, 0) = half * (-s3 + s2) + s2s3 - one4th;
    deriv(j, 5, 0) = half * (+s3 - s2) - s2s3 + one4th;
    deriv(j, 6, 0) = half * (+s3 + s2) + s2s3 + one4th;
    deriv(j, 7, 0) = half * (-s3 - s2) - s2s3 - one4th;

    deriv(j, 0, 1) = half * (s3 + s1) - s1s3 - one4th;
    deriv(j, 1, 1) = half * (s3 - s1) + s1s3 - one4th;
    deriv(j, 2, 1) = half * (-s3 + s1) - s1s3 + one4th;
    deriv(j, 3, 1) = half * (-s3 - s1) + s1s3 + one4th;
    deriv(j, 4, 1) = half * (-s3 + s1) + s1s3 - one4th;
    deriv(j, 5, 1) = half * (-s3 - s1) - s1s3 - one4th;
    deriv(j, 6, 1) = half * (s3 + s1) + s1s3 + one4th;
    deriv(j, 7, 1) = half * (s3 - s1) - s1s3 + one4th;

    deriv(j, 0, 2) = half * (s2 + s1) - s1s2 - one4th;
    deriv(j, 1, 2) = half * (s2 - s1) + s1s2 - one4th;
    deriv(j, 2, 2) = half * (-s2 - s1) - s1s2 - one4th;
    deriv(j, 3, 2) = half * (-s2 + s1) + s1s2 - one4th;
    deriv(j, 4, 2) = half * (-s2 - s1) + s1s2 + one4th;
    deriv(j, 5, 2) = half * (-s2 + s1) - s1s2 + one4th;
    deriv(j, 6, 2) = half * (s2 + s1) + s1s2 + one4th;
    deriv(j, 7, 2) = half * (s2 - s1) - s1s2 + one4th;
  }
}

int
hex_gradient_operator(
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
    3 == deriv.extent(2), "hex_gradient_operator: Error in derivative array");

  const unsigned nelem = cordel.extent(0);
  STK_ThrowRequireMsg(
    npe == cordel.extent(1),
    "hex_gradient_operator: Error in coorindate array");
  STK_ThrowRequireMsg(
    3 == cordel.extent(2), "hex_gradient_operator: Error in coorindate array");

  STK_ThrowRequireMsg(
    nint == gradop.extent(0), "hex_gradient_operator: Error in gradient array");
  STK_ThrowRequireMsg(
    nelem == gradop.extent(1),
    "hex_gradient_operator: Error in gradient array");
  STK_ThrowRequireMsg(
    npe == gradop.extent(2), "hex_gradient_operator: Error in gradient array");
  STK_ThrowRequireMsg(
    3 == gradop.extent(3), "hex_gradient_operator: Error in gradient array");

  STK_ThrowRequireMsg(
    nint == det_j.extent(0),
    "hex_gradient_operator: Error in determinent array");
  STK_ThrowRequireMsg(
    nelem == det_j.extent(1),
    "hex_gradient_operator: Error in determinent array");

  STK_ThrowRequireMsg(
    nelem == err.extent(0), "hex_gradient_operator: Error in error array");

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
KOKKOS_INLINE_FUNCTION void
hex8_shape_fcn(
  const int npts,
  const double* isoParCoord,
  SharedMemView<SCALAR**, SHMEM>& shape_fcn)
{
  const SCALAR half = 0.50;
  const SCALAR one4th = 0.25;
  const SCALAR one8th = 0.125;
  for (int j = 0; j < npts; ++j) {

    const SCALAR s1 = isoParCoord[j * 3];
    const SCALAR s2 = isoParCoord[j * 3 + 1];
    const SCALAR s3 = isoParCoord[j * 3 + 2];

    shape_fcn(j, 0) = one8th + one4th * (-s1 - s2 - s3) +
                      half * (s2 * s3 + s3 * s1 + s1 * s2) - s1 * s2 * s3;
    shape_fcn(j, 1) = one8th + one4th * (s1 - s2 - s3) +
                      half * (s2 * s3 - s3 * s1 - s1 * s2) + s1 * s2 * s3;
    shape_fcn(j, 2) = one8th + one4th * (s1 + s2 - s3) +
                      half * (-s2 * s3 - s3 * s1 + s1 * s2) - s1 * s2 * s3;
    shape_fcn(j, 3) = one8th + one4th * (-s1 + s2 - s3) +
                      half * (-s2 * s3 + s3 * s1 - s1 * s2) + s1 * s2 * s3;
    shape_fcn(j, 4) = one8th + one4th * (-s1 - s2 + s3) +
                      half * (-s2 * s3 - s3 * s1 + s1 * s2) + s1 * s2 * s3;
    shape_fcn(j, 5) = one8th + one4th * (s1 - s2 + s3) +
                      half * (-s2 * s3 + s3 * s1 - s1 * s2) - s1 * s2 * s3;
    shape_fcn(j, 6) = one8th + one4th * (s1 + s2 + s3) +
                      half * (s2 * s3 + s3 * s1 + s1 * s2) + s1 * s2 * s3;
    shape_fcn(j, 7) = one8th + one4th * (-s1 + s2 + s3) +
                      half * (s2 * s3 - s3 * s1 - s1 * s2) - s1 * s2 * s3;
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
HexSCV::HexSCV() : MasterElement()
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
HexSCV::ipNodeMap(int /*ordinal*/) const
{
  // define scv->node mappings
  return ipNodeMap_;
}

template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
HexSCV::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

KOKKOS_FUNCTION void
HexSCV::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
HexSCV::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
HexSCV::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}
KOKKOS_FUNCTION void
HexSCV::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
HexSCV::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
HexSCV::determinant_scv(
  const SharedMemView<DBLTYPE**, SHMEM>& coords,
  SharedMemView<DBLTYPE*, SHMEM>& volume) const
{
  constexpr int subDivisionTable[8][8] = {
    {0, 8, 12, 11, 19, 20, 26, 25},  {8, 1, 9, 12, 20, 18, 24, 26},
    {12, 9, 2, 10, 26, 24, 22, 23},  {11, 12, 10, 3, 25, 26, 23, 21},
    {19, 20, 26, 25, 4, 13, 17, 16}, {20, 18, 24, 26, 13, 5, 14, 17},
    {26, 24, 22, 23, 17, 14, 6, 15}, {25, 26, 23, 21, 16, 17, 15, 7}};

  DBLTYPE coordv[27][3];
  subdivide_hex_8(coords, coordv);

  constexpr int numSCV = 8;
  for (int ip = 0; ip < numSCV; ++ip) {
    DBLTYPE scvHex[8][3];
    for (int n = 0; n < 8; ++n) {
      const int subIndex = subDivisionTable[ip][n];
      for (int d = 0; d < 3; ++d) {
        scvHex[n][d] = coordv[subIndex][d];
      }
    }
    volume(ip) = hex_volume_grandy(scvHex);
  }
}

KOKKOS_FUNCTION void
HexSCV::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType*, DeviceShmem>& volume)
{
  determinant_scv(coords, volume);
}

void
HexSCV::determinant(
  const SharedMemView<double**>& coords, SharedMemView<double*>& volume)
{
  determinant_scv(coords, volume);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
HexSCV::grad_op(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>&)
{
  impl::grad_op<AlgTraitsHex8, QuadRank::SCV, QuadType::MID>(coords, gradop);
}

void
HexSCV::grad_op(
  const SharedMemView<double**>& coords,
  SharedMemView<double***>& gradop,
  SharedMemView<double***>&)
{
  impl::grad_op<AlgTraitsHex8, QuadRank::SCV, QuadType::MID>(coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
HexSCV::shifted_grad_op(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>&)
{
  impl::grad_op<AlgTraitsHex8, QuadRank::SCV, QuadType::SHIFTED>(
    coords, gradop);
}

//--------------------------------------------------------------------------
//-------- Mij -------------------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCV::Mij(const double* coords, double* metric, double* deriv)
{
  generic_Mij_3d<AlgTraitsHex8>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
KOKKOS_FUNCTION
void
HexSCV::Mij(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& metric,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  const SharedMemView<const double**, DeviceShmem> par_coord(
    intgLoc_, numIntPoints_, nDim_);
  hex8_derivative(par_coord, deriv);
  generic_Mij_3d<AlgTraitsHex8>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
HexSCS::HexSCS() : MasterElement(HexSCS::scaleToStandardIsoFac_)
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
HexSCS::ipNodeMap(int ordinal) const
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal * 4];
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
HexSCS::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}
KOKKOS_FUNCTION void
HexSCS::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
HexSCS::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
HexSCS::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}
KOKKOS_FUNCTION void
HexSCS::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
HexSCS::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
HexSCS::grad_op(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>&)
{
  impl::grad_op<AlgTraitsHex8, QuadRank::SCS, QuadType::MID>(coords, gradop);
}

void
HexSCS::grad_op(
  const SharedMemView<double**>& coords,
  SharedMemView<double***>& gradop,
  SharedMemView<double***>&)
{
  impl::grad_op<AlgTraitsHex8, QuadRank::SCS, QuadType::MID>(coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
HexSCS::shifted_grad_op(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>&)
{
  impl::grad_op<AlgTraitsHex8, QuadRank::SCS, QuadType::SHIFTED>(
    coords, gradop);
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename DBLTYPE, typename SHMEM>
KOKKOS_INLINE_FUNCTION void
HexSCS::determinant_scs(
  const SharedMemView<DBLTYPE**, SHMEM>& coords,
  SharedMemView<DBLTYPE**, SHMEM>& areav) const
{
  constexpr int hex_edge_facet_table[12][4] = {
    {20, 8, 12, 26},  {24, 9, 12, 26},  {10, 12, 26, 23}, {11, 25, 26, 12},
    {13, 20, 26, 17}, {17, 14, 24, 26}, {17, 15, 23, 26}, {16, 17, 26, 25},
    {19, 20, 26, 25}, {20, 18, 24, 26}, {22, 23, 26, 24}, {21, 25, 26, 23}};

  DBLTYPE coordv[27][3];
  subdivide_hex_8(coords, coordv);

  constexpr int npf = 4;
  constexpr int nscs = 12;
  for (int ics = 0; ics < nscs; ++ics) {
    DBLTYPE scscoords[4][3];
    for (int inode = 0; inode < npf; ++inode) {
      const int itrianglenode = hex_edge_facet_table[ics][inode];
      for (int d = 0; d < 3; ++d) {
        scscoords[inode][d] = coordv[itrianglenode][d];
      }
    }
    quad_area_by_triangulation(ics, scscoords, areav);
  }
}

KOKKOS_FUNCTION void
HexSCS::determinant(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType**, DeviceShmem>& areav)
{
  determinant_scs(coords, areav);
}

void
HexSCS::determinant(
  const SharedMemView<double**>& coords, SharedMemView<double**>& areav)
{
  determinant_scs(coords, areav);
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
KOKKOS_FUNCTION
//--------------------------------------------------------------------------
const int*
HexSCS::side_node_ordinals(int ordinal) const
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return sideNodeOrdinals_[ordinal];
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
template <bool shifted>
KOKKOS_FUNCTION void
HexSCS::face_grad_op_t(
  const int face_ordinal,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  using traits = AlgTraitsQuad4Hex8;
  const double* exp_face =
    shifted ? &intgExpFaceShift_[0][0][0] : &intgExpFace_[0][0][0];
  const int offset = traits::numFaceIp_ * traits::nDim_ * face_ordinal;

  const SharedMemView<const double**, DeviceShmem> par_coord(
    &exp_face[offset], traits::numFaceIp_, nDim_);
  hex8_derivative(par_coord, deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

KOKKOS_FUNCTION
void
HexSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  face_grad_op_t<false>(face_ordinal, coords, gradop, deriv);
}

//--------------------------------------------------------------------------
//-------- shifted_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
HexSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv)
{
  face_grad_op_t<true>(face_ordinal, coords, gradop, deriv);
}
//--------------------------------------------------------------------------
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
HexSCS::gij(
  const SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gupper,
  SharedMemView<DoubleType***, DeviceShmem>& glower,
  SharedMemView<DoubleType***, DeviceShmem>& /*deriv*/)
{
  constexpr auto deriv = elem_data_t<AlgTraitsHex8, QuadType::MID>::scs_deriv;
  generic_gij_3d<AlgTraitsHex8>(deriv, coords, gupper, glower);
}

//--------------------------------------------------------------------------
//-------- Mij -------------------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::Mij(const double* coords, double* metric, double* /*deriv*/)
{
  constexpr auto deriv = elem_data_t<AlgTraitsHex8, QuadType::MID>::scs_deriv;
  generic_Mij_3d<AlgTraitsHex8>(numIntPoints_, deriv.data(), coords, metric);
}
//-------------------------------------------------------------------------
KOKKOS_FUNCTION
void
HexSCS::Mij(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& metric,
  SharedMemView<DoubleType***, DeviceShmem>& /*deriv*/)
{
  constexpr auto deriv = elem_data_t<AlgTraitsHex8, QuadType::MID>::scs_deriv;
  generic_Mij_3d<AlgTraitsHex8>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
HexSCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_;
}

//--------------------------------------------------------------------------
//-------- scsIpEdgeOrd ----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
const int*
HexSCS::scsIpEdgeOrd()
{
  return &scsIpEdgeOrd_[0];
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
int
HexSCS::opposingNodes(const int ordinal, const int node)
{
  return oppNode_[ordinal * 4 + node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
int
HexSCS::opposingFace(const int ordinal, const int node)
{
  return oppFace_[ordinal * 4 + node];
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
HexSCS::isInElement(
  const double* elem_nodal_coor, // (8,3)
  const double* point_coor,      // (3)
  double* par_coor)
{
  const int maxNonlinearIter = 20;
  const double isInElemConverged = 1.0e-16;
  // Translate element so that (x,y,z) coordinates of the first node are (0,0,0)

  double x[] = {
    0.,
    0.125 * (elem_nodal_coor[1] - elem_nodal_coor[0]),
    0.125 * (elem_nodal_coor[2] - elem_nodal_coor[0]),
    0.125 * (elem_nodal_coor[3] - elem_nodal_coor[0]),
    0.125 * (elem_nodal_coor[4] - elem_nodal_coor[0]),
    0.125 * (elem_nodal_coor[5] - elem_nodal_coor[0]),
    0.125 * (elem_nodal_coor[6] - elem_nodal_coor[0]),
    0.125 * (elem_nodal_coor[7] - elem_nodal_coor[0])};
  double y[] = {
    0.,
    0.125 * (elem_nodal_coor[9] - elem_nodal_coor[8]),
    0.125 * (elem_nodal_coor[10] - elem_nodal_coor[8]),
    0.125 * (elem_nodal_coor[11] - elem_nodal_coor[8]),
    0.125 * (elem_nodal_coor[12] - elem_nodal_coor[8]),
    0.125 * (elem_nodal_coor[13] - elem_nodal_coor[8]),
    0.125 * (elem_nodal_coor[14] - elem_nodal_coor[8]),
    0.125 * (elem_nodal_coor[15] - elem_nodal_coor[8])};
  double z[] = {
    0.,
    0.125 * (elem_nodal_coor[17] - elem_nodal_coor[16]),
    0.125 * (elem_nodal_coor[18] - elem_nodal_coor[16]),
    0.125 * (elem_nodal_coor[19] - elem_nodal_coor[16]),
    0.125 * (elem_nodal_coor[20] - elem_nodal_coor[16]),
    0.125 * (elem_nodal_coor[21] - elem_nodal_coor[16]),
    0.125 * (elem_nodal_coor[22] - elem_nodal_coor[16]),
    0.125 * (elem_nodal_coor[23] - elem_nodal_coor[16])};

  // (xp,yp,zp) is the point at which we're searching for (xi,eta,zeta)
  // (must translate this also)

  double xp = point_coor[0] - elem_nodal_coor[0];
  double yp = point_coor[1] - elem_nodal_coor[8];
  double zp = point_coor[2] - elem_nodal_coor[16];

  // Newton-Raphson iteration for (xi,eta,zeta)
  double j[9];
  double f[3];
  double shapefct[8];
  double xinew = 0.5; // initial guess
  double etanew = 0.5;
  double zetanew = 0.5;
  double xicur = 0.5;
  double etacur = 0.5;
  double zetacur = 0.5;
  double xidiff[] = {1.0, 1.0, 1.0};
  int i = 0;

  do {
    j[0] = -(1.0 - etacur) * (1.0 - zetacur) * x[1] -
           (1.0 + etacur) * (1.0 - zetacur) * x[2] +
           (1.0 + etacur) * (1.0 - zetacur) * x[3] +
           (1.0 - etacur) * (1.0 + zetacur) * x[4] -
           (1.0 - etacur) * (1.0 + zetacur) * x[5] -
           (1.0 + etacur) * (1.0 + zetacur) * x[6] +
           (1.0 + etacur) * (1.0 + zetacur) * x[7];

    j[1] = (1.0 + xicur) * (1.0 - zetacur) * x[1] -
           (1.0 + xicur) * (1.0 - zetacur) * x[2] -
           (1.0 - xicur) * (1.0 - zetacur) * x[3] +
           (1.0 - xicur) * (1.0 + zetacur) * x[4] +
           (1.0 + xicur) * (1.0 + zetacur) * x[5] -
           (1.0 + xicur) * (1.0 + zetacur) * x[6] -
           (1.0 - xicur) * (1.0 + zetacur) * x[7];

    j[2] = (1.0 - etacur) * (1.0 + xicur) * x[1] +
           (1.0 + etacur) * (1.0 + xicur) * x[2] +
           (1.0 + etacur) * (1.0 - xicur) * x[3] -
           (1.0 - etacur) * (1.0 - xicur) * x[4] -
           (1.0 - etacur) * (1.0 + xicur) * x[5] -
           (1.0 + etacur) * (1.0 + xicur) * x[6] -
           (1.0 + etacur) * (1.0 - xicur) * x[7];

    j[3] = -(1.0 - etacur) * (1.0 - zetacur) * y[1] -
           (1.0 + etacur) * (1.0 - zetacur) * y[2] +
           (1.0 + etacur) * (1.0 - zetacur) * y[3] +
           (1.0 - etacur) * (1.0 + zetacur) * y[4] -
           (1.0 - etacur) * (1.0 + zetacur) * y[5] -
           (1.0 + etacur) * (1.0 + zetacur) * y[6] +
           (1.0 + etacur) * (1.0 + zetacur) * y[7];

    j[4] = (1.0 + xicur) * (1.0 - zetacur) * y[1] -
           (1.0 + xicur) * (1.0 - zetacur) * y[2] -
           (1.0 - xicur) * (1.0 - zetacur) * y[3] +
           (1.0 - xicur) * (1.0 + zetacur) * y[4] +
           (1.0 + xicur) * (1.0 + zetacur) * y[5] -
           (1.0 + xicur) * (1.0 + zetacur) * y[6] -
           (1.0 - xicur) * (1.0 + zetacur) * y[7];

    j[5] = (1.0 - etacur) * (1.0 + xicur) * y[1] +
           (1.0 + etacur) * (1.0 + xicur) * y[2] +
           (1.0 + etacur) * (1.0 - xicur) * y[3] -
           (1.0 - etacur) * (1.0 - xicur) * y[4] -
           (1.0 - etacur) * (1.0 + xicur) * y[5] -
           (1.0 + etacur) * (1.0 + xicur) * y[6] -
           (1.0 + etacur) * (1.0 - xicur) * y[7];

    j[6] = -(1.0 - etacur) * (1.0 - zetacur) * z[1] -
           (1.0 + etacur) * (1.0 - zetacur) * z[2] +
           (1.0 + etacur) * (1.0 - zetacur) * z[3] +
           (1.0 - etacur) * (1.0 + zetacur) * z[4] -
           (1.0 - etacur) * (1.0 + zetacur) * z[5] -
           (1.0 + etacur) * (1.0 + zetacur) * z[6] +
           (1.0 + etacur) * (1.0 + zetacur) * z[7];

    j[7] = (1.0 + xicur) * (1.0 - zetacur) * z[1] -
           (1.0 + xicur) * (1.0 - zetacur) * z[2] -
           (1.0 - xicur) * (1.0 - zetacur) * z[3] +
           (1.0 - xicur) * (1.0 + zetacur) * z[4] +
           (1.0 + xicur) * (1.0 + zetacur) * z[5] -
           (1.0 + xicur) * (1.0 + zetacur) * z[6] -
           (1.0 - xicur) * (1.0 + zetacur) * z[7];

    j[8] = (1.0 - etacur) * (1.0 + xicur) * z[1] +
           (1.0 + etacur) * (1.0 + xicur) * z[2] +
           (1.0 + etacur) * (1.0 - xicur) * z[3] -
           (1.0 - etacur) * (1.0 - xicur) * z[4] -
           (1.0 - etacur) * (1.0 + xicur) * z[5] -
           (1.0 + etacur) * (1.0 + xicur) * z[6] -
           (1.0 + etacur) * (1.0 - xicur) * z[7];

    double jdet = -(j[2] * j[4] * j[6]) + j[1] * j[5] * j[6] +
                  j[2] * j[3] * j[7] - j[0] * j[5] * j[7] - j[1] * j[3] * j[8] +
                  j[0] * j[4] * j[8];

    if (!jdet) {
      i = maxNonlinearIter;
      break;
    }
    shapefct[0] = (1.0 - etacur) * (1.0 - xicur) * (1.0 - zetacur);

    shapefct[1] = (1.0 - etacur) * (1.0 + xicur) * (1.0 - zetacur);

    shapefct[2] = (1.0 + etacur) * (1.0 + xicur) * (1.0 - zetacur);

    shapefct[3] = (1.0 + etacur) * (1.0 - xicur) * (1.0 - zetacur);

    shapefct[4] = (1.0 - etacur) * (1.0 - xicur) * (1.0 + zetacur);

    shapefct[5] = (1.0 - etacur) * (1.0 + xicur) * (1.0 + zetacur);

    shapefct[6] = (1.0 + etacur) * (1.0 + xicur) * (1.0 + zetacur);

    shapefct[7] = (1.0 + etacur) * (1.0 - xicur) * (1.0 + zetacur);

    f[0] = xp - shapefct[1] * x[1] - shapefct[2] * x[2] - shapefct[3] * x[3] -
           shapefct[4] * x[4] - shapefct[5] * x[5] - shapefct[6] * x[6] -
           shapefct[7] * x[7];

    f[1] = yp - shapefct[1] * y[1] - shapefct[2] * y[2] - shapefct[3] * y[3] -
           shapefct[4] * y[4] - shapefct[5] * y[5] - shapefct[6] * y[6] -
           shapefct[7] * y[7];

    f[2] = zp - shapefct[1] * z[1] - shapefct[2] * z[2] - shapefct[3] * z[3] -
           shapefct[4] * z[4] - shapefct[5] * z[5] - shapefct[6] * z[6] -
           shapefct[7] * z[7];

    xinew =
      (jdet * xicur + f[2] * (j[2] * j[4] - j[1] * j[5]) - f[1] * j[2] * j[7] +
       f[0] * j[5] * j[7] + f[1] * j[1] * j[8] - f[0] * j[4] * j[8]) /
      jdet;

    etanew = (etacur * jdet + f[2] * (-(j[2] * j[3]) + j[0] * j[5]) +
              f[1] * j[2] * j[6] - f[0] * j[5] * j[6] - f[1] * j[0] * j[8] +
              f[0] * j[3] * j[8]) /
             jdet;

    zetanew = (jdet * zetacur + f[2] * (j[1] * j[3] - j[0] * j[4]) -
               f[1] * j[1] * j[6] + f[0] * j[4] * j[6] + f[1] * j[0] * j[7] -
               f[0] * j[3] * j[7]) /
              jdet;

    xidiff[0] = xinew - xicur;
    xidiff[1] = etanew - etacur;
    xidiff[2] = zetanew - zetacur;
    xicur = xinew;
    etacur = etanew;
    zetacur = zetanew;

  } while (!within_tolerance(vector_norm_sq(xidiff, 3), isInElemConverged) &&
           ++i < maxNonlinearIter);

  par_coor[0] = par_coor[1] = par_coor[2] = std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if (i < maxNonlinearIter) {
    par_coor[0] = xinew;
    par_coor[1] = etanew;
    par_coor[2] = zetanew;

    std::array<double, 3> xtmp;
    xtmp[0] = par_coor[0];
    xtmp[1] = par_coor[1];
    xtmp[2] = par_coor[2];
    dist = parametric_distance(xtmp);
  }
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::interpolatePoint(
  const int& ncomp_field,
  const double* par_coord, // (3)
  const double* field,     // (8,ncomp_field)
  double* result)          // (ncomp_field)
{
  // 'field' is a flat array of dimension (8,ncomp_field) (Fortran ordering);
  double xi = par_coord[0];
  double eta = par_coord[1];
  double zeta = par_coord[2];

  // NOTE: this uses a [-1,1] definition of the reference element,
  // contrary to the rest of the code

  for (int i = 0; i < ncomp_field; i++) {
    // Base 'field array' index for ith component
    int b = 8 * i;

    result[i] = 0.125 * ((1 - xi) * (1 - eta) * (1 - zeta) * field[b + 0] +
                         (1 + xi) * (1 - eta) * (1 - zeta) * field[b + 1] +
                         (1 + xi) * (1 + eta) * (1 - zeta) * field[b + 2] +
                         (1 - xi) * (1 + eta) * (1 - zeta) * field[b + 3] +
                         (1 - xi) * (1 - eta) * (1 + zeta) * field[b + 4] +
                         (1 + xi) * (1 - eta) * (1 + zeta) * field[b + 5] +
                         (1 + xi) * (1 + eta) * (1 + zeta) * field[b + 6] +
                         (1 - xi) * (1 + eta) * (1 + zeta) * field[b + 7]);
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::general_shape_fcn(
  const int numIp, const double* isoParCoord, double* shpfc)
{
  // -1:1 isoparametric range
  const double npe = nodesPerElement_;
  for (int ip = 0; ip < numIp; ++ip) {

    const int rowIpc = 3 * ip;
    const int rowSfc = npe * ip;

    const double s1 = isoParCoord[rowIpc];
    const double s2 = isoParCoord[rowIpc + 1];
    const double s3 = isoParCoord[rowIpc + 2];
    shpfc[rowSfc] = 0.125 * (1.0 - s1) * (1.0 - s2) * (1.0 - s3);
    shpfc[rowSfc + 1] = 0.125 * (1.0 + s1) * (1.0 - s2) * (1.0 - s3);
    shpfc[rowSfc + 2] = 0.125 * (1.0 + s1) * (1.0 + s2) * (1.0 - s3);
    shpfc[rowSfc + 3] = 0.125 * (1.0 - s1) * (1.0 + s2) * (1.0 - s3);
    shpfc[rowSfc + 4] = 0.125 * (1.0 - s1) * (1.0 - s2) * (1.0 + s3);
    shpfc[rowSfc + 5] = 0.125 * (1.0 + s1) * (1.0 - s2) * (1.0 + s3);
    shpfc[rowSfc + 6] = 0.125 * (1.0 + s1) * (1.0 + s2) * (1.0 + s3);
    shpfc[rowSfc + 7] = 0.125 * (1.0 - s1) * (1.0 + s2) * (1.0 + s3);
  }
}

//--------------------------------------------------------------------------
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::general_face_grad_op(
  const int /* face_ordinal */,
  const double* isoParCoord,
  const double* coords,
  double* gradop,
  double* det_j,
  double* error)
{
  int lerr = 0;
  const int nface = 1;
  const int npe = nodesPerElement_;

  double dpsi[24];

  const SharedMemView<const double**, HostShmem> par_coord(
    isoParCoord, nface, 3);
  SharedMemView<double***, HostShmem> deriv(&dpsi[0], nface, npe, 3);

  hex8_derivative(par_coord, deriv);

  const SharedMemView<const double***, HostShmem> cordel(coords, nface, npe, 3);
  SharedMemView<double****, HostShmem> grad(gradop, nface, nface, npe, 3);
  SharedMemView<double**, HostShmem> det(det_j, nface, nface);
  SharedMemView<double*, HostShmem> err(error, nface);
  lerr = hex_gradient_operator(cordel, deriv, grad, det, err);

  if (lerr)
    NaluEnv::self().naluOutput()
      << "HexSCS::general_face_grad_op: issue.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::sidePcoords_to_elemPcoords(
  const int& side_ordinal,
  const int& npoints,
  const double* side_pcoords,
  double* elem_pcoords)
{
  switch (side_ordinal) {
  case 0:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] = 0.5 * side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 1] = -0.5;
      elem_pcoords[i * 3 + 2] = 0.5 * side_pcoords[2 * i + 1];
    }
    break;
  case 1:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] = 0.5;
      elem_pcoords[i * 3 + 1] = 0.5 * side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 2] = 0.5 * side_pcoords[2 * i + 1];
    }
    break;
  case 2:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] = -0.5 * side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 1] = 0.5;
      elem_pcoords[i * 3 + 2] = 0.5 * side_pcoords[2 * i + 1];
    }
    break;
  case 3:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] = -0.5;
      elem_pcoords[i * 3 + 1] = 0.5 * side_pcoords[2 * i + 1];
      elem_pcoords[i * 3 + 2] = 0.5 * side_pcoords[2 * i + 0];
    }
    break;
  case 4:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] = 0.5 * side_pcoords[2 * i + 1];
      elem_pcoords[i * 3 + 1] = 0.5 * side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 2] = -0.5;
    }
    break;
  case 5:
    for (int i = 0; i < npoints; i++) {
      elem_pcoords[i * 3 + 0] = 0.5 * side_pcoords[2 * i + 0];
      elem_pcoords[i * 3 + 1] = 0.5 * side_pcoords[2 * i + 1];
      elem_pcoords[i * 3 + 2] = 0.5;
    }
    break;
  default:
    throw std::runtime_error("HexSCS::sideMap invalid ordinal");
  }
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
HexSCS::parametric_distance(const std::array<double, 3>& x)
{
  std::array<double, 3> y;
  for (int i = 0; i < 3; ++i) {
    y[i] = std::fabs(x[i]);
  }

  double d = 0;
  for (int i = 0; i < 3; ++i) {
    if (d < y[i]) {
      d = y[i];
    }
  }
  return d;
}

} // namespace nalu
} // namespace sierra
