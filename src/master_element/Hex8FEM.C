// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <master_element/Hex8FEM.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/TensorOps.h>

#include <NaluEnv.h>

#include <cmath>
#include <iostream>

namespace sierra {
namespace nalu {

namespace {

template <
  typename AlgTraits,
  typename GradViewType,
  typename CoordViewType,
  typename DetjType,
  typename OutputViewType>
KOKKOS_FUNCTION void
generic_grad_op_3d(
  GradViewType referenceGradWeights,
  CoordViewType coords,
  OutputViewType weights,
  DetjType detj)
{
  /**
   * Given the reference gradient weights evaluated at the integration points
   * and the cooordinates, this method computes \nabla_\mathbf{x}|_{\alpha}  as
   * \left( J^{-T} \nabla_{\mathbf{x}^\hat} \right)|_\alpha,
   *
   *  This operation can be specialized for efficiency on hex topologies (as
   * tensor-contractions) or on tets (since the gradient is independent of
   * \alpha).  But this can work as a fallback.
   *
   *  Also saves detj at the integration points---useful for FEM but not used at
   * all in CVFEM
   */

  using ftype = typename CoordViewType::value_type;
  static_assert(
    std::is_same<ftype, typename GradViewType::value_type>::value,
    "Incompatiable value type for views");
  static_assert(
    std::is_same<ftype, typename OutputViewType::value_type>::value,
    "Incompatiable value type for views");
  static_assert(GradViewType::rank == 3, "grad view assumed to be 3D");
  static_assert(CoordViewType::rank == 2, "Coordinate view assumed to be 2D");
  static_assert(OutputViewType::rank == 3, "Weight view assumed to be 3D");
  static_assert(AlgTraits::nDim_ == 3, "3D method");

  STK_ThrowAssert(
    AlgTraits::nodesPerElement_ == referenceGradWeights.extent(1));
  STK_ThrowAssert(AlgTraits::nDim_ == referenceGradWeights.extent(2));
  STK_ThrowAssert(weights.extent(0) == referenceGradWeights.extent(0));
  STK_ThrowAssert(weights.extent(1) == referenceGradWeights.extent(1));
  STK_ThrowAssert(weights.extent(2) == referenceGradWeights.extent(2));

  for (unsigned ip = 0; ip < AlgTraits::numGp_; ++ip) {
    ftype jact[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    // compute Jacobian.  Stash away a local copy of the reference weights
    // since they're be used again soon
    ftype refGrad[AlgTraits::nodesPerElement_][3];
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      refGrad[n][0] = referenceGradWeights(ip, n, 0);
      refGrad[n][1] = referenceGradWeights(ip, n, 1);
      refGrad[n][2] = referenceGradWeights(ip, n, 2);

      jact[0][0] += refGrad[n][0] * coords(n, 0);
      jact[0][1] += refGrad[n][1] * coords(n, 0);
      jact[0][2] += refGrad[n][2] * coords(n, 0);

      jact[1][0] += refGrad[n][0] * coords(n, 1);
      jact[1][1] += refGrad[n][1] * coords(n, 1);
      jact[1][2] += refGrad[n][2] * coords(n, 1);

      jact[2][0] += refGrad[n][0] * coords(n, 2);
      jact[2][1] += refGrad[n][1] * coords(n, 2);
      jact[2][2] += refGrad[n][2] * coords(n, 2);
    }

    ftype invJac[3][3];
    adjugate_matrix33(jact, invJac);

    detj(ip) = jact[0][0] * invJac[0][0] + jact[1][0] * invJac[1][0] +
               jact[2][0] * invJac[2][0];
    STK_ThrowAssertMsg(
      stk::simd::are_any(detj(ip) > +tiny_positive_value()),
      "Problem with determinant");

    const ftype inv_detj = ftype(1.0) / detj(ip);
    for (int d_outer = 0; d_outer < 3; ++d_outer) {
      for (int d_inner = 0; d_inner < 3; ++d_inner) {
        invJac[d_outer][d_inner] *= inv_detj;
      }
    }

    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      weights(ip, n, 0) = invJac[0][0] * refGrad[n][0] +
                          invJac[0][1] * refGrad[n][1] +
                          invJac[0][2] * refGrad[n][2];
      weights(ip, n, 1) = invJac[1][0] * refGrad[n][0] +
                          invJac[1][1] * refGrad[n][1] +
                          invJac[1][2] * refGrad[n][2];
      weights(ip, n, 2) = invJac[2][0] * refGrad[n][0] +
                          invJac[2][1] * refGrad[n][1] +
                          invJac[2][2] * refGrad[n][2];
    }
  }
}
} // namespace

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
Hex8FEM::Hex8FEM() : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;
}

//--------------------------------------------------------------------------
//-------- determinant ------------------ n/a ------------------------------
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
#if 0
void
Hex8FEM::grad_op(
  const SharedMemView<double**>& coords,
  SharedMemView<double***>& gradop,
  SharedMemView<double***>& deriv)
{
   generic_grad_op<AlgTraits>(referenceGradWeights_, coords, gradop);
}
#endif

KOKKOS_FUNCTION
void
Hex8FEM::grad_op_fem(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv,
  SharedMemView<DoubleType*, DeviceShmem>& det_j)
{
  hex8_fem_derivative(numIntPoints_, intgLoc_, deriv);
  generic_grad_op_3d<AlgTraits>(deriv, coords, gradop, det_j);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
Hex8FEM::shifted_grad_op_fem(
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop,
  SharedMemView<DoubleType***, DeviceShmem>& deriv,
  SharedMemView<DoubleType*, DeviceShmem>& det_j)
{
  hex8_fem_derivative(numIntPoints_, intgLocShift_, deriv);
  generic_grad_op_3d<AlgTraits>(deriv, coords, gradop, det_j);
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::general_shape_fcn(
  const int numIp, const double* isoParCoord, double* shpfc)
{
  hex8_fem_shape_fcn(numIp, isoParCoord, shpfc);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Hex8FEM::shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  hex8_fem_shape_fcn(numIntPoints_, intgLoc_, shpfc);
}
KOKKOS_FUNCTION void
Hex8FEM::shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}
void
Hex8FEM::shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Hex8FEM::shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc)
{
  hex8_fem_shape_fcn(numIntPoints_, intgLocShift_, shpfc);
}
KOKKOS_FUNCTION void
Hex8FEM::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}
void
Hex8FEM::shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc)
{
  shifted_shape_fcn<>(shpfc);
}

//--------------------------------------------------------------------------
//-------- hex8_fem_shape_fcn ----------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::hex8_fem_shape_fcn(
  const int numIp, const double* isoParCoord, double* shpfc)
{
  // -1:1 isoparametric range
  const int npe = nodesPerElement_;
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
//-------- hex8_fem_shape_fcn ----------------------------------------------
//--------------------------------------------------------------------------
template <typename SCALAR, typename SHMEM>
KOKKOS_FUNCTION void
Hex8FEM::hex8_fem_shape_fcn(
  const int numIp,
  const double* isoParCoord,
  SharedMemView<SCALAR**, SHMEM> shpfc)
{
  // -1:1 isoparametric range
  for (int ip = 0; ip < numIp; ++ip) {
    const int rowIpc = 3 * ip;
    const SCALAR s1 = isoParCoord[rowIpc + 0];
    const SCALAR s2 = isoParCoord[rowIpc + 1];
    const SCALAR s3 = isoParCoord[rowIpc + 2];
    shpfc(ip, 0) = 0.125 * (1.0 - s1) * (1.0 - s2) * (1.0 - s3);
    shpfc(ip, 1) = 0.125 * (1.0 + s1) * (1.0 - s2) * (1.0 - s3);
    shpfc(ip, 2) = 0.125 * (1.0 + s1) * (1.0 + s2) * (1.0 - s3);
    shpfc(ip, 3) = 0.125 * (1.0 - s1) * (1.0 + s2) * (1.0 - s3);
    shpfc(ip, 4) = 0.125 * (1.0 - s1) * (1.0 - s2) * (1.0 + s3);
    shpfc(ip, 5) = 0.125 * (1.0 + s1) * (1.0 - s2) * (1.0 + s3);
    shpfc(ip, 6) = 0.125 * (1.0 + s1) * (1.0 + s2) * (1.0 + s3);
    shpfc(ip, 7) = 0.125 * (1.0 - s1) * (1.0 + s2) * (1.0 + s3);
  }
}

//--------------------------------------------------------------------------
//-------- hex8_fem_derivative ---------------------------------------------
//--------------------------------------------------------------------------
void
Hex8FEM::hex8_fem_derivative(
  const int npt, const double* par_coord, double* deriv)
{
  for (int i = 0; i < npt; ++i) {
    deriv[i * nodesPerElement_ * 3] =
      -0.125 * (1.0 - par_coord[i * 3 + 1]) * (1.0 - par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 3] =
      0.125 * (1.0 - par_coord[i * 3 + 1]) * (1.0 - par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 6] =
      0.125 * (1.0 + par_coord[i * 3 + 1]) * (1.0 - par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 9] =
      -0.125 * (1.0 + par_coord[i * 3 + 1]) * (1.0 - par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 12] =
      -0.125 * (1.0 - par_coord[i * 3 + 1]) * (1.0 + par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 15] =
      0.125 * (1.0 - par_coord[i * 3 + 1]) * (1.0 + par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 18] =
      0.125 * (1.0 + par_coord[i * 3 + 1]) * (1.0 + par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 21] =
      -0.125 * (1.0 + par_coord[i * 3 + 1]) * (1.0 + par_coord[i * 3 + 2]);

    deriv[i * nodesPerElement_ * 3 + 1] =
      -0.125 * (1.0 - par_coord[i * 3]) * (1.0 - par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 4] =
      -0.125 * (1.0 + par_coord[i * 3]) * (1.0 - par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 7] =
      0.125 * (1.0 + par_coord[i * 3]) * (1.0 - par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 10] =
      0.125 * (1.0 - par_coord[i * 3]) * (1.0 - par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 13] =
      -0.125 * (1.0 - par_coord[i * 3]) * (1.0 + par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 16] =
      -0.125 * (1.0 + par_coord[i * 3]) * (1.0 + par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 19] =
      0.125 * (1.0 + par_coord[i * 3]) * (1.0 + par_coord[i * 3 + 2]);
    deriv[i * nodesPerElement_ * 3 + 22] =
      0.125 * (1.0 - par_coord[i * 3]) * (1.0 + par_coord[i * 3 + 2]);

    deriv[i * nodesPerElement_ * 3 + 2] =
      -0.125 * (1.0 - par_coord[i * 3]) * (1.0 - par_coord[i * 3 + 1]);
    deriv[i * nodesPerElement_ * 3 + 5] =
      -0.125 * (1.0 + par_coord[i * 3]) * (1.0 - par_coord[i * 3 + 1]);
    deriv[i * nodesPerElement_ * 3 + 8] =
      -0.125 * (1.0 + par_coord[i * 3]) * (1.0 + par_coord[i * 3 + 1]);
    deriv[i * nodesPerElement_ * 3 + 11] =
      -0.125 * (1.0 - par_coord[i * 3]) * (1.0 + par_coord[i * 3 + 1]);
    deriv[i * nodesPerElement_ * 3 + 14] =
      0.125 * (1.0 - par_coord[i * 3]) * (1.0 - par_coord[i * 3 + 1]);
    deriv[i * nodesPerElement_ * 3 + 17] =
      0.125 * (1.0 + par_coord[i * 3]) * (1.0 - par_coord[i * 3 + 1]);
    deriv[i * nodesPerElement_ * 3 + 20] =
      0.125 * (1.0 + par_coord[i * 3]) * (1.0 + par_coord[i * 3 + 1]);
    deriv[i * nodesPerElement_ * 3 + 23] =
      0.125 * (1.0 - par_coord[i * 3]) * (1.0 + par_coord[i * 3 + 1]);
  }
}

KOKKOS_FUNCTION
void
Hex8FEM::hex8_fem_derivative(
  const int npt,
  const double* par_coord,
  SharedMemView<DoubleType***, DeviceShmem> deriv)
{
  for (int ip = 0; ip < npt; ++ip) {
    DoubleType x = par_coord[ip * 3 + 0];
    DoubleType y = par_coord[ip * 3 + 1];
    DoubleType z = par_coord[ip * 3 + 2];

    deriv(ip, 0, 0) = -0.125 * (1.0 - y) * (1.0 - z);
    deriv(ip, 1, 0) = 0.125 * (1.0 - y) * (1.0 - z);
    deriv(ip, 2, 0) = 0.125 * (1.0 + y) * (1.0 - z);
    deriv(ip, 3, 0) = -0.125 * (1.0 + y) * (1.0 - z);
    deriv(ip, 4, 0) = -0.125 * (1.0 - y) * (1.0 + z);
    deriv(ip, 5, 0) = 0.125 * (1.0 - y) * (1.0 + z);
    deriv(ip, 6, 0) = 0.125 * (1.0 + y) * (1.0 + z);
    deriv(ip, 7, 0) = -0.125 * (1.0 + y) * (1.0 + z);

    deriv(ip, 0, 1) = -0.125 * (1.0 - x) * (1.0 - z);
    deriv(ip, 1, 1) = -0.125 * (1.0 + x) * (1.0 - z);
    deriv(ip, 2, 1) = 0.125 * (1.0 + x) * (1.0 - z);
    deriv(ip, 3, 1) = 0.125 * (1.0 - x) * (1.0 - z);
    deriv(ip, 4, 1) = -0.125 * (1.0 - x) * (1.0 + z);
    deriv(ip, 5, 1) = -0.125 * (1.0 + x) * (1.0 + z);
    deriv(ip, 6, 1) = 0.125 * (1.0 + x) * (1.0 + z);
    deriv(ip, 7, 1) = 0.125 * (1.0 - x) * (1.0 + z);

    deriv(ip, 0, 2) = -0.125 * (1.0 - x) * (1.0 - y);
    deriv(ip, 1, 2) = -0.125 * (1.0 + x) * (1.0 - y);
    deriv(ip, 2, 2) = -0.125 * (1.0 + x) * (1.0 + y);
    deriv(ip, 3, 2) = -0.125 * (1.0 - x) * (1.0 + y);
    deriv(ip, 4, 2) = 0.125 * (1.0 - x) * (1.0 - y);
    deriv(ip, 5, 2) = 0.125 * (1.0 + x) * (1.0 - y);
    deriv(ip, 6, 2) = 0.125 * (1.0 + x) * (1.0 + y);
    deriv(ip, 7, 2) = 0.125 * (1.0 - x) * (1.0 + y);
  }
}

} // namespace nalu
} // namespace sierra
