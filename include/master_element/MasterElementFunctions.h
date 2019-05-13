/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MasterElementFunctions_h
#define MasterElementFunctions_h

#include <AlgTraits.h>

#include <master_element/MasterElement.h>
#include <master_element/TensorOps.h>

#include <SimdInterface.h>
#include <Kokkos_Core.hpp>

#include <EigenDecomposition.h>

#include <stk_util/util/ReportHandler.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>
#include <type_traits>

namespace sierra {
namespace nalu {

  template<typename ftype> KOKKOS_INLINE_FUNCTION void cofactorMatrix(ftype adjJac[][3], const ftype jact[][3]) {
    adjJac[0][0] = jact[1][1] * jact[2][2] - jact[2][1] * jact[1][2];
    adjJac[0][1] = jact[1][2] * jact[2][0] - jact[2][2] * jact[1][0];
    adjJac[0][2] = jact[1][0] * jact[2][1] - jact[2][0] * jact[1][1];

    adjJac[1][0] = jact[0][2] * jact[2][1] - jact[2][2] * jact[0][1];
    adjJac[1][1] = jact[0][0] * jact[2][2] - jact[2][0] * jact[0][2];
    adjJac[1][2] = jact[0][1] * jact[2][0] - jact[2][1] * jact[0][0];

    adjJac[2][0] = jact[0][1] * jact[1][2] - jact[1][1] * jact[0][2];
    adjJac[2][1] = jact[0][2] * jact[1][0] - jact[1][2] * jact[0][0];
    adjJac[2][2] = jact[0][0] * jact[1][1] - jact[1][0] * jact[0][1];
  }
  template<typename ftype> KOKKOS_INLINE_FUNCTION void cofactorMatrix(ftype adjJac[][2], const ftype jact[][2]) {
    adjJac[0][0] =  jact[1][1];
    adjJac[0][1] = -jact[1][0];
    adjJac[1][0] = -jact[0][1];
    adjJac[1][1] =  jact[0][0];
  }

  template <typename AlgTraits, typename GradViewType, typename CoordViewType, typename OutputViewType>
  KOKKOS_FUNCTION
  KOKKOS_FUNCTION void generic_grad_op(const GradViewType& referenceGradWeights, const CoordViewType& coords, OutputViewType& weights)
  {
    constexpr int dim = AlgTraits::nDim_;

    using ftype = typename CoordViewType::value_type;
    static_assert(std::is_same<ftype, typename GradViewType::value_type>::value,  "Incompatiable value type for views");
    static_assert(std::is_same<ftype, typename OutputViewType::value_type>::value,  "Incompatiable value type for views");
    static_assert(GradViewType::Rank   ==   3, "grad view assumed to be rank 3");
    static_assert(CoordViewType::Rank  ==   2, "Coordinate view assumed to be rank 2");
    static_assert(OutputViewType::Rank ==   3, "Weight view assumed to be rank 3");

    ThrowAssert(AlgTraits::nodesPerElement_ == referenceGradWeights.extent(1));
    ThrowAssert(AlgTraits::nDim_            == referenceGradWeights.extent(2));
    for (int i=0; i<dim; ++i)
      ThrowAssert(weights.extent(i) == referenceGradWeights.extent(i));

    for (unsigned ip = 0; ip < referenceGradWeights.extent(0); ++ip) {
      NALU_ALIGNED ftype jact[dim][dim];
      for (int i=0; i<dim; ++i)
        for (int j=0; j<dim; ++j)
          jact[i][j] = ftype(0.0);

      NALU_ALIGNED ftype refGrad[AlgTraits::nodesPerElement_][dim];
      for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
        for (int i=0; i<dim; ++i) {
          refGrad[n][i] = referenceGradWeights(ip, n, i);
        }
        for (int i=0; i<dim; ++i) {
          for (int j=0; j<dim; ++j) {
            jact[i][j] += refGrad[n][j] * coords(n, i);
          }
        }
      }

      NALU_ALIGNED ftype adjJac[dim][dim];
      cofactorMatrix(adjJac, jact);

      NALU_ALIGNED ftype det = ftype(0.0);
      for (int i=0; i<dim; ++i) det += jact[i][0] * adjJac[i][0];
      ThrowAssertMsg(
        stk::simd::are_any(det > tiny_positive_value()),
        "Problem with Jacobian determinant"
      );

      NALU_ALIGNED const ftype inv_detj = ftype(1.0) / det;

      for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
        for (int i=0; i<dim; ++i) {
          weights(ip, n, i) = ftype(0.0);
          for (int j=0; j<dim; ++j) {
            weights(ip, n, i) += adjJac[i][j] * refGrad[n][j];
          }
          weights(ip, n, i) *= inv_detj;
        }
      }
    }
  }

  template <typename AlgTraits, typename GradViewType, typename CoordViewType, typename OutputViewType>
  KOKKOS_FUNCTION void generic_gij_3d(
    const GradViewType& referenceGradWeights,
    const CoordViewType& coords,
    OutputViewType& gup,
    OutputViewType& glo)
  {
    using ftype = typename CoordViewType::value_type;
    static_assert(std::is_same<ftype, typename GradViewType::value_type>::value,
      "Incompatiable value type for views");
    static_assert(std::is_same<ftype, typename OutputViewType::value_type>::value,
      "Incompatiable value type for views");
    static_assert(GradViewType::Rank == 3, "grad view assumed to be 3D");
    static_assert(CoordViewType::Rank == 2, "Coordinate view assumed to be 2D");
    static_assert(OutputViewType::Rank == 3, "gij view assumed to be 3D");
    static_assert(AlgTraits::nDim_ == 3, "3D method");

    for (unsigned ip = 0; ip < referenceGradWeights.extent(0); ++ip) {

      NALU_ALIGNED ftype jac[3][3] = { {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} };
      for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
        jac[0][0] += referenceGradWeights(ip, n, 0) * coords(n, 0);
        jac[0][1] += referenceGradWeights(ip, n, 1) * coords(n, 0);
        jac[0][2] += referenceGradWeights(ip, n, 2) * coords(n, 0);

        jac[1][0] += referenceGradWeights(ip, n, 0) * coords(n, 1);
        jac[1][1] += referenceGradWeights(ip, n, 1) * coords(n, 1);
        jac[1][2] += referenceGradWeights(ip, n, 2) * coords(n, 1);

        jac[2][0] += referenceGradWeights(ip, n, 0) * coords(n, 2);
        jac[2][1] += referenceGradWeights(ip, n, 1) * coords(n, 2);
        jac[2][2] += referenceGradWeights(ip, n, 2) * coords(n, 2);
      }

      gup(ip, 0, 0) = jac[0][0] * jac[0][0] + jac[0][1] * jac[0][1] + jac[0][2] * jac[0][2];
      gup(ip, 0, 1) = jac[0][0] * jac[1][0] + jac[0][1] * jac[1][1] + jac[0][2] * jac[1][2];
      gup(ip, 0, 2) = jac[0][0] * jac[2][0] + jac[0][1] * jac[2][1] + jac[0][2] * jac[2][2];

      gup(ip, 1, 0) = gup(ip, 0, 1);
      gup(ip, 1, 1) = jac[1][0] * jac[1][0] + jac[1][1] * jac[1][1] + jac[1][2] * jac[1][2];
      gup(ip, 1, 2) = jac[1][0] * jac[2][0] + jac[1][1] * jac[2][1] + jac[1][2] * jac[2][2];

      gup(ip, 2, 0) = gup(ip, 0, 2);
      gup(ip, 2, 1) = gup(ip, 1, 2);
      gup(ip, 2, 2) = jac[2][0] * jac[2][0] + jac[2][1] * jac[2][1] + jac[2][2] * jac[2][2];

      // the covariant is the inverse of the contravariant by definition
      // gUpper is symmetric
      NALU_ALIGNED const ftype inv_detj = ftype(1.0) / (
            gup(ip, 0, 0) * ( gup(ip, 1, 1) * gup(ip, 2, 2) - gup(ip, 1, 2) * gup(ip, 1, 2) )
          - gup(ip, 0, 1) * ( gup(ip, 0, 1) * gup(ip, 2, 2) - gup(ip, 1, 2) * gup(ip, 0, 2) )
          + gup(ip, 0, 2) * ( gup(ip, 0, 1) * gup(ip, 1, 2) - gup(ip, 1, 1) * gup(ip, 0, 2) )
      );

      glo(ip, 0, 0) = inv_detj * (gup(ip, 1, 1) * gup(ip, 2, 2) - gup(ip, 1, 2) * gup(ip, 1, 2));
      glo(ip, 0, 1) = inv_detj * (gup(ip, 0, 2) * gup(ip, 1, 2) - gup(ip, 0, 1) * gup(ip, 2, 2));
      glo(ip, 0, 2) = inv_detj * (gup(ip, 0, 1) * gup(ip, 1, 2) - gup(ip, 0, 2) * gup(ip, 1, 1));

      glo(ip, 1, 0) = glo(ip, 0, 1);
      glo(ip, 1, 1) = inv_detj * (gup(ip, 0, 0) * gup(ip, 2, 2) - gup(ip, 0, 2) * gup(ip, 0, 2));
      glo(ip, 1, 2) = inv_detj * (gup(ip, 0, 2) * gup(ip, 0, 1) - gup(ip, 0, 0) * gup(ip, 1, 2));


      glo(ip, 2, 0) = glo(ip, 0, 2);
      glo(ip, 2, 1) = glo(ip, 1, 2);
      glo(ip, 2, 2) = inv_detj * (gup(ip, 0, 0) * gup(ip, 1, 1) - gup(ip, 0, 1) * gup(ip, 0, 1));
    }
  }

  template <typename AlgTraits>
  KOKKOS_FUNCTION void generic_Mij_2d(const int numIntPoints, const double *deriv,
                      const double *coords, double *metric) {
    static_assert(AlgTraits::nDim_ == 2, "2D method");

    const int npe = AlgTraits::nodesPerElement_;
    const int nint = numIntPoints;
    const int ndim = AlgTraits::nDim_;

    double dx_ds[2][2];
    double norm;
    double ev[2][2];

    // loop over integration points
    for (int ki = 0; ki < nint; ++ki) {
      dx_ds[0][0] = 0.0;
      dx_ds[0][1] = 0.0;
      dx_ds[1][0] = 0.0;
      dx_ds[1][1] = 0.0;

      // calculate the jacobian at the integration station by looping over nodes
      for (int kn = 0; kn < npe; ++kn) {
        dx_ds[0][0] +=
            deriv[(ki * npe + kn) * ndim + 0] * coords[kn * ndim + 0];
        dx_ds[0][1] +=
            deriv[(ki * npe + kn) * ndim + 1] * coords[kn * ndim + 0];
        dx_ds[1][0] +=
            deriv[(ki * npe + kn) * ndim + 0] * coords[kn * ndim + 1];
        dx_ds[1][1] +=
            deriv[(ki * npe + kn) * ndim + 1] * coords[kn * ndim + 1];
      }

      // Mij^2 = J J^T
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          metric[(ki * ndim + i) * ndim + j] =
              dx_ds[i][0] * dx_ds[j][0] + dx_ds[i][1] * dx_ds[j][1];
        }
      }

      // To get the square root of Mij^2, we note that this is a Sym Pos Def
      // matrix and thus Mij^(1/2) = A sqrt(L) A^T where Mij^2 = ALA^T is the
      // eigendecomposition of Mij^2
      const double trace = metric[(ki * ndim + 0) * ndim + 0] +
                           metric[(ki * ndim + 1) * ndim + 1];
      const double det = metric[(ki * ndim + 0) * ndim + 0] *
                             metric[(ki * ndim + 1) * ndim + 1] -
                         metric[(ki * ndim + 0) * ndim + 1] *
                             metric[(ki * ndim + 1) * ndim + 0];

      const double lambda1 =
          trace / 2.0 + stk::math::pow(trace * trace / 4.0 - det, 0.5);
      const double lambda2 =
          trace / 2.0 - stk::math::pow(trace * trace / 4.0 - det, 0.5);

      // calculate first eigenvector
      ev[0][0] = -metric[(ki * ndim + 0) * ndim + 1];
      ev[1][0] = metric[(ki * ndim + 0) * ndim + 0] - lambda1;

      norm = stk::math::sqrt(ev[0][0] * ev[0][0] + ev[1][0] * ev[1][0]);
      ev[0][0] = ev[0][0] / norm;
      ev[1][0] = ev[1][0] / norm;

      // calculate second eigenvector
      ev[0][1] = -(metric[(ki * ndim + 1) * ndim + 1] - lambda2);
      ev[1][1] = metric[(ki * ndim + 1) * ndim + 0];

      norm = stk::math::sqrt(ev[0][1] * ev[0][1] + ev[1][1] * ev[1][1]);
      ev[0][1] = ev[0][1] / norm;
      ev[1][1] = ev[1][1] / norm;

      // special case when diagonal entries were 0, we already had a diagonal
      // matrix
      ev[0][0] = stk::math::if_then_else(
          metric[(ki * ndim + 1) * ndim + 0] == 0.0, 1.0, ev[0][0]);
      ev[0][1] = stk::math::if_then_else(
          metric[(ki * ndim + 1) * ndim + 0] == 0.0, 0.0, ev[0][1]);
      ev[1][0] = stk::math::if_then_else(
          metric[(ki * ndim + 1) * ndim + 0] == 0.0, 0.0, ev[1][0]);
      ev[1][1] = stk::math::if_then_else(
          metric[(ki * ndim + 1) * ndim + 0] == 0.0, 1.0, ev[1][1]);

      // calculate sqrt of Mij^2 to get the metric tensor
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          metric[(ki * ndim + i) * ndim + j] =
              ev[i][0] * ev[j][0] * stk::math::sqrt(lambda1) +
              ev[i][1] * ev[j][1] * stk::math::sqrt(lambda2);
        }
      }
    }
  }

  template <typename AlgTraits, typename GradViewType, typename CoordViewType,
            typename OutputViewType>
  KOKKOS_FUNCTION void generic_Mij_2d(const GradViewType &referenceGradWeights,
                      const CoordViewType &coords, OutputViewType &metric) {
    using ftype = typename CoordViewType::value_type;
    static_assert(std::is_same<ftype, typename GradViewType::value_type>::value,
                  "Incompatiable value type for views");
    static_assert(
        std::is_same<ftype, typename OutputViewType::value_type>::value,
        "Incompatiable value type for views");
    static_assert(GradViewType::Rank == 3, "grad view assumed to be 3D");
    static_assert(CoordViewType::Rank == 2, "Coordinate view assumed to be 2D");
    static_assert(OutputViewType::Rank == 3, "Mij view assumed to be 3D");
    static_assert(AlgTraits::nDim_ == 2, "2D method");

    const int npe = AlgTraits::nodesPerElement_;
    const int nint = referenceGradWeights.extent(0);

    ftype dx_ds[2][2];
    ftype norm;
    ftype ev[2][2];

    // loop over integration points
    for (int ki = 0; ki < nint; ++ki) {
      dx_ds[0][0] = 0.0;
      dx_ds[0][1] = 0.0;
      dx_ds[1][0] = 0.0;
      dx_ds[1][1] = 0.0;

      // calculate the jacobian at the integration station by looping over nodes
      for (int kn = 0; kn < npe; ++kn) {
        dx_ds[0][0] += referenceGradWeights(ki, kn, 0) * coords(kn, 0);
        dx_ds[0][1] += referenceGradWeights(ki, kn, 1) * coords(kn, 0);
        dx_ds[1][0] += referenceGradWeights(ki, kn, 0) * coords(kn, 1);
        dx_ds[1][1] += referenceGradWeights(ki, kn, 1) * coords(kn, 1);
      }

      // Mij^2 = J J^T
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          metric(ki, j, i) =
              dx_ds[i][0] * dx_ds[j][0] + dx_ds[i][1] * dx_ds[j][1];
        }
      }

      // To get the square root of Mij^2, we note that this is a Sym Pos Def
      // matrix and thus Mij^(1/2) = A sqrt(L) A^T where Mij^2 = ALA^T is the
      // eigendecomposition of Mij^2
      const ftype trace = metric(ki, 0, 0) + metric(ki, 1, 1);
      const ftype det = metric(ki, 0, 0) * metric(ki, 1, 1) -
                        metric(ki, 0, 1) * metric(ki, 1, 0);

      // calculate eigenvalues
      const ftype lambda1 =
          trace / 2.0 + stk::math::pow(trace * trace / 4.0 - det, 0.5);
      const ftype lambda2 =
          trace / 2.0 - stk::math::pow(trace * trace / 4.0 - det, 0.5);

      // calculate first eigenvector
      ev[0][0] = -metric(ki, 0, 1);
      ev[1][0] = metric(ki, 0, 0) - lambda1;

      norm = stk::math::sqrt(ev[0][0] * ev[0][0] + ev[1][0] * ev[1][0]);
      ev[0][0] = ev[0][0] / norm;
      ev[1][0] = ev[1][0] / norm;

      // calculate second eigenvector
      ev[0][1] = -(metric(ki, 1, 1) - lambda2);
      ev[1][1] = metric(ki, 1, 0);

      norm = stk::math::sqrt(ev[0][1] * ev[0][1] + ev[1][1] * ev[1][1]);
      ev[0][1] = ev[0][1] / norm;
      ev[1][1] = ev[1][1] / norm;

      // special case when diagonal entries were 0, we already had a diagonal
      // matrix
      ev[0][0] =
          stk::math::if_then_else(metric(ki, 1, 0) == 0.0, 1.0, ev[0][0]);
      ev[0][1] =
          stk::math::if_then_else(metric(ki, 1, 0) == 0.0, 0.0, ev[0][1]);
      ev[1][0] =
          stk::math::if_then_else(metric(ki, 1, 0) == 0.0, 0.0, ev[1][0]);
      ev[1][1] =
          stk::math::if_then_else(metric(ki, 1, 0) == 0.0, 1.0, ev[1][1]);

      // calculate sqrt of Mij^2 to get the metric tensor
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          metric(ki, i, j) = ev[i][0] * ev[j][0] * stk::math::sqrt(lambda1) +
                             ev[i][1] * ev[j][1] * stk::math::sqrt(lambda2);
        }
      }
    }
  }

  template <typename AlgTraits>
  void generic_Mij_3d(
    const int numIntPoints,
    const double* deriv,
    const double* coords,
    double* metric)
  {
    static_assert(AlgTraits::nDim_ == 3, "3D method");

    for (int ip = 0; ip < numIntPoints; ++ip) {

      double jac[3][3] = { {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} };
      for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
        jac[0][0] += deriv[(ip * AlgTraits::nodesPerElement_ + n) * AlgTraits::nDim_ + 0] * coords[n * AlgTraits::nDim_ + 0];
        jac[0][1] += deriv[(ip * AlgTraits::nodesPerElement_ + n) * AlgTraits::nDim_ + 1] * coords[n * AlgTraits::nDim_ + 0];
        jac[0][2] += deriv[(ip * AlgTraits::nodesPerElement_ + n) * AlgTraits::nDim_ + 2] * coords[n * AlgTraits::nDim_ + 0];

        jac[1][0] += deriv[(ip * AlgTraits::nodesPerElement_ + n) * AlgTraits::nDim_ + 0] * coords[n * AlgTraits::nDim_ + 1];
        jac[1][1] += deriv[(ip * AlgTraits::nodesPerElement_ + n) * AlgTraits::nDim_ + 1] * coords[n * AlgTraits::nDim_ + 1];
        jac[1][2] += deriv[(ip * AlgTraits::nodesPerElement_ + n) * AlgTraits::nDim_ + 2] * coords[n * AlgTraits::nDim_ + 1];

        jac[2][0] += deriv[(ip * AlgTraits::nodesPerElement_ + n) * AlgTraits::nDim_ + 0] * coords[n * AlgTraits::nDim_ + 2];
        jac[2][1] += deriv[(ip * AlgTraits::nodesPerElement_ + n) * AlgTraits::nDim_ + 1] * coords[n * AlgTraits::nDim_ + 2];
        jac[2][2] += deriv[(ip * AlgTraits::nodesPerElement_ + n) * AlgTraits::nDim_ + 2] * coords[n * AlgTraits::nDim_ + 2];
      }

      // Here we calculate Mij^2 = J J^T
      double M[3][3];
      M[0][0] = jac[0][0] * jac[0][0] + jac[0][1] * jac[0][1] + jac[0][2] * jac[0][2];
      M[0][1] = jac[0][0] * jac[1][0] + jac[0][1] * jac[1][1] + jac[0][2] * jac[1][2];
      M[0][2] = jac[0][0] * jac[2][0] + jac[0][1] * jac[2][1] + jac[0][2] * jac[2][2];

      M[1][0] = M[0][1];
      M[1][1] = jac[1][0] * jac[1][0] + jac[1][1] * jac[1][1] + jac[1][2] * jac[1][2];
      M[1][2] = jac[1][0] * jac[2][0] + jac[1][1] * jac[2][1] + jac[1][2] * jac[2][2];

      M[2][0] = M[0][2];
      M[2][1] = M[1][2];
      M[2][2] = jac[2][0] * jac[2][0] + jac[2][1] * jac[2][1] + jac[2][2] * jac[2][2];

      // Now we take the sqrt(M^2) using eigenvalue decomposition, i.e. M = A sqrt(L) A^T
      // where M^2 = A L A^T since M^2 is symmetric positive definite as is M
      double Q[3][3];
      double D[3][3];
      EigenDecomposition::sym_diagonalize(M, Q, D);

      // At this point we have Q, the eigenvectors and D the eigenvalues of Mij^2, so to
      // create Mij, we use Q sqrt(D) Q^T
      for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
          metric[(ip * AlgTraits::nDim_ + i) * AlgTraits::nDim_ + j] =
            Q[i][0]*Q[j][0]*stk::math::sqrt(D[0][0]) +
            Q[i][1]*Q[j][1]*stk::math::sqrt(D[1][1]) +
            Q[i][2]*Q[j][2]*stk::math::sqrt(D[2][2]);
    }
  }

  template <typename AlgTraits, typename GradViewType, typename CoordViewType, typename OutputViewType>
  KOKKOS_FUNCTION void generic_Mij_3d(
    const GradViewType& referenceGradWeights,
    const CoordViewType& coords,
    OutputViewType& metric)
  {
    using ftype = typename CoordViewType::value_type;
    static_assert(std::is_same<ftype, typename GradViewType::value_type>::value,
      "Incompatiable value type for views");
    static_assert(std::is_same<ftype, typename OutputViewType::value_type>::value,
      "Incompatiable value type for views");
    static_assert(GradViewType::Rank == 3, "grad view assumed to be 3D");
    static_assert(CoordViewType::Rank == 2, "Coordinate view assumed to be 2D");
    static_assert(OutputViewType::Rank == 3, "Mij view assumed to be 3D");
    static_assert(AlgTraits::nDim_ == 3, "3D method");

    for (unsigned ip = 0; ip < referenceGradWeights.extent(0); ++ip) {

      ftype jac[3][3] = { {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} };
      for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
        jac[0][0] += referenceGradWeights(ip, n, 0) * coords(n, 0);
        jac[0][1] += referenceGradWeights(ip, n, 1) * coords(n, 0);
        jac[0][2] += referenceGradWeights(ip, n, 2) * coords(n, 0);

        jac[1][0] += referenceGradWeights(ip, n, 0) * coords(n, 1);
        jac[1][1] += referenceGradWeights(ip, n, 1) * coords(n, 1);
        jac[1][2] += referenceGradWeights(ip, n, 2) * coords(n, 1);

        jac[2][0] += referenceGradWeights(ip, n, 0) * coords(n, 2);
        jac[2][1] += referenceGradWeights(ip, n, 1) * coords(n, 2);
        jac[2][2] += referenceGradWeights(ip, n, 2) * coords(n, 2);
      }

      // Here we calculate Mij^2 = J J^T
      ftype M[3][3];
      M[0][0] = jac[0][0] * jac[0][0] + jac[0][1] * jac[0][1] + jac[0][2] * jac[0][2];
      M[0][1] = jac[0][0] * jac[1][0] + jac[0][1] * jac[1][1] + jac[0][2] * jac[1][2];
      M[0][2] = jac[0][0] * jac[2][0] + jac[0][1] * jac[2][1] + jac[0][2] * jac[2][2];

      M[1][0] = M[0][1];
      M[1][1] = jac[1][0] * jac[1][0] + jac[1][1] * jac[1][1] + jac[1][2] * jac[1][2];
      M[1][2] = jac[1][0] * jac[2][0] + jac[1][1] * jac[2][1] + jac[1][2] * jac[2][2];

      M[2][0] = M[0][2];
      M[2][1] = M[1][2];
      M[2][2] = jac[2][0] * jac[2][0] + jac[2][1] * jac[2][1] + jac[2][2] * jac[2][2];

      // Now we take the sqrt(M^2) using eigenvalue decomposition, i.e. M = A sqrt(L) A^T
      // where M^2 = A L A^T since M^2 is symmetric positive definite as is M
      ftype Q[3][3];
      ftype D[3][3];
      EigenDecomposition::sym_diagonalize(M, Q, D);

      // At this point we have Q, the eigenvectors and D the eigenvalues of Mij^2, so to
      // create Mij, we use Q sqrt(D) Q^T
      for (unsigned i = 0; i < 3; i++){
        for (unsigned j = 0; j < 3; j++){
          metric(ip,i,j) = Q[i][0]*Q[j][0]*stk::math::sqrt(D[0][0]) +
                           Q[i][1]*Q[j][1]*stk::math::sqrt(D[1][1]) +
                           Q[i][2]*Q[j][2]*stk::math::sqrt(D[2][2]);
          //std::cerr << metric(ip,i,j) << " " << Q[i][2] << " " << " " << Q[j][2] << " " << D[2][2] << std::endl;
        }
      }
    }
  }

  template <typename AlgTraits, typename GradViewType, typename CoordViewType, typename OutputViewType>
  KOKKOS_FUNCTION void generic_determinant_3d(GradViewType referenceGradWeights, CoordViewType coords, OutputViewType detj)
  {
    using ftype = typename CoordViewType::value_type;
    static_assert(std::is_same<ftype, typename GradViewType::value_type>::value,  "Incompatiable value type for views");
    static_assert(std::is_same<ftype, typename OutputViewType::value_type>::value,  "Incompatiable value type for views");
    static_assert(GradViewType::Rank == 3, "grad view assumed to be 3D");
    static_assert(CoordViewType::Rank == 2, "Coordinate view assumed to be 2D");
    static_assert(OutputViewType::Rank == 1, "Weight view assumed to be 1D");
    static_assert(AlgTraits::nDim_ == 3, "3D method");

    ThrowAssert(AlgTraits::nodesPerElement_ == referenceGradWeights.extent(1));
    ThrowAssert(AlgTraits::nDim_ == referenceGradWeights.extent(2));

    ThrowAssert(detj.extent(0) == referenceGradWeights.extent(0));

    for (unsigned ip = 0; ip < referenceGradWeights.extent(0); ++ip) {
      NALU_ALIGNED ftype jac[3][3] = { {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} };
      for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
        jac[0][0] += referenceGradWeights(ip, n, 0) * coords(n, 0);
        jac[0][1] += referenceGradWeights(ip, n, 1) * coords(n, 0);
        jac[0][2] += referenceGradWeights(ip, n, 2) * coords(n, 0);

        jac[1][0] += referenceGradWeights(ip, n, 0) * coords(n, 1);
        jac[1][1] += referenceGradWeights(ip, n, 1) * coords(n, 1);
        jac[1][2] += referenceGradWeights(ip, n, 2) * coords(n, 1);

        jac[2][0] += referenceGradWeights(ip, n, 0) * coords(n, 2);
        jac[2][1] += referenceGradWeights(ip, n, 1) * coords(n, 2);
        jac[2][2] += referenceGradWeights(ip, n, 2) * coords(n, 2);
      }
      detj(ip) = determinant33(&jac[0][0]);
    }
  }

} // namespace nalu
} // namespace Sierra

#endif
