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

#include <stk_util/util/ReportHandler.hpp>

#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>
#include <type_traits>

namespace sierra {
namespace nalu {

  template<typename ftype> inline void cofactorMatrix(ftype adjJac[][3], const ftype jact[][3]) {
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
  template<typename ftype> inline void cofactorMatrix(ftype adjJac[][2], const ftype jact[][2]) {
    adjJac[0][0] =  jact[1][1];
    adjJac[0][1] = -jact[1][0];
    adjJac[1][0] = -jact[0][1];
    adjJac[1][1] =  jact[0][0];
  }

  template <typename AlgTraits, typename GradViewType, typename CoordViewType, typename OutputViewType>
  void generic_grad_op(const GradViewType& referenceGradWeights, const CoordViewType& coords, OutputViewType& weights)
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
  void generic_gij_3d(
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

  template <typename AlgTraits, typename GradViewType, typename CoordViewType, typename OutputViewType>
  void generic_determinant_3d(GradViewType referenceGradWeights, CoordViewType coords, OutputViewType detj)
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
