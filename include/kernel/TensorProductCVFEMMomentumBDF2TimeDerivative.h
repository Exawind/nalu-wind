/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductCVFEMMomentumBDF2TimeDerivative_h
#define TensorProductCVFEMMomentumBDF2TimeDerivative_h

#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/DirectionMacros.h>
#include <CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int poly_order, typename Scalar>
void momentum_dt_lhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& metric,
  double gamma1_div_dt,
  const nodal_scalar_view<poly_order, Scalar>& rho_p1,
  matrix_vector_view<poly_order, Scalar>& lhs)
{
  constexpr int n1D = poly_order + 1;
  const auto& weight = ops.mat_.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      for (int l = 0; l < n1D; ++l) {
        const int rowIndices[3] = {
            idx<n1D>(XH, n, m, l),
            idx<n1D>(YH, n, m, l),
            idx<n1D>(ZH, n, m, l)
        };

        for (int k = 0; k < n1D; ++k) {
          const Scalar gammaWnk = gamma1_div_dt * weight(n, k);
          for (int j = 0; j < n1D; ++j) {
            auto gammaWnkWmj = gammaWnk * weight(m, j);
            for (int i = 0; i < n1D; ++i) {
              const Scalar lhsfac = gammaWnkWmj * weight(l, i) * metric(k, j, i) * rho_p1(k, j, i);
              lhs(rowIndices[XH], idx<n1D>(XH, k, j, i)) += lhsfac;
              lhs(rowIndices[YH], idx<n1D>(YH, k, j, i)) += lhsfac;
              lhs(rowIndices[ZH], idx<n1D>(ZH, k, j, i)) += lhsfac;
            }
          }
        }

      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename Scalar>
void momentum_dt_lhs_lumped(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& metric,
  double gamma1_div_dt,
  const nodal_scalar_view<poly_order, Scalar>& rho_p1,
  matrix_vector_view<poly_order, Scalar>& lhs)
{
  constexpr int n1D = poly_order + 1;
  const auto& weight = ops.mat_.lumpedNodalWeights;

  for (int n = 0; n < n1D; ++n) {
    const Scalar Wn = gamma1_div_dt * weight(n,n);
    for (int m = 0; m < n1D; ++m) {
      const Scalar WnWm = Wn * weight(m,m);
      for (int l = 0; l < n1D; ++l) {
        const auto lhsfac = WnWm * weight(l,l) * metric(n,m,l) * rho_p1(n,m,l);
        const int rowIndices[3] = {
            idx<n1D>(XH, n, m, l),
            idx<n1D>(YH, n, m, l),
            idx<n1D>(ZH, n, m, l)
        };

        lhs(rowIndices[XH], rowIndices[XH]) += lhsfac;
        lhs(rowIndices[YH], rowIndices[YH]) += lhsfac;
        lhs(rowIndices[ZH], rowIndices[ZH]) += lhsfac;
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename Scalar>
void momentum_dt_rhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& metric,
  double gamma_div_dt[3],
  const nodal_vector_view<poly_order, Scalar>& Gp,
  const nodal_scalar_view<poly_order, Scalar>& rhom1,
  const nodal_scalar_view<poly_order, Scalar>& rhop0,
  const nodal_scalar_view<poly_order, Scalar>& rhop1,
  const nodal_vector_view<poly_order, Scalar>& velm1,
  const nodal_vector_view<poly_order, Scalar>& velp0,
  const nodal_vector_view<poly_order, Scalar>& velp1,
  nodal_vector_view<poly_order, Scalar>& rhs)
{
  constexpr int n1D = poly_order + 1;

  nodal_vector_workview<poly_order, Scalar> work_drhoudt(0);
  auto& drhoudt = work_drhoudt.view();

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const Scalar local_vol = metric(k, j, i);
        const Scalar facp1 = gamma_div_dt[0] * rhop1(k, j, i);
        const Scalar facp0 = gamma_div_dt[1] * rhop0(k, j, i);
        const Scalar facm1 = gamma_div_dt[2] * rhom1(k, j, i);
        for (int d = 0; d < 3; ++d) {
          drhoudt(k, j, i, d) =
              -(facp1 * velp1(k, j, i, d)
              + facp0 * velp0(k, j, i, d)
              + facm1 * velm1(k, j, i, d)
              + Gp(k, j, i, d)
              ) * local_vol;
        }
      }
    }
  }
  ops.volume(drhoudt, rhs);
}

}
}
}

#endif
