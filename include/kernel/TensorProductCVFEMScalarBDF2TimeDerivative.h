/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderScalarBDF2TimeDerivativeQuad_h
#define HighOrderScalarBDF2TimeDerivativeQuad_h

#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/DirectionMacros.h>
#include <CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int poly_order, typename Scalar>
void scalar_dt_lhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& vol,
  double gamma1_div_dt,
  const nodal_scalar_view<poly_order, Scalar>& rho_p1,
  matrix_view<poly_order, Scalar>& lhs)
{
  constexpr int n1D = poly_order + 1;
  const auto& weight = ops.mat_.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      for (int l = 0; l < n1D; ++l) {
        auto rowIndex = idx<n1D>(n, m, l);
        for (int k = 0; k < n1D; ++k) {
          auto gammaWnk = gamma1_div_dt * weight(n, k);
          for (int j = 0; j < n1D; ++j) {
            auto gammWnkWmj = gammaWnk * weight(m, j);
            for (int i = 0; i < n1D; ++i) {
              lhs(rowIndex, idx<n1D>(k, j, i)) += gammWnkWmj * weight(l, i) * vol(k, j, i) * rho_p1(k, j, i);
            }
          }
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename Scalar>
void scalar_dt_rhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& metric,
  double gamma_div_dt[3],
  const nodal_scalar_view<poly_order, Scalar>& rhom1,
  const nodal_scalar_view<poly_order, Scalar>& rhop0,
  const nodal_scalar_view<poly_order, Scalar>& rhop1,
  const nodal_scalar_view<poly_order, Scalar>& phim1,
  const nodal_scalar_view<poly_order, Scalar>& phip0,
  const nodal_scalar_view<poly_order, Scalar>& phip1,
  nodal_scalar_view<poly_order, Scalar>& rhs)
{
  constexpr int n1D = poly_order + 1;

  nodal_scalar_workview<poly_order, Scalar> work_drhoudt(0);
  auto& drhoqdt = work_drhoudt.view();

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
          drhoqdt(k, j, i) = -(
                gamma_div_dt[0] * rhop1(k, j, i) * phip1(k, j, i)
              + gamma_div_dt[1] * rhop0(k, j, i) * phip0(k, j, i)
              + gamma_div_dt[2] * rhom1(k, j, i) * phim1(k, j, i)
              ) * metric(k, j, i);
      }
    }
  }
  ops.volume(drhoqdt, rhs);
}
//--------------------------------------------------------------------------
template <int poly_order, typename Scalar>
void density_dt_rhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& metric,
  double gamma_div_dt[3],
  const nodal_scalar_view<poly_order, Scalar>& rhom1,
  const nodal_scalar_view<poly_order, Scalar>& rhop0,
  const nodal_scalar_view<poly_order, Scalar>& rhop1,
  nodal_scalar_view<poly_order, Scalar>& rhs)
{
  constexpr int n1D = poly_order + 1;

  nodal_scalar_workview<poly_order, Scalar> work_drhodt(0);
  auto& drhodt = work_drhodt.view();

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
          drhodt(k, j, i) = -(
                gamma_div_dt[0] * rhop1(k, j, i)
              + gamma_div_dt[1] * rhop0(k, j, i)
              + gamma_div_dt[2] * rhom1(k, j, i)
              ) * metric(k, j, i);
      }
    }
  }
  ops.volume(drhodt, rhs);
}


}
}
}

#endif
