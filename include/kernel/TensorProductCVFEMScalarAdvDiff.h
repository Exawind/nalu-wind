/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductDiffusion_h
#define TensorProductDiffusion_h

#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/DirectionMacros.h>
#include <CVFEMTypeDefs.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int poly_order, typename Scalar>
void scalar_advdiff_xhat_contrib(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const scs_vector_view<poly_order, Scalar>& metric,
  matrix_view<poly_order, Scalar>& lhs,
  bool reduced_sens)
{
  constexpr int n1D = poly_order + 1;
  const auto& mat = ops.mat_;
  const auto& nodalWeights = (reduced_sens) ? mat.lumpedNodalWeights : mat.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      constexpr int l_minus = 0;

      auto rowIndexMinus = idx<n1D>(n, m, l_minus);
      for (int k = 0; k < n1D; ++k) {
        const Scalar Wnk = nodalWeights(n, k);

        for (int j = 0; j < n1D; ++j) {
          const Scalar WnkWmj = Wnk * nodalWeights(m, j);
          const Scalar orth = WnkWmj * metric(XH, k, j, l_minus, XH);
          const Scalar adv = WnkWmj * mdot(XH, k, j, l_minus);

          Scalar non_orth_y = 0.0;
          Scalar non_orth_z = 0.0;
          for (int q = 0; q < n1D; ++q) {
             non_orth_y += nodalWeights(n, k) * nodalWeights(m, q) * mat.nodalDeriv(q, j)
                      * metric(XH, k, q, l_minus, YH);
             non_orth_z += nodalWeights(n, q) * nodalWeights(m, j) * mat.nodalDeriv(q, k)
                      * metric(XH, q, j, l_minus, ZH);
           }

          for (int i = 0; i < n1D; ++i) {
            lhs(rowIndexMinus, idx<n1D>(k, j, i)) += orth * mat.scsDeriv(l_minus, i)
                          + mat.scsInterp(l_minus, i) * (-adv + non_orth_y + non_orth_z);
          }
        }
      }

      for (int l = 1; l < n1D - 1; ++l) {
        auto rowIndex = idx<n1D>(n, m, l);
        for (int k = 0; k < n1D; ++k) {
          const Scalar Wnk = nodalWeights(n, k);
          for (int j = 0; j < n1D; ++j) {
            const Scalar WnkWmj = Wnk * nodalWeights(m, j);
            const Scalar orthm1 = WnkWmj * metric(XH, k, j, l - 1, XH);
            const Scalar orthp0 = WnkWmj * metric(XH, k, j, l + 0, XH);
            const Scalar advm1 =  WnkWmj * mdot(XH, k, j, l - 1);
            const Scalar advp0 =  WnkWmj * mdot(XH, k, j, l + 0);

            Scalar non_orth_y_m1 = 0.0;
            Scalar non_orth_y_p0 = 0.0;
            Scalar non_orth_z_m1 = 0.0;
            Scalar non_orth_z_p0 = 0.0;

            for (int q = 0; q < n1D; ++q) {
              const Scalar wy = nodalWeights(n, k) * nodalWeights(m, q) * mat.nodalDeriv(q, j);
              non_orth_y_m1 += wy * metric(XH, k, q, l - 1, YH);
              non_orth_y_p0 += wy * metric(XH, k, q, l + 0, YH);

              const Scalar wz = nodalWeights(n, q) * nodalWeights(m, j) * mat.nodalDeriv(q, k);
              non_orth_z_m1 += wz * metric(XH, q, j, l - 1, ZH);
              non_orth_z_p0 += wz * metric(XH, q, j, l + 0, ZH);
            }

            for (int i = 0; i < n1D; ++i) {
              const Scalar integrated_flux_m = orthm1 * mat.scsDeriv(l - 1, i)
                                    + mat.scsInterp(l - 1, i) * (-advm1 + non_orth_y_m1 + non_orth_z_m1);
              const Scalar integrated_flux_p = orthp0 * mat.scsDeriv(l + 0, i)
                                    + mat.scsInterp(l + 0, i) * (-advp0 + non_orth_y_p0 + non_orth_z_p0);
              lhs(rowIndex, idx<n1D>(k, j, i)) += integrated_flux_p - integrated_flux_m;
            }
          }
        }
      }

      constexpr int l_plus = n1D - 1;
      auto rowIndexPlus = idx<n1D>(n, m, l_plus);
      for (int k = 0; k < n1D; ++k) {
        const Scalar Wnk = nodalWeights(n, k);
        for (int j = 0; j < n1D; ++j) {
          const Scalar WnkWmj = Wnk * nodalWeights(m, j);
          const Scalar orth = WnkWmj * metric(XH, k, j, l_plus - 1, XH);
          const Scalar adv = WnkWmj * mdot(XH, k, j, l_plus - 1);


          Scalar non_orth_y = 0.0;
          Scalar non_orth_z = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_y += nodalWeights(n, k) * nodalWeights(m, q) * mat.nodalDeriv(q, j)
                      * metric(XH, k, q, l_plus - 1, YH);
            non_orth_z += nodalWeights(n, q) * nodalWeights(m, j) * mat.nodalDeriv(q, k)
                      * metric(XH, q, j, l_plus - 1, ZH);
          }

          for (int i = 0; i < n1D; ++i) {
            lhs(rowIndexPlus, idx<n1D>(k, j, i)) -= orth * mat.scsDeriv(l_plus - 1, i)
                          + mat.scsInterp(l_plus - 1, i) * (-adv + non_orth_y + non_orth_z);
          }
        }
      }
    }
  }
}

template <int poly_order, typename Scalar>
void scalar_advdiff_yhat_contrib(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const scs_vector_view<poly_order, Scalar>& metric,
  matrix_view<poly_order, Scalar>& lhs,
  bool reduced_sens)
{
  constexpr int n1D = poly_order + 1;
  const auto& mat = ops.mat_;
  const auto& nodalWeights = (reduced_sens) ? mat.lumpedNodalWeights : mat.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int l = 0; l < n1D; ++l) {
      constexpr int m_minus = 0;
      auto rowIndexMinus = idx<n1D>(n, m_minus, l);
      for (int k = 0; k < n1D; ++k) {
        const Scalar Wnk = nodalWeights(n, k);
        for (int i = 0; i < n1D; ++i) {
          const Scalar WnkWli = Wnk * nodalWeights(l, i);
          const Scalar orth =  WnkWli * metric(YH,k, m_minus, i, YH);
          const Scalar adv = WnkWli * mdot(YH, k, m_minus, i);

          Scalar non_orth_x = 0.0;
          Scalar non_orth_z = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_x += nodalWeights(n, k) * nodalWeights(l, q) * mat.nodalDeriv(q, i)
                          * metric(YH,k, m_minus, q, XH);
            non_orth_z += nodalWeights(n, q) * nodalWeights(l, i) * mat.nodalDeriv(q, k)
                          * metric(YH,q, m_minus, i, ZH);
          }

          for (int j = 0; j < n1D; ++j) {
            lhs(rowIndexMinus, idx<n1D>(k, j, i)) += orth * mat.scsDeriv(m_minus, j)
                          + mat.scsInterp(m_minus, j) * (-adv + non_orth_x + non_orth_z);
          }
        }
      }

      for (int m = 1; m < n1D - 1; ++m) {
        auto rowIndex = idx<n1D>(n, m, l);
        for (int k = 0; k < n1D; ++k) {
          const Scalar Wnk = nodalWeights(n, k);
          for (int i = 0; i < n1D; ++i) {
            const Scalar WnkWli = Wnk * nodalWeights(l, i);
            const Scalar orthm1 = WnkWli * metric(YH, k, m - 1, i, YH);
            const Scalar orthp0 = WnkWli * metric(YH, k, m + 0, i, YH);
            const Scalar advm1 =  WnkWli * mdot(YH, k, m - 1, i);
            const Scalar advp0 =  WnkWli * mdot(YH, k, m + 0, i);

            Scalar non_orth_x_m1 = 0.0;
            Scalar non_orth_x_p0 = 0.0;
            Scalar non_orth_z_m1 = 0.0;
            Scalar non_orth_z_p0 = 0.0;

            for (int q = 0; q < n1D; ++q) {
              const Scalar wx = nodalWeights(n, k) * nodalWeights(l, q) * mat.nodalDeriv(q, i);
              non_orth_x_m1 += wx * metric(YH, k, m - 1, q, XH);
              non_orth_x_p0 += wx * metric(YH, k, m + 0, q, XH);

              const Scalar wz = nodalWeights(n, q) * nodalWeights(l, i) * mat.nodalDeriv(q, k);
              non_orth_z_m1 += wz * metric(YH, q, m - 1, i, ZH);
              non_orth_z_p0 += wz * metric(YH, q, m + 0, i, ZH);
            }

            for (int j = 0; j < n1D; ++j) {
              const Scalar integrated_flux_m = orthm1 * mat.scsDeriv(m - 1, j)
                            + mat.scsInterp(m - 1, j) * (-advm1 + non_orth_x_m1 + non_orth_z_m1);
              const Scalar integrated_flux_p = orthp0 * mat.scsDeriv(m + 0, j)
                            + mat.scsInterp(m + 0, j) * (-advp0 + non_orth_x_p0 + non_orth_z_p0);
              lhs(rowIndex, idx<n1D>(k, j, i)) += integrated_flux_p - integrated_flux_m;
            }
          }
        }
      }

      constexpr int m_plus = n1D - 1;
      auto rowIndexPlus = idx<n1D>(n, m_plus, l);
      for (int k = 0; k < n1D; ++k) {
        const Scalar Wnk = nodalWeights(n, k);
        for (int i = 0; i < n1D; ++i) {
          const Scalar WnkWli = Wnk * nodalWeights(l, i);
          const Scalar orth = WnkWli * metric(YH, k, m_plus - 1, i, YH);
          const Scalar adv = WnkWli * mdot(YH, k, m_plus - 1, i);


          Scalar non_orth_x = 0.0;
          Scalar non_orth_z = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_x += nodalWeights(n, k) * nodalWeights(l, q) * mat.nodalDeriv(q, i)
                          * metric(YH, k, m_plus - 1, q, XH);
            non_orth_z += nodalWeights(n, q) * nodalWeights(l, i) * mat.nodalDeriv(q, k)
                          * metric(YH, q, m_plus - 1, i, ZH);
          }

          for (int j = 0; j < n1D; ++j) {
            lhs(rowIndexPlus, idx<n1D>(k, j, i)) -= orth * mat.scsDeriv(m_plus - 1, j)
                          + mat.scsInterp(m_plus - 1, j) * (-adv + non_orth_x + non_orth_z);
          }
        }
      }
    }
  }
}

template <int poly_order, typename Scalar>
void scalar_advdiff_zhat_contrib(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const scs_vector_view<poly_order, Scalar>& metric,
  matrix_view<poly_order, Scalar>& lhs,
  bool reduced_sens)
{
  constexpr int n1D = poly_order + 1;
  const auto& mat = ops.mat_;
  const auto& nodalWeights = (reduced_sens) ? mat.lumpedNodalWeights : mat.nodalWeights;

  for (int m = 0; m < n1D; ++m) {
    for (int l = 0; l < n1D; ++l) {
      constexpr int n_minus = 0;
      auto rowIndexMinus = idx<n1D>(n_minus, m, l);
      for (int j = 0; j < n1D; ++j) {
        const Scalar Wmj = nodalWeights(m, j);
        for (int i = 0; i < n1D; ++i) {
          const Scalar WmjWli = Wmj * nodalWeights(l, i);
          const Scalar orth = WmjWli * metric(ZH, n_minus, j, i, ZH);
          const Scalar adv = WmjWli * mdot(ZH, n_minus, j, i);

          Scalar non_orth_x = 0.0;
          Scalar non_orth_y = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_x += nodalWeights(m, j) * nodalWeights(l, q) * mat.nodalDeriv(q, i)
                          * metric(ZH, n_minus, j, q, XH);
            non_orth_y += nodalWeights(m, q) * nodalWeights(l, i) * mat.nodalDeriv(q, j)
                          * metric(ZH, n_minus, q, i, YH);
          }

          for (int k = 0; k < n1D; ++k) {
            lhs(rowIndexMinus, idx<n1D>(k, j, i)) += orth * mat.scsDeriv(n_minus, k)
                          + mat.scsInterp(n_minus, k) * (-adv + non_orth_x + non_orth_y);
          }
        }
      }

      for (int n = 1; n < n1D - 1; ++n) {
        auto rowIndex = idx<n1D>(n, m, l);
        for (int j = 0; j < n1D; ++j) {
          const Scalar Wmj = nodalWeights(m, j);
          for (int i = 0; i < n1D; ++i) {
            const Scalar WmjWli = Wmj * nodalWeights(l, i);
            const Scalar orthm1 = WmjWli * metric(ZH, n - 1, j, i, ZH);
            const Scalar orthp0 = WmjWli * metric(ZH, n + 0, j, i, ZH);
            const Scalar advm1 = WmjWli * mdot(ZH, n - 1, j,  i);
            const Scalar advp0 = WmjWli * mdot(ZH, n + 0, j,  i);

            Scalar non_orth_x_m1 = 0.0;
            Scalar non_orth_x_p0 = 0.0;
            Scalar non_orth_y_m1 = 0.0;
            Scalar non_orth_y_p0 = 0.0;

            for (int q = 0; q < n1D; ++q) {
              const Scalar wx = nodalWeights(m, j) * nodalWeights(l, q) * mat.nodalDeriv(q, i);
              non_orth_x_m1 += wx * metric(ZH, n - 1, j, q, XH);
              non_orth_x_p0 += wx * metric(ZH, n + 0, j, q, XH);

              const Scalar wy = nodalWeights(m, q) * nodalWeights(l, i) * mat.nodalDeriv(q, j);
              non_orth_y_m1 += wy * metric(ZH, n - 1, q, i, YH);
              non_orth_y_p0 += wy * metric(ZH, n + 0, q, i, YH);
            }

            for (int k = 0; k < n1D; ++k) {
              const Scalar integrated_flux_m = orthm1 * mat.scsDeriv(n - 1, k)
                            + mat.scsInterp(n - 1, k) * (-advm1 + non_orth_x_m1 + non_orth_y_m1);
              const Scalar integrated_flux_p = orthp0 * mat.scsDeriv(n + 0, k)
                            + mat.scsInterp(n + 0, k) * (-advp0 + non_orth_x_p0 + non_orth_y_p0);

              lhs(rowIndex, idx<n1D>(k, j, i)) += integrated_flux_p - integrated_flux_m;
            }
          }
        }
      }

      constexpr int n_plus = n1D - 1;
      auto rowIndexPlus = idx<n1D>(n_plus, m, l);
      for (int j = 0; j < n1D; ++j) {
        const Scalar Wmj = nodalWeights(m, j);
        for (int i = 0; i < n1D; ++i) {
          const Scalar WmjWli = Wmj * nodalWeights(l, i);
          const Scalar orth = WmjWli * metric(ZH, n_plus - 1, j, i, ZH);
          const Scalar adv =  WmjWli * mdot(ZH, n_plus - 1, j, i);

          Scalar non_orth_x = 0.0;
          Scalar non_orth_y = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_x += nodalWeights(m, j) * nodalWeights(l, q) * mat.nodalDeriv(q, i)
                      * metric(ZH, n_plus - 1, j, q, XH);
            non_orth_y += nodalWeights(m, q) * nodalWeights(l, i) * mat.nodalDeriv(q, j)
                      * metric(ZH, n_plus - 1, q, i, YH);
          }

          for (int k = 0; k < n1D; ++k) {
            lhs(rowIndexPlus, idx<n1D>(k, j, i)) -= orth * mat.scsDeriv(n_plus - 1, k)
                          + mat.scsInterp(n_plus - 1, k) * (-adv + non_orth_x + non_orth_y);
          }
        }
      }
    }
  }
}

template <int poly_order, typename Scalar>
void scalar_advdiff_lhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const scs_vector_view<poly_order, Scalar>& metric,
  matrix_view<poly_order, Scalar>& lhs,
  bool reduced_sens = false)
{
  scalar_advdiff_xhat_contrib(ops, mdot, metric, lhs, reduced_sens);
  scalar_advdiff_yhat_contrib(ops, mdot, metric, lhs, reduced_sens);
  scalar_advdiff_zhat_contrib(ops, mdot, metric, lhs, reduced_sens);
}

template <int poly_order, typename Scalar>
void scalar_advdiff_rhs(
  CVFEMOperators<poly_order, Scalar>& ops,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const scs_vector_view<poly_order, Scalar>& metric,
  const nodal_scalar_view<poly_order, Scalar>& scalar,
  nodal_scalar_view<poly_order, Scalar>& rhs)
{
  constexpr int n1D = poly_order + 1;
  constexpr int nscs = poly_order;

  nodal_scalar_workview<poly_order, Scalar> work_phi_scs(0);
  auto& phi_scs = work_phi_scs.view();

  nodal_vector_workview<poly_order, Scalar> work_grad_phi_scs(0);
  auto& grad_phi_scs = work_grad_phi_scs.view();

  nodal_scalar_workview<poly_order, Scalar> work_integrand(0);
  auto& v_integrand = work_integrand.view();

  ops.scs_xhat_interp(scalar, phi_scs);
  ops.scs_xhat_grad(scalar, grad_phi_scs);
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < nscs; ++i) {
        v_integrand(k,j,i) = metric(XH, k, j, i, XH) * grad_phi_scs(k, j, i, XH)
                           + metric(XH, k, j, i, YH) * grad_phi_scs(k, j, i, YH)
                           + metric(XH, k, j, i, ZH) * grad_phi_scs(k, j, i, ZH)
                           - mdot(XH, k, j, i) * phi_scs(k, j, i);
      }
    }
  }
  ops.integrate_and_diff_xhat(v_integrand, rhs);

  ops.scs_yhat_interp(scalar, phi_scs);
  ops.scs_yhat_grad(scalar, grad_phi_scs);
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < nscs; ++j) {
      for (int i = 0; i < n1D; ++i) {
        v_integrand(k, j, i) = metric(YH, k, j, i, XH) * grad_phi_scs(k, j, i, XH)
                             + metric(YH, k, j, i, YH) * grad_phi_scs(k, j, i, YH)
                             + metric(YH, k, j, i, ZH) * grad_phi_scs(k, j, i, ZH)
                             - mdot(YH, k, j, i) * phi_scs(k, j, i);
      }
    }
  }
  ops.integrate_and_diff_yhat(v_integrand, rhs);

  ops.scs_zhat_interp(scalar, phi_scs);
  ops.scs_zhat_grad(scalar, grad_phi_scs);
  for (int k = 0; k < nscs; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        v_integrand(k, j, i) = metric(ZH, k, j, i, XH) * grad_phi_scs(k, j, i, XH)
                             + metric(ZH, k, j, i, YH) * grad_phi_scs(k, j, i, YH)
                             + metric(ZH, k, j, i, ZH) * grad_phi_scs(k, j, i, ZH)
                             - mdot(ZH, k, j, i) * phi_scs(k, j, i);
      }
    }
  }
  ops.integrate_and_diff_zhat(v_integrand, rhs);
}

} // namespace HighOrderLaplacianQuad
} // namespace naluUnit
} // namespace Sierra

#endif
