/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductCVFEMAdvection_h
#define TensorProductCVFEMAdvection_h

#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/Hex8GeometryFunctions.h>
#include <master_element/DirectionMacros.h>
#include <CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int poly_order, typename Scalar>
void momentum_advdiff_xhat_contrib(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const scs_vector_view<poly_order, Scalar>& metric,
  matrix_vector_view<poly_order, Scalar>& lhs,
  bool reduced_sens = false)
{
  constexpr int n1D = poly_order + 1;
  const auto& mat = ops.mat_;

  const auto& nodalWeights = (reduced_sens) ? mat.lumpedNodalWeights : mat.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      constexpr int l_minus = 0;
      for (int k = 0; k < n1D; ++k) {
        const Scalar Wnk = nodalWeights(n, k);

        for (int j = 0; j < n1D; ++j) {
          const Scalar WnkWmj = Wnk * nodalWeights(m, j);
          const Scalar orth = WnkWmj * metric(XH, k, j, l_minus, XH);
          const Scalar adv = WnkWmj * mdot(XH, k, j, l_minus);

          Scalar non_orth_y = 0.0;
          Scalar non_orth_z = 0.0;
          for (int q = 0; q < n1D; ++q) {
             non_orth_y += mat.nodalWeights(n, k) * mat.nodalWeights(m, q) * mat.nodalDeriv(q, j)
                      * metric(XH, k, q, l_minus, YH);
             non_orth_z += mat.nodalWeights(n, q) * mat.nodalWeights(m, j) * mat.nodalDeriv(q, k)
                      * metric(XH, q, j, l_minus, ZH);
           }

          for (int i = 0; i < n1D; ++i) {
            const Scalar fp = orth * mat.scsDeriv(l_minus, i)
                              + mat.scsInterp(l_minus, i) * (-adv + non_orth_y + non_orth_z);
            for (int d = 0; d < 3; ++d) {
              lhs(idx<n1D>(d, n, m, l_minus), idx<n1D>(d, k, j, i)) += fp;
            }
          }
        }
      }

      for (int l = 1; l < n1D - 1; ++l) {
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
              const Scalar wy = mat.nodalWeights(n, k) * mat.nodalWeights(m, q) * mat.nodalDeriv(q, j);
              non_orth_y_m1 += wy * metric(XH, k, q, l - 1, YH);
              non_orth_y_p0 += wy * metric(XH, k, q, l + 0, YH);

              const Scalar wz = mat.nodalWeights(n, q) * mat.nodalWeights(m, j) * mat.nodalDeriv(q, k);
              non_orth_z_m1 += wz * metric(XH, q, j, l - 1, ZH);
              non_orth_z_p0 += wz * metric(XH, q, j, l + 0, ZH);
            }

            for (int i = 0; i < n1D; ++i) {
              const Scalar integrated_flux_m = orthm1 * mat.scsDeriv(l - 1, i)
                                    + mat.scsInterp(l - 1, i) * (-advm1 + non_orth_y_m1 + non_orth_z_m1);
              const Scalar integrated_flux_p = orthp0 * mat.scsDeriv(l + 0, i)
                                    + mat.scsInterp(l + 0, i) * (-advp0 + non_orth_y_p0 + non_orth_z_p0);

              const Scalar flux_diff = integrated_flux_p - integrated_flux_m;
              lhs(idx<n1D>(XH, n, m, l), idx<n1D>(XH, k, j, i)) += flux_diff;
              lhs(idx<n1D>(YH, n, m, l), idx<n1D>(YH, k, j, i)) += flux_diff;
              lhs(idx<n1D>(ZH, n, m, l), idx<n1D>(ZH, k, j, i)) += flux_diff;
            }
          }
        }
      }

      constexpr int l_plus = n1D - 1;
      for (int k = 0; k < n1D; ++k) {
        const Scalar Wnk = nodalWeights(n, k);
        for (int j = 0; j < n1D; ++j) {
          const Scalar WnkWmj = Wnk * nodalWeights(m, j);
          const Scalar orth = WnkWmj * metric(XH, k, j, l_plus - 1, XH);
          const Scalar adv = WnkWmj * mdot(XH, k, j, l_plus - 1);

          Scalar non_orth_y = 0.0;
          Scalar non_orth_z = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_y += mat.nodalWeights(n, k) * mat.nodalWeights(m, q) * mat.nodalDeriv(q, j)
                      * metric(XH, k, q, l_plus - 1, YH);
            non_orth_z += mat.nodalWeights(n, q) * mat.nodalWeights(m, j) * mat.nodalDeriv(q, k)
                      * metric(XH, q, j, l_plus - 1, ZH);
          }

          for (int i = 0; i < n1D; ++i) {
            const Scalar fm = orth * mat.scsDeriv(l_plus - 1, i)
                                  + mat.scsInterp(l_plus - 1, i) * (-adv + non_orth_y + non_orth_z);
            for (int d = 0; d < 3; ++d) {
              lhs(idx<n1D>(d, n, m, l_plus), idx<n1D>(d, k, j, i)) -= fm;
            }
          }
        }
      }
    }
  }
}

template <int poly_order, typename Scalar>
void momentum_advdiff_yhat_contrib(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const scs_vector_view<poly_order, Scalar>& metric,
  matrix_vector_view<poly_order, Scalar>& lhs,
  bool reduced_sens = false)
{
  constexpr int n1D = poly_order + 1;
  const auto& mat = ops.mat_;
  const auto& nodalWeights = (reduced_sens) ? mat.lumpedNodalWeights : mat.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int l = 0; l < n1D; ++l) {
      constexpr int m_minus = 0;
      for (int k = 0; k < n1D; ++k) {
        const Scalar Wnk = nodalWeights(n, k);
        for (int i = 0; i < n1D; ++i) {
          const Scalar WnkWli = Wnk * nodalWeights(l, i);
          const Scalar orth =  WnkWli * metric(YH,k, m_minus, i, YH);
          const Scalar adv = WnkWli * mdot(YH, k, m_minus, i);

          Scalar non_orth_x = 0.0;
          Scalar non_orth_z = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_x += mat.nodalWeights(n, k) * mat.nodalWeights(l, q) * mat.nodalDeriv(q, i)
                          * metric(YH,k, m_minus, q, XH);
            non_orth_z += mat.nodalWeights(n, q) * mat.nodalWeights(l, i) * mat.nodalDeriv(q, k)
                          * metric(YH,q, m_minus, i, ZH);
          }
          for (int j = 0; j < n1D; ++j) {
            const auto fp = orth * mat.scsDeriv(m_minus, j)
                              + mat.scsInterp(m_minus, j) * (-adv + non_orth_x + non_orth_z);
            for (int d = 0; d < 3; ++d) {
              lhs(idx<n1D>(d, n, m_minus, l), idx<n1D>(d, k, j, i)) += fp;
            }
          }
        }
      }

      for (int m = 1; m < n1D - 1; ++m) {
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
              const Scalar wx = mat.nodalWeights(n, k) * mat.nodalWeights(l, q) * mat.nodalDeriv(q, i);
              non_orth_x_m1 += wx * metric(YH, k, m - 1, q, XH);
              non_orth_x_p0 += wx * metric(YH, k, m + 0, q, XH);

              const Scalar wz = mat.nodalWeights(n, q) * mat.nodalWeights(l, i) * mat.nodalDeriv(q, k);
              non_orth_z_m1 += wz * metric(YH, q, m - 1, i, ZH);
              non_orth_z_p0 += wz * metric(YH, q, m + 0, i, ZH);
            }

            for (int j = 0; j < n1D; ++j) {
              const Scalar integrated_flux_m = orthm1 * mat.scsDeriv(m - 1, j)
                            + mat.scsInterp(m - 1, j) * (-advm1 + non_orth_x_m1 + non_orth_z_m1);
              const Scalar integrated_flux_p = orthp0 * mat.scsDeriv(m + 0, j)
                            + mat.scsInterp(m + 0, j) * (-advp0 + non_orth_x_p0 + non_orth_z_p0);
              const Scalar flux_diff = integrated_flux_p - integrated_flux_m;
              lhs(idx<n1D>(XH, n, m, l), idx<n1D>(XH, k, j, i)) += flux_diff;
              lhs(idx<n1D>(YH, n, m, l), idx<n1D>(YH, k, j, i)) += flux_diff;
              lhs(idx<n1D>(ZH, n, m, l), idx<n1D>(ZH, k, j, i)) += flux_diff;
            }
          }
        }
      }

      constexpr int m_plus = n1D - 1;
      for (int k = 0; k < n1D; ++k) {
        const Scalar Wnk = nodalWeights(n, k);
        for (int i = 0; i < n1D; ++i) {
          const Scalar WnkWli = Wnk * nodalWeights(l, i);
          const Scalar orth = WnkWli * metric(YH, k, m_plus - 1, i, YH);
          const Scalar adv = WnkWli * mdot(YH, k, m_plus - 1, i);

          Scalar non_orth_x = 0.0;
          Scalar non_orth_z = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_x += mat.nodalWeights(n, k) * mat.nodalWeights(l, q) * mat.nodalDeriv(q, i)
                          * metric(YH, k, m_plus - 1, q, XH);
            non_orth_z += mat.nodalWeights(n, q) * mat.nodalWeights(l, i) * mat.nodalDeriv(q, k)
                          * metric(YH, q, m_plus - 1, i, ZH);
          }

          for (int j = 0; j < n1D; ++j) {
            const Scalar fm = orth * mat.scsDeriv(m_plus - 1, j)
                                  + mat.scsInterp(m_plus - 1, j) * (-adv + non_orth_x + non_orth_z);
            for (int d = 0; d < 3; ++d) {
              lhs(idx<n1D>(d, n, m_plus, l), idx<n1D>(d, k, j, i)) -= fm;
            }
          }
        }
      }
    }
  }
}

template <int poly_order, typename Scalar>
void momentum_advdiff_zhat_contrib(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const scs_vector_view<poly_order, Scalar>& metric,
  matrix_vector_view<poly_order, Scalar>& lhs,
  bool reduced_sens = false)
{
  constexpr int n1D = poly_order + 1;
  const auto& mat = ops.mat_;

  const auto& nodalWeights = (reduced_sens) ? mat.lumpedNodalWeights : mat.nodalWeights;

  for (int m = 0; m < n1D; ++m) {
    for (int l = 0; l < n1D; ++l) {
      constexpr int n_minus = 0;
      for (int j = 0; j < n1D; ++j) {
        const Scalar Wmj = nodalWeights(m, j);
        for (int i = 0; i < n1D; ++i) {
          const Scalar WmjWli = Wmj * nodalWeights(l, i);
          const Scalar orth = WmjWli * metric(ZH, n_minus, j, i, ZH);
          const Scalar adv = WmjWli * mdot(ZH, n_minus, j, i);


          Scalar non_orth_x = 0.0;
          Scalar non_orth_y = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_x += mat.nodalWeights(m, j) * mat.nodalWeights(l, q) * mat.nodalDeriv(q, i)
                          * metric(ZH, n_minus, j, q, XH);
            non_orth_y += mat.nodalWeights(m, q) * mat.nodalWeights(l, i) * mat.nodalDeriv(q, j)
                          * metric(ZH, n_minus, q, i, YH);
          }

          for (int k = 0; k < n1D; ++k) {
            const Scalar fp = orth * mat.scsDeriv(n_minus, k)
                              + mat.scsInterp(n_minus, k) * (-adv + non_orth_x + non_orth_y);

            for (int d = 0; d < 3; ++d) {
              lhs(idx<n1D>(d, n_minus, m, l), idx<n1D>(d, k, j, i)) += fp;
            }
          }
        }
      }

      for (int n = 1; n < n1D - 1; ++n) {
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
              const Scalar wx = mat.nodalWeights(m, j) * mat.nodalWeights(l, q) * mat.nodalDeriv(q, i);
              non_orth_x_m1 += wx * metric(ZH, n - 1, j, q, XH);
              non_orth_x_p0 += wx * metric(ZH, n + 0, j, q, XH);

              const Scalar wy = mat.nodalWeights(m, q) * mat.nodalWeights(l, i) * mat.nodalDeriv(q, j);
              non_orth_y_m1 += wy * metric(ZH, n - 1, q, i, YH);
              non_orth_y_p0 += wy * metric(ZH, n + 0, q, i, YH);
            }

            for (int k = 0; k < n1D; ++k) {
              const Scalar integrated_flux_m = orthm1 * mat.scsDeriv(n - 1, k)
                            + mat.scsInterp(n - 1, k) * (-advm1 + non_orth_x_m1 + non_orth_y_m1);
              const Scalar integrated_flux_p = orthp0 * mat.scsDeriv(n + 0, k)
                            + mat.scsInterp(n + 0, k) * (-advp0 + non_orth_x_p0 + non_orth_y_p0);

              const Scalar flux_diff = integrated_flux_p - integrated_flux_m;
              lhs(idx<n1D>(XH, n, m, l), idx<n1D>(XH, k, j, i)) += flux_diff;
              lhs(idx<n1D>(YH, n, m, l), idx<n1D>(YH, k, j, i)) += flux_diff;
              lhs(idx<n1D>(ZH, n, m, l), idx<n1D>(ZH, k, j, i)) += flux_diff;
            }
          }
        }
      }

      constexpr int n_plus = n1D - 1;
      for (int j = 0; j < n1D; ++j) {
        const Scalar Wmj = nodalWeights(m, j);
        for (int i = 0; i < n1D; ++i) {
          const Scalar WmjWli = Wmj * nodalWeights(l, i);
          const Scalar orth = WmjWli * metric(ZH, n_plus - 1, j, i, ZH);
          const Scalar adv =  WmjWli * mdot(ZH, n_plus - 1, j, i);

          Scalar non_orth_x = 0.0;
          Scalar non_orth_y = 0.0;
          for (int q = 0; q < n1D; ++q) {
            non_orth_x += mat.nodalWeights(m, j) * mat.nodalWeights(l, q) * mat.nodalDeriv(q, i)
                      * metric(ZH, n_plus - 1, j, q, XH);
            non_orth_y += mat.nodalWeights(m, q) * mat.nodalWeights(l, i) * mat.nodalDeriv(q, j)
                      * metric(ZH, n_plus - 1, q, i, YH);
          }
          for (int k = 0; k < n1D; ++k) {
            const Scalar fm = orth * mat.scsDeriv(n_plus - 1, k)
                                  + mat.scsInterp(n_plus - 1, k) * (-adv + non_orth_x + non_orth_y);
            for (int d = 0; d < 3; ++d) {
              lhs(idx<n1D>(d, n_plus, m, l), idx<n1D>(d, k, j, i)) -= fm;
            }
          }
        }
      }
    }
  }
}

template <int p, typename Scalar>
void area_weighted_face_normal_shear_stress(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_vector_view<p, Scalar>& xc,
  const nodal_scalar_view<p, Scalar>& visc,
  const nodal_vector_view<p, Scalar>& vel,
  scs_vector_view<p, Scalar>& tau_dot_a)
{
  // haven't decided on a more general way to separate out the space transformation
  // from the assembly

  NALU_ALIGNED Scalar base_box[3][8];
  hex_vertex_coordinates<p, Scalar>(xc, base_box);

  const auto& nodalInterp = ops.mat_.linearNodalInterp;
  const auto& scsInterp = ops.mat_.linearScsInterp;

  nodal_tensor_workview<p, Scalar> work_gradh(0);
  auto& gradu_scs = work_gradh.view();

  nodal_scalar_workview<p, Scalar> work_visc_scs(0);
  auto& visc_scs = work_visc_scs.view();

  ops.scs_xhat_grad(vel, gradu_scs);
  ops.scs_xhat_interp(visc, visc_scs);
  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { scsInterp(0, i), scsInterp(1, i) };

        NALU_ALIGNED Scalar jact[3][3];
        hex_jacobian_t(base_box, interpi, interpj, interpk, jact);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jact, adjJac);

        NALU_ALIGNED Scalar local_grad[3][3];
        for (int d = 0; d < 3; ++d) {
          local_grad[d][XH] = adjJac[XH][XH] * gradu_scs(k, j, i, d, XH)
              + adjJac[XH][YH] * gradu_scs(k, j, i, d, YH) + adjJac[XH][ZH] * gradu_scs(k, j, i, d, ZH);
          local_grad[d][YH] = adjJac[YH][XH] * gradu_scs(k, j, i, d, XH)
              + adjJac[YH][YH] * gradu_scs(k, j, i, d, YH) + adjJac[YH][ZH] * gradu_scs(k, j, i, d, ZH);
          local_grad[d][ZH] = adjJac[ZH][XH] * gradu_scs(k, j, i, d, XH)
              + adjJac[ZH][YH] * gradu_scs(k, j, i, d, YH) + adjJac[ZH][ZH] * gradu_scs(k, j, i, d, ZH);
        }

        NALU_ALIGNED Scalar areav[3];
        areav_from_jacobian_t<XH>(jact, areav);

        const Scalar inv_detj = 2 * visc_scs(k, j, i) /
            (jact[0][0] * adjJac[0][0] + jact[1][0] * adjJac[1][0] + jact[2][0] * adjJac[2][0]);

        areav[XH] *= inv_detj;
        areav[YH] *= inv_detj;
        areav[ZH] *= inv_detj;

        const Scalar one_third_divu = 1.0/3.0 * (
            local_grad[XH][XH]  + local_grad[YH][YH] + local_grad[ZH][ZH]);

        tau_dot_a(XH, k, j, i, XH) =
            (local_grad[XH][XH] - one_third_divu) * areav[XH]
            + 0.5 * (local_grad[XH][YH] + local_grad[YH][XH]) * areav[YH]
            + 0.5 * (local_grad[XH][ZH] + local_grad[ZH][XH]) * areav[ZH];

        tau_dot_a(XH, k, j, i, YH) =
            0.5 * (local_grad[YH][XH] + local_grad[XH][YH]) * areav[XH]
          + (local_grad[YH][YH] - one_third_divu) * areav[YH]
          + 0.5 * (local_grad[YH][ZH] + local_grad[ZH][YH]) * areav[ZH];

        tau_dot_a(XH, k, j, i, ZH) =
            0.5 * (local_grad[XH][ZH] + local_grad[ZH][XH]) * areav[XH]
          + 0.5 * (local_grad[ZH][YH] + local_grad[YH][ZH]) * areav[YH]
          + (local_grad[ZH][ZH] - one_third_divu)* areav[ZH];
      }
    }
  }

  ops.scs_yhat_grad(vel, gradu_scs);
  ops.scs_yhat_interp(visc, visc_scs);
  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { scsInterp(0, j), scsInterp(1, j) };
      for (int i = 0; i < p+1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jact[3][3];
        hex_jacobian_t(base_box, interpi, interpj, interpk, jact);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jact, adjJac);

        NALU_ALIGNED Scalar local_grad[3][3];
        for (int d = 0; d < 3; ++d) {
          local_grad[d][XH] = adjJac[XH][XH] * gradu_scs(k, j, i, d, XH)
              + adjJac[XH][YH] * gradu_scs(k, j, i, d, YH) + adjJac[XH][ZH] * gradu_scs(k, j, i, d, ZH);
          local_grad[d][YH] = adjJac[YH][XH] * gradu_scs(k, j, i, d, XH)
              + adjJac[YH][YH] * gradu_scs(k, j, i, d, YH) + adjJac[YH][ZH] * gradu_scs(k, j, i, d, ZH);
          local_grad[d][ZH] = adjJac[ZH][XH] * gradu_scs(k, j, i, d, XH)
              + adjJac[ZH][YH] * gradu_scs(k, j, i, d, YH) + adjJac[ZH][ZH] * gradu_scs(k, j, i, d, ZH);
        }

        NALU_ALIGNED Scalar areav[3];
        areav_from_jacobian_t<YH>(jact, areav);

        const Scalar inv_detj = 2 * visc_scs(k, j, i) /
            (jact[0][0] * adjJac[0][0] + jact[1][0] * adjJac[1][0] + jact[2][0] * adjJac[2][0]);

        areav[XH] *= inv_detj;
        areav[YH] *= inv_detj;
        areav[ZH] *= inv_detj;

        const Scalar one_third_divu = 1.0/3.0 * (
            local_grad[XH][XH]  + local_grad[YH][YH] + local_grad[ZH][ZH]);

        tau_dot_a(YH, k, j, i, XH) =
            (local_grad[XH][XH] - one_third_divu) * areav[XH]
            + 0.5 * (local_grad[XH][YH] + local_grad[YH][XH]) * areav[YH]
            + 0.5 * (local_grad[XH][ZH] + local_grad[ZH][XH]) * areav[ZH];

        tau_dot_a(YH, k, j, i, YH) =
            0.5 * (local_grad[YH][XH] + local_grad[XH][YH]) * areav[XH]
          + (local_grad[YH][YH] - one_third_divu) * areav[YH]
          + 0.5 * (local_grad[YH][ZH] + local_grad[ZH][YH]) * areav[ZH];

        tau_dot_a(YH, k, j, i, ZH) =
            0.5 * (local_grad[XH][ZH] + local_grad[ZH][XH]) * areav[XH]
          + 0.5 * (local_grad[ZH][YH] + local_grad[YH][ZH]) * areav[YH]
          + (local_grad[ZH][ZH] - one_third_divu) * areav[ZH];
      }
    }
  }

  ops.scs_zhat_grad(vel, gradu_scs);
  ops.scs_zhat_interp(visc, visc_scs);
  for (int k = 0; k < p ; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { scsInterp(0, k), scsInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p + 1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jact[3][3];
        hex_jacobian_t(base_box, interpi, interpj, interpk, jact);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jact, adjJac);

        NALU_ALIGNED Scalar local_grad[3][3];
        for (int d = 0; d < 3; ++d) {
          local_grad[d][XH] = adjJac[XH][XH] * gradu_scs(k, j, i, d, XH)
              + adjJac[XH][YH] * gradu_scs(k, j, i, d, YH) + adjJac[XH][ZH] * gradu_scs(k, j, i, d, ZH);
          local_grad[d][YH] = adjJac[YH][XH] * gradu_scs(k, j, i, d, XH)
              + adjJac[YH][YH] * gradu_scs(k, j, i, d, YH) + adjJac[YH][ZH] * gradu_scs(k, j, i, d, ZH);
          local_grad[d][ZH] = adjJac[ZH][XH] * gradu_scs(k, j, i, d, XH)
              + adjJac[ZH][YH] * gradu_scs(k, j, i, d, YH) + adjJac[ZH][ZH] * gradu_scs(k, j, i, d, ZH);
        }

        NALU_ALIGNED Scalar areav[3];
        areav_from_jacobian_t<ZH>(jact, areav);
        const Scalar inv_detj = 2 * visc_scs(k, j, i) /
            (jact[0][0] * adjJac[0][0] + jact[1][0] * adjJac[1][0] + jact[2][0] * adjJac[2][0]);

        areav[XH] *= inv_detj;
        areav[YH] *= inv_detj;
        areav[ZH] *= inv_detj;

        const Scalar one_third_divu = 1.0/3.0 * (
            local_grad[XH][XH]  + local_grad[YH][YH] + local_grad[ZH][ZH]);

        tau_dot_a(ZH, k, j, i, XH) =
            (local_grad[XH][XH] - one_third_divu) * areav[XH]
            + 0.5 * (local_grad[XH][YH] + local_grad[YH][XH]) * areav[YH]
            + 0.5 * (local_grad[XH][ZH] + local_grad[ZH][XH]) * areav[ZH];

        tau_dot_a(ZH, k, j, i, YH) =
            0.5 * (local_grad[YH][XH] + local_grad[XH][YH]) * areav[XH]
          + (local_grad[YH][YH] - one_third_divu) * areav[YH]
          + 0.5 * (local_grad[YH][ZH] + local_grad[ZH][YH]) * areav[ZH];

        tau_dot_a(ZH, k, j, i, ZH) =
            0.5 * (local_grad[XH][ZH] + local_grad[ZH][XH]) * areav[XH]
          + 0.5 * (local_grad[ZH][YH] + local_grad[YH][ZH]) * areav[YH]
          + (local_grad[ZH][ZH] - one_third_divu) * areav[ZH];
      }
    }
  }
}

template <int poly_order, typename Scalar>
void momentum_advdiff_lhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const scs_vector_view<poly_order, Scalar>& metric,
  matrix_vector_view<poly_order, Scalar>& lhs,
  bool reduced_sens = false)
{
  momentum_advdiff_xhat_contrib(ops, mdot, metric, lhs, reduced_sens);
  momentum_advdiff_yhat_contrib(ops, mdot, metric, lhs, reduced_sens);
  momentum_advdiff_zhat_contrib(ops, mdot, metric, lhs, reduced_sens);
}

template <int poly_order, typename Scalar>
void momentum_advdiff_rhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_vector_view<poly_order, Scalar>& tau_dot_a,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  const nodal_vector_view<poly_order, Scalar>& velocity,
  nodal_vector_view<poly_order, Scalar>& rhs)
{
  constexpr int n1D = poly_order + 1;
  constexpr int nscs = poly_order;

  nodal_vector_workview<poly_order, Scalar> work_integrand(0);
  auto& integrand = work_integrand.view();

  nodal_vector_workview<poly_order, Scalar> work_vel_scs(0);
  auto& vel_scs = work_vel_scs.view();

  ops.scs_xhat_interp(velocity, vel_scs);
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < nscs; ++i) {
        integrand(k, j, i, XH) = tau_dot_a(XH, k, j, i, XH) - mdot(XH, k, j, i) * vel_scs(k, j, i, XH);
        integrand(k, j, i, YH) = tau_dot_a(XH, k, j, i, YH) - mdot(XH, k, j, i) * vel_scs(k, j, i, YH);
        integrand(k, j, i, ZH) = tau_dot_a(XH, k, j, i, ZH) - mdot(XH, k, j, i) * vel_scs(k, j, i, ZH);
      }
    }
  }
  ops.integrate_and_diff_xhat(integrand, rhs);

  ops.scs_yhat_interp(velocity, vel_scs);
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < nscs; ++j) {
      for (int i = 0; i < n1D; ++i) {
        integrand(k, j, i, XH) = tau_dot_a(YH, k, j, i, XH) - mdot(YH, k, j, i) * vel_scs(k, j, i, XH);
        integrand(k, j, i, YH) = tau_dot_a(YH, k, j, i, YH) - mdot(YH, k, j, i) * vel_scs(k, j, i, YH);
        integrand(k, j, i, ZH) = tau_dot_a(YH, k, j, i, ZH) - mdot(YH, k, j, i) * vel_scs(k, j, i, ZH);
      }
    }
  }
  ops.integrate_and_diff_yhat(integrand, rhs);

  ops.scs_zhat_interp(velocity, vel_scs);
  for (int k = 0; k < nscs; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        integrand(k, j, i, XH) = tau_dot_a(ZH, k, j, i, XH) - mdot(ZH, k, j, i) * vel_scs(k, j, i, XH);
        integrand(k, j, i, YH) = tau_dot_a(ZH, k, j, i, YH) - mdot(ZH, k, j, i) * vel_scs(k, j, i, YH);
        integrand(k, j, i, ZH) = tau_dot_a(ZH, k, j, i, ZH) - mdot(ZH, k, j, i) * vel_scs(k, j, i, ZH);
      }
    }
  }
  ops.integrate_and_diff_zhat(integrand, rhs);
}

}
}
}
#endif
