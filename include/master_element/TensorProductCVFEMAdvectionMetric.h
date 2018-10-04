/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductCVFEMAdvectionMetric_h
#define TensorProductCVFEMAdvectionMetric_h

#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/DirectionMacros.h>
#include <master_element/TensorOps.h>
#include <master_element/Hex8GeometryFunctions.h>
#include <AlgTraits.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {
namespace high_order_metrics {

  template <int p, typename Scalar>
  void compute_area_linear(
    const CVFEMOperators<p, Scalar>& ops,
    const nodal_vector_view<p, Scalar>& xc,
    scs_vector_view<p, Scalar>& metric)
  {
    const auto& nodalInterp = ops.mat_.linearNodalInterp;
    const auto& scsInterp = ops.mat_.linearScsInterp;

    NALU_ALIGNED Scalar base_box[3][8];
    hex_vertex_coordinates<p, Scalar>(xc, base_box);

    for (int k = 0; k < p + 1; ++k) {
      NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
      for (int j = 0; j < p + 1; ++j) {
        NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
        for (int i = 0; i < p; ++i) {
          NALU_ALIGNED const Scalar interpi[2] = { scsInterp(0, i), scsInterp(1, i) };
          NALU_ALIGNED Scalar areav[3];
          hex_areav_x(base_box, interpi, interpj, interpk, areav);

          metric(XH, k, j, i, XH) = areav[XH];
          metric(XH, k, j, i, YH) = areav[YH];
          metric(XH, k, j, i, ZH) = areav[ZH];
        }
      }
    }

    for (int k = 0; k < p + 1; ++k) {
      NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
      for (int j = 0; j < p; ++j) {
        NALU_ALIGNED const Scalar interpj[2] = { scsInterp(0, j), scsInterp(1, j) };
        for (int i = 0; i < p + 1; ++i) {
          NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };
          NALU_ALIGNED Scalar areav[3];
          hex_areav_y(base_box, interpi, interpj, interpk, areav);

          metric(YH, k, j, i, XH) = areav[XH];
          metric(YH, k, j, i, YH) = areav[YH];
          metric(YH, k, j, i, ZH) = areav[ZH];
        }
      }
    }

    for (int k = 0; k < p ; ++k) {
      NALU_ALIGNED const Scalar interpk[2] = { scsInterp(0, k), scsInterp(1, k) };
      for (int j = 0; j < p + 1; ++j) {
        NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
        for (int i = 0; i < p + 1; ++i) {
          NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };
          NALU_ALIGNED Scalar areav[3];
          hex_areav_z(base_box, interpi, interpj, interpk, areav);

          metric(ZH, k, j, i, XH) = areav[XH];
          metric(ZH, k, j, i, YH) = areav[YH];
          metric(ZH, k, j, i, ZH) = areav[ZH];
        }
      }
    }
  }

  template <int p, typename Scalar>
    void compute_mdot_linear(
      const CVFEMOperators<p, Scalar>& ops,
      const nodal_vector_view<p, Scalar>& xc,
      const scs_vector_view<p, Scalar>& laplacian_metric,
      double projTimeScale,
      const nodal_scalar_view<p, Scalar>& density,
      const nodal_vector_view<p, Scalar>& velocity,
      const nodal_vector_view<p, Scalar>& proj_pressure_gradient,
      const nodal_scalar_view<p, Scalar>& pressure,
      scs_scalar_view<p, Scalar>& mdot)
    {
      const auto& nodalInterp = ops.mat_.linearNodalInterp;
      const auto& scsInterp = ops.mat_.linearScsInterp;

      NALU_ALIGNED Scalar base_box[3][8];
      hex_vertex_coordinates<p, Scalar>(xc, base_box);

      nodal_vector_workview<p, Scalar> work_rhou_corr;
      auto& rhou_corr = work_rhou_corr.view();

      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            const auto rho = density(k, j, i);
            for (int d = 0; d < 3; ++d) {
              rhou_corr(k, j, i, d) = rho * velocity(k, j, i, d) + projTimeScale * proj_pressure_gradient(k, j, i, d);
            }
          }
        }
      }

      nodal_vector_workview<p, Scalar> work_rhou_corrIp;
      auto& rhou_corrIp = work_rhou_corrIp.view();

      nodal_vector_workview<p, Scalar> work_dpdxhIp;
      auto& dpdxhIp = work_dpdxhIp.view();

      ops.scs_xhat_interp(rhou_corr, rhou_corrIp);
      ops.scs_xhat_grad(pressure, dpdxhIp);

      for (int k = 0; k < p + 1; ++k) {
        NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
        for (int j = 0; j < p + 1; ++j) {
          NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
          for (int i = 0; i < p; ++i) {
            NALU_ALIGNED const Scalar interpi[2] = { scsInterp(0, i), scsInterp(1, i) };
            NALU_ALIGNED Scalar areav[3];
            hex_areav_x(base_box, interpi, interpj, interpk, areav);

            const auto dpdxIp_dot_A = laplacian_metric(XH, k, j, i, XH) * dpdxhIp(k, j, i, XH)
                                    + laplacian_metric(XH, k, j, i, YH) * dpdxhIp(k, j, i, YH)
                                    + laplacian_metric(XH, k, j, i, ZH) * dpdxhIp(k, j, i, ZH);

            const auto rhouCorr_dot_A = rhou_corrIp(k, j, i, XH) * areav[XH]
                                      + rhou_corrIp(k, j, i, YH) * areav[YH]
                                      + rhou_corrIp(k, j, i, ZH) * areav[ZH];

            mdot(XH, k, j, i) = rhouCorr_dot_A - projTimeScale * dpdxIp_dot_A;
          }
        }
      }

      ops.scs_yhat_interp(rhou_corr, rhou_corrIp);
      ops.scs_yhat_grad(pressure, dpdxhIp);

      for (int k = 0; k < p + 1; ++k) {
        NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
        for (int j = 0; j < p; ++j) {
          NALU_ALIGNED const Scalar interpj[2] = { scsInterp(0, j), scsInterp(1, j) };
          for (int i = 0; i < p + 1; ++i) {
            NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };
            NALU_ALIGNED Scalar areav[3];
            hex_areav_y(base_box, interpi, interpj, interpk, areav);

            const auto dpdxIp_dot_A = laplacian_metric(YH, k, j, i, XH) * dpdxhIp(k, j, i, XH)
                                    + laplacian_metric(YH, k, j, i, YH) * dpdxhIp(k, j, i, YH)
                                    + laplacian_metric(YH, k, j, i, ZH) * dpdxhIp(k, j, i, ZH);

            const auto rhouCorr_dot_A = rhou_corrIp(k, j, i, XH) * areav[XH]
                                      + rhou_corrIp(k, j, i, YH) * areav[YH]
                                      + rhou_corrIp(k, j, i, ZH) * areav[ZH];

            mdot(YH, k, j, i) = rhouCorr_dot_A - projTimeScale * dpdxIp_dot_A;
          }
        }
      }

      ops.scs_zhat_interp(rhou_corr, rhou_corrIp);
      ops.scs_zhat_grad(pressure, dpdxhIp);
      for (int k = 0; k < p ; ++k) {
        NALU_ALIGNED const Scalar interpk[2] = { scsInterp(0, k), scsInterp(1, k) };
        for (int j = 0; j < p + 1; ++j) {
          NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
          for (int i = 0; i < p + 1; ++i) {
            NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };
            NALU_ALIGNED Scalar areav[3];
            hex_areav_z(base_box, interpi, interpj, interpk, areav);

            const auto dpdxIp_dot_A = laplacian_metric(ZH, k, j, i, XH) * dpdxhIp(k, j, i, XH)
                                    + laplacian_metric(ZH, k, j, i, YH) * dpdxhIp(k, j, i, YH)
                                    + laplacian_metric(ZH, k, j, i, ZH) * dpdxhIp(k, j, i, ZH);

            const auto rhouCorr_dot_A = rhou_corrIp(k, j, i, XH) * areav[XH]
                                      + rhou_corrIp(k, j, i, YH) * areav[YH]
                                      + rhou_corrIp(k, j, i, ZH) * areav[ZH];

            mdot(ZH, k, j, i) = rhouCorr_dot_A - projTimeScale * dpdxIp_dot_A;
          }
        }
      }
    }
} // namespace HighOrderGeometryQuad
} // namespace naluUnit
} // namespace Sierra

#endif
