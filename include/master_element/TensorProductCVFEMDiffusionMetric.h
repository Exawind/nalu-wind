/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductCVFEMDiffusionMetric_h
#define TensorProductCVFEMDiffusionMetric_h

#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/DirectionMacros.h>
#include <master_element/TensorOps.h>
#include <master_element/Hex8GeometryFunctions.h>

#include <CVFEMTypeDefs.h>
#include <AlgTraits.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {
namespace high_order_metrics
{

template <int p, typename Scalar>
void compute_laplacian_metric_linear(
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

        NALU_ALIGNED Scalar jac[3][3];
        hex_jacobian(base_box, interpi, interpj, interpk, jac);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jac, adjJac);

        const Scalar inv_detj = 1.0 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);

        metric(XH, k, j, i, XH) = -inv_detj*(adjJac[XH][XH] * adjJac[XH][XH] + adjJac[XH][YH] * adjJac[XH][YH] + adjJac[XH][ZH] * adjJac[XH][ZH]);
        metric(XH, k, j, i, YH) = -inv_detj*(adjJac[XH][XH] * adjJac[YH][XH] + adjJac[XH][YH] * adjJac[YH][YH] + adjJac[XH][ZH] * adjJac[YH][ZH]);
        metric(XH, k, j, i, ZH) = -inv_detj*(adjJac[XH][XH] * adjJac[ZH][XH] + adjJac[XH][YH] * adjJac[ZH][YH] + adjJac[XH][ZH] * adjJac[ZH][ZH]);
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { scsInterp(0, j), scsInterp(1, j) };
      for (int i = 0; i < p+1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jac[3][3];
        hex_jacobian(base_box, interpi, interpj, interpk, jac);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jac, adjJac);

        const Scalar inv_detj = 1.0 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);

        metric(YH, k, j, i, XH) = -inv_detj*(adjJac[YH][XH] * adjJac[XH][XH] + adjJac[YH][YH] * adjJac[XH][YH] + adjJac[YH][ZH] * adjJac[XH][ZH]);
        metric(YH, k, j, i, YH) = -inv_detj*(adjJac[YH][XH] * adjJac[YH][XH] + adjJac[YH][YH] * adjJac[YH][YH] + adjJac[YH][ZH] * adjJac[YH][ZH]);
        metric(YH, k, j, i, ZH) = -inv_detj*(adjJac[YH][XH] * adjJac[ZH][XH] + adjJac[YH][YH] * adjJac[ZH][YH] + adjJac[YH][ZH] * adjJac[ZH][ZH]);

      }
    }
  }

  for (int k = 0; k < p ; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { scsInterp(0, k), scsInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p + 1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jac[3][3];
        hex_jacobian(base_box, interpi, interpj, interpk, jac);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jac, adjJac);

        const Scalar inv_detj = 1.0 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);

        metric(ZH, k, j, i, XH) = -inv_detj*(adjJac[ZH][XH] * adjJac[XH][XH] + adjJac[ZH][YH] * adjJac[XH][YH] + adjJac[ZH][ZH] * adjJac[XH][ZH]);
        metric(ZH, k, j, i, YH) = -inv_detj*(adjJac[ZH][XH] * adjJac[YH][XH] + adjJac[ZH][YH] * adjJac[YH][YH] + adjJac[ZH][ZH] * adjJac[YH][ZH]);
        metric(ZH, k, j, i, ZH) = -inv_detj*(adjJac[ZH][XH] * adjJac[ZH][XH] + adjJac[ZH][YH] * adjJac[ZH][YH] + adjJac[ZH][ZH] * adjJac[ZH][ZH]);
      }
    }
  }
}

template <int p, typename Scalar>
void compute_laplacian_metric_lineart(
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

        NALU_ALIGNED Scalar jac[3][3];
        hex_jacobiant(base_box, interpi, interpj, interpk, jac);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jac, adjJac);

        const Scalar inv_detj = 1.0 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);

        metric(XH, k, j, i, XH) = -inv_detj*(adjJac[XH][XH] * adjJac[XH][XH] + adjJac[XH][YH] * adjJac[XH][YH] + adjJac[XH][ZH] * adjJac[XH][ZH]);
        metric(XH, k, j, i, YH) = -inv_detj*(adjJac[XH][XH] * adjJac[YH][XH] + adjJac[XH][YH] * adjJac[YH][YH] + adjJac[XH][ZH] * adjJac[YH][ZH]);
        metric(XH, k, j, i, ZH) = -inv_detj*(adjJac[XH][XH] * adjJac[ZH][XH] + adjJac[XH][YH] * adjJac[ZH][YH] + adjJac[XH][ZH] * adjJac[ZH][ZH]);
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { scsInterp(0, j), scsInterp(1, j) };
      for (int i = 0; i < p+1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jac[3][3];
        hex_jacobiant(base_box, interpi, interpj, interpk, jac);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jac, adjJac);

        const Scalar inv_detj = 1.0 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);

        metric(YH, k, j, i, XH) = -inv_detj*(adjJac[YH][XH] * adjJac[XH][XH] + adjJac[YH][YH] * adjJac[XH][YH] + adjJac[YH][ZH] * adjJac[XH][ZH]);
        metric(YH, k, j, i, YH) = -inv_detj*(adjJac[YH][XH] * adjJac[YH][XH] + adjJac[YH][YH] * adjJac[YH][YH] + adjJac[YH][ZH] * adjJac[YH][ZH]);
        metric(YH, k, j, i, ZH) = -inv_detj*(adjJac[YH][XH] * adjJac[ZH][XH] + adjJac[YH][YH] * adjJac[ZH][YH] + adjJac[YH][ZH] * adjJac[ZH][ZH]);

      }
    }
  }

  for (int k = 0; k < p ; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { scsInterp(0, k), scsInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p + 1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jac[3][3];
        hex_jacobiant(base_box, interpi, interpj, interpk, jac);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jac, adjJac);

        const Scalar inv_detj = 1.0 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);

        metric(ZH, k, j, i, XH) = -inv_detj*(adjJac[ZH][XH] * adjJac[XH][XH] + adjJac[ZH][YH] * adjJac[XH][YH] + adjJac[ZH][ZH] * adjJac[XH][ZH]);
        metric(ZH, k, j, i, YH) = -inv_detj*(adjJac[ZH][XH] * adjJac[YH][XH] + adjJac[ZH][YH] * adjJac[YH][YH] + adjJac[ZH][ZH] * adjJac[YH][ZH]);
        metric(ZH, k, j, i, ZH) = -inv_detj*(adjJac[ZH][XH] * adjJac[ZH][XH] + adjJac[ZH][YH] * adjJac[ZH][YH] + adjJac[ZH][ZH] * adjJac[ZH][ZH]);
      }
    }
  }
}

template <int p, typename Scalar>
void compute_diffusion_metric_linear(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_vector_view<p, Scalar>& xc,
  const nodal_scalar_view<p, Scalar>& diffusivity,
  scs_vector_view<p, Scalar>& metric)
{
  const auto& nodalInterp = ops.mat_.linearNodalInterp;
  const auto& scsInterp = ops.mat_.linearScsInterp;

  NALU_ALIGNED Scalar base_box[3][8];
  hex_vertex_coordinates<p, Scalar>(xc, base_box);

  nodal_scalar_workview<p, Scalar> diffIp_wsv(0);
  auto& diffIp = diffIp_wsv.view();

  ops.scs_xhat_interp(diffusivity, diffIp);

  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { scsInterp(0, i), scsInterp(1, i) };

        NALU_ALIGNED Scalar jac[3][3];
        hex_jacobian(base_box, interpi, interpj, interpk, jac);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jac, adjJac);

        const Scalar inv_detj = diffIp(k,j,i) / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);

        metric(XH, k, j, i, XH) = -inv_detj*(adjJac[XH][XH] * adjJac[XH][XH] + adjJac[XH][YH] * adjJac[XH][YH] + adjJac[XH][ZH] * adjJac[XH][ZH]);
        metric(XH, k, j, i, YH) = -inv_detj*(adjJac[XH][XH] * adjJac[YH][XH] + adjJac[XH][YH] * adjJac[YH][YH] + adjJac[XH][ZH] * adjJac[YH][ZH]);
        metric(XH, k, j, i, ZH) = -inv_detj*(adjJac[XH][XH] * adjJac[ZH][XH] + adjJac[XH][YH] * adjJac[ZH][YH] + adjJac[XH][ZH] * adjJac[ZH][ZH]);
      }
    }
  }

  ops.scs_yhat_interp(diffusivity, diffIp);

  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { scsInterp(0, j), scsInterp(1, j) };
      for (int i = 0; i < p+1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jac[3][3];
        hex_jacobian(base_box, interpi, interpj, interpk, jac);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jac, adjJac);

        const Scalar inv_detj = diffIp(k,j,i) / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);

        metric(YH, k, j, i, XH) =  -inv_detj*(adjJac[YH][XH] * adjJac[XH][XH] + adjJac[YH][YH] * adjJac[XH][YH] + adjJac[YH][ZH] * adjJac[XH][ZH]);
        metric(YH, k, j, i, YH) =  -inv_detj*(adjJac[YH][XH] * adjJac[YH][XH] + adjJac[YH][YH] * adjJac[YH][YH] + adjJac[YH][ZH] * adjJac[YH][ZH]);
        metric(YH, k, j, i, ZH) =  -inv_detj*(adjJac[YH][XH] * adjJac[ZH][XH] + adjJac[YH][YH] * adjJac[ZH][YH] + adjJac[YH][ZH] * adjJac[ZH][ZH]);

      }
    }
  }

  ops.scs_zhat_interp(diffusivity, diffIp);

  for (int k = 0; k < p ; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { scsInterp(0, k), scsInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p + 1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jac[3][3];
        hex_jacobian(base_box, interpi, interpj, interpk, jac);

        NALU_ALIGNED Scalar adjJac[3][3];
        adjugate_matrix33(jac, adjJac);

        const Scalar inv_detj = diffIp(k,j,i) / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);

        metric(ZH, k, j, i, XH) = -inv_detj*(adjJac[ZH][XH] * adjJac[XH][XH] + adjJac[ZH][YH] * adjJac[XH][YH] + adjJac[ZH][ZH] * adjJac[XH][ZH]);
        metric(ZH, k, j, i, YH) = -inv_detj*(adjJac[ZH][XH] * adjJac[YH][XH] + adjJac[ZH][YH] * adjJac[YH][YH] + adjJac[ZH][ZH] * adjJac[YH][ZH]);
        metric(ZH, k, j, i, ZH) = -inv_detj*(adjJac[ZH][XH] * adjJac[ZH][XH] + adjJac[ZH][YH] * adjJac[ZH][YH] + adjJac[ZH][ZH] * adjJac[ZH][ZH]);
      }
    }
  }
}

template <int p, typename Scalar>
void scale_metric(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_scalar_view<p, Scalar>& diffusivity,
  scs_vector_view<p, Scalar>& metric)
{
  constexpr int n1D = p + 1;
  constexpr int nscs = p;

  nodal_scalar_workview<p, Scalar> diffIp_wsv(0);
  auto& diffIp = diffIp_wsv.view();

  ops.scs_xhat_interp(diffusivity, diffIp);

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < nscs; ++i) {
        const Scalar mu = diffIp(k, j, i);
        for (int d = 0; d < 3; ++d) {
          metric(XH, k, j, i, d) *= mu;
        }
      }
    }
  }

  ops.scs_yhat_interp(diffusivity, diffIp);

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < nscs; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const Scalar mu = diffIp(k, j, i);
        for (int d = 0; d < 3; ++d) {
          metric(YH, k, j, i, d) *= mu;
        }
      }
    }
  }

  ops.scs_zhat_interp(diffusivity, diffIp);

  for (int k = 0; k < nscs; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const Scalar mu = diffIp(k, j, i);
        for (int d = 0; d < 3; ++d) {
          metric(ZH, k, j, i, d) *= mu;
        }
      }
    }
  }
}

}
}
}

#endif
