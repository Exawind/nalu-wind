// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TensorProductCVFEMPressurePoisson_h
#define TensorProductCVFEMPressurePoisson_h

#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/DirectionMacros.h>
#include <CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int poly_order, typename Scalar>
void pressure_poisson_rhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  double projTimeScale,
  const scs_scalar_view<poly_order, Scalar>& mdot,
  nodal_scalar_view<poly_order, Scalar>& rhs)
{
  constexpr int n1D = poly_order + 1;
  constexpr int nscs = poly_order;

  nodal_scalar_workview<poly_order, Scalar> work_integrand(0);
  auto& integrand = work_integrand.view();

  const double inv_projTimeScale = 1.0 / projTimeScale;

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < nscs; ++i) {
        integrand(k, j, i) = -mdot(XH, k, j, i) * inv_projTimeScale;
      }
    }
  }
  ops.integrate_and_diff_xhat(integrand, rhs);

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < nscs; ++j) {
      for (int i = 0; i < n1D; ++i) {
        integrand(k,j,i) = -mdot(YH, k, j, i) * inv_projTimeScale;
      }
    }
  }
  ops.integrate_and_diff_yhat(integrand, rhs);

  for (int k = 0; k < nscs; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        integrand(k,j,i) = -mdot(ZH, k, j, i) * inv_projTimeScale;
      }
    }
  }
  ops.integrate_and_diff_zhat(integrand, rhs);
}

}
}
}
#endif
