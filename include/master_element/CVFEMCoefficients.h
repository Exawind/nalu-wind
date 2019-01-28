/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderCoefficients_h
#define HighOrderCoefficients_h

#include <element_promotion/QuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>
#include <CVFEMTypeDefs.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {
namespace coefficients {
/* Computes 1D coefficient matrices (e.g. for the derivative) for CVFEM */

template <int p, typename Scalar = DoubleType>
nodal_matrix_view<p, Scalar> nodal_integration_weights(const double* /* nodeLocs */, const double* scsLocs)
{
  constexpr int nodes1D = p+1;
  auto scsEndLoc = pad_end_points(p, scsLocs);
  auto weightvec = SGL_quadrature_rule(nodes1D, scsEndLoc.data()).second;

  // copy over to a 2D view
  nodal_matrix_view<p, Scalar> weights("nodal_integration_weights");
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      weights(j,i) = Scalar(weightvec[j*nodes1D +i]);
    }
  }
  return weights;
}
template <int p, typename Scalar = DoubleType>
nodal_matrix_view<p, Scalar> lumped_nodal_integration_weights(const double* nodeLocs, const double* scsLocs)
{
  constexpr int nodes1D = p+1;
  const auto weights = nodal_integration_weights<p,Scalar>(nodeLocs, scsLocs);

  Kokkos::Array<Scalar, p + 1> lumped_weights;
  for (int j = 0; j < p+1; ++j) {
    lumped_weights[j] = 0;
  }
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      lumped_weights[j] += weights(j,i);
    }
  }
  nodal_matrix_view<p, Scalar> lumped_weight_matrix("lumped_nodal_integration_weights");

  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      lumped_weight_matrix(j,i) = (i == j) ? lumped_weights[j] : 0;
    }
  }
  return lumped_weight_matrix;
}

template <int p, typename Scalar = DoubleType>
scs_matrix_view<p, Scalar> scs_interpolation_weights(const double* nodeLocs, const double* scsLocs)
{
  constexpr int nodes1D = p+1;
  scs_matrix_view<p, Scalar> scsInterp("subcontrol surface interpolation matrix");

  auto basis1D = Lagrange1D(nodeLocs, p);
  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      scsInterp(j,i) = basis1D.interpolation_weight(scsLocs[j], i);
    }
  }
  return scsInterp;
}

template <int p, typename Scalar = DoubleType>
scs_matrix_view<p, Scalar> scs_derivative_weights(const double* nodeLocs, const double* scsLocs)
{
  constexpr int nodes1D = p+1;
  scs_matrix_view<p, Scalar> scsDeriv("subcontrol surface derivative matrix");

  auto basis1D = Lagrange1D(nodeLocs, p);
  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      scsDeriv(j,i) = basis1D.derivative_weight(scsLocs[j], i);
    }
  }
  return scsDeriv;
}

template <int p,typename Scalar = DoubleType>
nodal_matrix_view<p, Scalar> nodal_derivative_weights(const double* nodeLocs)
{
  constexpr int nodes1D = p+1;
  nodal_matrix_view<p, Scalar> nodalDeriv("nodal derivative matrix");

  auto basis1D = Lagrange1D(nodeLocs, p);
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      nodalDeriv(j,i) = basis1D.derivative_weight(nodeLocs[j],i);
    }
  }
  return nodalDeriv;
}

template <int p, typename Scalar = DoubleType>
linear_scs_matrix_view<p, Scalar> linear_scs_interpolation_weights(const double* scsLocs)
{
  linear_scs_matrix_view<p, Scalar> linear_scs_interp("linscs");

  for (int j = 0; j < p; ++j) {
    linear_scs_interp(0,j) = 0.5*(1 - scsLocs[j]);
    linear_scs_interp(1,j) = 0.5*(1 + scsLocs[j]);
  }

  return linear_scs_interp;
}

template <int p, typename Scalar = DoubleType>
linear_nodal_matrix_view<p, Scalar> linear_nodal_interpolation_weights(const double* nodeLocs)
{
  linear_nodal_matrix_view<p, Scalar> linear_nodal_interp("linnodal");

  for (int j = 0; j < p+1; ++j) {
    linear_nodal_interp(0,j) = 0.5*(1 - nodeLocs[j]);
    linear_nodal_interp(1,j) = 0.5*(1 + nodeLocs[j]);
  }
  return linear_nodal_interp;
}

template <int p, typename Scalar = DoubleType>
nodal_matrix_view<p, Scalar> nodal_integration_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  auto scsLocs  = gauss_legendre_rule(p).first;

  return nodal_integration_weights<p, Scalar>(nodeLocs.data(), scsLocs.data());
}


template <int p, typename Scalar = DoubleType>
nodal_matrix_view<p, Scalar> lumped_nodal_integration_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  auto scsLocs  = gauss_legendre_rule(p).first;

  return lumped_nodal_integration_weights<p, Scalar>(nodeLocs.data(), scsLocs.data());
}

template <int p, typename Scalar = DoubleType>
nodal_matrix_view<p, Scalar> nodal_derivative_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  return nodal_derivative_weights<p, Scalar>(nodeLocs.data());
}

template <int p,typename Scalar = DoubleType>
scs_matrix_view<p, Scalar> scs_derivative_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  auto scsLocs  = gauss_legendre_rule(p).first;

  return scs_derivative_weights<p, Scalar>(nodeLocs.data(), scsLocs.data());
}

template <int p, typename Scalar = DoubleType>
scs_matrix_view<p, Scalar> scs_interpolation_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  auto scsLocs  = gauss_legendre_rule(p).first;

  return scs_interpolation_weights<p, Scalar>(nodeLocs.data(), scsLocs.data());
}

template <int p, typename Scalar = DoubleType>
linear_scs_matrix_view<p, Scalar> linear_scs_interpolation_weights()
{
  auto scsLocs  = gauss_legendre_rule(p).first;
  return linear_scs_interpolation_weights<p, Scalar>(scsLocs.data());
}

template <int p, typename Scalar = DoubleType>
linear_nodal_matrix_view<p, Scalar> linear_nodal_interpolation_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  return linear_nodal_interpolation_weights<p, Scalar>(nodeLocs.data());
}

template <int p, typename Scalar = DoubleType>
nodal_matrix_view<p, Scalar> difference_matrix()
{
  nodal_matrix_view<p, Scalar> scatt{"diff"};
  Kokkos::deep_copy(scatt, Scalar(0.0));

  scatt(0, 0)  = -1;
  scatt(p,p-1) = +1;

  for (int j = 1; j < p; ++j) {
    scatt(j,j+0) = -1;
    scatt(j,j-1) = +1;
  }
  return scatt;
}

template <int p, typename Scalar = DoubleType>
nodal_matrix_view<p, Scalar> identity_matrix()
{
  nodal_matrix_view<p, Scalar> id{ "" };
  Kokkos::deep_copy(id,Scalar(0.0));
  for (int j = 0; j < p+1; ++j) {
    id(j,j) = 1.0;
  }

  return id;
}

} // namespace CoefficientMatrices
} // namespace naluUnit
} // namespace Sierra

#endif
