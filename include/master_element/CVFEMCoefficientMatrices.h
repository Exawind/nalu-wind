/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef CoefficientMatrices_h
#define CoefficientMatrices_h

#include <master_element/CVFEMCoefficients.h>
#include <CVFEMTypeDefs.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu{

template <int p, typename Scalar = DoubleType>
struct CoefficientMatrices
{
  using value_type = Scalar;
  constexpr static int poly_order = p;
  CoefficientMatrices(const double* nodeLocs, const double* scsLocs)
  : scsDeriv(coefficients::scs_derivative_weights<p, Scalar>(nodeLocs, scsLocs)),
    scsInterp(coefficients::scs_interpolation_weights<p, Scalar>(nodeLocs, scsLocs)),
    nodalWeights(coefficients::nodal_integration_weights<p, Scalar>(nodeLocs, scsLocs)),
    nodalDeriv(coefficients::nodal_derivative_weights<p, Scalar>(nodeLocs)),
    linearNodalInterp(coefficients::linear_nodal_interpolation_weights<p, Scalar>(nodeLocs)),
    linearScsInterp(coefficients::linear_scs_interpolation_weights<p, Scalar>(scsLocs)),
    lumpedNodalWeights(coefficients::lumped_nodal_integration_weights<p,Scalar>(nodeLocs,scsLocs))
  {};

  CoefficientMatrices()
  : scsDeriv(coefficients::scs_derivative_weights<p, Scalar>()),
    scsInterp(coefficients::scs_interpolation_weights<p, Scalar>()),
    nodalWeights(coefficients::nodal_integration_weights<p, Scalar>()),
    nodalDeriv(coefficients::nodal_derivative_weights<p, Scalar>()),
    difference(coefficients::difference_matrix<p, Scalar>()),
    linearNodalInterp(coefficients::linear_nodal_interpolation_weights<p, Scalar>()),
    linearScsInterp(coefficients::linear_scs_interpolation_weights<p, Scalar>()),
    lumpedNodalWeights(coefficients::lumped_nodal_integration_weights<p,Scalar>())
  {};

  const scs_matrix_view<p, Scalar> scsDeriv;
  const scs_matrix_view<p, Scalar> scsInterp;
  const nodal_matrix_view<p, Scalar> nodalWeights;
  const nodal_matrix_view<p, Scalar> nodalDeriv;
  const nodal_matrix_view<p, Scalar> difference;
  const linear_nodal_matrix_view<p, Scalar> linearNodalInterp;
  const linear_scs_matrix_view<p, Scalar> linearScsInterp;
  const nodal_matrix_view<p, Scalar> lumpedNodalWeights;
};

} // namespace naluUnit
} // namespace Sierra

#endif
