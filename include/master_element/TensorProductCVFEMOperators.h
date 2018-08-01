/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductCVFEMOperators_h
#define TensorProductCVFEMOperators_h

#include <master_element/TensorProductOperatorsHexInternal.h>

#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/DirectionMacros.h>
#include <CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {

template<int nodes1D>
KOKKOS_FORCEINLINE_FUNCTION
int idx(int i, int j, int k)
{
  return ((i * nodes1D + j) * nodes1D + k);
}

template<int nodes1D>
KOKKOS_FORCEINLINE_FUNCTION
int idx(int d, int i, int j, int k)
{
  return (((i * nodes1D + j) * nodes1D + k) * 3 + d);
}

template<int p, typename Scalar = DoubleType>
class CVFEMOperators
{
public:
using ViewTypes = CVFEMViews<p, Scalar>;
DeclareCVFEMTypeDefs(ViewTypes);

static constexpr int n = p + 1;

const CoefficientMatrices<p, Scalar> mat_{};

void nodal_grad(const nodal_scalar_view& f, nodal_vector_view& grad) const
{
  tensor_internal::apply_x<p, Scalar, n, n, n>(mat_.nodalDeriv, f, grad, XH);
  tensor_internal::apply_y<p, Scalar, n, n, n>(mat_.nodalDeriv, f, grad, YH);
  tensor_internal::apply_z<p, Scalar, n, n, n>(mat_.nodalDeriv, f, grad, ZH);
}

void nodal_grad(const nodal_vector_view& f, nodal_tensor_view& grad) const
{
  tensor_internal::apply_x<p, Scalar, n, n, n>(mat_.nodalDeriv, f, grad, XH);
  tensor_internal::apply_y<p, Scalar, n, n, n>(mat_.nodalDeriv, f, grad, YH);
  tensor_internal::apply_z<p, Scalar, n, n, n>(mat_.nodalDeriv, f, grad, ZH);
}

void scs_xhat_grad(const nodal_scalar_view& f, nodal_vector_view& grad) const
{
  nodal_scalar_workview work_scratch;
  auto& scratch = work_scratch.view();
  tensor_internal::apply_x<p, Scalar, p, n, n>(mat_.scsInterp, f, scratch);

  tensor_internal::apply_x<p, Scalar, p, n, n>(mat_.scsDeriv, f, grad, XH);
  tensor_internal::apply_y<p, Scalar, p, n, n>(mat_.nodalDeriv, scratch, grad, YH);
  tensor_internal::apply_z<p, Scalar, p, n, n>(mat_.nodalDeriv, scratch, grad, ZH);
}

void scs_xhat_grad(const nodal_vector_view& f, nodal_tensor_view& grad) const
{
  nodal_vector_workview work_scratch;
  auto& scratch = work_scratch.view();
  tensor_internal::apply_x<p, Scalar, p, n, n>(mat_.scsInterp, f, scratch);

  tensor_internal::apply_x<p, Scalar, p, n, n>(mat_.scsDeriv, f, grad, XH);
  tensor_internal::apply_y<p, Scalar, p, n, n>(mat_.nodalDeriv, scratch, grad, YH);
  tensor_internal::apply_z<p, Scalar, p, n, n>(mat_.nodalDeriv, scratch, grad, ZH);
}

void scs_yhat_grad(const nodal_scalar_view& f, nodal_vector_view& grad) const
{
  nodal_scalar_workview work_scratch;
  auto& scratch = work_scratch.view();
  tensor_internal::apply_y<p, Scalar, n, p, n>(mat_.scsInterp, f, scratch);

  tensor_internal::apply_x<p, Scalar, n, p, n>(mat_.nodalDeriv, scratch, grad, XH);
  tensor_internal::apply_y<p, Scalar, n, p, n>(mat_.scsDeriv, f, grad, YH);
  tensor_internal::apply_z<p, Scalar, n, p, n>(mat_.nodalDeriv, scratch, grad, ZH);
}
void scs_yhat_grad(const nodal_vector_view& f, nodal_tensor_view& grad) const
{
  nodal_vector_workview work_scratch;
  auto& scratch = work_scratch.view();
  tensor_internal::apply_y<p, Scalar, n, p, n>(mat_.scsInterp, f, scratch);

  tensor_internal::apply_x<p, Scalar, n, p, n>(mat_.nodalDeriv, scratch, grad, XH);
  tensor_internal::apply_y<p, Scalar, n, p, n>(mat_.scsDeriv, f, grad, YH);
  tensor_internal::apply_z<p, Scalar, n, p, n>(mat_.nodalDeriv, scratch, grad, ZH);
}

void scs_zhat_grad(const nodal_scalar_view& f, nodal_vector_view& grad) const
{
  nodal_scalar_workview work_scratch;
  auto& scratch = work_scratch.view();
  tensor_internal::apply_z<p, Scalar, n, n, p>(mat_.scsInterp, f, scratch);

  tensor_internal::apply_x<p, Scalar, n, n, p>(mat_.nodalDeriv, scratch, grad, XH);
  tensor_internal::apply_y<p, Scalar, n, n, p>(mat_.nodalDeriv, scratch, grad, YH);
  tensor_internal::apply_z<p, Scalar, n, n, p>(mat_.scsDeriv, f, grad, ZH);
}

void scs_zhat_grad(const nodal_vector_view& f, nodal_tensor_view& grad) const
{
  nodal_vector_workview work_scratch;
  auto& scratch = work_scratch.view();
  tensor_internal::apply_z<p, Scalar, n, n, p>(mat_.scsInterp, f, scratch);

  tensor_internal::apply_x<p, Scalar, n, n, p>(mat_.nodalDeriv, scratch, grad, XH);
  tensor_internal::apply_y<p, Scalar, n, n, p>(mat_.nodalDeriv, scratch, grad, YH);
  tensor_internal::apply_z<p, Scalar, n, n, p>(mat_.scsDeriv, f, grad, ZH);
}

void scs_xhat_interp(const nodal_scalar_view& f, nodal_scalar_view& fIp) const
{
  tensor_internal::apply_x<p, Scalar, p, n, n>(mat_.scsInterp, f, fIp);
}
void scs_xhat_interp(const nodal_vector_view& f, nodal_vector_view& fIp) const
{
  tensor_internal::apply_x<p, Scalar, p, n, n>(mat_.scsInterp, f, fIp);
}

void scs_yhat_interp(const nodal_scalar_view& f, nodal_scalar_view& fIp) const
{
  tensor_internal::apply_y<p, Scalar, n, p, n>(mat_.scsInterp, f, fIp);
}

void scs_yhat_interp(const nodal_vector_view& f, nodal_vector_view& fIp) const
{
  tensor_internal::apply_y<p, Scalar, n, p, n>(mat_.scsInterp, f, fIp);
}

void scs_zhat_interp(const nodal_scalar_view& f, nodal_scalar_view& fIp) const
{
  tensor_internal::apply_z<p, Scalar, n, n, p>(mat_.scsInterp, f, fIp);
}

void scs_zhat_interp(const nodal_vector_view& f, nodal_vector_view& fIp) const
{
  tensor_internal::apply_z<p, Scalar, n, n, p>(mat_.scsInterp, f, fIp);
}

void volume(const nodal_scalar_view& f, nodal_scalar_view& f_bar) const
{
  nodal_scalar_workview work_scratch_1;
  auto& scratch_1 = work_scratch_1.view();
  tensor_internal::apply_x<p, Scalar, n, n, n>(mat_.nodalWeights, f, scratch_1);

  nodal_scalar_workview work_scratch_2;
  auto& scratch_2 = work_scratch_2.view();

  tensor_internal::apply_y<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_1, scratch_2);
  tensor_internal::apply_z<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_2, scratch_1);

  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        f_bar(k, j, i) += scratch_1(k, j, i);
      }
    }
  }
}

void volume(const nodal_vector_view& f, nodal_vector_view& f_bar) const
{
  nodal_vector_workview work_scratch_1;
  auto& scratch_1 = work_scratch_1.view();
  tensor_internal::apply_x<p, Scalar, n, n, n>(mat_.nodalWeights, f, scratch_1);

  nodal_vector_workview work_scratch_2;
  auto& scratch_2 = work_scratch_2.view();

  tensor_internal::apply_y<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_1, scratch_2);
  tensor_internal::apply_z<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_2, scratch_1);

  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        f_bar(k, j, i, XH) += scratch_1(k, j, i, XH);
        f_bar(k, j, i, YH) += scratch_1(k, j, i, YH);
        f_bar(k, j, i, ZH) += scratch_1(k, j, i, ZH);
      }
    }
  }
}

void integrate_and_diff_xhat(const nodal_scalar_view& f, nodal_scalar_view& f_bar) const
{
  nodal_scalar_workview work_scratch_1;
  auto& scratch_1 = work_scratch_1.view();
  tensor_internal::apply_y<p, Scalar, n, n, n>(mat_.nodalWeights, f, scratch_1);

  nodal_scalar_workview work_scratch_2;
  auto& scratch_2 = work_scratch_2.view();
  tensor_internal::apply_z<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_1, scratch_2);
  tensor_internal::difference_x<p, Scalar>(scratch_2, f_bar);
}

void integrate_and_diff_xhat(const nodal_vector_view& f, nodal_vector_view& f_bar) const
{
  nodal_vector_workview work_scratch_1;
  auto& scratch_1 = work_scratch_1.view();
  tensor_internal::apply_y<p, Scalar, n, n, n>(mat_.nodalWeights, f, scratch_1);

  nodal_vector_workview work_scratch_2;
  auto& scratch_2 = work_scratch_2.view();
  tensor_internal::apply_z<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_1, scratch_2);
  tensor_internal::difference_x<p, Scalar>(scratch_2, f_bar);
}

void integrate_and_diff_yhat(const nodal_scalar_view& f, nodal_scalar_view& f_bar) const
{
  nodal_scalar_workview work_scratch_1;
  auto& scratch_1 = work_scratch_1.view();
  tensor_internal::apply_x<p, Scalar, n, n, n>(mat_.nodalWeights, f, scratch_1);

  nodal_scalar_workview work_scratch_2;
  auto& scratch_2 = work_scratch_2.view();
  tensor_internal::apply_z<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_1, scratch_2);
  tensor_internal::difference_y<p, Scalar>(scratch_2, f_bar);
}

void integrate_and_diff_yhat(const nodal_vector_view& f, nodal_vector_view& f_bar) const
{
  nodal_vector_workview work_scratch_1;
  auto& scratch_1 = work_scratch_1.view();
  tensor_internal::apply_x<p, Scalar, n, n, n>(mat_.nodalWeights, f, scratch_1);

  nodal_vector_workview work_scratch_2;
  auto& scratch_2 = work_scratch_2.view();
  tensor_internal::apply_z<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_1, scratch_2);
  tensor_internal::difference_y<p, Scalar>(scratch_2, f_bar);
}

void integrate_and_diff_zhat(const nodal_scalar_view& f, nodal_scalar_view& f_bar) const
{
  nodal_scalar_workview work_scratch_1;
  auto& scratch_1 = work_scratch_1.view();
  tensor_internal::apply_x<p, Scalar, n, n, n>(mat_.nodalWeights, f, scratch_1);

  nodal_scalar_workview work_scratch_2;
  auto& scratch_2 = work_scratch_2.view();
  tensor_internal::apply_y<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_1, scratch_2);
  tensor_internal::difference_z<p, Scalar>(scratch_2, f_bar);
}

void integrate_and_diff_zhat(const nodal_vector_view& f, nodal_vector_view& f_bar) const
{
  nodal_vector_workview work_scratch_1;
  auto& scratch_1 = work_scratch_1.view();
  tensor_internal::apply_x<p, Scalar, n, n, n>(mat_.nodalWeights, f, scratch_1);

  nodal_vector_workview work_scratch_2;
  auto& scratch_2 = work_scratch_2.view();
  tensor_internal::apply_y<p, Scalar, n, n, n>(mat_.nodalWeights, scratch_1, scratch_2);
  tensor_internal::difference_z<p, Scalar>(scratch_2, f_bar);
}

};


} // namespace naluUnit
} // namespace Sierra

#endif

