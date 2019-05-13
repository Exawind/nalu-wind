/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <user_functions/VariableDensityEnthalpyMMSHOElemKernel.h>
#include <kernel/TensorProductCVFEMSource.h>
#include <master_element/TensorProductCVFEMVolumeMetric.h>

#include <kernel/Kernel.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>

#include <BuildTemplates.h>
#include <ScratchViews.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

template <class AlgTraits>
VariableDensityEnthalpyMMSHOElemKernel<AlgTraits>::VariableDensityEnthalpyMMSHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
: Kernel(),
  unot_(1.0),
  vnot_(1.0),
  wnot_(1.0),
  hnot_(1.0),
  a_(20.0),
  ah_(10.0),
  visc_(0.00125),
  Pref_(100.0),
  MW_(30.0),
  R_(10.0),
  Tref_(300.0),
  Cp_(0.01),
  Pr_(0.8),
  pi_(M_PI)
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
}
//--------------------------------------------------------------------------
template<class AlgTraits>
void
VariableDensityEnthalpyMMSHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& /*lhs*/,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);
  nodal_scalar_workview work_vol(0);
  auto& vol = work_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops_, coords, vol);

  nodal_scalar_view v_rhs(rhs.data());
  tensor_assembly::add_volumetric_source_func(ops_,vol, v_rhs, [&](int k, int j, int i) {
    const auto x = coords(k,j,i,XH);
    const auto y = coords(k,j,i,YH);
    const auto z = coords(k,j,i,ZH);
    const auto src  = -Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * unot_ * stk::math::cos(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) * hnot_ * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::pow(stk::math::cos(ah_ * pi_ * y), 0.2e1) * stk::math::pow(stk::math::cos(ah_ * pi_ * z), 0.2e1) * stk::math::sin(ah_ * pi_ * x) * ah_ * pi_ / Cp_ + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * unot_ * stk::math::sin(a_ * pi_ * x) * a_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * unot_ * stk::math::cos(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) * hnot_ * stk::math::sin(ah_ * pi_ * x) * ah_ * pi_ * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) + Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * vnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) * hnot_ * hnot_ * stk::math::pow(stk::math::cos(ah_ * pi_ * x), 0.2e1) * stk::math::cos(ah_ * pi_ * y) * stk::math::pow(stk::math::cos(ah_ * pi_ * z), 0.2e1) * stk::math::sin(ah_ * pi_ * y) * ah_ * pi_ / Cp_ - Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * vnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * a_ * pi_ * stk::math::sin(a_ * pi_ * z) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) - Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * vnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::sin(ah_ * pi_ * y) * ah_ * pi_ * stk::math::cos(ah_ * pi_ * z) - Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * wnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z) * hnot_ * hnot_ * stk::math::pow(stk::math::cos(ah_ * pi_ * x), 0.2e1) * stk::math::pow(stk::math::cos(ah_ * pi_ * y), 0.2e1) * stk::math::cos(ah_ * pi_ * z) * stk::math::sin(ah_ * pi_ * z) * ah_ * pi_ / Cp_ + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) * a_ * pi_ * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::sin(ah_ * pi_ * z) * ah_ * pi_ + 0.3e1 * visc_ / Pr_ * hnot_ * stk::math::cos(ah_ * pi_ * x) * ah_ * ah_ * pi_ * pi_ * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z);
    return src;
  });
}

INSTANTIATE_KERNEL_HOSGL(VariableDensityEnthalpyMMSHOElemKernel)

} // namespace nalu
} // namespace Sierra
