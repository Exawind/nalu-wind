/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <user_functions/VariableDensityMomentumMMSHOElemKernel.h>
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
VariableDensityMomentumMMSHOElemKernel<AlgTraits>::VariableDensityMomentumMMSHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
: Kernel(),
  unot_(1.0),
  vnot_(1.0),
  wnot_(1.0),
  pnot_(1.0),
  hnot_(1.0),
  a_(20.0),
  ah_(10.0),
  visc_(0.00125),
  Pref_(100.0),
  MW_(30.0),
  R_(10.0),
  Tref_(300.0),
  Cp_(0.01),
  pi_(M_PI),
  twoThirds_(2.0/3.0),
  rhoRef_(1.0),
  gx_(0.0),
  gy_(0.0),
  gz_(0.0)
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);

  rhoRef_ = solnOpts.referenceDensity_;
  gx_ = solnOpts.gravity_.at(0);
  gy_ = solnOpts.gravity_.at(1);
  gz_ = solnOpts.gravity_.at(2);
}
//--------------------------------------------------------------------------
template<class AlgTraits>
void
VariableDensityMomentumMMSHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& /*lhs*/,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);

  nodal_scalar_workview work_vol(0);
  auto& vol = work_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops_, coords, vol);

  nodal_vector_view v_rhs(rhs.data());

  nodal_vector_workview work_sourceVec(0);
  auto& sourceVec = work_sourceVec.view();

  for (int k = 0; k < AlgTraits::nodes1D_; ++k) {
    for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
      for (int i = 0; i <AlgTraits::nodes1D_; ++i) {
        const auto x = coords(k, j, i, XH);
        const auto y = coords(k, j, i, YH);
        const auto z = coords(k, j, i, ZH);

        sourceVec(k,j,i,XH) = Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * unot_ * unot_ * stk::math::pow(stk::math::cos(a_ * pi_ * x), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * hnot_ * stk::math::sin(ah_ * pi_ * x) * ah_ * pi_ * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ - 0.2e1 * Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * unot_ * unot_ * stk::math::cos(a_ * pi_ * x) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * stk::math::sin(a_ * pi_ * x) * a_ * pi_ - Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * vnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::cos(a_ * pi_ * y) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * unot_ * stk::math::cos(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::sin(ah_ * pi_ * y) * ah_ * pi_ * stk::math::cos(ah_ * pi_ * z) / Cp_ + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * vnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * a_ * pi_ * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * unot_ * stk::math::cos(a_ * pi_ * x) - Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * vnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::pow(stk::math::cos(a_ * pi_ * y), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * unot_ * stk::math::cos(a_ * pi_ * x) * a_ * pi_ + Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * wnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::cos(a_ * pi_ * z) * unot_ * stk::math::cos(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * z) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::sin(ah_ * pi_ * z) * ah_ * pi_ / Cp_ - Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * a_ * pi_ * unot_ * stk::math::cos(a_ * pi_ * x) + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::pow(stk::math::cos(a_ * pi_ * z), 0.2e1) * unot_ * stk::math::cos(a_ * pi_ * x) * a_ * pi_ - visc_ * (-(unot_ * stk::math::cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) - vnot_ * stk::math::cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) + wnot_ * stk::math::cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z)) * twoThirds_ + 0.2e1 * unot_ * stk::math::cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z)) - visc_ * (unot_ * stk::math::cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) - vnot_ * stk::math::cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z)) - visc_ * (unot_ * stk::math::cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) + wnot_ * stk::math::cos(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z)) + 0.50e0 * pnot_ * stk::math::sin(0.2e1 * a_ * pi_ * x) * a_ * pi_ - (Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) - rhoRef_) * gx_;
        sourceVec(k,j,i,YH) = -Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * vnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::cos(a_ * pi_ * y) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * unot_ * stk::math::cos(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * hnot_ * stk::math::sin(ah_ * pi_ * x) * ah_ * pi_ * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ - Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * vnot_ * stk::math::pow(stk::math::cos(a_ * pi_ * x), 0.2e1) * a_ * pi_ * stk::math::cos(a_ * pi_ * y) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * unot_ * stk::math::sin(a_ * pi_ * y) + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * vnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::cos(a_ * pi_ * y) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * unot_ * a_ * pi_ * stk::math::sin(a_ * pi_ * y) + Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * vnot_ * vnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::pow(stk::math::cos(a_ * pi_ * y), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::sin(ah_ * pi_ * y) * ah_ * pi_ * stk::math::cos(ah_ * pi_ * z) / Cp_ - 0.2e1 * Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * vnot_ * vnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::cos(a_ * pi_ * y) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * stk::math::sin(a_ * pi_ * y) * a_ * pi_ - Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * wnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z) * vnot_ * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::sin(ah_ * pi_ * z) * ah_ * pi_ / Cp_ + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::sin(a_ * pi_ * y) * stk::math::pow(stk::math::sin(a_ * pi_ * z), 0.2e1) * a_ * pi_ * vnot_ * stk::math::cos(a_ * pi_ * y) - Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::sin(a_ * pi_ * y) * stk::math::pow(stk::math::cos(a_ * pi_ * z), 0.2e1) * vnot_ * stk::math::cos(a_ * pi_ * y) * a_ * pi_ - visc_ * (unot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) - vnot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z)) - visc_ * (-(unot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) - vnot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) + wnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::cos(a_ * pi_ * y) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * z)) * twoThirds_ - 0.2e1 * vnot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z)) - visc_ * (-vnot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) + wnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::cos(a_ * pi_ * y) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * z)) + 0.50e0 * pnot_ * stk::math::sin(0.2e1 * a_ * pi_ * y) * a_ * pi_ - (Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) - rhoRef_) * gy_;
        sourceVec(k,j,i,ZH) = Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * wnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::cos(a_ * pi_ * z) * unot_ * stk::math::cos(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * z) * hnot_ * stk::math::sin(ah_ * pi_ * x) * ah_ * pi_ * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::pow(stk::math::cos(a_ * pi_ * x), 0.2e1) * a_ * pi_ * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::cos(a_ * pi_ * z) * unot_ * stk::math::sin(a_ * pi_ * z) - Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::cos(a_ * pi_ * z) * unot_ * a_ * pi_ * stk::math::sin(a_ * pi_ * z) - Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * wnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z) * vnot_ * stk::math::cos(a_ * pi_ * y) * stk::math::sin(a_ * pi_ * z) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::sin(ah_ * pi_ * y) * ah_ * pi_ * stk::math::cos(ah_ * pi_ * z) / Cp_ - Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::pow(stk::math::cos(a_ * pi_ * y), 0.2e1) * a_ * pi_ * stk::math::cos(a_ * pi_ * z) * vnot_ * stk::math::sin(a_ * pi_ * z) + Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::cos(a_ * pi_ * z) * vnot_ * a_ * pi_ * stk::math::sin(a_ * pi_ * z) + Pref_ * MW_ / R_ * stk::math::pow(hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_, -0.2e1) * wnot_ * wnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::pow(stk::math::cos(a_ * pi_ * z), 0.2e1) * hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::sin(ah_ * pi_ * z) * ah_ * pi_ / Cp_ - 0.2e1 * Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) * wnot_ * wnot_ * stk::math::pow(stk::math::sin(a_ * pi_ * x), 0.2e1) * stk::math::pow(stk::math::sin(a_ * pi_ * y), 0.2e1) * stk::math::cos(a_ * pi_ * z) * stk::math::sin(a_ * pi_ * z) * a_ * pi_ - visc_ * (unot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z) + wnot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z)) - visc_ * (-vnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * a_ * a_ * pi_ * pi_ * stk::math::cos(a_ * pi_ * z) + wnot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z)) - visc_ * (-(unot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z) - vnot_ * stk::math::sin(a_ * pi_ * x) * stk::math::sin(a_ * pi_ * y) * a_ * a_ * pi_ * pi_ * stk::math::cos(a_ * pi_ * z) + wnot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z)) * twoThirds_ + 0.2e1 * wnot_ * stk::math::sin(a_ * pi_ * x) * a_ * a_ * pi_ * pi_ * stk::math::sin(a_ * pi_ * y) * stk::math::cos(a_ * pi_ * z)) + 0.50e0 * pnot_ * stk::math::sin(0.2e1 * a_ * pi_ * z) * a_ * pi_ - (Pref_ * MW_ / R_ / (hnot_ * stk::math::cos(ah_ * pi_ * x) * stk::math::cos(ah_ * pi_ * y) * stk::math::cos(ah_ * pi_ * z) / Cp_ + Tref_) - rhoRef_) * gz_;

        for (int d = 0; d < 3; ++d) {
          sourceVec(k, j, i, d) *= vol(k, j, i);
        }
      }
    }
  }
  ops_.volume(sourceVec, v_rhs);
}

INSTANTIATE_KERNEL_HOSGL(VariableDensityMomentumMMSHOElemKernel)

} // namespace nalu
} // namespace Sierra
