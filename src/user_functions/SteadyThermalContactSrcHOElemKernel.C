/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/SteadyThermalContactSrcHOElemKernel.h>
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
SteadyThermalContactSrcHOElemKernel<AlgTraits>::SteadyThermalContactSrcHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
: Kernel()
{
  // save off fields
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
}
//--------------------------------------------------------------------------
template<class AlgTraits>
void
SteadyThermalContactSrcHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& /*lhs*/,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  const auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);

  nodal_scalar_workview l_vol(0);
  auto& vol = l_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops_, coords, vol);

  nodal_scalar_view v_rhs(rhs.data());
  const double wave = 2.0 * k_ *  M_PI;
  const double coeff = a_ * wave * wave / 4.0;

  tensor_assembly::add_volumetric_source_func<AlgTraits::polyOrder_, DoubleType>(ops_, vol, v_rhs,
    [&coords, &coeff, &wave](int k, int j, int i) {
    return coeff * (
      stk::math::cos(wave * coords(k, j, i, XH))
    + stk::math::cos(wave * coords(k, j, i, YH))
    + stk::math::cos(wave * coords(k, j, i, ZH))
    );
  });
}

INSTANTIATE_KERNEL_HOSGL(SteadyThermalContactSrcHOElemKernel)

} // namespace nalu
} // namespace Sierra
