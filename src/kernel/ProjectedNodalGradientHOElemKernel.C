/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <kernel/ProjectedNodalGradientHOElemKernel.h>
#include <kernel/TensorProductCVFEMPNG.h>
#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <master_element/TensorProductCVFEMAdvectionMetric.h>

#include <element_promotion/ElementDescription.h>

#include <kernel/Kernel.h>
#include <SolutionOptions.h>
#include <FieldTypeDef.h>
#include <AlgTraits.h>

// template and scratch space
#include <BuildTemplates.h>
#include <ScratchViewsHO.h>

// stk_mesh/base/fem
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

#include <KokkosInterface.h>
#include <SimdInterface.h>

namespace sierra{
namespace nalu{

template<class AlgTraits>
ProjectedNodalGradientHOElemKernel<AlgTraits>::ProjectedNodalGradientHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  std::string inputDofName, std::string gradDofName,
  ElemDataRequests& dataPreReqs)
  : Kernel()
{
  // save off fields
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);

  q_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, inputDofName);
  ThrowRequireMsg(q_ != nullptr, inputDofName + " field not valid");
  dataPreReqs.add_gathered_nodal_field(*q_, 1);

  dqdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, gradDofName);
  ThrowRequireMsg(dqdx_ != nullptr, gradDofName + " field not valid");
  dataPreReqs.add_gathered_nodal_field(*dqdx_, AlgTraits::nDim_);
}
//--------------------------------------------------------------------------
template <class AlgTraits> void
ProjectedNodalGradientHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  const auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);

  nodal_scalar_workview work_vol(0);
  auto& vol = work_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops_, coords, vol);

  scs_vector_workview work_area;
  auto& area = work_area.view();
  high_order_metrics::compute_area_linear(ops_, coords, area);

  const auto q = scratchViews.get_scratch_view<nodal_scalar_view>(*q_);
  const auto dqdx = scratchViews.get_scratch_view<nodal_vector_view>(*dqdx_);

  nodal_vector_view v_rhs(rhs.data());
  tensor_assembly::green_gauss_rhs(ops_, area, vol, q, dqdx, v_rhs);

  matrix_vector_view v_lhs(lhs.data());
  tensor_assembly::green_gauss_lhs(ops_, vol, v_lhs);
}

INSTANTIATE_KERNEL_HOSGL(ProjectedNodalGradientHOElemKernel)

} // namespace nalu
} // namespace Sierra
