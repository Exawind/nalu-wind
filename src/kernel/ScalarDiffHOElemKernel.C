/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <kernel/ScalarDiffHOElemKernel.h>
#include <kernel/TensorProductCVFEMDiffusion.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>

#include <SolutionOptions.h>

#include <kernel/Kernel.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// template and scratch space
#include <BuildTemplates.h>
#include <ScratchViews.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

// topology
#include <stk_topology/topology.hpp>

// Kokkos
#include <KokkosInterface.h>
#include <SimdInterface.h>

namespace sierra{
namespace nalu{

template <typename AlgTraits>
ScalarDiffHOElemKernel<AlgTraits>::ScalarDiffHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ScalarFieldType *scalarQ,
  ScalarFieldType *diffFluxCoeff,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    scalarQ_(scalarQ),
    diffFluxCoeff_(diffFluxCoeff)
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  dataPreReqs.add_gathered_nodal_field(*coordinates_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*scalarQ, 1);
  dataPreReqs.add_gathered_nodal_field(*diffFluxCoeff, 1);
}
//--------------------------------------------------------------------------
template <typename AlgTraits> void
ScalarDiffHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto v_coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);
  auto v_diff = scratchViews.get_scratch_view<nodal_scalar_view>(*diffFluxCoeff_);
  scs_vector_workview work_metric(0);
  auto& metric = work_metric.view();

  auto start_time_diff = std::chrono::steady_clock::now();
  high_order_metrics::compute_diffusion_metric_linear(ops_, v_coords, v_diff, metric);
  auto end_time_diff = std::chrono::steady_clock::now();
  timer_diff += std::chrono::duration_cast<std::chrono::duration<double>>(end_time_diff-start_time_diff).count();

  matrix_view v_lhs(lhs.data());
  auto start_time_jac = std::chrono::steady_clock::now();
  tensor_assembly::scalar_diffusion_lhs(ops_, metric, v_lhs);
  auto end_time_jac = std::chrono::steady_clock::now();
  timer_jac += std::chrono::duration_cast<std::chrono::duration<double>>(end_time_jac-start_time_jac).count();

  auto scalar = scratchViews.get_scratch_view<nodal_scalar_view>(*scalarQ_);
  nodal_scalar_view v_rhs(rhs.data());
  auto start_time_resid = std::chrono::steady_clock::now();
  tensor_assembly::scalar_diffusion_rhs(ops_, metric, scalar, v_rhs);
  auto end_time_resid = std::chrono::steady_clock::now();
  timer_resid += std::chrono::duration_cast<std::chrono::duration<double>>(end_time_resid - start_time_resid).count();
}

INSTANTIATE_KERNEL_HOSGL(ScalarDiffHOElemKernel)

} // namespace nalu
} // namespace Sierra
