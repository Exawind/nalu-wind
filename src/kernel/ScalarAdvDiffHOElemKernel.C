/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <kernel/ScalarAdvDiffHOElemKernel.h>
#include <kernel/TensorProductCVFEMScalarAdvDiff.h>

#include <master_element/TensorProductCVFEMAdvectionMetric.h>
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
ScalarAdvDiffHOElemKernel<AlgTraits>::ScalarAdvDiffHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ScalarFieldType *scalarQ,
  ScalarFieldType *diffFluxCoeff,
  ElemDataRequests& dataPreReqs)
: Kernel()
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();

  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_gathered_nodal_field(*coordinates_, AlgTraits::nDim_);

  scalarQ_ = &scalarQ->field_of_state(stk::mesh::StateNP1);
  dataPreReqs.add_gathered_nodal_field(*scalarQ_, 1);

  diffFluxCoeff_ = &diffFluxCoeff->field_of_state(stk::mesh::StateNone);
  dataPreReqs.add_gathered_nodal_field(*diffFluxCoeff_, 1);

  Gp_ = &meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx")
      ->field_of_state(stk::mesh::StateNone);
  dataPreReqs.add_gathered_nodal_field(*Gp_, AlgTraits::nDim_);

  pressure_ = &meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure")
      ->field_of_state(stk::mesh::StateNone);
  dataPreReqs.add_gathered_nodal_field(*pressure_, 1);

  density_ = &meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density")
      ->field_of_state(stk::mesh::StateNP1);
  dataPreReqs.add_gathered_nodal_field(*density_, 1);

  velocity_ = &meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity")
      ->field_of_state(stk::mesh::StateNP1);
  dataPreReqs.add_gathered_nodal_field(*velocity_, AlgTraits::nDim_);
}
//--------------------------------------------------------------------------
template <typename AlgTraits> void
ScalarAdvDiffHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);

  scs_vector_workview work_metric(0);
  auto& metric = work_metric.view();
  high_order_metrics::compute_laplacian_metric_linear(ops_, coords, metric);

  scs_scalar_workview work_mdot(0);
  auto& mdot = work_mdot.view();
  {
    // FIXME: no ability to save off the new mdot in kernel ATM.  So we just recompute it
    auto vel = scratchViews.get_scratch_view<nodal_vector_view>(*velocity_);
    auto pressure = scratchViews.get_scratch_view<nodal_scalar_view>(*pressure_);
    auto rho = scratchViews.get_scratch_view<nodal_scalar_view>(*density_);
    auto Gp = scratchViews.get_scratch_view<nodal_vector_view>(*Gp_);
    high_order_metrics::compute_mdot_linear(ops_, coords, metric, projTimeScale_,  rho, vel, Gp, pressure, mdot);
  }
  auto diff = scratchViews.get_scratch_view<nodal_scalar_view>(*diffFluxCoeff_);
  high_order_metrics::scale_metric(ops_, diff, metric);

  matrix_view v_lhs(lhs.data());
  tensor_assembly::scalar_advdiff_lhs(ops_, mdot, metric, v_lhs);

  auto scalar = scratchViews.get_scratch_view<nodal_scalar_view>(*scalarQ_);
  nodal_scalar_view v_rhs(rhs.data());
  tensor_assembly::scalar_advdiff_rhs(ops_, mdot, metric, scalar, v_rhs);
}

INSTANTIATE_KERNEL_HOSGL(ScalarAdvDiffHOElemKernel)

} // namespace nalu
} // namespace Sierra
