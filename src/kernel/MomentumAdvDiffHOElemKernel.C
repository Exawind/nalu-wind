/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <kernel/MomentumAdvDiffHOElemKernel.h>
#include <kernel/TensorProductCVFEMMomentumAdvDiff.h>

#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/TensorProductCVFEMAdvectionMetric.h>

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
MomentumAdvDiffHOElemKernel<AlgTraits>::MomentumAdvDiffHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  VectorFieldType* velocity,
  ScalarFieldType* viscosity,
  ElemDataRequests& dataPreReqs,
  bool reduced_sens)
  : Kernel(),
    reduced_sens_(reduced_sens)
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();

  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_gathered_nodal_field(*coordinates_, AlgTraits::nDim_);

  viscosity_ = &viscosity->field_of_state(stk::mesh::StateNone);
  dataPreReqs.add_gathered_nodal_field(*viscosity_, 1);

  Gp_ = &meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx")
      ->field_of_state(stk::mesh::StateNone);
  dataPreReqs.add_gathered_nodal_field(*Gp_, AlgTraits::nDim_);

  pressure_ = &meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure")
      ->field_of_state(stk::mesh::StateNone);
  dataPreReqs.add_gathered_nodal_field(*pressure_, 1);

  density_ = &meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density")
      ->field_of_state(stk::mesh::StateNP1);
  dataPreReqs.add_gathered_nodal_field(*density_, 1);

  velocity_ = &velocity->field_of_state(stk::mesh::StateNP1);
  dataPreReqs.add_gathered_nodal_field(*velocity_, AlgTraits::nDim_);
}

//--------------------------------------------------------------------------
template <typename AlgTraits> void
MomentumAdvDiffHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);

  scs_vector_workview work_metric;
  auto& laplacian_metric = work_metric.view();
  high_order_metrics::compute_laplacian_metric_linear(ops_, coords, laplacian_metric);

  auto vel = scratchViews.get_scratch_view<nodal_vector_view>(*velocity_);

  scs_scalar_workview work_mdot;
  auto& mdot = work_mdot.view();
  {
    // FIXME: no ability to save off the new mdot in kernel ATM.  So we just recompute it
    auto pressure = scratchViews.get_scratch_view<nodal_scalar_view>(*pressure_);
    auto rho = scratchViews.get_scratch_view<nodal_scalar_view>(*density_);
    auto Gp = scratchViews.get_scratch_view<nodal_vector_view>(*Gp_);
    high_order_metrics::compute_mdot_linear(ops_, coords, laplacian_metric, projTimeScale_,  rho, vel, Gp, pressure, mdot);
  }

  auto viscosity = scratchViews.get_scratch_view<nodal_scalar_view>(*viscosity_);

  scs_vector_workview work_tau_dot_a;
  auto& tau_dot_a = work_tau_dot_a.view();
  tensor_assembly::area_weighted_face_normal_shear_stress(ops_, coords, viscosity, vel, tau_dot_a);

  nodal_vector_view v_rhs(rhs.data());
  tensor_assembly::momentum_advdiff_rhs(ops_, tau_dot_a,  mdot, vel, v_rhs);

  high_order_metrics::scale_metric(ops_, viscosity, laplacian_metric);
  matrix_vector_view v_lhs(lhs.data()); // reshape lhs
  tensor_assembly::momentum_advdiff_lhs(ops_, mdot, laplacian_metric, v_lhs, reduced_sens_);
}

INSTANTIATE_KERNEL_HOSGL(MomentumAdvDiffHOElemKernel)

} // namespace nalu
} // namespace Sierra
