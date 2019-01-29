/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <kernel/PressurePoissonHOElemKernel.h>
#include <kernel/TensorProductCVFEMDiffusion.h>
#include <kernel/TensorProductCVFEMPressurePoisson.h>
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
PressurePoissonHOElemKernel<AlgTraits>::PressurePoissonHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs,
  bool reduced_sens)
  : Kernel(),
    reduced_sens_(reduced_sens)
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  pressure_ = &meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure")
      ->field_of_state(stk::mesh::StateNone);
  Gp_ = &meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx")
      ->field_of_state(stk::mesh::StateNone);
  density_ = &meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density")
      ->field_of_state(stk::mesh::StateNP1);
  velocity_ = &meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity")
      ->field_of_state(stk::mesh::StateNP1);

  dataPreReqs.add_gathered_nodal_field(*coordinates_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*pressure_, 1);
  dataPreReqs.add_gathered_nodal_field(*density_, 1);
  dataPreReqs.add_gathered_nodal_field(*Gp_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*velocity_, AlgTraits::nDim_);
}

//--------------------------------------------------------------------------
template <typename AlgTraits> void
PressurePoissonHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);

  scs_vector_workview l_metric(0);
  auto& metric = l_metric.view();
  high_order_metrics::compute_laplacian_metric_linear(ops_, coords, metric);


  auto rho = scratchViews.get_scratch_view<nodal_scalar_view>(*density_);
  auto vel = scratchViews.get_scratch_view<nodal_vector_view>(*velocity_);
  auto Gp = scratchViews.get_scratch_view<nodal_vector_view>(*Gp_);
  auto pressure = scratchViews.get_scratch_view<nodal_scalar_view>(*pressure_);
  scs_scalar_workview l_mdot(0);
  auto& mdot = l_mdot.view();
  high_order_metrics::compute_mdot_linear(ops_, coords, metric, projTimeScale_, rho, vel, Gp, pressure, mdot);

  nodal_scalar_view v_rhs(rhs.data());
  tensor_assembly::pressure_poisson_rhs(ops_, projTimeScale_, l_mdot.view(), v_rhs);

  matrix_view v_lhs(lhs.data());
  tensor_assembly::scalar_diffusion_lhs(ops_, metric, v_lhs, reduced_sens_);
}

INSTANTIATE_KERNEL_HOSGL(PressurePoissonHOElemKernel)

} // namespace nalu
} // namespace Sierra
