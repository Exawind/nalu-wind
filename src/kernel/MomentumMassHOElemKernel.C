/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <kernel/MomentumMassHOElemKernel.h>
#include <kernel/TensorProductCVFEMMomentumBDF2TimeDerivative.h>
#include <master_element/TensorProductCVFEMVolumeMetric.h>


#include <element_promotion/ElementDescription.h>

#include <kernel/Kernel.h>
#include <SolutionOptions.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <AlgTraits.h>
#include <TimeIntegrator.h>

// template and scratch space
#include <BuildTemplates.h>
#include <ScratchViewsHO.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

#include <Teuchos_BLAS.hpp>

// topology
#include <stk_topology/topology.hpp>

#include <KokkosInterface.h>
#include <SimdInterface.h>

namespace sierra{
namespace nalu{

template<class AlgTraits>
MomentumMassHOElemKernel<AlgTraits>::MomentumMassHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel()
{
  // save off fields
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_gathered_nodal_field(*coordinates_, AlgTraits::nDim_);

  VectorFieldType *velocity = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  velocityN_ = &(velocity->field_of_state(stk::mesh::StateN));
  velocityNp1_ = &(velocity->field_of_state(stk::mesh::StateNP1));
  if (velocity->number_of_states() == 2) {
    velocityNm1_ = velocityN_;
  }
  else {
    velocityNm1_ = &(velocity->field_of_state(stk::mesh::StateNM1));

  }
  dataPreReqs.add_gathered_nodal_field(*velocityNm1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*velocityN_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*velocityNp1_, AlgTraits::nDim_);

  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");

  densityN_ = &(density->field_of_state(stk::mesh::StateN));
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  if (density->number_of_states() == 2) {
    densityNm1_ = densityN_;
  }
  else {
    densityNm1_ = &(density->field_of_state(stk::mesh::StateNM1));
  }
  dataPreReqs.add_gathered_nodal_field(*densityNm1_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityN_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_, 1);

  Gp_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
  dataPreReqs.add_gathered_nodal_field(*Gp_, AlgTraits::nDim_);
}
////--------------------------------------------------------------------------
template<typename AlgTraits>
void
MomentumMassHOElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  gamma_[0] = timeIntegrator.get_gamma1() / timeIntegrator.get_time_step();
  gamma_[1] = timeIntegrator.get_gamma2() / timeIntegrator.get_time_step();
  gamma_[2] = timeIntegrator.get_gamma3() / timeIntegrator.get_time_step();
}
//--------------------------------------------------------------------------
template <class AlgTraits> void
MomentumMassHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto v_coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);
  nodal_scalar_workview work_vol(0);
  auto& vol = work_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops_, v_coords, vol);

  auto velm1 = scratchViews.get_scratch_view<nodal_vector_view>(*velocityNm1_);
  auto velp0 = scratchViews.get_scratch_view<nodal_vector_view>(*velocityN_);
  auto velp1 = scratchViews.get_scratch_view<nodal_vector_view>(*velocityNp1_);

  auto rhom1 = scratchViews.get_scratch_view<nodal_scalar_view>(*densityNm1_);
  auto rhop0 = scratchViews.get_scratch_view<nodal_scalar_view>(*densityN_);
  auto rhop1 = scratchViews.get_scratch_view<nodal_scalar_view>(*densityNp1_);

  auto Gp = scratchViews.get_scratch_view<nodal_vector_view>(*Gp_);

  nodal_vector_view v_rhs(rhs.data());
  tensor_assembly::momentum_dt_rhs(ops_, vol, gamma_, Gp, rhom1, rhop0, rhop1, velm1, velp0, velp1, v_rhs);

  matrix_vector_view v_lhs(lhs.data());
  tensor_assembly::momentum_dt_lhs(ops_, vol, gamma_[0], rhop1, v_lhs);
}

INSTANTIATE_KERNEL_HOSGL(MomentumMassHOElemKernel)

} // namespace nalu
} // namespace Sierra
