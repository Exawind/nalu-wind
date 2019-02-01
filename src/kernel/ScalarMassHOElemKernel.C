/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <kernel/ScalarMassHOElemKernel.h>
#include <kernel/TensorProductCVFEMScalarBDF2TimeDerivative.h>
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
ScalarMassHOElemKernel<AlgTraits>::ScalarMassHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ScalarFieldType *scalarQ,
  ElemDataRequests& dataPreReqs)
  : Kernel()
{
  // save off fields
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);

  ThrowRequireMsg(scalarQ != nullptr, "Scalar field not valid");
  scalarNp1_ = &(scalarQ->field_of_state(stk::mesh::StateNP1));
  dataPreReqs.add_gathered_nodal_field(*scalarNp1_, 1);

  scalarN_   = &(scalarQ->field_of_state(stk::mesh::StateN));
  dataPreReqs.add_gathered_nodal_field(*scalarN_, 1);

  scalarNm1_ = (scalarQ->number_of_states() == 2) ? scalarN_ : &(scalarQ->field_of_state(stk::mesh::StateNM1));
  dataPreReqs.add_gathered_nodal_field(*scalarNm1_, 1);


  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityN_ = &(density->field_of_state(stk::mesh::StateN));
  dataPreReqs.add_gathered_nodal_field(*densityN_, 1);

  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  dataPreReqs.add_gathered_nodal_field(*densityNp1_, 1);

  densityNm1_ = (scalarQ->number_of_states() == 2) ? densityN_ : &(density->field_of_state(stk::mesh::StateNM1));
  dataPreReqs.add_gathered_nodal_field(*densityNm1_, 1);
}
//--------------------------------------------------------------------------
template<typename AlgTraits>
void
ScalarMassHOElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  gamma_[0] = timeIntegrator.get_gamma1() / timeIntegrator.get_time_step();
  gamma_[1] = timeIntegrator.get_gamma2() / timeIntegrator.get_time_step();
  gamma_[2] = timeIntegrator.get_gamma3() / timeIntegrator.get_time_step();
}

//--------------------------------------------------------------------------
template <class AlgTraits> void
ScalarMassHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);
  nodal_scalar_workview l_vol(0);
  auto& vol = l_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops_, coords, vol);

  auto phim1 = scratchViews.get_scratch_view<nodal_scalar_view>(*scalarNm1_);
  auto phip0 = scratchViews.get_scratch_view<nodal_scalar_view>(*scalarN_);
  auto phip1 = scratchViews.get_scratch_view<nodal_scalar_view>(*scalarNp1_);

  auto rhom1 = scratchViews.get_scratch_view<nodal_scalar_view>(*densityNm1_);
  auto rhop0 = scratchViews.get_scratch_view<nodal_scalar_view>(*densityN_);
  auto rhop1 = scratchViews.get_scratch_view<nodal_scalar_view>(*densityNp1_);

  nodal_scalar_view v_rhs(rhs.data());
  tensor_assembly::scalar_dt_rhs(ops_, vol, gamma_, rhom1, rhop0, rhop1, phim1, phip0, phip1, v_rhs);

  matrix_view v_lhs(lhs.data());
  tensor_assembly::scalar_dt_lhs(ops_, vol, gamma_[0], rhop1, v_lhs);
}

INSTANTIATE_KERNEL_HOSGL(ScalarMassHOElemKernel)

} // namespace nalu
} // namespace Sierra
