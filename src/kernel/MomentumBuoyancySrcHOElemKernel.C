/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <kernel/MomentumBuoyancySrcHOElemKernel.h>
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
MomentumBuoyancySrcHOElemKernel<AlgTraits>::MomentumBuoyancySrcHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    rhoRef_(solnOpts.referenceDensity_),
    gravity_({{solnOpts.gravity_.at(0), solnOpts.gravity_.at(1), solnOpts.gravity_.at(2)}})
{
  ThrowRequire(solnOpts.gravity_.size() == 3u);

  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);

  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  density_ = &(density->field_of_state(stk::mesh::StateNP1));
  dataPreReqs.add_gathered_nodal_field(*density_, 1);

}
//--------------------------------------------------------------------------
template <class AlgTraits> void
MomentumBuoyancySrcHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>&  /* lhs */,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);
  nodal_scalar_workview l_vol(0);
  auto& vol = l_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops_, coords, vol);

  auto rho = scratchViews.get_scratch_view<nodal_scalar_view>(*density_);

  nodal_vector_workview work_buoyancy_src;
  auto& buoyancy_src = work_buoyancy_src.view();

  for (int k = 0; k < AlgTraits::nodes1D_; ++k) {
    for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
      for (int i = 0; i < AlgTraits::nodes1D_; ++i) {
        const DoubleType deltaRhoVol = (rho(k, j, i) - rhoRef_) * vol(k, j, i);
        for (int d = 0; d < 3; ++d) {
          buoyancy_src(k, j, i, d) = gravity_[d] * deltaRhoVol;
        }
      }
    }
  }

  nodal_vector_view v_rhs(rhs.data());
  ops_.volume(buoyancy_src, v_rhs);
}

INSTANTIATE_KERNEL_HOSGL(MomentumBuoyancySrcHOElemKernel)

} // namespace nalu
} // namespace Sierra
