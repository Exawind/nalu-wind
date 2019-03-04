/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <user_functions/TGMMSHOElemKernel.h>
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
TGMMSHOElemKernel<AlgTraits>::TGMMSHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
: Kernel()
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
}
//--------------------------------------------------------------------------
template<class AlgTraits>
void
TGMMSHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& /*lhs*/,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto v_coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);

  nodal_scalar_workview work_vol(0);
  auto& v_vol = work_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops_, v_coords, v_vol);

  nodal_vector_view v_rhs(rhs.data());

  nodal_vector_workview work_sourceVec(0);
  auto& sourceVec = work_sourceVec.view();

  for (int k = 0; k < AlgTraits::nodes1D_; ++k) {
    for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
      for (int i = 0; i <AlgTraits::nodes1D_; ++i) {
        const auto x = v_coords(k,j, i,XH);
        const auto y = v_coords(k,j, i,YH);
        const auto z = v_coords(k,j, i, ZH);
        const auto mu = visc_;

        sourceVec(k,j,i,XH) = (-(M_PI*(1 + stk::math::cos(4*M_PI*y) - 2*stk::math::cos(4*M_PI*z))*stk::math::sin(4*M_PI*x))/8.
            + 6*mu*(M_PI * M_PI)*stk::math::cos(2*M_PI*x)*stk::math::sin(2*M_PI*y)*stk::math::sin(2*M_PI*z));

        sourceVec(k,j,i,YH) = (M_PI*((-2 + stk::math::cos(4*M_PI*x) + stk::math::cos(4*M_PI*z))*stk::math::sin(4*M_PI*y)
            - 48*mu*M_PI*stk::math::cos(2*M_PI*y)*stk::math::sin(2*M_PI*x)*stk::math::sin(2*M_PI*z)))/4.;

        sourceVec(k,j,i,ZH) = (M_PI*(24*mu*M_PI*stk::math::cos(2*M_PI*z)*stk::math::sin(2*M_PI*x)*stk::math::sin(2*M_PI*y)
            + (stk::math::cos(4*M_PI*x) - (stk::math::cos(2*M_PI*y) * stk::math::cos(2*M_PI*y)))*stk::math::sin(4*M_PI*z)))/4.;

        for (int d = 0; d < 3; ++d) {
          sourceVec(k,j,i,d) *= v_vol(k,j,i);
        }
      }
    }
  }
  ops_.volume(sourceVec, v_rhs);
}

INSTANTIATE_KERNEL_HOSGL(TGMMSHOElemKernel)

} // namespace nalu
} // namespace Sierra
