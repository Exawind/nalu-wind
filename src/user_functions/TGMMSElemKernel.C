/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <user_functions/TGMMSElemKernel.h>
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
TGMMSElemKernel<AlgTraits>::TGMMSElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
: Kernel(),
  ipNodeMap_(MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap())

{
  ThrowRequireMsg(AlgTraits::nDim_ == 3, "Only 3D");

  coordinates_ = bulkData.mesh_meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);

  MasterElement* meSCV = MasterElementRepo::get_volume_master_element(AlgTraits::topo_);
  dataPreReqs.add_cvfem_volume_me(meSCV);
  get_scv_shape_fn_data<AlgTraits>([&](double* ptr){ meSCV->shape_fcn(ptr); }, v_shape_function_);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
}
//--------------------------------------------------------------------------
template<class AlgTraits>
void
TGMMSElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& /*lhs*/,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType*>& vol = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;
  SharedMemView<DoubleType**>& v_coords = scratchViews.get_scratch_view_2D(*coordinates_);

  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
    DoubleType x = 0;
    DoubleType y = 0;
    DoubleType z = 0;
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      const DoubleType r = v_shape_function_(ip, n);
      x += r * v_coords(n, 0);
      y += r * v_coords(n, 1);
      z += r * v_coords(n, 2);
    }

    const DoubleType dv = vol(ip);
    const int offset = ipNodeMap_[ip] * AlgTraits::nDim_;

    rhs(offset + 0) += dv*(-(M_PI*(1 + stk::math::cos(4*M_PI*y) - 2*stk::math::cos(4*M_PI*z))*stk::math::sin(4*M_PI*x))/8.
        + 6*mu*(M_PI * M_PI)*stk::math::cos(2*M_PI*x)*stk::math::sin(2*M_PI*y)*stk::math::sin(2*M_PI*z));

    rhs(offset + 1) += dv*(M_PI*((-2 + stk::math::cos(4*M_PI*x) + stk::math::cos(4*M_PI*z))*stk::math::sin(4*M_PI*y)
        - 48*mu*M_PI*stk::math::cos(2*M_PI*y)*stk::math::sin(2*M_PI*x)*stk::math::sin(2*M_PI*z)))/4.;

    rhs(offset + 2) += dv*(M_PI*(24*mu*M_PI*stk::math::cos(2*M_PI*z)*stk::math::sin(2*M_PI*x)*stk::math::sin(2*M_PI*y)
        + (stk::math::cos(4*M_PI*x) - (stk::math::cos(2*M_PI*y) * stk::math::cos(2*M_PI*y)))*stk::math::sin(4*M_PI*z)))/4.;


  }
}

INSTANTIATE_KERNEL(TGMMSElemKernel)

} // namespace nalu
} // namespace Sierra
