/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumCoriolisSrcElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "master_element/TensorOps.h"
#include "SolutionOptions.h"
#include "CoriolisSrc.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

template<class AlgTraits>
MomentumCoriolisSrcElemKernel<AlgTraits>::MomentumCoriolisSrcElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  VectorFieldType* velocity,
  ElemDataRequests& dataPreReqs,
  bool lumped)
  : Kernel(),
    cor_(solnOpts),
    velocityNp1_(velocity->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal()),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap())
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  ThrowRequireMsg(metaData.spatial_dimension() == 3u, "Coriolis source term only available in 3D");

  densityNp1_ = get_field_ordinal(metaData, "density", stk::mesh::StateNP1);
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  MasterElement* meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);

  if ( lumped ) {
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shifted_shape_fcn(ptr);}, v_shape_function_);
  }
  else {
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_);
  }

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
}

template<class AlgTraits>
MomentumCoriolisSrcElemKernel<AlgTraits>::~MomentumCoriolisSrcElemKernel() {}

template<class AlgTraits>
void
MomentumCoriolisSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType**>& v_velocityNp1 = scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType*>& v_densityNp1 = scratchViews.get_scratch_view_1D(densityNp1_);
  SharedMemView<DoubleType*>& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
    const int nnDim = ipNodeMap_[ip] * AlgTraits::nDim_;
    NALU_ALIGNED DoubleType uIp[3] = { 0.0, 0.0, 0.0 };
    DoubleType rhoIp = 0.0;
    for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
      DoubleType r = v_shape_function_(ip,n);
      uIp[0] += r * v_velocityNp1(n, 0);
      uIp[1] += r * v_velocityNp1(n, 1);
      uIp[2] += r * v_velocityNp1(n, 2);
      rhoIp += r * v_densityNp1(n);
    }

    const DoubleType ue = cor_.eastVector_[0] * uIp[0] + cor_.eastVector_[1] * uIp[1] + cor_.eastVector_[2] * uIp[2];
    const DoubleType un = cor_.northVector_[0] * uIp[0] + cor_.northVector_[1] * uIp[1] + cor_.northVector_[2] * uIp[2];
    const DoubleType uu = cor_.upVector_[0] * uIp[0] + cor_.upVector_[1] * uIp[1] + cor_.upVector_[2] * uIp[2];

    const DoubleType ae = +cor_.corfac_ * (un * cor_.sinphi_ - uu * cor_.cosphi_);
    const DoubleType an = -cor_.corfac_ * ue * cor_.sinphi_;
    const DoubleType au = +cor_.corfac_ * ue * cor_.cosphi_;

    const DoubleType ax = ae*cor_.eastVector_[0] + an*cor_.northVector_[0] + au*cor_.upVector_[0];
    const DoubleType ay = ae*cor_.eastVector_[1] + an*cor_.northVector_[1] + au*cor_.upVector_[1];
    const DoubleType az = ae*cor_.eastVector_[2] + an*cor_.northVector_[2] + au*cor_.upVector_[2];

    const DoubleType rho_dvol = rhoIp * v_scv_volume(ip);
    rhs(nnDim + 0) += rho_dvol * ax;
    rhs(nnDim + 1) += rho_dvol * ay;
    rhs(nnDim + 2) += rho_dvol * az;

    // constant Jacobian
    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const int icNdim = ic * AlgTraits::nDim_;
      const DoubleType fac = v_shape_function_(ip, ic) * rho_dvol;

      lhs(nnDim + 0, icNdim + 0) += 0.0; // diagonal terms are zero
      lhs(nnDim + 0, icNdim + 1) += fac * cor_.Jxy_;
      lhs(nnDim + 0, icNdim + 2) += fac * cor_.Jxz_;

      lhs(nnDim + 1, icNdim + 0) -= fac * cor_.Jxy_; // Jyx = - Jxy
      lhs(nnDim + 1, icNdim + 1) += 0.0; // diagonal terms are zero
      lhs(nnDim + 1, icNdim + 2) += fac * cor_.Jyz_;

      lhs(nnDim + 2, icNdim + 0) -= fac * cor_.Jxz_; // Jzx = - Jxz
      lhs(nnDim + 2, icNdim + 1) -= fac * cor_.Jyz_; // Jzy = - Jyz
      lhs(nnDim + 2, icNdim + 2) += 0.0; // diagonal terms are zero
    }
  }
}

INSTANTIATE_KERNEL(MomentumCoriolisSrcElemKernel)

} // namespace nalu
} // namespace Sierra
