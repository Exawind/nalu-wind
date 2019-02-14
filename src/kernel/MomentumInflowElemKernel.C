/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumInflowElemKernel.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"
#include "master_element/TensorOps.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename BcAlgTraits>
MomentumInflowElemKernel<BcAlgTraits>::MomentumInflowElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions &solnOpts,
  ElemDataRequests& faceDataPreReqs,
  ElemDataRequests& elemDataPreReqs)
  : Kernel(),
    skewSymmetric_(solnOpts.get_skew_symmetric("velocity")),
    includeDivU_(solnOpts.includeDivU_),
    projTimeScale_(1.0),
    ipNodeMap_(MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_)->ipNodeMap())
 {
  const auto& meta = bulkData.mesh_meta_data();

  auto ubcName = solnOpts.activateOpenMdotCorrection_ ? "velocity_bc" : "cont_velocity_bc";
  velocityBC_ = meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, ubcName);
  faceDataPreReqs.add_gathered_nodal_field(*velocityBC_, dim);

  coordinates_ = meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  elemDataPreReqs.add_coordinates_field(*coordinates_, dim, CURRENT_COORDINATES);

  velocity_ = &meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity")
          ->field_of_state(stk::mesh::StateNP1);
  faceDataPreReqs.add_gathered_nodal_field(*velocity_, dim);
  elemDataPreReqs.add_gathered_nodal_field(*velocity_, dim);

  density_ = &meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density")
      ->field_of_state(stk::mesh::StateNP1);
  faceDataPreReqs.add_gathered_nodal_field(*density_, 1);

  auto viscName = solnOpts.turbulenceModel_ == LAMINAR ? "viscosity" : "effective_viscosity";
  visc_ = meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);
  faceDataPreReqs.add_gathered_nodal_field(*visc_, 1);
  elemDataPreReqs.add_gathered_nodal_field(*visc_, 1);

  areav_ = meta.get_field<GenericFieldType>(meta.side_rank(), "exposed_area_vector");
  faceDataPreReqs.add_face_field(*areav_, BcAlgTraits::numFaceIp_, dim);

  auto* meFC = MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_);
  faceDataPreReqs.add_cvfem_face_me(meFC);

  meSCS_ = MasterElementRepo::get_surface_master_element(BcAlgTraits::elemTopo_);
  elemDataPreReqs.add_cvfem_surface_me(meSCS_);

  elemDataPreReqs.add_master_element_call(SCS_FACE_GRAD_OP, CURRENT_COORDINATES);
  get_face_shape_fn_data<BcAlgTraits>([&](double* ptr) { meFC->shape_fcn(ptr);}, vf_shape_function_);
  get_face_shape_fn_data<BcAlgTraits>([&](double* ptr) {
    skewSymmetric_ ? meFC->shifted_shape_fcn(ptr) : meFC->shape_fcn(ptr);}, vf_adv_shape_function_);
}

template<typename BcAlgTraits>
void
MomentumInflowElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& faceViews,
  ScratchViews<DoubleType>& elemViews,
  int elemFaceOrdinal)
{
  const int* face_node_ordinals = meSCS_->side_node_ordinals(elemFaceOrdinal);

  SharedMemView<DoubleType*>& face_rho = faceViews.get_scratch_view_1D(*density_);
  SharedMemView<DoubleType*>& face_mu = faceViews.get_scratch_view_1D(*visc_);
  SharedMemView<DoubleType**>& face_velocity = faceViews.get_scratch_view_2D(*velocity_);
  SharedMemView<DoubleType**>& face_velocityBC = faceViews.get_scratch_view_2D(*velocityBC_);
  SharedMemView<DoubleType**>& face_areav = faceViews.get_scratch_view_2D(*areav_);

  SharedMemView<DoubleType***>& v_dndx = elemViews.get_me_views(CURRENT_COORDINATES).dndx_fc_scs;
  SharedMemView<DoubleType**>& elem_velocity = elemViews.get_scratch_view_2D(*velocity_);

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    const int nn = meSCS_->ipNodeMap(elemFaceOrdinal)[ip]; // "Right"

    NALU_ALIGNED Kokkos::Array<DoubleType, 3> uIp = {{0,0,0}};
    NALU_ALIGNED Kokkos::Array<DoubleType, 3> uBCIp = {{0,0,0}};
    NALU_ALIGNED Kokkos::Array<DoubleType, 3> rhouBCIp = {{0,0,0}};

    DoubleType viscIp = 0;
    for (int n = 0; n <  BcAlgTraits::nodesPerFace_; ++n) {
      const auto r = vf_shape_function_(ip, n);
      const auto rAdv = vf_adv_shape_function_(ip, n);
      viscIp += r * face_mu(n);
      for (int d = 0; d < dim; ++d) {
        uIp[d] += rAdv * face_velocity(n, d);
        uBCIp[d] += rAdv * face_velocityBC(n, d);
        rhouBCIp[d] += rAdv * face_rho(n) * face_velocityBC(n, d);
      }
    }
    const auto* areavIp = &face_areav(ip, 0);
    const auto mdotBCIp = ddot(rhouBCIp.data(), areavIp, dim);

    auto areaWeightedInverseLengthScale = DoubleType(0.0);
    for (int n = 0; n < BcAlgTraits::nodesPerFace_; ++n) {
      areaWeightedInverseLengthScale += ddot(&v_dndx(ip, face_node_ordinals[n], 0), areavIp, dim);
    }

    // no need for a penalty if it's outflow
    const auto inflow_penalty = stk::math::if_then_else_zero(mdotBCIp <= 0, DoubleType(inviscid_penalty));
    const auto penaltyFac =  inflow_penalty * mdotBCIp - viscous_penalty * viscIp * areaWeightedInverseLengthScale;

    for (int d = 0; d < dim; ++d) {
      const int indexR = nn * dim + d;
      for (int n = 0; n <  BcAlgTraits::nodesPerFace_; ++n) {
        lhs(indexR, face_node_ordinals[n] * dim + d) += vf_adv_shape_function_(ip, n) * mdotBCIp;
      }
      rhs(nn * dim + d) -= mdotBCIp * uIp[d];
    }

    for (int d = 0; d < dim; ++d) {
      const int indexR = nn * dim + d;
      for (int n = 0; n < BcAlgTraits::nodesPerFace_; ++n) {
        lhs(indexR, face_node_ordinals[n] * dim + d) += -vf_adv_shape_function_(ip, n) * penaltyFac;
      }
      rhs(indexR) -= -penaltyFac * (uIp[d] - uBCIp[d]);
    }

    DoubleType divU = 0;
    for (int n = 0; n < BcAlgTraits::nodesPerElement_; ++n) {
      for (int d = 0; d < dim; ++d) {
        divU += elem_velocity(n, d) * v_dndx(ip, n, d);
      }
    }

    NALU_ALIGNED Kokkos::Array<DoubleType, 9> viscFluxIp = {{0,0,0,0,0,0,0,0,0}};
    for (int d = 0; d < dim; ++d) {
      viscFluxIp[d*dim+d] = -includeDivU_ * 2.0/dim * viscIp * divU;
    }

    for ( int n = 0; n < BcAlgTraits::nodesPerElement_; ++n ) {
      for ( int j = 0; j < dim; ++j ) {
        for ( int i = 0; i < dim; ++i ) {
          viscFluxIp[j*dim+i] += viscIp * (v_dndx(ip, n, j) * elem_velocity(n, i) + v_dndx(ip, n, i) * elem_velocity(n, j));
        }
      }
    }

    NALU_ALIGNED Kokkos::Array<DoubleType, 3> faceViscFlux = {{0,0,0}};
    for ( int d = 0; d < dim; ++d ) {
      faceViscFlux[d] = -ddot(&viscFluxIp[d*dim], areavIp, dim);
    }

    for (int j = 0; j < dim; ++j) {
      for (int n = 0; n < BcAlgTraits::nodesPerElement_; ++n) {
        lhs(nn * dim + j, n * dim + j) += -viscIp * ddot(&v_dndx(ip, n, 0), areavIp, dim);
        for (int i = 0; i < dim; ++i) {
          lhs(nn * dim + j, n * dim + i) += -viscIp * v_dndx(ip, n, i) * areavIp[j];
        }
      }
      rhs(nn * dim + j) -= faceViscFlux[j];
    }
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(MomentumInflowElemKernel)

}  // nalu
}  // sierra
