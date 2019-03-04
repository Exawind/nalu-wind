/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "nso/MomentumNSOSijElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename AlgTraits>
MomentumNSOSijElemKernel<AlgTraits>::MomentumNSOSijElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  VectorFieldType*,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)->adjacentNodes()),
    includeDivU_(solnOpts.includeDivU_),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity"))
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  velocityNp1_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity");
  densityNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  pressure_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "pressure");

  if (solnOpts.does_mesh_move())
    velocityRTM_ = metaData.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = metaData.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity");

  pressure_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "pressure");

  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  Gjp_ = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");

  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){meSCS->shape_fcn(ptr);}, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS);

  // fields
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*velocityRTM_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*Gjp_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_,1);
  dataPreReqs.add_gathered_nodal_field(*pressure_,1);

  // master element data
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  if ( shiftedGradOp_ )
    dataPreReqs.add_master_element_call(SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCS_GIJ, CURRENT_COORDINATES);
}

template<typename AlgTraits>
void
MomentumNSOSijElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_ke         [AlgTraits::nodesPerElement_];
  NALU_ALIGNED DoubleType w_rhoVrtmScs [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_uNp1Scs    [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_dpdxScs    [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_GjpScs     [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_dkedxScs   [AlgTraits::nDim_];

  SharedMemView<DoubleType**>& v_uNp1 = scratchViews.get_scratch_view_2D(*velocityNp1_);
  SharedMemView<DoubleType**>& v_velocityRTM = scratchViews.get_scratch_view_2D(*velocityRTM_);
  SharedMemView<DoubleType**>& v_Gjp = scratchViews.get_scratch_view_2D(*Gjp_);
  SharedMemView<DoubleType*>& v_rhoNp1 = scratchViews.get_scratch_view_1D(*densityNp1_);
  SharedMemView<DoubleType*>& v_pressure = scratchViews.get_scratch_view_1D(*pressure_);

  SharedMemView<DoubleType**>& v_scs_areav = scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_
    ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted
    : scratchViews.get_me_views(CURRENT_COORDINATES).dndx;
  SharedMemView<DoubleType***>& v_gijUpper = scratchViews.get_me_views(CURRENT_COORDINATES).gijUpper;
  SharedMemView<DoubleType***>& v_gijLower = scratchViews.get_me_views(CURRENT_COORDINATES).gijLower;

  // compute nodal ke
  for ( int n = 0; n < AlgTraits::nodesPerElement_; ++n ) {
    DoubleType ke = 0.0;
    for ( int j = 0; j < AlgTraits::nDim_; ++j )
      ke += v_uNp1(n,j)*v_uNp1(n,j)/2.0;
    w_ke[n] = ke;
  }

  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];

    // save off some offsets
    const int ilNdim = il*AlgTraits::nDim_;
    const int irNdim = ir*AlgTraits::nDim_;

    // zero out vector that prevail over all components
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      w_rhoVrtmScs[i] = 0.0;
      w_uNp1Scs[i] = 0.0;
      w_dpdxScs[i] = 0.0;
      w_GjpScs[i] = 0.0;
      w_dkedxScs[i] = 0.0;
    }
    DoubleType rhoScs = 0.0;
    DoubleType divU = 0.0;

    // determine scs values of interest
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      // save off shape function
      const DoubleType r = v_shape_function_(ip,ic);

      // compute scs derivatives and flux derivative
      const DoubleType pressureIC = v_pressure(ic);
      const DoubleType rhoIC = v_rhoNp1(ic);
      const DoubleType keIC = w_ke[ic];
      rhoScs += r*rhoIC;
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        const DoubleType dnj = v_dndx(ip,ic,j);
        const DoubleType vrtmj = v_velocityRTM(ic,j);
        const DoubleType uNp1 = v_uNp1(ic,j);
        const DoubleType Gjp = v_Gjp(ic,j);

        w_rhoVrtmScs[j] += r*rhoIC*vrtmj;
        w_uNp1Scs[j] += r*uNp1;
        w_dpdxScs[j] += pressureIC*dnj;
        w_GjpScs[j] += r*Gjp;
        w_dkedxScs[j] += keIC*dnj;
        divU += uNp1*dnj;
      }
    }

    // form ke residual (based on fine scale momentum residual used in Pstab)
    DoubleType keResidual = 0.0;
    for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
      keResidual += w_uNp1Scs[j]*(w_dpdxScs[j] - w_GjpScs[j])/rhoScs/2.0;
    }

    // denominator for nu as well as terms for "upwind" nu
    DoubleType gUpperMagGradQ = 0.0;
    DoubleType rhoVrtmiGLowerRhoVrtmj = 0.0;
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      const DoubleType dkedxScsi = w_dkedxScs[i];
      const DoubleType rhoVrtmi = w_rhoVrtmScs[i];
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        gUpperMagGradQ += dkedxScsi*v_gijUpper(ip,i,j)*w_dkedxScs[j];
        rhoVrtmiGLowerRhoVrtmj += rhoVrtmi*v_gijLower(ip,i,j)*w_rhoVrtmScs[j];
      }
    }

    // construct nu from ke residual
    const DoubleType nuResidual = rhoScs*stk::math::sqrt((keResidual*keResidual)/(gUpperMagGradQ+small_));

    // construct nu from first-order-like approach; SNL-internal write-up (eq 209)
    // for now, only include advection as full set of terms is too diffuse
    const DoubleType nuFirstOrder = stk::math::sqrt(rhoVrtmiGLowerRhoVrtmj);

    // limit based on first order; Cupw_ is a fudge factor similar to Guermond's approach
    const DoubleType nu = stk::math::min(Cupw_*nuFirstOrder, nuResidual);

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      const int icNdim = ic*AlgTraits::nDim_;

      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {

        // divU stress term
        const DoubleType divUstress = 2.0/3.0*nu*divU*v_scs_areav(ip,i)*v_gijUpper(ip,i,i)*includeDivU_;

        const int indexL = ilNdim + i;
        const int indexR = irNdim + i;

        // NSO diffusion-like term; -2*nu*S^*_ij*g^ij = -nu*g^ij*(dui/dxj + duj/dxi - 2/3*nu*rho*divU*del_ij)*Aj
        DoubleType lhs_riC_i = 0.0;
        for ( int j = 0; j < AlgTraits::nDim_; ++j ) {

          const DoubleType axj = v_scs_areav(ip,j);
          const DoubleType uj = v_uNp1(ic,j);

          // -nu*g^ij*dui/dxj*A_j; fixed i over j loop; see below..
          const DoubleType lhsfacDiff_i = -nu*v_gijUpper(ip,i,j)*v_dndx(ip,ic,j)*axj;
          // lhs; il then ir
          lhs_riC_i += lhsfacDiff_i;

          // -nu*g^ij*duj/dxi*A_j
          const DoubleType lhsfacDiff_j = -nu*v_gijUpper(ip,i,j)*v_dndx(ip,ic,i)*axj;
          // lhs; il then ir
          lhs(indexL,icNdim+j) += lhsfacDiff_j;
          lhs(indexR,icNdim+j) -= lhsfacDiff_j;
          // rhs; il then ir
          rhs(indexL) -= lhsfacDiff_j*uj;
          rhs(indexR) += lhsfacDiff_j*uj;
        }

        // deal with accumulated lhs and flux for -nu*g^ij*dui/dxj*Aj
        lhs(indexL,icNdim+i) += lhs_riC_i;
        lhs(indexR,icNdim+i) -= lhs_riC_i;
        const DoubleType ui = v_uNp1(ic,i);
        rhs(indexL) -= lhs_riC_i*ui + divUstress;
        rhs(indexR) += lhs_riC_i*ui + divUstress;
      }
    }
  }
}

INSTANTIATE_KERNEL(MomentumNSOSijElemKernel)

}  // nalu
}  // sierra
