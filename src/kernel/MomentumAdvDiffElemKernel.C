/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumAdvDiffElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<class AlgTraits>
MomentumAdvDiffElemKernel<AlgTraits>::MomentumAdvDiffElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  VectorFieldType* velocity,
  ScalarFieldType* viscosity,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    viscosity_(viscosity->mesh_meta_data_ordinal()),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)->adjacentNodes()),
    includeDivU_(solnOpts.includeDivU_),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(velocity->name()))
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  velocityNp1_ = velocity->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  massFlowRate_ = get_field_ordinal(metaData, "mass_flow_rate_scs", stk::topology::ELEM_RANK);

  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);

  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){meSCS->shape_fcn(ptr);}, v_shape_function_);
  const bool skewSymmetric = solnOpts.get_skew_symmetric(velocity->name());
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){skewSymmetric ? meSCS->shifted_shape_fcn(ptr) : meSCS->shape_fcn(ptr);}, 
                                   v_adv_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS);

  // fields and data; mdot not gathered as element data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  dataPreReqs.add_element_field(massFlowRate_, AlgTraits::numScsIp_);
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  if ( shiftedGradOp_ )
    dataPreReqs.add_master_element_call(SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);
}

template<class AlgTraits>
MomentumAdvDiffElemKernel<AlgTraits>::~MomentumAdvDiffElemKernel()
{}

template<class AlgTraits>
void
MomentumAdvDiffElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_uIp[AlgTraits::nDim_];

  SharedMemView<DoubleType**>& v_uNp1 = scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType*>& v_viscosity = scratchViews.get_scratch_view_1D(viscosity_);
  SharedMemView<DoubleType*>& v_mdot = scratchViews.get_scratch_view_1D(massFlowRate_);

  SharedMemView<DoubleType**>& v_scs_areav = scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_
    ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted 
    : scratchViews.get_me_views(CURRENT_COORDINATES).dndx;

  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];

    // save off some offsets
    const int ilNdim = il*AlgTraits::nDim_;
    const int irNdim = ir*AlgTraits::nDim_;

    // save off mdot
    const DoubleType tmdot = v_mdot(ip);

    // compute scs point values; sneak in divU
    DoubleType muIp = 0.0;
    DoubleType divU = 0.0;
    for ( int i = 0; i < AlgTraits::nDim_; ++i )
      w_uIp[i] = 0.0;

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_shape_function_(ip,ic);
      const DoubleType rAdv = v_adv_shape_function_(ip,ic);
      muIp += r*v_viscosity(ic);
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        const DoubleType uj = v_uNp1(ic,j);
        w_uIp[j] += rAdv*uj;
        divU += uj*v_dndx(ip,ic,j);
      }
    }

    // assemble advection; rhs only; add in divU stress (explicit)
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {

      // 2nd order central
      const DoubleType uiIp = w_uIp[i];

      // total advection; (pressure contribution in time term)
      const DoubleType aflux = tmdot*uiIp;

      // divU stress term
      const DoubleType divUstress = 2.0/3.0*muIp*divU*v_scs_areav(ip,i)*includeDivU_;

      const int indexL = ilNdim + i;
      const int indexR = irNdim + i;

      // right hand side; L and R
      rhs(indexL) -= aflux + divUstress;
      rhs(indexR) += aflux + divUstress;
    }

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      const int icNdim = ic*AlgTraits::nDim_;

      // advection and diffusion
      const DoubleType lhsfacAdv = v_adv_shape_function_(ip,ic)*tmdot;

      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {

        const int indexL = ilNdim + i;
        const int indexR = irNdim + i;

        // advection operator lhs; rhs handled above
        // lhs; il then ir
        lhs(indexL,icNdim+i) += lhsfacAdv;
        lhs(indexR,icNdim+i) -= lhsfacAdv;

        // viscous stress
        DoubleType lhs_riC_i = 0.0;
        for ( int j = 0; j < AlgTraits::nDim_; ++j ) {

          const DoubleType axj = v_scs_areav(ip,j);
          const DoubleType uj = v_uNp1(ic,j);

          // -mu*dui/dxj*A_j; fixed i over j loop; see below..
          const DoubleType lhsfacDiff_i = -muIp*v_dndx(ip,ic,j)*axj;
          // lhs; il then ir
          lhs_riC_i += lhsfacDiff_i;

          // -mu*duj/dxi*A_j
          const DoubleType lhsfacDiff_j = -muIp*v_dndx(ip,ic,i)*axj;
          // lhs; il then ir
          lhs(indexL,icNdim+j) += lhsfacDiff_j;
          lhs(indexR,icNdim+j) -= lhsfacDiff_j;
          // rhs; il then ir
          rhs(indexL) -= lhsfacDiff_j*uj;
          rhs(indexR) += lhsfacDiff_j*uj;
        }

        // deal with accumulated lhs and flux for -mu*dui/dxj*Aj
        lhs(indexL,icNdim+i) += lhs_riC_i;
        lhs(indexR,icNdim+i) -= lhs_riC_i;
        const DoubleType ui = v_uNp1(ic,i);
        rhs(indexL) -= lhs_riC_i*ui;
        rhs(indexR) += lhs_riC_i*ui;
      }
    }
  }
}

INSTANTIATE_KERNEL(MomentumAdvDiffElemKernel)

}  // nalu
}  // sierra
