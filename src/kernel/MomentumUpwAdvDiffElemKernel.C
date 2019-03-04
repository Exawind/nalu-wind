/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumUpwAdvDiffElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"
#include "EquationSystem.h"
#include "PecletFunction.h"

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
MomentumUpwAdvDiffElemKernel<AlgTraits>::MomentumUpwAdvDiffElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  EquationSystem* eqSystem,
  VectorFieldType* velocity,
  ScalarFieldType* viscosity,
  GenericFieldType* Gju,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    solnOpts_(solnOpts),
    viscosity_(viscosity->mesh_meta_data_ordinal()),
    Gju_(Gju->mesh_meta_data_ordinal()),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)->adjacentNodes()),
    dofName_(velocity->name()),
    alpha_(solnOpts.get_alpha_factor(dofName_)),
    alphaUpw_(solnOpts.get_alpha_upw_factor(dofName_)),
    hoUpwind_(solnOpts.get_upw_factor(dofName_)),
    useLimiter_(solnOpts.primitive_uses_limiter(dofName_)),
    om_alpha_(1.0 - alpha_),
    om_alphaUpw_(1.0 - alphaUpw_),
    includeDivU_(solnOpts.includeDivU_),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(velocity->name())),
    pecletFunction_(eqSystem->create_peclet_function<DoubleType>(dofName_))
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  const std::string vrtmName = solnOpts_.does_mesh_move()? "velocity_rtm" : "velocity";

  velocityNp1_ = velocity->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();
  velocityRTM_ = get_field_ordinal(metaData, vrtmName);
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  density_ = get_field_ordinal(metaData, "density");
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
  dataPreReqs.add_gathered_nodal_field(Gju_, AlgTraits::nDim_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(velocityRTM_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(density_, 1);
  dataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  dataPreReqs.add_element_field(massFlowRate_, AlgTraits::numScsIp_);
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  if ( shiftedGradOp_ )
    dataPreReqs.add_master_element_call(SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);
}

template<class AlgTraits>
MomentumUpwAdvDiffElemKernel<AlgTraits>::~MomentumUpwAdvDiffElemKernel()
{
  delete pecletFunction_;
}

template<typename AlgTraits>
void
MomentumUpwAdvDiffElemKernel<AlgTraits>::setup(const TimeIntegrator&)
{
  alpha_ = solnOpts_.get_alpha_factor(dofName_);
  alphaUpw_ = solnOpts_.get_alpha_upw_factor(dofName_);
  hoUpwind_ = solnOpts_.get_upw_factor(dofName_);
  useLimiter_ = solnOpts_.primitive_uses_limiter(dofName_);

  // one minus flavor..
  om_alpha_ = 1.0-alpha_;
  om_alphaUpw_ = 1.0-alphaUpw_;
}

template<class AlgTraits>
void
MomentumUpwAdvDiffElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_uIp[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_uIpL[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_uIpR[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_coordIp[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_duL[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_duR[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_limitL[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_limitR[AlgTraits::nDim_];
  for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
    w_limitL[i] = 1.0;
    w_limitR[i] = 1.0;
  }
  
  SharedMemView<DoubleType***>& v_Gju = scratchViews.get_scratch_view_3D(Gju_);
  SharedMemView<DoubleType**>& v_coordinates = scratchViews.get_scratch_view_2D(coordinates_);
  SharedMemView<DoubleType**>& v_uNp1 = scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType*>& v_density = scratchViews.get_scratch_view_1D(density_);
  SharedMemView<DoubleType*>& v_viscosity = scratchViews.get_scratch_view_1D(viscosity_);
  SharedMemView<DoubleType*>& v_mdot = scratchViews.get_scratch_view_1D(massFlowRate_);
  SharedMemView<DoubleType**>& v_velocityRTM = scratchViews.get_scratch_view_2D(velocityRTM_);
 
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
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      w_uIp[i] = 0.0;
      w_coordIp[i] = 0.0;
    }

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_shape_function_(ip,ic);
      const DoubleType rAdv = v_adv_shape_function_(ip,ic);
      muIp += r*v_viscosity(ic);
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        const DoubleType uj = v_uNp1(ic,j);
        w_uIp[j] += rAdv*uj;
        w_coordIp[j] += rAdv*v_coordinates(ic,j);
        divU += uj*v_dndx(ip,ic,j);
      }
    }

    // udotx; left and right extrapolation
    DoubleType udotx = 0.0;
    for (int i = 0; i < AlgTraits::nDim_; ++i ) {
      // udotx
      const DoubleType dxi = v_coordinates(ir,i) - v_coordinates(il,i);
      const DoubleType ui = 0.5*(v_velocityRTM(il,i) + v_velocityRTM(ir,i));
      udotx += ui*dxi;
      // extrapolation du
      w_duL[i] = 0.0;
      w_duR[i] = 0.0;
      for(int j = 0; j < AlgTraits::nDim_; ++j ) {
        const DoubleType dxjL = w_coordIp[j] - v_coordinates(il,j);
        const DoubleType dxjR = v_coordinates(ir,j) - w_coordIp[j];
        w_duL[i] += dxjL*v_Gju(il,i,j);
        w_duR[i] += dxjR*v_Gju(ir,i,j);
      }
    }
    
    // Peclet factor; along the edge is fine
    const DoubleType diffIp = 0.5*(v_viscosity(il)/v_density(il)
                                   + v_viscosity(ir)/v_density(ir));
    const DoubleType pecFuncArg = stk::math::abs(udotx)/(diffIp+small_);
    const DoubleType pecfac = pecletFunction_->execute(pecFuncArg);
    const DoubleType om_pecfac = 1.0-pecfac;
    
    // determine limiter if applicable
    if ( useLimiter_ ) {
      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
        const DoubleType dq = v_uNp1(ir,i) - v_uNp1(il,i);
        const DoubleType dqMl = 2.0*2.0*w_duL[i] - dq;
        const DoubleType dqMr = 2.0*2.0*w_duR[i] - dq;
        w_limitL[i] = van_leer(dqMl, dq);
        w_limitR[i] = van_leer(dqMr, dq);
      }
    }
    
    // final upwind extrapolation; with limiter
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      w_uIpL[i] = v_uNp1(il,i) + w_duL[i]*hoUpwind_*w_limitL[i];
      w_uIpR[i] = v_uNp1(ir,i) - w_duR[i]*hoUpwind_*w_limitR[i];
    }
    
    // assemble advection; rhs only; add in divU stress (explicit)
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {

      // 2nd order central
      const DoubleType uiIp = w_uIp[i];

      // upwind
      const DoubleType uiUpwind = stk::math::if_then_else(tmdot > 0, 
                                                          alphaUpw_*w_uIpL[i] + (om_alphaUpw_)*uiIp,
                                                          alphaUpw_*w_uIpR[i] + (om_alphaUpw_)*uiIp);

      // generalized central (2nd and 4th order)
      const DoubleType uiHatL = alpha_*w_uIpL[i] + om_alpha_*uiIp;
      const DoubleType uiHatR = alpha_*w_uIpR[i] + om_alpha_*uiIp;
      const DoubleType uiCds = 0.5*(uiHatL + uiHatR);
      
      // total advection; pressure contribution in time term
      const DoubleType aflux = tmdot*(pecfac*uiUpwind + om_pecfac*uiCds);

      // divU stress term
      const DoubleType divUstress = 2.0/3.0*muIp*divU*v_scs_areav(ip,i)*includeDivU_;

      const int indexL = ilNdim + i;
      const int indexR = irNdim + i;

      // right hand side; L and R
      rhs(indexL) -= aflux + divUstress;
      rhs(indexR) += aflux + divUstress;

      // advection operator sens; all but central
      
      // upwind advection (includes 4th); left node
      const DoubleType alhsfacL = 0.5*(tmdot+stk::math::abs(tmdot))*pecfac*alphaUpw_
        + 0.5*alpha_*om_pecfac*tmdot;
      lhs(indexL, indexL) += alhsfacL;
      lhs(indexR, indexL) -= alhsfacL;
      
      // upwind advection (includes 4th); right node
      const DoubleType alhsfacR = 0.5*(tmdot-stk::math::abs(tmdot))*pecfac*alphaUpw_
        + 0.5*alpha_*om_pecfac*tmdot;
      lhs(indexR, indexR) -= alhsfacR;
      lhs(indexL, indexR) += alhsfacR;
    }

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      const int icNdim = ic*AlgTraits::nDim_;

      // advection and diffusion
      
      // upwind (il/ir) handled above; collect terms on alpha and alphaUpw
      const DoubleType lhsfacAdv = v_adv_shape_function_(ip,ic)*tmdot*(pecfac*om_alphaUpw_ + om_pecfac*om_alpha_);

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

template<class AlgTraits>
DoubleType
MomentumUpwAdvDiffElemKernel<AlgTraits>::van_leer(
  const DoubleType &dqm,
  const DoubleType &dqp)
{
  DoubleType limit = (2.0*(dqm*dqp+stk::math::abs(dqm*dqp))) /
    ((dqm+dqp)*(dqm+dqp)+small_);
  return limit;
}

INSTANTIATE_KERNEL(MomentumUpwAdvDiffElemKernel)

}  // nalu
}  // sierra
