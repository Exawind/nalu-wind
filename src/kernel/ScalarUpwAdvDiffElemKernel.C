/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ScalarUpwAdvDiffElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
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

template<typename AlgTraits>
ScalarUpwAdvDiffElemKernel<AlgTraits>::ScalarUpwAdvDiffElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  EquationSystem* eqSystem,
  ScalarFieldType* scalarQ,
  VectorFieldType* Gjq,
  ScalarFieldType* diffFluxCoeff,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    solnOpts_(solnOpts),
    scalarQ_(scalarQ->mesh_meta_data_ordinal()),
    Gjq_(Gjq->mesh_meta_data_ordinal()),
    diffFluxCoeff_(diffFluxCoeff->mesh_meta_data_ordinal()),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)->adjacentNodes()),
    dofName_(scalarQ->name()),
    alpha_(solnOpts.get_alpha_factor(dofName_)),
    alphaUpw_(solnOpts.get_alpha_upw_factor(dofName_)),
    hoUpwind_(solnOpts.get_upw_factor(dofName_)),
    useLimiter_(solnOpts.primitive_uses_limiter(dofName_)),
    om_alpha_(1.0 - alpha_),
    om_alphaUpw_(1.0 - alphaUpw_),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(scalarQ->name())),
    pecletFunction_(eqSystem->create_peclet_function<DoubleType>(dofName_))
{
  // Save of required fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  massFlowRate_ = get_field_ordinal(metaData, "mass_flow_rate_scs", stk::topology::ELEM_RANK);
  density_ = get_field_ordinal(metaData, "density");

  const std::string vrtm_name = solnOpts.does_mesh_move()? "velocity_rtm" : "velocity";
  velocityRTM_ = get_field_ordinal(metaData, vrtm_name);

  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
  
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){meSCS->shape_fcn(ptr);}, v_shape_function_);
  const bool skewSymmetric = solnOpts.get_skew_symmetric(scalarQ->name());
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){skewSymmetric ? meSCS->shifted_shape_fcn(ptr) : meSCS->shape_fcn(ptr);}, 
                                   v_adv_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS);

  // fields and data; mdot not gathered
  dataPreReqs.add_gathered_nodal_field(velocityRTM_, AlgTraits::nDim_);
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(Gjq_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(scalarQ_, 1);
  dataPreReqs.add_gathered_nodal_field(density_, 1);
  dataPreReqs.add_gathered_nodal_field(diffFluxCoeff_, 1);
  dataPreReqs.add_element_field(massFlowRate_, AlgTraits::numScsIp_);
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  if ( shiftedGradOp_ )
    dataPreReqs.add_master_element_call(SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);
}

template<typename AlgTraits>
ScalarUpwAdvDiffElemKernel<AlgTraits>::~ScalarUpwAdvDiffElemKernel()
{
  delete pecletFunction_;
}

template<typename AlgTraits>
void
ScalarUpwAdvDiffElemKernel<AlgTraits>::setup(const TimeIntegrator&)
{
  alpha_ = solnOpts_.get_alpha_factor(dofName_);
  alphaUpw_ = solnOpts_.get_alpha_upw_factor(dofName_);
  hoUpwind_ = solnOpts_.get_upw_factor(dofName_);
  useLimiter_ = solnOpts_.primitive_uses_limiter(dofName_);

  // one minus flavor..
  om_alpha_ = 1.0-alpha_;
  om_alphaUpw_ = 1.0-alphaUpw_;
}

template<typename AlgTraits>
void
ScalarUpwAdvDiffElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  /// Scratch space to hold coordinates at the integration point
  NALU_ALIGNED DoubleType w_coordIp[AlgTraits::nDim_];

  SharedMemView<DoubleType**>& v_velocityRTM = scratchViews.get_scratch_view_2D(velocityRTM_);
  SharedMemView<DoubleType**>& v_coordinates = scratchViews.get_scratch_view_2D(coordinates_);
  SharedMemView<DoubleType**>& v_Gjq = scratchViews.get_scratch_view_2D(Gjq_);
  SharedMemView<DoubleType*>& v_scalarQ = scratchViews.get_scratch_view_1D(scalarQ_);
  SharedMemView<DoubleType*>& v_density = scratchViews.get_scratch_view_1D(density_);
  SharedMemView<DoubleType*>& v_diffFluxCoeff = scratchViews.get_scratch_view_1D(diffFluxCoeff_);
  SharedMemView<DoubleType*>& v_mdot = scratchViews.get_scratch_view_1D(massFlowRate_);

  SharedMemView<DoubleType**>& v_scs_areav = scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_
    ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted
    : scratchViews.get_me_views(CURRENT_COORDINATES).dndx;

  // start the assembly
  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];

    // save off mdot
    const DoubleType tmdot = v_mdot(ip);

    // zero out values of interest for this ip
    for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
      w_coordIp[j] = 0.0;
    }

    // compute ip property and
    DoubleType qIp = 0.0;
    DoubleType diffFluxCoeffIp = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_shape_function_(ip,ic);
      const DoubleType rAdv = v_adv_shape_function_(ip,ic);
      qIp += rAdv*v_scalarQ(ic);
      diffFluxCoeffIp += r*v_diffFluxCoeff(ic);
      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
        w_coordIp[i] += rAdv*v_coordinates(ic,i);
      }
    }

    // Peclet factor; along the edge
    const DoubleType diffIp = 0.5*(v_diffFluxCoeff(il)/v_density(il)
                               + v_diffFluxCoeff(ir)/v_density(ir));
    DoubleType udotx = 0.0;
    for(int j = 0; j < AlgTraits::nDim_; ++j ) {
      const DoubleType dxj = v_coordinates(ir,j) - v_coordinates(il,j);;
      const DoubleType uj = 0.5*(v_velocityRTM(il,j) + v_velocityRTM(ir,j));
      udotx += uj*dxj;
    }
    const DoubleType tmp = stk::math::abs(udotx)/(diffIp+small_);
    const DoubleType pecfac = pecletFunction_->execute(tmp);
    const DoubleType om_pecfac = 1.0-pecfac;

    // left and right extrapolation
    DoubleType dqL = 0.0;
    DoubleType dqR = 0.0;
    for(int j = 0; j < AlgTraits::nDim_; ++j ) {
      const DoubleType dxjL = w_coordIp[j] - v_coordinates(il,j);
      const DoubleType dxjR = v_coordinates(ir,j) - w_coordIp[j];
      dqL += dxjL*v_Gjq(il,j);
      dqR += dxjR*v_Gjq(ir,j);
    }

    // add limiter if appropriate
    DoubleType limitL = 1.0;
    DoubleType limitR = 1.0;
    if ( useLimiter_ ) {
      const DoubleType dq = v_scalarQ(ir) - v_scalarQ(il);
      const DoubleType dqMl = 2.0*2.0*dqL - dq;
      const DoubleType dqMr = 2.0*2.0*dqR - dq;
      limitL = van_leer(dqMl, dq);
      limitR = van_leer(dqMr, dq);
    }

    // extrapolated; for now limit (along edge is fine)
    const DoubleType qIpL = v_scalarQ(il) + dqL*hoUpwind_*limitL;
    const DoubleType qIpR = v_scalarQ(ir) - dqR*hoUpwind_*limitR;

    // upwind
    const DoubleType qUpwind = stk::math::if_then_else(tmdot > 0,
                                                       alphaUpw_*qIpL + om_alphaUpw_*qIp,
                                                       alphaUpw_*qIpR + om_alphaUpw_*qIp);

    // generalized central (2nd and 4th order)
    const DoubleType qHatL = alpha_*qIpL + om_alpha_*qIp;
    const DoubleType qHatR = alpha_*qIpR + om_alpha_*qIp;
    const DoubleType qCds = 0.5*(qHatL + qHatR);

    // total advection
    const DoubleType aflux = tmdot*(pecfac*qUpwind + om_pecfac*qCds);

    // right hand side; L and R
    rhs(il) -= aflux;
    rhs(ir) += aflux;

    // upwind advection (includes 4th); left node
    const DoubleType alhsfacL = 0.5*(tmdot+stk::math::abs(tmdot))*pecfac*alphaUpw_
      + 0.5*alpha_*om_pecfac*tmdot;
    lhs(il,il) += alhsfacL;
    lhs(ir,il) -= alhsfacL;

    // upwind advection; right node
    const DoubleType alhsfacR = 0.5*(tmdot-stk::math::abs(tmdot))*pecfac*alphaUpw_
      + 0.5*alpha_*om_pecfac*tmdot;
    lhs(ir,ir) -= alhsfacR;
    lhs(il,ir) += alhsfacR;

    // advection and diffusion
    DoubleType qDiff = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      // upwind (il/ir) handled above; collect terms on alpha and alphaUpw
      const DoubleType lhsfacAdv = v_adv_shape_function_(ip,ic)*tmdot*(pecfac*om_alphaUpw_ + om_pecfac*om_alpha_);

      // advection operator lhs; rhs handled above
      lhs(il,ic) += lhsfacAdv;
      lhs(ir,ic) -= lhsfacAdv;

      // diffusion
      DoubleType lhsfacDiff = 0.0;
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        lhsfacDiff += -diffFluxCoeffIp*v_dndx(ip,ic,j)*v_scs_areav(ip,j);
      }
      qDiff += lhsfacDiff*v_scalarQ(ic);

      // lhs; il then ir
      lhs(il,ic) +=  lhsfacDiff;
      lhs(ir,ic) -= lhsfacDiff;
    }

    // rhs; il then ir
    rhs(il) -= qDiff;
    rhs(ir) += qDiff;
  }
}

template<class AlgTraits>
DoubleType
ScalarUpwAdvDiffElemKernel<AlgTraits>::van_leer(
  const DoubleType &dqm,
  const DoubleType &dqp)
{
  DoubleType limit = (2.0*(dqm*dqp+stk::math::abs(dqm*dqp))) /
    ((dqm+dqp)*(dqm+dqp)+small_);
  return limit;
}

INSTANTIATE_KERNEL(ScalarUpwAdvDiffElemKernel)

}  // nalu
}  // sierra
