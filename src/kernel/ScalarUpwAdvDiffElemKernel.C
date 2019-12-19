// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


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
  ElemDataRequests& dataPreReqs,
  const bool useAvgMdot /*=false*/)
  : scalarQ_(scalarQ->mesh_meta_data_ordinal()),
    Gjq_(Gjq->mesh_meta_data_ordinal()),
    diffFluxCoeff_(diffFluxCoeff->mesh_meta_data_ordinal()),
    alpha_(solnOpts.get_alpha_factor(scalarQ->name())),
    alphaUpw_(solnOpts.get_alpha_upw_factor(scalarQ->name())),
    hoUpwind_(solnOpts.get_upw_factor(scalarQ->name())),
    useLimiter_(solnOpts.primitive_uses_limiter(scalarQ->name())),
    om_alpha_(1.0 - alpha_),
    om_alphaUpw_(1.0 - alphaUpw_),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(scalarQ->name())),
    skewSymmetric_(solnOpts.get_skew_symmetric(scalarQ->name())),
    pecletFunction_(eqSystem->ngp_create_peclet_function(scalarQ->name()))
{
  // Save of required fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  if (useAvgMdot) {
    massFlowRate_ = get_field_ordinal(metaData, "average_mass_flow_rate_scs", stk::topology::ELEM_RANK);
  } else {
    massFlowRate_ = get_field_ordinal(metaData, "mass_flow_rate_scs", stk::topology::ELEM_RANK);
  }
  density_ = get_field_ordinal(metaData, "density");

  const std::string vrtm_name = solnOpts.does_mesh_move()? "velocity_rtm" : "velocity";
  velocityRTM_ = get_field_ordinal(metaData, vrtm_name);

  meSCS_ = sierra::nalu::MasterElementRepo::get_surface_master_element<AlgTraits>();
  
  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS_);

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

  dataPreReqs.add_master_element_call(SCS_SHAPE_FCN, CURRENT_COORDINATES);
  if (skewSymmetric_)
    dataPreReqs.add_master_element_call(SCS_SHIFTED_SHAPE_FCN, CURRENT_COORDINATES);
}

template<typename AlgTraits>
void
ScalarUpwAdvDiffElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  /// Scratch space to hold coordinates at the integration point
  NALU_ALIGNED DoubleType w_coordIp[AlgTraits::nDim_];

  const auto& v_velocityRTM = scratchViews.get_scratch_view_2D(velocityRTM_);
  const auto& v_coordinates = scratchViews.get_scratch_view_2D(coordinates_);
  const auto& v_Gjq = scratchViews.get_scratch_view_2D(Gjq_);
  const auto& v_scalarQ = scratchViews.get_scratch_view_1D(scalarQ_);
  const auto& v_density = scratchViews.get_scratch_view_1D(density_);
  const auto& v_diffFluxCoeff = scratchViews.get_scratch_view_1D(diffFluxCoeff_);
  const auto& v_mdot = scratchViews.get_scratch_view_1D(massFlowRate_);

  const auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  const auto& v_scs_areav = meViews.scs_areav;
  const auto& v_dndx = shiftedGradOp_ ? meViews.dndx_shifted : meViews.dndx;
  const auto& v_shape_function = meViews.scs_shape_fcn;
  const auto& v_adv_shape_function = skewSymmetric_ ? meViews.scs_shifted_shape_fcn : meViews.scs_shape_fcn;

  const int* lrscv = meSCS_->adjacentNodes();

  // start the assembly
  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv[2*ip];
    const int ir = lrscv[2*ip+1];

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
      const DoubleType r = v_shape_function(ip,ic);
      const DoubleType rAdv = v_adv_shape_function(ip,ic);
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
      const DoubleType lhsfacAdv = v_adv_shape_function(ip,ic)*tmdot*(pecfac*om_alphaUpw_ + om_pecfac*om_alpha_);

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
