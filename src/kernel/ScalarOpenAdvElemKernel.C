/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ScalarOpenAdvElemKernel.h"
#include "EquationSystem.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "PecletFunction.h"
#include "SolutionOptions.h"
#include "BuildTemplates.h"

// template and scratch space
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename BcAlgTraits>
ScalarOpenAdvElemKernel<BcAlgTraits>::ScalarOpenAdvElemKernel(
  const stk::mesh::MetaData &metaData,
  const SolutionOptions &solnOpts,
  EquationSystem *eqSystem,
  ScalarFieldType *scalarQ,
  ScalarFieldType *bcScalarQ,
  VectorFieldType *Gjq,
  ScalarFieldType *diffFluxCoeff,
  ElemDataRequests &faceDataPreReqs,
  ElemDataRequests &elemDataPreReqs)
  : Kernel(),
    scalarQ_(scalarQ->mesh_meta_data_ordinal()),
    bcScalarQ_(bcScalarQ->mesh_meta_data_ordinal()),
    Gjq_(Gjq->mesh_meta_data_ordinal()),
    diffFluxCoeff_(diffFluxCoeff->mesh_meta_data_ordinal()),
    alphaUpw_(solnOpts.get_alpha_upw_factor(scalarQ->name())),
    om_alphaUpw_(1.0-alphaUpw_),
    hoUpwind_(solnOpts.get_upw_factor(scalarQ->name())),
    faceIpNodeMap_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_)->ipNodeMap()),
    meSCS_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::elemTopo_)),
    pecletFunction_(eqSystem->create_peclet_function<DoubleType>(scalarQ->name()))
{
  // save off fields
  const std::string vrtm_name = solnOpts.does_mesh_move()? "velocity_rtm" : "velocity";
  velocityRTM_ = get_field_ordinal(metaData, vrtm_name);
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  density_ = get_field_ordinal(metaData, "density");
  openMassFlowRate_ = get_field_ordinal(metaData, "open_mass_flow_rate", metaData.side_rank());
  
  // extract master elements
  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_);
  
  // add master elements
  faceDataPreReqs.add_cvfem_face_me(meFC);
  elemDataPreReqs.add_cvfem_surface_me(meSCS_);

  // fields and data; face and then element
  faceDataPreReqs.add_coordinates_field(coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceDataPreReqs.add_gathered_nodal_field(Gjq_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_gathered_nodal_field(scalarQ_, 1);
  faceDataPreReqs.add_gathered_nodal_field(bcScalarQ_, 1);
  faceDataPreReqs.add_face_field(openMassFlowRate_, BcAlgTraits::numFaceIp_);

  elemDataPreReqs.add_coordinates_field(coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemDataPreReqs.add_gathered_nodal_field(velocityRTM_, BcAlgTraits::nDim_);
  elemDataPreReqs.add_gathered_nodal_field(diffFluxCoeff_, 1);
  elemDataPreReqs.add_gathered_nodal_field(density_, 1);

  // never shift properties
  const bool skewSymmetric = solnOpts.get_skew_symmetric(scalarQ->name());
  get_face_shape_fn_data<BcAlgTraits>(
    [&](double* ptr) {
      skewSymmetric ? meFC->shifted_shape_fcn(ptr) : meFC->shape_fcn(ptr);
    },
    vf_adv_shape_function_);
}

template<typename BcAlgTraits>
ScalarOpenAdvElemKernel<BcAlgTraits>::~ScalarOpenAdvElemKernel()
{
  delete pecletFunction_;
}


template<typename BcAlgTraits>
void
ScalarOpenAdvElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**> &lhs,
  SharedMemView<DoubleType *> &rhs,
  ScratchViews<DoubleType> &faceScratchViews,
  ScratchViews<DoubleType> &elemScratchViews,
  int elemFaceOrdinal)
{
  DoubleType w_coordBip[BcAlgTraits::nDim_];

  const int *face_node_ordinals = meSCS_->side_node_ordinals(elemFaceOrdinal);
 
  // face
  SharedMemView<DoubleType*>& vf_scalarQ = faceScratchViews.get_scratch_view_1D(scalarQ_);
  SharedMemView<DoubleType*>& vf_bcScalarQ = faceScratchViews.get_scratch_view_1D(bcScalarQ_);
  SharedMemView<DoubleType**>& vf_Gjq = faceScratchViews.get_scratch_view_2D(Gjq_);
  SharedMemView<DoubleType**>& vf_coordinates = faceScratchViews.get_scratch_view_2D(coordinates_);
  SharedMemView<DoubleType*>& vf_openMassFlowRate = faceScratchViews.get_scratch_view_1D(openMassFlowRate_);
  
  // element
  SharedMemView<DoubleType**>& v_coordinates = elemScratchViews.get_scratch_view_2D(coordinates_);
  SharedMemView<DoubleType**>& v_vrtm = elemScratchViews.get_scratch_view_2D(velocityRTM_);
  SharedMemView<DoubleType*>& v_diffFluxCoeff = elemScratchViews.get_scratch_view_1D(diffFluxCoeff_);
  SharedMemView<DoubleType*>& v_density = elemScratchViews.get_scratch_view_1D(density_);

  for (int ip=0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    
    const int opposingNode = meSCS_->opposingNodes(elemFaceOrdinal,ip); // "Left"
    const int nearestNode = meSCS_->ipNodeMap(elemFaceOrdinal)[ip]; // "Right"
    const int localFaceNode = faceIpNodeMap_[ip];
    
    // zero out vector quantities
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      w_coordBip[j] = 0.0;
    }
    
    // interpolate to bip
    DoubleType qIp = 0.0;
    DoubleType qIpEntrain = 0.0;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType rAdv = vf_adv_shape_function_(ip,ic);
      qIp += rAdv*vf_scalarQ(ic);
      qIpEntrain += rAdv*vf_bcScalarQ(ic);
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        w_coordBip[j] += rAdv*vf_coordinates(ic,j);
      }
    }
    
    // udotx; extrapolation
    DoubleType udotx = 0.0;
    DoubleType dqR = 0.0;
    for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {
      const DoubleType dxi = v_coordinates(nearestNode,i) - v_coordinates(opposingNode,i);
      udotx += 0.5*dxi*(v_vrtm(opposingNode,i) + v_vrtm(nearestNode,i));
      DoubleType dxBip = w_coordBip[i] - v_coordinates(nearestNode,i);
      dqR += dxBip*vf_Gjq(localFaceNode,i)*hoUpwind_;
    }
    const DoubleType qIpUpw = vf_scalarQ(localFaceNode) + dqR;
    
    // Peclet factor; along the edge is fine
    const DoubleType diffIp = 0.5*(v_diffFluxCoeff(opposingNode)/v_density(opposingNode)
                                   + v_diffFluxCoeff(nearestNode)/v_density(nearestNode));
    const DoubleType pecFuncArg = stk::math::abs(udotx)/(diffIp+small_);
    const DoubleType pecfac = pecletFunction_->execute(pecFuncArg);
    const DoubleType om_pecfac = 1.0-pecfac;
   
    const DoubleType tmdot = vf_openMassFlowRate(ip);

    // account for both cases, i.e., leaving or entering to avoid hard loop over SIMD length
    const DoubleType mdotLeaving = stk::math::if_then_else(tmdot > 0, 1.0, 0.0);
    const DoubleType om_mdotLeaving = 1.0 - mdotLeaving;

    //================================
    // flow is leaving
    //================================
    
    // central is simply qIp
    
    // upwind
    const DoubleType qUpwind = alphaUpw_*qIpUpw + om_alphaUpw_*qIp;
    
    // total advection
    const DoubleType afluxLeaving = tmdot*(pecfac*qUpwind+om_pecfac*qIp);

    rhs(nearestNode) -= afluxLeaving*mdotLeaving;

    // upwind lhs
    lhs(nearestNode,nearestNode) += tmdot*pecfac*alphaUpw_*mdotLeaving;

    // central part
    const DoubleType fac = tmdot*(pecfac*om_alphaUpw_+om_pecfac)*mdotLeaving;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const int nn = face_node_ordinals[ic];
      lhs(nearestNode,nn) += vf_adv_shape_function_(ip,ic)*fac;
    }

    //================================
    // flow is entering
    //================================

    // extrainment; advect in from specified value
    const DoubleType afluxEntraining = tmdot*qIpEntrain;
    rhs(nearestNode) -= afluxEntraining*om_mdotLeaving;
    
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(ScalarOpenAdvElemKernel)

}  // nalu
}  // sierra
