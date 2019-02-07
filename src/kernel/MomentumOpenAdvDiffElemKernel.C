/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumOpenAdvDiffElemKernel.h"
#include "EquationSystem.h"
#include "master_element/MasterElement.h"
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
MomentumOpenAdvDiffElemKernel<BcAlgTraits>::MomentumOpenAdvDiffElemKernel(
  const stk::mesh::MetaData &metaData,
  const SolutionOptions &solnOpts,
  EquationSystem *eqSystem,
  VectorFieldType *velocity,
  GenericFieldType *Gjui,
  ScalarFieldType *viscosity,
  ElemDataRequests &faceDataPreReqs,
  ElemDataRequests &elemDataPreReqs)
  : Kernel(),
    viscosity_(viscosity->mesh_meta_data_ordinal()),
    Gjui_(Gjui->mesh_meta_data_ordinal()),
    alphaUpw_(solnOpts.get_alpha_upw_factor("velocity")),
    om_alphaUpw_(1.0-alphaUpw_),
    hoUpwind_(solnOpts.get_upw_factor("velocity")),
    nfEntrain_(solnOpts.nearestFaceEntrain_),
    om_nfEntrain_(1.0-nfEntrain_),
    includeDivU_(solnOpts.includeDivU_),
    nocFac_(solnOpts.get_noc_usage("velocity") ? 1.0 : 0.0),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(velocity->name())),
    faceIpNodeMap_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_)->ipNodeMap()),
    meSCS_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::elemTopo_)),
    pecletFunction_(eqSystem->create_peclet_function<DoubleType>(velocity->name()))
{
  // save off fields
  velocityNp1_ = velocity->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();
  if ( solnOpts.does_mesh_move() )
    velocityRTM_ = get_field_ordinal(metaData, "velocity_rtm");
  else
    velocityRTM_ = velocity->mesh_meta_data_ordinal();
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  density_ = get_field_ordinal(metaData, "density");
  exposedAreaVec_ = get_field_ordinal(metaData, "exposed_area_vector", metaData.side_rank());
  openMassFlowRate_ = get_field_ordinal(metaData, "open_mass_flow_rate", metaData.side_rank());
  velocityBc_ = get_field_ordinal(metaData, "open_velocity_bc");
  
  // extract master elements
  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_);
  
  // add master elements
  faceDataPreReqs.add_cvfem_face_me(meFC);
  elemDataPreReqs.add_cvfem_surface_me(meSCS_);

  // fields and data; face and then element
  faceDataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  faceDataPreReqs.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_gathered_nodal_field(velocityBc_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_gathered_nodal_field(Gjui_, BcAlgTraits::nDim_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_coordinates_field(coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceDataPreReqs.add_face_field(exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_face_field(openMassFlowRate_, BcAlgTraits::numFaceIp_);

  elemDataPreReqs.add_coordinates_field(coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemDataPreReqs.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
  elemDataPreReqs.add_gathered_nodal_field(velocityRTM_, BcAlgTraits::nDim_);
  elemDataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  elemDataPreReqs.add_gathered_nodal_field(density_, 1);
 
  if ( shiftedGradOp_ )
    elemDataPreReqs.add_master_element_call(SCS_SHIFTED_FACE_GRAD_OP, CURRENT_COORDINATES);
  else
    elemDataPreReqs.add_master_element_call(SCS_FACE_GRAD_OP, CURRENT_COORDINATES);

  // never shift properties
  get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shape_fcn(ptr);}, vf_shape_function_);
  get_scs_shape_fn_data<BcAlgTraits>([&](double* ptr){meSCS_->shape_fcn(ptr);}, v_shape_function_);

  const bool skewSymmetric = solnOpts.get_skew_symmetric(velocity->name());
  get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){skewSymmetric ? meFC->shifted_shape_fcn(ptr) : meFC->shape_fcn(ptr);}, 
                                      vf_adv_shape_function_);
  get_scs_shape_fn_data<BcAlgTraits>([&](double* ptr){skewSymmetric ? meSCS_->shifted_shape_fcn(ptr) : meSCS_->shape_fcn(ptr);}, 
                                     v_adv_shape_function_);
}

template<typename BcAlgTraits>
MomentumOpenAdvDiffElemKernel<BcAlgTraits>::~MomentumOpenAdvDiffElemKernel()
{
  delete pecletFunction_;
}


template<typename BcAlgTraits>
void
MomentumOpenAdvDiffElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**> &lhs,
  SharedMemView<DoubleType *> &rhs,
  ScratchViews<DoubleType> &faceScratchViews,
  ScratchViews<DoubleType> &elemScratchViews,
  int elemFaceOrdinal)
{
  NALU_ALIGNED DoubleType w_uBip[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_uScs[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_uBipExtrap[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_uspecBip[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_coordBip[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_nx[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_GuBip[BcAlgTraits::nDim_*BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_NOC[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_coordScs[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_dxBip[BcAlgTraits::nDim_];

  const int *face_node_ordinals = meSCS_->side_node_ordinals(elemFaceOrdinal);
 
  // face
  SharedMemView<DoubleType*>& vf_viscosity = faceScratchViews.get_scratch_view_1D(viscosity_);
  SharedMemView<DoubleType**>& vf_velocityNp1 = faceScratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType**>& vf_bcVelocity = faceScratchViews.get_scratch_view_2D(velocityBc_);
  SharedMemView<DoubleType***>& vf_Gjui = faceScratchViews.get_scratch_view_3D(Gjui_);
  SharedMemView<DoubleType**>& vf_coordinates = faceScratchViews.get_scratch_view_2D(coordinates_);
  SharedMemView<DoubleType**>& vf_exposedAreaVec = faceScratchViews.get_scratch_view_2D(exposedAreaVec_);
  SharedMemView<DoubleType*>& vf_openMassFlowRate = faceScratchViews.get_scratch_view_1D(openMassFlowRate_);
  
  // element
  SharedMemView<DoubleType**>& v_coordinates = elemScratchViews.get_scratch_view_2D(coordinates_);
  SharedMemView<DoubleType**>& v_velocityNp1 = elemScratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType**>& v_vrtm = elemScratchViews.get_scratch_view_2D(velocityRTM_);
  SharedMemView<DoubleType*>& v_viscosity = elemScratchViews.get_scratch_view_1D(viscosity_);
  SharedMemView<DoubleType*>& v_density = elemScratchViews.get_scratch_view_1D(density_);
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_
    ? elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted_fc_scs
    : elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_fc_scs;

  for (int ip=0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    
    const int opposingNode = meSCS_->opposingNodes(elemFaceOrdinal,ip); // "Left"
    const int nearestNode = meSCS_->ipNodeMap(elemFaceOrdinal)[ip]; // "Right"
    const int opposingScsIp = meSCS_->opposingFace(elemFaceOrdinal,ip);
    const int localFaceNode = faceIpNodeMap_[ip];
    
    // zero out vector quantities
    DoubleType asq = 0.0;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      w_uBip[j] = 0.0;
      w_uScs[j] = 0.0;
      w_uspecBip[j] = 0.0;
      w_coordBip[j] = 0.0;
      w_coordScs[j] = 0.0;
      const DoubleType axj = vf_exposedAreaVec(ip,j);
      asq += axj*axj;
    }
    const DoubleType amag = stk::math::sqrt(asq);
    
    // interpolate to bip
    DoubleType viscBip = 0.0;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType r = vf_shape_function_(ip,ic);
      const DoubleType rAdv = vf_adv_shape_function_(ip,ic);
      viscBip += r*vf_viscosity(ic);
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        w_uspecBip[j] += rAdv*vf_bcVelocity(ic,j);
        w_uBip[j] += rAdv*vf_velocityNp1(ic,j);
        w_coordBip[j] += rAdv*vf_coordinates(ic,j);
      }
    }
    
    // data at interior opposing face
    for ( int ic = 0; ic < BcAlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_adv_shape_function_(opposingScsIp,ic);
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        w_uScs[j] += r*v_velocityNp1(ic,j);
        w_coordScs[j] += r*v_coordinates(ic,j);
      }
    }
    
    // Peclet factor; along the edge is fine  
    DoubleType udotx = 0.0;
    for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {
      const DoubleType dxi = v_coordinates(nearestNode,i) - v_coordinates(opposingNode,i);
      udotx += 0.5*dxi*(v_vrtm(opposingNode,i) + v_vrtm(nearestNode,i));
      w_nx[i] = vf_exposedAreaVec(ip,i)/amag; 
      // extrapolation
      DoubleType duR = 0.0;
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        DoubleType dxj = w_coordBip[j] - v_coordinates(nearestNode,j);
        duR += dxj*vf_Gjui(localFaceNode,i,j)*hoUpwind_;
      }
      w_uBipExtrap[i] = v_velocityNp1(nearestNode,i) + duR;
    }
    
    // Peclet factor; along the edge is fine
    const DoubleType diffIp = 0.5*(v_viscosity(opposingNode)/v_density(opposingNode)
                                   + v_viscosity(nearestNode)/v_density(nearestNode));
    const DoubleType pecFuncArg = stk::math::abs(udotx)/(diffIp+small_);
    const DoubleType pecfac = pecletFunction_->execute(pecFuncArg);
    const DoubleType om_pecfac = 1.0-pecfac;
   
    //================================
    // advection first
    //================================
    const DoubleType tmdot = vf_openMassFlowRate(ip);

    // account for both cases, i.e., leaving or entering to avoid hard loop over SIMD length
    const DoubleType mdotLeaving = stk::math::if_then_else(tmdot > 0, 1.0, 0.0);
    const DoubleType om_mdotLeaving = 1.0 - mdotLeaving;
    
    // NOC: compute dxBip and zero tensor GjUi
    for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {
      w_dxBip[i] = w_coordBip[i] - w_coordScs[i];
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        w_GuBip[i*BcAlgTraits::nDim_+j] = 0.0;
      }
    }
    
    // NOC: compute Gjui at bip
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType rAdv = vf_adv_shape_function_(ip,ic);
      for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) { 
        for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
          w_GuBip[i*BcAlgTraits::nDim_+j] += rAdv*vf_Gjui(ic,i,j);
        }
      }
    }
        
    // NOC term for each component
    for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {
      DoubleType GlUidxl = 0.0;
      DoubleType Gn = 0.0;
      DoubleType an = 0.0;
      DoubleType adx = 0.0;
      for ( int l = 0; l < BcAlgTraits::nDim_; ++l ) {
        GlUidxl += w_GuBip[i*BcAlgTraits::nDim_+l]*w_dxBip[l];
        Gn += w_GuBip[i*BcAlgTraits::nDim_+l]*w_nx[l];
        adx += vf_exposedAreaVec(ip,l)*w_dxBip[l];
        an += vf_exposedAreaVec(ip,l)*w_nx[l];
      }
      w_NOC[i] = GlUidxl - Gn*adx/an;
    }
    
    for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {
      
      const int indexR = nearestNode*BcAlgTraits::nDim_ + i;
      
      //================
      // flow is leaving
      //================

      // central
      const DoubleType uiIp = w_uBip[i];

      // upwind
      const DoubleType uiUpwind = alphaUpw_*w_uBipExtrap[i] + om_alphaUpw_*uiIp;

      // total advection; pressure contribution in time expression
      const DoubleType afluxLeaving = tmdot*(pecfac*uiUpwind+om_pecfac*uiIp);

      rhs(indexR) -= afluxLeaving*mdotLeaving;

      // upwind lhs
      lhs(indexR,nearestNode*BcAlgTraits::nDim_+i) += tmdot*pecfac*alphaUpw_*mdotLeaving;

      // central part
      const DoubleType fac = tmdot*(pecfac*om_alphaUpw_+om_pecfac)*mdotLeaving;
      for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
        const int nn = face_node_ordinals[ic];
        lhs(indexR,nn*BcAlgTraits::nDim_+i) += vf_adv_shape_function_(ip,ic)*fac;
      }

      //===================
      // flow is extraining
      //===================

      DoubleType ubipnx = 0.0;
      DoubleType ubipExtrapnx = 0.0;
      DoubleType uscsnx = 0.0;
      DoubleType uspecbipnx = 0.0;
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        const DoubleType nj = w_nx[j];
        ubipnx += w_uBip[j]*nj;
        ubipExtrapnx += w_uBipExtrap[j]*nj;
        uscsnx += (w_uScs[j] + w_NOC[j]*nocFac_)*nj;
        uspecbipnx += w_uspecBip[j]*nj;
      }

      const DoubleType nxi = w_nx[i];
      
      // total advection; with tangeant entrain
      const DoubleType afluxEntraining =
        tmdot*(pecfac*ubipExtrapnx+om_pecfac*(nfEntrain_*ubipnx + om_nfEntrain_*uscsnx))*nxi
        + tmdot*(w_uspecBip[i] - uspecbipnx*nxi);

      rhs(indexR) -= afluxEntraining*om_mdotLeaving;
      
      // upwind and central
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        const DoubleType nxinxj = nxi*w_nx[j];

        // upwind
        lhs(indexR,nearestNode*BcAlgTraits::nDim_+j) += tmdot*pecfac*alphaUpw_*nxinxj*om_mdotLeaving;

        // central part; exposed face
        DoubleType fac = tmdot*(pecfac*om_alphaUpw_+om_pecfac*nfEntrain_)*nxinxj*om_mdotLeaving;
        for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
          const int nn = face_node_ordinals[ic];
          lhs(indexR,nn*BcAlgTraits::nDim_+j) += vf_adv_shape_function_(ip,ic)*fac;
        }

        // central part; scs face
        fac = tmdot*om_pecfac*om_nfEntrain_*nxinxj*om_mdotLeaving;
        for ( int ic = 0; ic < BcAlgTraits::nodesPerElement_; ++ic ) {
          lhs(indexR,ic*BcAlgTraits::nDim_+j) += v_shape_function_(opposingScsIp,ic)*fac;
        }
      }
    }

    //================================
    // diffusion second
    //================================
    for ( int ic = 0; ic < BcAlgTraits::nodesPerElement_; ++ic ) {
      
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        
        const DoubleType axj = vf_exposedAreaVec(ip,j);
        const DoubleType dndxj = v_dndx(ip,ic,j);
        const DoubleType uxj = v_velocityNp1(ic,j);
        
        const DoubleType divUstress = 2.0/3.0*viscBip*dndxj*uxj*axj*includeDivU_;
        
        for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {
          
          // matrix entries
          int indexR = nearestNode*BcAlgTraits::nDim_ + i;
          
          const DoubleType dndxi = v_dndx(ip,ic,i);
          const DoubleType uxi = v_velocityNp1(ic,i);
          const DoubleType nxi = w_nx[i];
          const DoubleType om_nxinxi = 1.0-nxi*nxi;
          
          // -mu*dui/dxj*Aj(1.0-nini); sneak in divU (explicit)
          DoubleType lhsfac = -viscBip*dndxj*axj*om_nxinxi;
          lhs(indexR,ic*BcAlgTraits::nDim_+i) += lhsfac;
          rhs(indexR) -= lhsfac*uxi + divUstress*om_nxinxi;
          
          // -mu*duj/dxi*Aj(1.0-nini)
          lhsfac = -viscBip*dndxi*axj*om_nxinxi;
          lhs(indexR,ic*BcAlgTraits::nDim_+j) += lhsfac;
          rhs(indexR) -= lhsfac*uxj;
          
          // now we need the -nx*ny*Fy - nx*nz*Fz part
          for ( int l = 0; l < BcAlgTraits::nDim_; ++l ) {
            
            if ( i != l ) {
              const DoubleType nxinxl = nxi*w_nx[l];
              const DoubleType uxl = v_velocityNp1(ic,l);
              const DoubleType dndxl = v_dndx(ip,ic,l);
              
              // +ni*nl*mu*dul/dxj*Aj; sneak in divU (explicit)
              lhsfac = viscBip*dndxj*axj*nxinxl;
              lhs(indexR,ic*BcAlgTraits::nDim_+l) += lhsfac;
              rhs(indexR) -= lhsfac*uxl + divUstress*nxinxl;
              
              // +ni*nl*mu*duj/dxl*Aj
              lhsfac = viscBip*dndxl*axj*nxinxl;
              lhs(indexR,ic*BcAlgTraits::nDim_+j) += lhsfac;
              rhs(indexR) -= lhsfac*uxj;
            }
          }
        }
      }
    }
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(MomentumOpenAdvDiffElemKernel)

}  // nalu
}  // sierra
