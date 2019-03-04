/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ContinuityOpenElemKernel.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename BcAlgTraits>
ContinuityOpenElemKernel<BcAlgTraits>::ContinuityOpenElemKernel(
  const stk::mesh::MetaData &metaData,
  const SolutionOptions &solnOpts,
  ElemDataRequests &faceDataPreReqs,
  ElemDataRequests &elemDataPreReqs)
  : Kernel(),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("pressure")),
    reducedSensitivities_(solnOpts.cvfemReducedSensPoisson_),
    pstabFac_(solnOpts.activateOpenMdotCorrection_ ? 0.0 : 1.0),
    interpTogether_(solnOpts.get_mdot_interp()),
    om_interpTogether_(1.0 - interpTogether_),
    meSCS_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::elemTopo_))
{
  const std::string vrtm_name =
    solnOpts.does_mesh_move() ? "velocity_rtm" : "velocity";
  const std::string pbc_name =
    solnOpts.activateOpenMdotCorrection_ ? "pressure" : "pressure_bc";
  velocityRTM_ = get_field_ordinal(metaData, vrtm_name);
  Gpdx_ = get_field_ordinal(metaData, "dpdx");
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  pressure_ = get_field_ordinal(metaData, "pressure");
  pressureBc_ = get_field_ordinal(metaData, pbc_name);
  density_ = get_field_ordinal(metaData, "density");
  Udiag_ = get_field_ordinal(metaData, "momentum_diag");
  exposedAreaVec_ = get_field_ordinal(metaData, "exposed_area_vector", metaData.side_rank());
  
  // extract master elements
  MasterElement* meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_);
  
  // add master elements
  faceDataPreReqs.add_cvfem_face_me(meFC);
  elemDataPreReqs.add_cvfem_surface_me(meSCS_);

  // fields and data; face and then element
  faceDataPreReqs.add_gathered_nodal_field(pressure_, 1);
  faceDataPreReqs.add_gathered_nodal_field(pressureBc_, 1);
  faceDataPreReqs.add_gathered_nodal_field(density_, 1);
  faceDataPreReqs.add_gathered_nodal_field(Udiag_, 1);
  faceDataPreReqs.add_gathered_nodal_field(velocityRTM_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_gathered_nodal_field(Gpdx_, BcAlgTraits::nDim_);  
  faceDataPreReqs.add_face_field(exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  elemDataPreReqs.add_coordinates_field(coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemDataPreReqs.add_gathered_nodal_field(pressure_, 1);

  // manage dndx
  if ( !shiftedGradOp_ || !reducedSensitivities_ )
    elemDataPreReqs.add_master_element_call(SCS_FACE_GRAD_OP, CURRENT_COORDINATES);
  if ( shiftedGradOp_ || reducedSensitivities_ )
    elemDataPreReqs.add_master_element_call(SCS_SHIFTED_FACE_GRAD_OP, CURRENT_COORDINATES);

  if ( solnOpts.cvfemShiftMdot_ )
    get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shifted_shape_fcn(ptr);}, vf_shape_function_);
  else
    get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shape_fcn(ptr);}, vf_shape_function_);
}

template<typename BcAlgTraits>
ContinuityOpenElemKernel<BcAlgTraits>::~ContinuityOpenElemKernel()
{}

template<typename BcAlgTraits>
void
ContinuityOpenElemKernel<BcAlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  const double dt = timeIntegrator.get_time_step();
  const double gamma1 = timeIntegrator.get_gamma1();
  projTimeScale_ = dt/gamma1;
}

template<typename BcAlgTraits>
void
ContinuityOpenElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**> &lhs,
  SharedMemView<DoubleType *> &rhs,
  ScratchViews<DoubleType> &faceScratchViews,
  ScratchViews<DoubleType> &elemScratchViews,
  int elemFaceOrdinal)
{
  NALU_ALIGNED DoubleType w_uBip[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_rho_uBip[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_GpdxBip[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_dpdxBip[BcAlgTraits::nDim_];
 
  const int *face_node_ordinals = meSCS_->side_node_ordinals(elemFaceOrdinal);
 
  // face
  SharedMemView<DoubleType*>& vf_pressure = faceScratchViews.get_scratch_view_1D(pressure_);
  SharedMemView<DoubleType*>& vf_pressureBc = faceScratchViews.get_scratch_view_1D(pressureBc_);
  SharedMemView<DoubleType**>& vf_Gpdx = faceScratchViews.get_scratch_view_2D(Gpdx_);
  SharedMemView<DoubleType*>& vf_density = faceScratchViews.get_scratch_view_1D(density_);
  SharedMemView<DoubleType*>& vf_udiag = faceScratchViews.get_scratch_view_1D(Udiag_);
  SharedMemView<DoubleType**>& vf_vrtm = faceScratchViews.get_scratch_view_2D(velocityRTM_);
  SharedMemView<DoubleType**>& vf_exposedAreaVec = faceScratchViews.get_scratch_view_2D(exposedAreaVec_);
 
  // element
  SharedMemView<DoubleType*>& v_pressure = elemScratchViews.get_scratch_view_1D(pressure_);
 
  // dndx for both rhs and lhs
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_ 
    ? elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted_fc_scs
    : elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_fc_scs;
  SharedMemView<DoubleType***>& v_dndx_lhs = (shiftedGradOp_ || reducedSensitivities_)
    ? elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted_fc_scs
    : elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_fc_scs;

  for (int ip=0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    
    const int nearestNode = meSCS_->ipNodeMap(elemFaceOrdinal)[ip]; // "Right"
    
    // zero out vector quantities; form aMag
    DoubleType aMag = 0.0;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      w_uBip[j] = 0.0;
      w_rho_uBip[j] = 0.0;
      w_GpdxBip[j] = 0.0;
      w_dpdxBip[j] = 0.0;
      const DoubleType axj = vf_exposedAreaVec(ip,j);
      aMag += axj*axj;
    }
    aMag = stk::math::sqrt(aMag);
    
    // form L^-1
    DoubleType inverseLengthScale = 0.0;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const int faceNodeNumber = face_node_ordinals[ic];
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        inverseLengthScale += v_dndx(ip,faceNodeNumber,j)*vf_exposedAreaVec(ip,j);
      }
    }        
    inverseLengthScale /= aMag;

    // interpolate to bip
    DoubleType pBip = 0.0;
    DoubleType pbcBip = 0.0;
    DoubleType rhoBip = 0.0;
    DoubleType projTimeScaleBip = 0.0;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType r = vf_shape_function_(ip,ic);
      const DoubleType rhoIC = vf_density(ic);
      const DoubleType udiagInv = 1.0 / vf_udiag(ic);
      rhoBip += r*rhoIC;
      pBip += r*vf_pressure(ic);
      pbcBip += r*vf_pressureBc(ic);
      projTimeScaleBip += r * udiagInv;
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        w_uBip[j] += r*vf_vrtm(ic,j);
        w_rho_uBip[j] += r*rhoIC*vf_vrtm(ic,j);
        w_GpdxBip[j] += r*vf_Gpdx(ic,j) * udiagInv;
      }
    }
    
    // form dpdxBip
    for ( int ic = 0; ic < BcAlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType pIc = v_pressure(ic);
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        w_dpdxBip[j] += v_dndx(ip,ic,j)*pIc;
      }
    }
    
    // form mdot; rho*uj*Aj - projT*(dpdxj - Gjp)*Aj + penaltyFac*projTimeScale*invL*(pBip - pbcBip)*aMag
    DoubleType mdot = -mdotCorrection_ + penaltyFac_*projTimeScaleBip*inverseLengthScale*(pBip - pbcBip)*aMag*pstabFac_;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      const DoubleType axj = vf_exposedAreaVec(ip,j);
      mdot += (interpTogether_*w_rho_uBip[j] + om_interpTogether_*rhoBip*w_uBip[j] 
               - (projTimeScaleBip * w_dpdxBip[j] - w_GpdxBip[j])*pstabFac_)*axj;
    }
    
    // face-based penalty; divide by projTimeScale
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const int faceNodeNumber = face_node_ordinals[ic];
      const DoubleType r = vf_shape_function_(ip,ic);
      lhs(nearestNode,faceNodeNumber) += r*penaltyFac_*inverseLengthScale*aMag*pstabFac_ * projTimeScaleBip / projTimeScale_;
    }
    
    // element-based gradient; divide by projTimeScale
    for ( int ic = 0; ic < BcAlgTraits::nodesPerElement_; ++ic ) {
      DoubleType lhsFac = 0.0;
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j )
        lhsFac += -v_dndx_lhs(ip,ic,j)*vf_exposedAreaVec(ip,j);
      lhs(nearestNode,ic) += lhsFac*pstabFac_ * projTimeScaleBip / projTimeScale_;
    }
    
    // residual
    rhs(nearestNode) -= mdot / projTimeScale_;
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(ContinuityOpenElemKernel)

}  // nalu
}  // sierra
