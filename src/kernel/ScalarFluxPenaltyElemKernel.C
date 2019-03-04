/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ScalarFluxPenaltyElemKernel.h"
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
ScalarFluxPenaltyElemKernel<BcAlgTraits>::ScalarFluxPenaltyElemKernel(
  const stk::mesh::MetaData &metaData,
  const SolutionOptions &solnOpts,
  ScalarFieldType *scalarQ,
  ScalarFieldType *bcScalarQ,
  ScalarFieldType *diffFluxCoeff,
  ElemDataRequests &faceDataPreReqs,
  ElemDataRequests &elemDataPreReqs)
  : Kernel(),
    scalarQ_(scalarQ->mesh_meta_data_ordinal()),
    bcScalarQ_(bcScalarQ->mesh_meta_data_ordinal()),
    diffFluxCoeff_(diffFluxCoeff->mesh_meta_data_ordinal()),
    penaltyFac_(2.0),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("pressure")),
    meSCS_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::elemTopo_))
{
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  exposedAreaVec_ = get_field_ordinal(metaData, "exposed_area_vector", metaData.side_rank());
  
  // extract master elements
  MasterElement* meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_);
  
  // add master elements
  faceDataPreReqs.add_cvfem_face_me(meFC);
  elemDataPreReqs.add_cvfem_surface_me(meSCS_);

  // fields and data; face and then element
  faceDataPreReqs.add_gathered_nodal_field(scalarQ_, 1);
  faceDataPreReqs.add_gathered_nodal_field(bcScalarQ_, 1);
  faceDataPreReqs.add_gathered_nodal_field(diffFluxCoeff_, 1);
  faceDataPreReqs.add_face_field(exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  elemDataPreReqs.add_coordinates_field(coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemDataPreReqs.add_gathered_nodal_field(scalarQ_, 1);
  
  // manage dndx
  if ( shiftedGradOp_ )
    elemDataPreReqs.add_master_element_call(SCS_SHIFTED_FACE_GRAD_OP, CURRENT_COORDINATES);
  else
    elemDataPreReqs.add_master_element_call(SCS_FACE_GRAD_OP, CURRENT_COORDINATES);

  get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shape_fcn(ptr);}, vf_shape_function_);
}

template<typename BcAlgTraits>
ScalarFluxPenaltyElemKernel<BcAlgTraits>::~ScalarFluxPenaltyElemKernel()
{}

template<typename BcAlgTraits>
void
ScalarFluxPenaltyElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**> &lhs,
  SharedMemView<DoubleType *> &rhs,
  ScratchViews<DoubleType> &faceScratchViews,
  ScratchViews<DoubleType> &elemScratchViews,
  int elemFaceOrdinal)
{
  NALU_ALIGNED DoubleType w_dqdxBip[BcAlgTraits::nDim_];
 
  const int *face_node_ordinals = meSCS_->side_node_ordinals(elemFaceOrdinal);
 
  // face
  SharedMemView<DoubleType*>& vf_scalarQ = faceScratchViews.get_scratch_view_1D(scalarQ_);
  SharedMemView<DoubleType*>& vf_bcScalarQ = faceScratchViews.get_scratch_view_1D(bcScalarQ_);
  SharedMemView<DoubleType*>& vf_diffFluxCoeff = faceScratchViews.get_scratch_view_1D(diffFluxCoeff_);
  SharedMemView<DoubleType**>& vf_exposedAreaVec = faceScratchViews.get_scratch_view_2D(exposedAreaVec_);
 
  // element
  SharedMemView<DoubleType*>& v_scalarQ = elemScratchViews.get_scratch_view_1D(scalarQ_);

  // dndx for both rhs and lhs
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_ 
    ? elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted_fc_scs
    : elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_fc_scs;

  for (int ip=0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    
    const int nearestNode = meSCS_->ipNodeMap(elemFaceOrdinal)[ip]; // "Right"
    
    // zero out vector quantities; form aMag
    DoubleType aMag = 0.0;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      w_dqdxBip[j] = 0.0;
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
    DoubleType qBip = 0.0;
    DoubleType qbcBip = 0.0;
    DoubleType diffFluxCoeffBip = 0.0;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType r = vf_shape_function_(ip,ic);
      qBip += r*vf_scalarQ(ic);
      qbcBip += r*vf_bcScalarQ(ic);
      diffFluxCoeffBip += r*vf_diffFluxCoeff(ic);
    }
    
    // form dqdxBip
    for ( int ic = 0; ic < BcAlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType qIc = v_scalarQ(ic);
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        w_dqdxBip[j] += v_dndx(ip,ic,j)*qIc;
      }
    }
    
    // form integrated flux; -diffFluxCoeffBip*dqdxj*Aj + penaltyFac*diffFluxCoeffBip*invL*(qBip - qbcBip)*aMag
    DoubleType intFlux = penaltyFac_*diffFluxCoeffBip*inverseLengthScale*(qBip - qbcBip)*aMag;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      const DoubleType axj = vf_exposedAreaVec(ip,j);
      intFlux -= diffFluxCoeffBip*w_dqdxBip[j]*axj;
    }

    // residual
    rhs(nearestNode) -= intFlux;

    // face-based penalty
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const int faceNodeNumber = face_node_ordinals[ic];
      const DoubleType r = vf_shape_function_(ip,ic);
      lhs(nearestNode,faceNodeNumber) += r*penaltyFac_*diffFluxCoeffBip*inverseLengthScale*aMag;
    }
    
    // element-based gradient
    for ( int ic = 0; ic < BcAlgTraits::nodesPerElement_; ++ic ) {
      DoubleType lhsFac = 0.0;
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j )
        lhsFac += -v_dndx(ip,ic,j)*vf_exposedAreaVec(ip,j);
      lhs(nearestNode,ic) += diffFluxCoeffBip*lhsFac;
    }    
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(ScalarFluxPenaltyElemKernel)

}  // nalu
}  // sierra
