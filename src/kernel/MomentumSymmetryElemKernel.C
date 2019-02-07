/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumSymmetryElemKernel.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"

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
MomentumSymmetryElemKernel<BcAlgTraits>::MomentumSymmetryElemKernel(
  const stk::mesh::MetaData &metaData,
  const SolutionOptions &solnOpts,
  VectorFieldType *velocity,
  ScalarFieldType *viscosity,
  ElemDataRequests &faceDataPreReqs,
  ElemDataRequests &elemDataPreReqs)
  : Kernel(),
    viscosity_(viscosity->mesh_meta_data_ordinal()),
    includeDivU_(solnOpts.includeDivU_),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(velocity->name())),
    meSCS_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::elemTopo_))
{
  velocityNp1_ = velocity->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  exposedAreaVec_ = get_field_ordinal(metaData, "exposed_area_vector", metaData.side_rank());

  // extract master elements
  MasterElement* meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_);
  
  // add master elements
  faceDataPreReqs.add_cvfem_face_me(meFC);
  elemDataPreReqs.add_cvfem_surface_me(meSCS_);

  // fields and data; face and then element
  faceDataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  faceDataPreReqs.add_face_field(exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  elemDataPreReqs.add_coordinates_field(coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemDataPreReqs.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);

  if ( shiftedGradOp_ )
    elemDataPreReqs.add_master_element_call(SCS_SHIFTED_FACE_GRAD_OP, CURRENT_COORDINATES);
  else
    elemDataPreReqs.add_master_element_call(SCS_FACE_GRAD_OP, CURRENT_COORDINATES);

  // never shift properties
  get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shape_fcn(ptr);}, vf_shape_function_);
}

template<typename BcAlgTraits>
MomentumSymmetryElemKernel<BcAlgTraits>::~MomentumSymmetryElemKernel()
{}


template<typename BcAlgTraits>
void
MomentumSymmetryElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**> &lhs,
  SharedMemView<DoubleType *> &rhs,
  ScratchViews<DoubleType> &faceScratchViews,
  ScratchViews<DoubleType> &elemScratchViews,
  int elemFaceOrdinal)
{
  DoubleType w_nx[BcAlgTraits::nDim_];

  // face
  SharedMemView<DoubleType*>& vf_viscosity = faceScratchViews.get_scratch_view_1D(viscosity_);
  SharedMemView<DoubleType**>& vf_exposedAreaVec = faceScratchViews.get_scratch_view_2D(exposedAreaVec_);
 
  // element
  SharedMemView<DoubleType**>& v_uNp1 = elemScratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_
    ? elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted_fc_scs
    : elemScratchViews.get_me_views(CURRENT_COORDINATES).dndx_fc_scs;

  for (int ip=0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    
    const int nearestNode = meSCS_->ipNodeMap(elemFaceOrdinal)[ip]; // "Right"
    
    // form unit normal
    DoubleType asq = 0.0;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      const DoubleType axj = vf_exposedAreaVec(ip,j);
      asq += axj*axj;
    }
    const DoubleType amag = stk::math::sqrt(asq);
    for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {
      w_nx[i] = vf_exposedAreaVec(ip,i)/amag;
    }
    
    DoubleType viscBip = 0.0;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType r = vf_shape_function_(ip,ic);
      viscBip += r*vf_viscosity(ic);
    }
    
    for ( int ic = 0; ic < BcAlgTraits::nodesPerElement_; ++ic ) {
            
      const int icNdim = ic*BcAlgTraits::nDim_;

      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        
        const DoubleType axj = vf_exposedAreaVec(ip,j);
        const DoubleType dndxj = v_dndx(ip,ic,j);
        const DoubleType uxj = v_uNp1(ic,j);
        
        const DoubleType divUstress = 2.0/3.0*viscBip*dndxj*uxj*axj*includeDivU_;
        
        for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {

          const int indexR = nearestNode*BcAlgTraits::nDim_ +i;

          const DoubleType dndxi = v_dndx(ip,ic,i);
          const DoubleType uxi = v_uNp1(ic,i);
          const DoubleType nxi = w_nx[i];
          const DoubleType nxinxi = nxi*nxi;
          
          // -mu*dui/dxj*Aj*ni*ni; sneak in divU (explicit)
          DoubleType lhsfac = -viscBip*dndxj*axj*nxinxi;
          lhs(indexR,icNdim+i) += lhsfac;
          rhs(indexR) -= lhsfac*uxi + divUstress*nxinxi;
          
          // -mu*duj/dxi*Aj*ni*ni
          lhsfac = -viscBip*dndxi*axj*nxinxi;
          lhs(indexR,icNdim+j) += lhsfac;
          rhs(indexR) -= lhsfac*uxj;
          
          // now we need the +nx*ny*Fy + nx*nz*Fz part
          for ( int l = 0; l < BcAlgTraits::nDim_; ++l ) {
            
            if ( i != l ) {
              const DoubleType nxinxl = nxi*w_nx[l];
              const DoubleType uxl = v_uNp1(ic,l);
              const DoubleType dndxl = v_dndx(ip,ic,l);
              
              // -ni*nl*mu*dul/dxj*Aj; sneak in divU (explicit)
              lhsfac = -viscBip*dndxj*axj*nxinxl;
              lhs(indexR,icNdim+l) += lhsfac;
              rhs(indexR) -= lhsfac*uxl + divUstress*nxinxl;
              
              // -ni*nl*mu*duj/dxl*Aj
              lhsfac = -viscBip*dndxl*axj*nxinxl;
              lhs(indexR,icNdim+j) += lhsfac;
              rhs(indexR) -= lhsfac*uxj;
            }
          }
        }
      }
    }
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(MomentumSymmetryElemKernel)

}  // nalu
}  // sierra
