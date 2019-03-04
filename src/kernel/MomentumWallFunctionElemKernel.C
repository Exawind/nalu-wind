/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumWallFunctionElemKernel.h"
#include "master_element/MasterElement.h"
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

template<class BcAlgTraits>
MomentumWallFunctionElemKernel<BcAlgTraits>::MomentumWallFunctionElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    elog_(solnOpts.get_turb_model_constant(TM_elog)),
    kappa_(solnOpts.get_turb_model_constant(TM_kappa)),
    yplusCrit_(solnOpts.get_turb_model_constant(TM_yplus_crit)),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::topo_)->ipNodeMap())
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  velocityNp1_ = get_field_ordinal(metaData, "velocity", stk::mesh::StateNP1);
  bcVelocity_ = get_field_ordinal(metaData, "wall_velocity_bc");
  density_ = get_field_ordinal(metaData, "density");
  viscosity_ = get_field_ordinal(metaData, "viscosity");
  exposedAreaVec_ =
    get_field_ordinal(metaData, "exposed_area_vector", metaData.side_rank());
  wallFrictionVelocityBip_ = get_field_ordinal(
    metaData, "wall_friction_velocity_bip", metaData.side_rank());
  wallNormalDistanceBip_ = get_field_ordinal(
    metaData, "wall_normal_distance_bip", metaData.side_rank());
  unsigned coordinates = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::topo_);
 
  // compute and save shape function
  get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shape_fcn(ptr);}, vf_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_face_me(meFC);
 
  // fields and data; mdot not gathered as element data
  dataPreReqs.add_coordinates_field(coordinates, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(bcVelocity_, BcAlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(density_, 1);
  dataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  dataPreReqs.add_face_field(exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  dataPreReqs.add_face_field(wallFrictionVelocityBip_, BcAlgTraits::numFaceIp_);
  dataPreReqs.add_face_field(wallNormalDistanceBip_, BcAlgTraits::numFaceIp_);
}

template<class BcAlgTraits>
MomentumWallFunctionElemKernel<BcAlgTraits>::~MomentumWallFunctionElemKernel()
{}

template<class BcAlgTraits>
void
MomentumWallFunctionElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType **>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  DoubleType w_uNp1Bip[BcAlgTraits::nDim_];
  DoubleType w_uBcBip[BcAlgTraits::nDim_];
  DoubleType w_unitNormal[BcAlgTraits::nDim_];

  SharedMemView<DoubleType**>& v_uNp1 = scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType**>& v_bcVelocity = scratchViews.get_scratch_view_2D(bcVelocity_);
  SharedMemView<DoubleType*>& v_density = scratchViews.get_scratch_view_1D(density_);
  SharedMemView<DoubleType*>& v_viscosity = scratchViews.get_scratch_view_1D(viscosity_);
  SharedMemView<DoubleType**>& vf_exposedAreaVec = scratchViews.get_scratch_view_2D(exposedAreaVec_);
  SharedMemView<DoubleType*>& vf_utau = scratchViews.get_scratch_view_1D(wallFrictionVelocityBip_);
  SharedMemView<DoubleType*>& vf_yp = scratchViews.get_scratch_view_1D(wallNormalDistanceBip_);

  for ( int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip ) {
        
    const int nearestNode = ipNodeMap_[ip];
    
    // zero out vector quantities; squeeze in aMag
    DoubleType aMag = 0.0;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      w_uNp1Bip[j] = 0.0;
      w_uBcBip[j] = 0.0;
      const DoubleType axj = vf_exposedAreaVec(ip,j);
      aMag += axj*axj;
    }
    aMag = stk::math::sqrt(aMag);
    
    // interpolate to bip
    DoubleType rhoBip = 0.0;
    DoubleType muBip = 0.0;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType r = vf_shape_function_(ip,ic);
      rhoBip += r*v_density(ic);
      muBip += r*v_viscosity(ic);
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        w_uNp1Bip[j] += r*v_uNp1(ic,j);
        w_uBcBip[j] += r*v_bcVelocity(ic,j);
      }
    }
    
    // form unit normal
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      w_unitNormal[j] = vf_exposedAreaVec(ip,j)/aMag;
    }
    
    // extract bip data
    const DoubleType yp = vf_yp(ip);
    const DoubleType utau = vf_utau(ip);
    
    // determine yplus and law of the wall scaling
    const DoubleType yplus = rhoBip*yp*utau/muBip;
    const DoubleType lambda = stk::math::if_then_else( yplus > yplusCrit_, 
                                                       rhoBip*kappa_*utau/stk::math::log(elog_*yplus)*aMag, 
                                                       muBip/yp*aMag);
    
    // start the lhs assembly
    for ( int i = 0; i < BcAlgTraits::nDim_; ++i ) {
      
      const int indexR = nearestNode*BcAlgTraits::nDim_ + i;
      
      DoubleType uiTan = 0.0;
      DoubleType uiBcTan = 0.0;
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        const DoubleType ninj = w_unitNormal[i]*w_unitNormal[j];
        if ( i==j ) {
          const DoubleType om_nini = 1.0 - ninj;
          uiTan += om_nini*w_uNp1Bip[j];
          uiBcTan += om_nini*w_uBcBip[j];
          for (int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic)
            lhs(indexR,ic*BcAlgTraits::nDim_+i) += lambda*om_nini*vf_shape_function_(ip,ic);
        }
        else {
          uiTan -= ninj*w_uNp1Bip[j];
          uiBcTan -= ninj*w_uBcBip[j];
          for (int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic)
            lhs(indexR,ic*BcAlgTraits::nDim_+j) -= lambda*ninj*vf_shape_function_(ip,ic);
        }
      }
      rhs(indexR) -= lambda*(uiTan-uiBcTan);
    }
  }  
}

INSTANTIATE_KERNEL_FACE(MomentumWallFunctionElemKernel)

}  // nalu
}  // sierra
