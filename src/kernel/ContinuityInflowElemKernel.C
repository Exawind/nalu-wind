/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ContinuityInflowElemKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
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
ContinuityInflowElemKernel<BcAlgTraits>::ContinuityInflowElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions &solnOpts,
  const bool &useShifted,
  ElemDataRequests &dataPreReqs)
  : Kernel(),
    useShifted_(useShifted),
    projTimeScale_(1.0),
    interpTogether_(solnOpts.get_mdot_interp()),
    om_interpTogether_(1.0 - interpTogether_),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::topo_)->ipNodeMap())
 {
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  const std::string velbc_name =
    solnOpts.activateOpenMdotCorrection_ ? "velocity_bc" : "cont_velocity_bc";
  velocityBC_ = get_field_ordinal(metaData, velbc_name);
  densityBC_ = get_field_ordinal(metaData, "density");
  exposedAreaVec_ = get_field_ordinal(metaData, "exposed_area_vector", metaData.side_rank());
  
  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::topo_);
  
  // add master elements
  dataPreReqs.add_cvfem_face_me(meFC);
  
  // required fields
  dataPreReqs.add_gathered_nodal_field(velocityBC_, BcAlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(densityBC_, 1);
  dataPreReqs.add_face_field(exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  
  if ( useShifted )
    get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shifted_shape_fcn(ptr);}, vf_shape_function_);
  else
    get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shape_fcn(ptr);}, vf_shape_function_);
 }
  
template<typename BcAlgTraits>
ContinuityInflowElemKernel<BcAlgTraits>::~ContinuityInflowElemKernel()
{}

template<typename BcAlgTraits>
void
ContinuityInflowElemKernel<BcAlgTraits>::setup(const TimeIntegrator &timeIntegrator)
{
  const double dt = timeIntegrator.get_time_step();
  const double gamma1 = timeIntegrator.get_gamma1();
  projTimeScale_ = dt/gamma1;
}

template<typename BcAlgTraits>
void
ContinuityInflowElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType **>&/*lhs*/,
  SharedMemView<DoubleType *>&rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_uBip[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_rho_uBip[BcAlgTraits::nDim_];
  
  SharedMemView<DoubleType**>& vf_velocityBC = scratchViews.get_scratch_view_2D(velocityBC_);
  SharedMemView<DoubleType*>& vf_density = scratchViews.get_scratch_view_1D(densityBC_);
  SharedMemView<DoubleType**>& vf_exposedAreaVec = scratchViews.get_scratch_view_2D(exposedAreaVec_);
  
  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    
    // nearest node (to which we will assemble RHS)
    const int nearestNode = ipNodeMap_[ip];
    
    // zero out vector quantities
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      w_uBip[j] = 0.0;
      w_rho_uBip[j] = 0.0;
    }
    DoubleType rhoBip = 0.0;
    
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType r = vf_shape_function_(ip,ic);
      const DoubleType rhoIC = vf_density(ic);
      for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
        w_uBip[j] += r*vf_velocityBC(ic,j);
        w_rho_uBip[j] += r*rhoIC*vf_velocityBC(ic,j);
      }
    }
    
    DoubleType mDot = 0.0;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      mDot += (interpTogether_*w_rho_uBip[j] + om_interpTogether_*rhoBip*w_uBip[j])*vf_exposedAreaVec(ip,j);
    }
    
    rhs(nearestNode) -= mDot / projTimeScale_;
  } 
}

INSTANTIATE_KERNEL_FACE(ContinuityInflowElemKernel)

}  // nalu
}  // sierra
