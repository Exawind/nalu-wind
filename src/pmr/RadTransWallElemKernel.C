/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "pmr/RadTransWallElemKernel.h"
#include "pmr/RadiativeTransportEquationSystem.h"
#include "master_element/MasterElement.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename BcAlgTraits>
RadTransWallElemKernel<BcAlgTraits>::RadTransWallElemKernel(
  const stk::mesh::BulkData& bulkData,
  RadiativeTransportEquationSystem *radEqSystem,
  const bool & /* useShifted */,
  ElemDataRequests &dataPreReqs)
  : Kernel(),
    radEqSystem_(radEqSystem),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::topo_)->ipNodeMap())
 {
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  bcIntensity_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "intensity_bc");
  intensity_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "intensity");
  exposedAreaVec_ = metaData.get_field<GenericFieldType>(metaData.side_rank(), "exposed_area_vector");

  MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::topo_);
 
  // compute and save shape function
  get_face_shape_fn_data<BcAlgTraits>([&](double* ptr){meFC->shape_fcn(ptr);}, vf_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_face_me(meFC);
  
  // required fields
  dataPreReqs.add_gathered_nodal_field(*intensity_, 1);
  dataPreReqs.add_gathered_nodal_field(*bcIntensity_, 1);
  dataPreReqs.add_face_field(*exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);

  // vf
}

template<typename BcAlgTraits>
RadTransWallElemKernel<BcAlgTraits>::~RadTransWallElemKernel()
{}

template<typename BcAlgTraits>
void
RadTransWallElemKernel<BcAlgTraits>::setup(const TimeIntegrator& /*timeIntegrator*/)
{
  // extract ordinate direction and copy to Double type
  std::vector<double> Sk(BcAlgTraits::nDim_,0.0);
  radEqSystem_->get_current_ordinate(&Sk[0]);
  for ( int j = 0; j < BcAlgTraits::nDim_; ++j )
    v_Sk_(j) = Sk[j];
}

template<typename BcAlgTraits>
void
RadTransWallElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType **>&lhs,
  SharedMemView<DoubleType *>&rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType*>& v_intensity = scratchViews.get_scratch_view_1D(
    *intensity_);
  SharedMemView<DoubleType*>& v_bcIntensity = scratchViews.get_scratch_view_1D(
    *bcIntensity_); 
  SharedMemView<DoubleType**>& vf_exposedAreaVec = scratchViews.get_scratch_view_2D(*exposedAreaVec_);

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {

    // nearest node (to which we will assemble RHS)
    const int nearestNode = ipNodeMap_[ip];

    // interpolate to bip; both intensity flavors
    DoubleType iBip = 0.0;
    DoubleType iBcBip = 0.0;
    for ( int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic ) {
      const DoubleType r = vf_shape_function_(ip,ic);
      iBip += r*v_intensity(ic);
      iBcBip += r*v_bcIntensity(ic);
    }
    
    // determine in or out intensity bc based on sign of ajsj
    DoubleType ajsj = 0.0;
    for ( int j = 0; j < BcAlgTraits::nDim_; ++j ) {
      ajsj += vf_exposedAreaVec(ip,j)*v_Sk_(j);
    }

    // account for both cases, i.e., Int sj I njdS; flavor of I depends on sign of ajsj
    const DoubleType ajsjFac = stk::math::if_then_else(ajsj > 0, 1.0, 0.0);
    const DoubleType om_ajsjFac = 1.0 - ajsjFac;
   
    // use dof intensity (ajsjFac  == unity); requires LHS assemble
    for (int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic) {
      lhs(nearestNode,ic) += ajsj*vf_shape_function_(ip,ic)*ajsjFac;
    }
    rhs(nearestNode) -= iBip*ajsj*ajsjFac;
    
    // use bc intensity (ajsjFac == 0); requires NO LHS assembly
    rhs(nearestNode) -= iBcBip*ajsj*om_ajsjFac;
  } 
}

INSTANTIATE_KERNEL_FACE(RadTransWallElemKernel)

}  // nalu
}  // sierra
