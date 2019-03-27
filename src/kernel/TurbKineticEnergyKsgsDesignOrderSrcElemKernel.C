/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/TurbKineticEnergyKsgsDesignOrderSrcElemKernel.h"
#include "AlgTraits.h"
#include "Enums.h"
#include "SolutionOptions.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"

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
TurbKineticEnergyKsgsDesignOrderSrcElemKernel<AlgTraits>::TurbKineticEnergyKsgsDesignOrderSrcElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    cEps_(solnOpts.get_turb_model_constant(TM_cEps)),
    tkeProdLimitRatio_(solnOpts.get_turb_model_constant(TM_tkeProdLimitRatio)),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap())
{
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  tkeNp1_ = get_field_ordinal(metaData, "turbulent_ke", stk::mesh::StateNP1);
  densityNp1_ = get_field_ordinal(metaData, "density", stk::mesh::StateNP1);
  velocityNp1_ = get_field_ordinal(metaData, "velocity", stk::mesh::StateNP1);
  tvisc_ = get_field_ordinal(metaData, "turbulent_viscosity");
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  dualNodalVolume_ = get_field_ordinal(metaData, "dual_nodal_volume");

  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);
  get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // required fields
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(tvisc_, 1);
  dataPreReqs.add_gathered_nodal_field(dualNodalVolume_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCV_GRAD_OP, CURRENT_COORDINATES);
}

template<typename AlgTraits>
TurbKineticEnergyKsgsDesignOrderSrcElemKernel<AlgTraits>::~TurbKineticEnergyKsgsDesignOrderSrcElemKernel()
{}

template<typename AlgTraits>
void
TurbKineticEnergyKsgsDesignOrderSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>&lhs,
  SharedMemView<DoubleType *>&rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_dudx [AlgTraits::nDim_][AlgTraits::nDim_];
 
  SharedMemView<DoubleType**>& v_velocityNp1 = scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType*>& v_tkeNp1 = scratchViews.get_scratch_view_1D(
    tkeNp1_);
  SharedMemView<DoubleType*>& v_densityNp1 = scratchViews.get_scratch_view_1D(
    densityNp1_);
  SharedMemView<DoubleType*>& v_tvisc = scratchViews.get_scratch_view_1D(
    tvisc_);
  SharedMemView<DoubleType*>& v_dualNodalVolume = scratchViews.get_scratch_view_1D(
    dualNodalVolume_);
  SharedMemView<DoubleType*>& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;
  SharedMemView<DoubleType***>& v_dndx = scratchViews.get_me_views(CURRENT_COORDINATES).dndx_scv;

  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {
    
    const int nearestNode = ipNodeMap_[ip];
    
    // save off scvol
    const DoubleType scV = v_scv_volume(ip);
    
    // everyting lives at the quadrature points
    DoubleType tkeIp = 0.0;
    DoubleType rhoIp = 0.0;
    DoubleType tviscIp = 0.0;
    DoubleType dualNodalVolIp = 0.0;

    for ( int i = 0; i < AlgTraits::nDim_; ++i )
      for ( int j = 0; j < AlgTraits::nDim_; ++j )
        w_dudx[i][j] = 0.0;

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      
      const DoubleType r = v_shape_function_(ip, ic);      
      tkeIp += r*v_tkeNp1(ic);
      rhoIp += r*v_densityNp1(ic);
      tviscIp += r*v_tvisc(ic);
      dualNodalVolIp += r*v_dualNodalVolume(ic);

      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
        const DoubleType ui = v_velocityNp1(ic,i);
        for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
          w_dudx[i][j] += v_dndx(ip,ic,j)*ui;
        }
      }
    }  
   
    DoubleType Pk = 0.0;
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        Pk += w_dudx[i][j]*(w_dudx[i][j] + w_dudx[j][i]);
      }
    }
    Pk *= stk::math::max(tviscIp,0);
    tkeIp = stk::math::max(tkeIp,0);
    dualNodalVolIp = stk::math::max(dualNodalVolIp,0);
    rhoIp = stk::math::max(rhoIp,0);


    // tke factor
    const DoubleType tkeFac = (AlgTraits::nDim_ == 2) 
      ? cEps_*rhoIp*stk::math::sqrt(tkeIp/dualNodalVolIp)
      : cEps_*rhoIp*stk::math::sqrt(tkeIp)/stk::math::cbrt(dualNodalVolIp);
    
    // dissipation and production (limited)
    DoubleType Dk = tkeFac * tkeIp;
    Pk = stk::math::min(Pk, tkeProdLimitRatio_*Dk);
    
    // assemble RHS and LHS
    rhs(nearestNode) += (Pk - Dk)*scV;   
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      lhs(nearestNode, ic) += 1.5*v_shape_function_(ip,ic)*tkeFac*scV;
    }
  }
} 
  
INSTANTIATE_KERNEL(TurbKineticEnergyKsgsDesignOrderSrcElemKernel)

}  // nalu
}  // sierra
