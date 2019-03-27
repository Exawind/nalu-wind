/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumActuatorSrcElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
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

template<typename AlgTraits>
MomentumActuatorSrcElemKernel<AlgTraits>::MomentumActuatorSrcElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs,
  bool lumped)
  : Kernel(),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap())
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  actuator_source_ = get_field_ordinal(metaData, "actuator_source");
  actuator_source_lhs_ = get_field_ordinal(metaData, "actuator_source_lhs");
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  MasterElement* meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);

  if ( lumped ) {
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shifted_shape_fcn(ptr);}, v_shape_function_);
  }
  else {
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_);
  }

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(actuator_source_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(actuator_source_lhs_, AlgTraits::nDim_);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
}

template<typename AlgTraits>
MomentumActuatorSrcElemKernel<AlgTraits>::~MomentumActuatorSrcElemKernel()
{}

template<typename AlgTraits>
void
MomentumActuatorSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType**>& v_actuator_source = scratchViews.get_scratch_view_2D(actuator_source_);
  SharedMemView<DoubleType**>& v_actuator_source_lhs = scratchViews.get_scratch_view_2D(actuator_source_lhs_);
  SharedMemView<DoubleType*>& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {

    const int nearestNode = ipNodeMap_[ip];

    NALU_ALIGNED DoubleType actuatorSourceIp[AlgTraits::nDim_];
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      actuatorSourceIp[i] = 0.0;
    }

    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function_(ip, ic);
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
          const DoubleType uj = v_actuator_source(ic,j);
          actuatorSourceIp[j] += r * uj;
      }
    }

    // Compute LHS and RHS contributions
    // LHS contribution is always lumped
    const DoubleType scV = v_scv_volume(ip);
    const int nnNdim = nearestNode * AlgTraits::nDim_;
    for (int j=0; j < AlgTraits::nDim_; ++j) {
      rhs(nnNdim + j) += actuatorSourceIp[j] * scV;
      lhs(nnNdim + j, nnNdim + j) += v_actuator_source_lhs(nearestNode, j) * scV;
    }


  }
}

INSTANTIATE_KERNEL(MomentumActuatorSrcElemKernel)

}  // nalu
}  // sierra
