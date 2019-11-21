// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


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
  bool lumpedMass)
  : lumpedMass_(lumpedMass)
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  actuator_source_ = get_field_ordinal(metaData, "actuator_source");
  actuator_source_lhs_ = get_field_ordinal(metaData, "actuator_source_lhs");
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  meSCV_ = MasterElementRepo::get_volume_master_element<AlgTraits>();

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV_);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(actuator_source_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(actuator_source_lhs_, AlgTraits::nDim_);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  if (lumpedMass_)
    dataPreReqs.add_master_element_call(SCV_SHIFTED_SHAPE_FCN, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCV_SHAPE_FCN, CURRENT_COORDINATES);
}

template<typename AlgTraits>
void
MomentumActuatorSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  const auto& v_actuator_source = scratchViews.get_scratch_view_2D(actuator_source_);
  const auto& v_actuator_source_lhs = scratchViews.get_scratch_view_2D(actuator_source_lhs_);
  const auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  const auto& v_scv_volume = meViews.scv_volume;
  const auto& v_shape_function = lumpedMass_ ? meViews.scv_shifted_shape_fcn : meViews.scv_shape_fcn;
  const auto* ipNodeMap = meSCV_->ipNodeMap();

  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {

    const int nearestNode = ipNodeMap[ip];

    NALU_ALIGNED DoubleType actuatorSourceIp[AlgTraits::nDim_];
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      actuatorSourceIp[i] = 0.0;
    }

    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function(ip, ic);
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
