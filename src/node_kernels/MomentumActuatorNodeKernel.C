/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/MomentumActuatorNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"

#include "SolutionOptions.h"

namespace sierra{
namespace nalu{

MomentumActuatorNodeKernel::MomentumActuatorNodeKernel(
  const stk::mesh::BulkData& bulk 
) : NGPNodeKernel<MomentumActuatorNodeKernel>(),
    nDim_(bulk.mesh_meta_data().spatial_dimension()),
    dualNodalVolumeID_ (get_field_ordinal(bulk.mesh_meta_data(), "dual_nodal_volume")),
    actuatorSrcID_     (get_field_ordinal(bulk.mesh_meta_data(), "actuator_source")),
    actuatorSrcLHSID_  (get_field_ordinal(bulk.mesh_meta_data(), "actuator_source_lhs"))
{
}

void MomentumActuatorNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  actuatorSrc_     = fieldMgr.get_field<double>(actuatorSrcID_);
  actuatorSrcLHS_  = fieldMgr.get_field<double>(actuatorSrcLHSID_);
}

void
MomentumActuatorNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  for ( int i = 0; i < nDim_; ++i ) {
    const NodeKernelTraits::DblType src      = actuatorSrc_.    get(node,i);
    const NodeKernelTraits::DblType srcLHS   = actuatorSrcLHS_. get(node,i);
    rhs(i)   += dualVolume*src;
    lhs(i,i) += dualVolume*srcLHS; 
  }
}

} // namespace nalu
} // namespace Sierra
