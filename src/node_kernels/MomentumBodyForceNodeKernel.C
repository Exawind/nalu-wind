/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/MomentumBodyForceNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

MomentumBodyForceNodeKernel::MomentumBodyForceNodeKernel(
  const stk::mesh::BulkData& bulk,
  const std::vector<double>& params
) : NGPNodeKernel<MomentumBodyForceNodeKernel>(),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
  const auto& meta = bulk.mesh_meta_data();

  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");

  for (int i=0; i < nDim_; ++i)
    forceVector_[i] = params[i];
}

void MomentumBodyForceNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
}

void
MomentumBodyForceNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  for (int i=0; i < nDim_; ++i)
    rhs(i) += dualVolume * forceVector_[i];

  // No LHS contributions
}

}  // nalu
}  // sierra
