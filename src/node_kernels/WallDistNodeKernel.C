/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/WallDistNodeKernel.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

WallDistNodeKernel::WallDistNodeKernel(
  stk::mesh::BulkData& bulk
) : NGPNodeKernel<WallDistNodeKernel>(),
    dualNodalVolumeID_(get_field_ordinal(bulk.mesh_meta_data(), "dual_nodal_volume"))
{}

void
WallDistNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
}

void
WallDistNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType dualVol = dualNodalVolume_.get(node, 0);

  rhs(0) += dualVol;
  // No LHS contributions
}

}  // nalu
}  // sierra
