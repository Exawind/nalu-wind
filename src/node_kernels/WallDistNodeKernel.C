// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/WallDistNodeKernel.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

WallDistNodeKernel::WallDistNodeKernel(stk::mesh::BulkData& bulk)
  : NGPNodeKernel<WallDistNodeKernel>(),
    dualNodalVolumeID_(
      get_field_ordinal(bulk.mesh_meta_data(), "dual_nodal_volume"))
{
}

void
WallDistNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
}

KOKKOS_FUNCTION
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

} // namespace nalu
} // namespace sierra
