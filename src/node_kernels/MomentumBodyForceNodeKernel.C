// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumBodyForceNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

MomentumBodyForceNodeKernel::MomentumBodyForceNodeKernel(
  const stk::mesh::BulkData& bulk, const std::vector<double>& params)
  : NGPNodeKernel<MomentumBodyForceNodeKernel>(),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
  const auto& meta = bulk.mesh_meta_data();

  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");

  for (int i = 0; i < nDim_; ++i)
    forceVector_[i] = params[i];
}

void
MomentumBodyForceNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
}

KOKKOS_FUNCTION
void
MomentumBodyForceNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  for (int i = 0; i < nDim_; ++i)
    rhs(i) += dualVolume * forceVector_[i];

  // No LHS contributions
}

} // namespace nalu
} // namespace sierra
