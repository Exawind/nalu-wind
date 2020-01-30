// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumBodyForceBoxNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

MomentumBodyForceBoxNodeKernel::MomentumBodyForceBoxNodeKernel(
  const stk::mesh::BulkData& bulk,
  const std::string coordName,
  const std::vector<double>& forces,
  const std::vector<double>& box)
  : NGPNodeKernel<MomentumBodyForceBoxNodeKernel>(),
    coordinatesID_(get_field_ordinal(bulk.mesh_meta_data(), coordName)),
    dualNodalVolumeID_(
      get_field_ordinal(bulk.mesh_meta_data(), "dual_nodal_volume")),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
  for (int i = 0; i < nDim_; ++i)
    forceVector_[i] = forces[i];
  for (int i = 0; i < nDim_; ++i) {
    lo_[i] = box[i];
    hi_[i] = box[nDim_ + i];
  }
}

void
MomentumBodyForceBoxNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
}

void
MomentumBodyForceBoxNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{

  bool is_inside = true;
  for (int i = 0; i < nDim_; ++i)
    is_inside =
      (is_inside and (lo_[i] <= coordinates_.get(node, i)) and
       (coordinates_.get(node, i) <= hi_[i]));

  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);
  for (int i = 0; i < nDim_; ++i)
    rhs(i) += (is_inside) ? dualVolume * forceVector_[i] : 0.0;
}

} // namespace nalu
} // namespace sierra
