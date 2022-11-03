// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumABLForceNodeKernel.h"
#include "wind_energy/ABLForcingAlgorithm.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

MomentumABLForceNodeKernel::MomentumABLForceNodeKernel(
  const stk::mesh::BulkData& bulk, const SolutionOptions& solnOpts)
  : NGPNodeKernel<MomentumABLForceNodeKernel>(),
    coordinatesID_(get_field_ordinal(
      bulk.mesh_meta_data(), solnOpts.get_coordinates_name())),
    dualNodalVolumeID_(
      get_field_ordinal(bulk.mesh_meta_data(), "dual_nodal_volume")),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
}

void
MomentumABLForceNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  ablSrc_ = realm.ablForcingAlg_->velocity_source_interpolator();
}

KOKKOS_FUNCTION
void
MomentumABLForceNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  NALU_ALIGNED NodeKernelTraits::DblType momSrc[NodeKernelTraits::NDimMax];

  const NodeKernelTraits::DblType dualVol = dualNodalVolume_.get(node, 0);

  ablSrc_(coordinates_.get(node, nDim_ - 1), momSrc);

  for (int i = 0; i < nDim_; ++i)
    rhs(i) += dualVol * momSrc[i];
}

} // namespace nalu
} // namespace sierra
