// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumBuoyancyNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"

#include "SolutionOptions.h"

namespace sierra {
namespace nalu {

MomentumBuoyancyNodeKernel::MomentumBuoyancyNodeKernel(
  const stk::mesh::BulkData& bulk, const SolutionOptions& solnOpts)
  : NGPNodeKernel<MomentumBuoyancyNodeKernel>(),
    nDim_(bulk.mesh_meta_data().spatial_dimension()),
    use_balanced_buoyancy_(solnOpts.use_balanced_buoyancy_force_),
    rhoRef_(solnOpts.referenceDensity_)
{
  const auto& meta = bulk.mesh_meta_data();

  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
  densityNp1ID_ = get_field_ordinal(meta, "density");
  if (use_balanced_buoyancy_)
    sourceID_ = get_field_ordinal(meta, "buoyancy_source");

  const std::vector<double>& solnOptsGravity =
    solnOpts.get_gravity_vector(nDim_);
  for (int i = 0; i < nDim_; i++)
    gravity_[i] = solnOptsGravity[i];
}

void
MomentumBuoyancyNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  densityNp1_ = fieldMgr.get_field<double>(densityNp1ID_);
  source_ = fieldMgr.get_field<double>(sourceID_);
}

KOKKOS_FUNCTION
void
MomentumBuoyancyNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType rhoNp1 = densityNp1_.get(node, 0);
  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);
  const double fac = (rhoNp1 - rhoRef_) * dualVolume;

  if (use_balanced_buoyancy_) {
    for (int i = 0; i < nDim_; ++i) {
      rhs(i) += source_.get(node, i) * dualVolume;
    }
  } else {
    for (int i = 0; i < nDim_; ++i) {
      rhs(i) += fac * gravity_[i];
    }
  }
}

} // namespace nalu
} // namespace sierra
