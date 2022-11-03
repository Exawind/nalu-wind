// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumBoussinesqNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"

#include "SolutionOptions.h"

namespace sierra {
namespace nalu {

MomentumBoussinesqNodeKernel::MomentumBoussinesqNodeKernel(
  const stk::mesh::BulkData& bulk, const SolutionOptions& solnOpts)
  : NGPNodeKernel<MomentumBoussinesqNodeKernel>(),
    nDim_(bulk.mesh_meta_data().spatial_dimension()),
    tRef_(solnOpts.referenceTemperature_),
    rhoRef_(solnOpts.referenceDensity_),
    beta_(solnOpts.thermalExpansionCoeff_)
{
  const auto& meta = bulk.mesh_meta_data();

  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
  temperatureID_ = get_field_ordinal(meta, "temperature");

  const std::vector<double>& solnOptsGravity =
    solnOpts.get_gravity_vector(nDim_);
  for (int i = 0; i < nDim_; i++)
    gravity_[i] = solnOptsGravity[i];
}

void
MomentumBoussinesqNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  temperature_ = fieldMgr.get_field<double>(temperatureID_);
}

KOKKOS_FUNCTION
void
MomentumBoussinesqNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType temperature = temperature_.get(node, 0);
  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);
  const double fac = -rhoRef_ * beta_ * (temperature - tRef_) * dualVolume;

  for (int i = 0; i < nDim_; ++i) {
    rhs(i) += fac * gravity_[i];
  }
}

} // namespace nalu
} // namespace sierra
