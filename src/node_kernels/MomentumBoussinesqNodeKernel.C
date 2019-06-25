/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/MomentumBoussinesqNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"

#include "SolutionOptions.h"

namespace sierra{
namespace nalu{

MomentumBoussinesqNodeKernel::MomentumBoussinesqNodeKernel(
  const stk::mesh::BulkData& bulk,
  const std::vector<double>& params,
  const SolutionOptions& solnOpts
) : NGPNodeKernel<MomentumBoussinesqNodeKernel>(),
    nDim_(bulk.mesh_meta_data().spatial_dimension()),
    tRef_(solnOpts.referenceTemperature_),
    rhoRef_(solnOpts.referenceDensity_),
    beta_(solnOpts.thermalExpansionCoeff_)
{
  const auto& meta = bulk.mesh_meta_data();

  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
  temperatureID_ = get_field_ordinal(meta, "temperature");

  for (int i=0; i < nDim_; ++i)
    forceVector_[i] = params[i];

  const std::vector<double>& solnOptsGravity = solnOpts.get_gravity_vector(nDim_);
  for (int i = 0; i < nDim_; i++)
    gravity_[i] = solnOptsGravity[i];
}

void MomentumBoussinesqNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  temperature_ = fieldMgr.get_field<double>(temperatureID_);
}

void
MomentumBoussinesqNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType temperature = temperature_.get(node, 0);
  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);
  const double fac = -rhoRef_*beta_*(temperature - tRef_)*dualVolume;

  for ( int i = 0; i < nDim_; ++i ) {
    rhs(i) += fac*gravity_[i];
  }
}

} // namespace nalu
} // namespace Sierra
