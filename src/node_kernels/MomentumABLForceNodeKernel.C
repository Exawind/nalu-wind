/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/MomentumABLForceNodeKernel.h"
#include "wind_energy/ABLForcingAlgorithm.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

MomentumABLForceNodeKernel::MomentumABLForceNodeKernel(
  const stk::mesh::BulkData& bulk,
  const SolutionOptions& solnOpts
) : NGPNodeKernel<MomentumABLForceNodeKernel>(),
    coordinatesID_(get_field_ordinal(bulk.mesh_meta_data(), solnOpts.get_coordinates_name())),
    dualNodalVolumeID_(get_field_ordinal(bulk.mesh_meta_data(), "dual_nodal_volume")),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{}

void MomentumABLForceNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  ablSrc_ = realm.ablForcingAlg_->velocity_source_interpolator();
}

void MomentumABLForceNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  NALU_ALIGNED NodeKernelTraits::DblType momSrc[NodeKernelTraits::NDimMax];

  const NodeKernelTraits::DblType dualVol = dualNodalVolume_.get(node, 0);

  ablSrc_(coordinates_.get(node, nDim_ - 1), momSrc);

  for (int i=0; i < nDim_; ++i)
    rhs(i) += dualVol * momSrc[i];
}

}  // nalu
}  // sierra
