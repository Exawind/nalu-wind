/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/ContinuityMassBackwardEulerNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"

namespace sierra{
namespace nalu{

ContinuityMassBackwardEulerNodeKernel::ContinuityMassBackwardEulerNodeKernel(
  const stk::mesh::BulkData& bulk
) : NGPNodeKernel<ContinuityMassBackwardEulerNodeKernel>()
{
  const auto& meta = bulk.mesh_meta_data();

  densityNID_ = get_field_ordinal(meta, "density", stk::mesh::StateN);
  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
}

void
ContinuityMassBackwardEulerNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  densityN_ = fieldMgr.get_field<double>(densityNID_);
  densityNp1_ = fieldMgr.get_field<double>(densityNp1ID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  dt_ = realm.get_time_step();
}

void
ContinuityMassBackwardEulerNodeKernel::execute(
  NodeKernelTraits::LhsType& /*lhs*/,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType projTimeScale = dt_;
  const NodeKernelTraits::DblType rhoN       = densityN_.get(node, 0);
  const NodeKernelTraits::DblType rhoNp1     = densityNp1_.get(node, 0);
  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);
  rhs(0) -= (rhoNp1 - rhoN)*dualVolume/dt_/projTimeScale;
  //lhs(0, 0) += 0.0;
}

} // namespace nalu
} // namespace Sierra
