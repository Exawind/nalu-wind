/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/ScalarMassBDF2NodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"

namespace sierra{
namespace nalu{

ScalarMassBDF2NodeKernel::ScalarMassBDF2NodeKernel(
  const stk::mesh::BulkData& bulk,
  const std::vector<double>&,
  ScalarFieldType *scalarQ
) : NGPNodeKernel<ScalarMassBDF2NodeKernel>()
{
  const auto& meta = bulk.mesh_meta_data();
  scalarQNm1ID_ = scalarQ->field_of_state(stk::mesh::StateNM1).mesh_meta_data_ordinal();
  scalarQNID_ = scalarQ->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal();
  scalarQNp1ID_ = scalarQ->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();
  densityNm1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNM1);
  densityNID_ = get_field_ordinal(meta, "density", stk::mesh::StateN);
  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
}

void
ScalarMassBDF2NodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  scalarQNm1_ = fieldMgr.get_field<double>(scalarQNm1ID_);
  scalarQN_ = fieldMgr.get_field<double>(scalarQNID_);
  scalarQNp1_ = fieldMgr.get_field<double>(scalarQNp1ID_);
  densityNm1_ = fieldMgr.get_field<double>(densityNm1ID_);
  densityN_ = fieldMgr.get_field<double>(densityNID_);
  densityNp1_ = fieldMgr.get_field<double>(densityNp1ID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  dt_ = realm.get_time_step();
  gamma1_ = realm.get_gamma1();
  gamma2_ = realm.get_gamma2();
  gamma3_ = realm.get_gamma3();
}

void
ScalarMassBDF2NodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType qNm1       = scalarQNm1_.get(node, 0);
  const NodeKernelTraits::DblType qN         = scalarQN_.get(node, 0);
  const NodeKernelTraits::DblType qNp1       = scalarQNp1_.get(node, 0);
  const NodeKernelTraits::DblType rhoNm1     = densityNm1_.get(node, 0);
  const NodeKernelTraits::DblType rhoN       = densityN_.get(node, 0);
  const NodeKernelTraits::DblType rhoNp1     = densityNp1_.get(node, 0);
  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);
  const NodeKernelTraits::DblType lhsTime    = gamma1_*rhoNp1*dualVolume/dt_;
  rhs(0) -= (gamma1_*rhoNp1*qNp1 + gamma2_*qN*rhoN + gamma3_*qNm1*rhoNm1)*dualVolume/dt_;
  lhs(0, 0) += lhsTime;
}

} // namespace nalu
} // namespace Sierra
