/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/ContinuityMassBDFNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"

namespace sierra{
namespace nalu{

ContinuityMassBDFNodeKernel::ContinuityMassBDFNodeKernel(
  const stk::mesh::BulkData& bulk
) : NGPNodeKernel<ContinuityMassBDFNodeKernel>()
{
  const auto& meta = bulk.mesh_meta_data();

  const ScalarFieldType *density = meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");

  densityNID_ = get_field_ordinal(meta, "density", stk::mesh::StateN);

  if (density->number_of_states() == 2)
    densityNm1ID_ = densityNID_;
  else
    densityNm1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNM1);

  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
}

void
ContinuityMassBDFNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

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
ContinuityMassBDFNodeKernel::execute(
  NodeKernelTraits::LhsType& /*lhs*/,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType rhoNm1        = densityNm1_.get(node, 0);
  const NodeKernelTraits::DblType rhoN          = densityN_.get(node, 0);
  const NodeKernelTraits::DblType rhoNp1        = densityNp1_.get(node, 0);
  const NodeKernelTraits::DblType dualVolume    = dualNodalVolume_.get(node, 0);

  //original kernel
  //const NodeKernelTraits::DblType projTimeScale = dt_/gamma1_;
  //rhs(0) -= (gamma1_*rhoNp1 + gamma2_*rhoN + gamma3_*rhoNm1)*dualVolume/dt_/projTimeScale;
  //lhs(0, 0) += 0.0;

  //simplified kernel
  rhs(0) -= (gamma1_*rhoNp1 + gamma2_*rhoN + gamma3_*rhoNm1)*(dualVolume/dt_)*(gamma1_/dt_);
}

} // namespace nalu
} // namespace Sierra
