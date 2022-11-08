// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/ContinuityMassBDFNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"
#include "utils/FieldHelpers.h"

namespace sierra {
namespace nalu {

ContinuityMassBDFNodeKernel::ContinuityMassBDFNodeKernel(
  const stk::mesh::BulkData& bulk)
  : NGPNodeKernel<ContinuityMassBDFNodeKernel>()
{
  const auto& meta = bulk.mesh_meta_data();

  const ScalarFieldType* density =
    meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");

  densityNID_ = get_field_ordinal(meta, "density", stk::mesh::StateN);

  if (density->number_of_states() == 2)
    densityNm1ID_ = densityNID_;
  else
    densityNm1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNM1);

  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);

  dnvNp1ID_ = get_field_ordinal(meta, "dual_nodal_volume", stk::mesh::StateNP1);
  populate_dnv_states(meta, dnvNm1ID_, dnvNID_, dnvNp1ID_);
}

void
ContinuityMassBDFNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  densityNm1_ = fieldMgr.get_field<double>(densityNm1ID_);
  densityN_ = fieldMgr.get_field<double>(densityNID_);
  densityNp1_ = fieldMgr.get_field<double>(densityNp1ID_);

  dnvNp1_ = fieldMgr.get_field<double>(dnvNp1ID_);
  dnvN_ = fieldMgr.get_field<double>(dnvNID_);
  dnvNm1_ = fieldMgr.get_field<double>(dnvNm1ID_);
  dt_ = realm.get_time_step();
  gamma1_ = realm.get_gamma1();
  gamma2_ = realm.get_gamma2();
  gamma3_ = realm.get_gamma3();
}

KOKKOS_FUNCTION
void
ContinuityMassBDFNodeKernel::execute(
  NodeKernelTraits::LhsType& /*lhs*/,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType rhoNm1 = densityNm1_.get(node, 0);
  const NodeKernelTraits::DblType rhoN = densityN_.get(node, 0);
  const NodeKernelTraits::DblType rhoNp1 = densityNp1_.get(node, 0);
  const NodeKernelTraits::DblType dnvNp1 = dnvNp1_.get(node, 0);
  const NodeKernelTraits::DblType dnvN = dnvN_.get(node, 0);
  const NodeKernelTraits::DblType dnvNm1 = dnvNm1_.get(node, 0);

  // original kernel
  // const NodeKernelTraits::DblType projTimeScale = dt_/gamma1_;
  // rhs(0) -= (gamma1_*rhoNp1 + gamma2_*rhoN +
  // gamma3_*rhoNm1)*dualVolume/dt_/projTimeScale; lhs(0, 0) += 0.0;

  // simplified kernel
  rhs(0) -= (gamma1_ * rhoNp1 * dnvNp1 + gamma2_ * rhoN * dnvN +
             gamma3_ * rhoNm1 * dnvNm1) /
            dt_ * (gamma1_ / dt_);
}

} // namespace nalu
} // namespace sierra
