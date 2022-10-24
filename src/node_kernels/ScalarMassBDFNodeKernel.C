// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/ScalarMassBDFNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"
#include "utils/FieldHelpers.h"

namespace sierra {
namespace nalu {

ScalarMassBDFNodeKernel::ScalarMassBDFNodeKernel(
  const stk::mesh::BulkData& bulk, ScalarFieldType* scalarQ)
  : NGPNodeKernel<ScalarMassBDFNodeKernel>()
{
  const auto& meta = bulk.mesh_meta_data();

  scalarQNID_ =
    scalarQ->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal();

  if (scalarQ->number_of_states() == 2)
    scalarQNm1ID_ = scalarQNID_;
  else
    scalarQNm1ID_ =
      scalarQ->field_of_state(stk::mesh::StateNM1).mesh_meta_data_ordinal();

  scalarQNp1ID_ =
    scalarQ->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();

  densityNID_ = get_field_ordinal(meta, "density", stk::mesh::StateN);

  if (scalarQ->number_of_states() == 2)
    densityNm1ID_ = densityNID_;
  else
    densityNm1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNM1);

  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);

  dnvNp1ID_ = get_field_ordinal(meta, "dual_nodal_volume", stk::mesh::StateNP1);
  populate_dnv_states(meta, dnvNm1ID_, dnvNID_, dnvNp1ID_);
}

void
ScalarMassBDFNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  scalarQNm1_ = fieldMgr.get_field<double>(scalarQNm1ID_);
  scalarQN_ = fieldMgr.get_field<double>(scalarQNID_);
  scalarQNp1_ = fieldMgr.get_field<double>(scalarQNp1ID_);
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
ScalarMassBDFNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType qNm1 = scalarQNm1_.get(node, 0);
  const NodeKernelTraits::DblType qN = scalarQN_.get(node, 0);
  const NodeKernelTraits::DblType qNp1 = scalarQNp1_.get(node, 0);
  const NodeKernelTraits::DblType rhoNm1 = densityNm1_.get(node, 0);
  const NodeKernelTraits::DblType rhoN = densityN_.get(node, 0);
  const NodeKernelTraits::DblType rhoNp1 = densityNp1_.get(node, 0);
  const NodeKernelTraits::DblType dnvNp1 = dnvNp1_.get(node, 0);
  const NodeKernelTraits::DblType dnvN = dnvN_.get(node, 0);
  const NodeKernelTraits::DblType dnvNm1 = dnvNm1_.get(node, 0);

  const NodeKernelTraits::DblType lhsTime = gamma1_ * rhoNp1 * dnvNp1 / dt_;
  rhs(0) -= (gamma1_ * rhoNp1 * qNp1 * dnvNp1 + gamma2_ * qN * rhoN * dnvN +
             gamma3_ * qNm1 * rhoNm1 * dnvNm1) /
            dt_;
  lhs(0, 0) += lhsTime;
}

} // namespace nalu
} // namespace sierra
