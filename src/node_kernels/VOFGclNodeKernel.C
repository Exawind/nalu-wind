// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/VOFGclNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

VOFGclNodeKernel::VOFGclNodeKernel(
  const stk::mesh::BulkData& bulk, ScalarFieldType* scalarQ)
  : NGPNodeKernel<VOFGclNodeKernel>()
{
  const auto& meta = bulk.mesh_meta_data();

  scalarQNp1ID_ =
    scalarQ->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();

  divVID_ = get_field_ordinal(meta, "div_mesh_velocity");
  dualNdVolNID_ =
    get_field_ordinal(meta, "dual_nodal_volume", stk::mesh::StateN);

  if (scalarQ->number_of_states() == 2)
    dualNdVolNm1ID_ = dualNdVolNID_;
  else
    dualNdVolNm1ID_ =
      get_field_ordinal(meta, "dual_nodal_volume", stk::mesh::StateNM1);

  dualNdVolNp1ID_ =
    get_field_ordinal(meta, "dual_nodal_volume", stk::mesh::StateNP1);
}

void
VOFGclNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  scalarQNp1_ = fieldMgr.get_field<double>(scalarQNp1ID_);
  divV_ = fieldMgr.get_field<double>(divVID_);
  dualNdVolNm1_ = fieldMgr.get_field<double>(dualNdVolNm1ID_);
  dualNdVolN_ = fieldMgr.get_field<double>(dualNdVolNID_);
  dualNdVolNp1_ = fieldMgr.get_field<double>(dualNdVolNp1ID_);

  gamma1_ = realm.get_gamma1();
  gamma2_ = realm.get_gamma2();
  gamma3_ = realm.get_gamma3();

  dt_ = realm.get_time_step();
}

KOKKOS_FUNCTION
void
VOFGclNodeKernel::execute(
  NodeKernelTraits::LhsType& /*lhs*/,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  // rhs -= rho*scalarQ*div(v)*dV
  const NodeKernelTraits::DblType scalarQNp1 = scalarQNp1_.get(node, 0);
  const NodeKernelTraits::DblType divV = divV_.get(node, 0);
  const NodeKernelTraits::DblType dualNdVolNm1 = dualNdVolNm1_.get(node, 0);
  const NodeKernelTraits::DblType dualNdVolN = dualNdVolN_.get(node, 0);
  const NodeKernelTraits::DblType dualNdVolNp1 = dualNdVolNp1_.get(node, 0);

  const NodeKernelTraits::DblType volRate =
    (gamma1_ * dualNdVolNp1 + gamma2_ * dualNdVolN + gamma3_ * dualNdVolNm1) /
    dt_ / dualNdVolNp1;

  // the term divV comes from the Reynold's transport theorem for moving bodies
  // with changing volume the term (volRate-divV) is the GCL law which presents
  // non-zero errors in a discretized setting
  rhs(0) -= scalarQNp1 * (divV - (volRate - divV)) * dualNdVolNp1;
}

} // namespace nalu
} // namespace sierra
