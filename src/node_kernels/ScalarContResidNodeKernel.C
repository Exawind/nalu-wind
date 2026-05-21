// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/ScalarContResidNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"

namespace sierra::kynema_ugf {

ScalarContResNodeKernel::ScalarContResNodeKernel(
  const stk::mesh::BulkData& bulk, ScalarFieldType* scalarQ)
  : NGPNodeKernel<ScalarContResNodeKernel>()
{
  const auto& meta = bulk.mesh_meta_data();

  qID_ = scalarQ->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();
  contresID_ = get_field_ordinal(meta, "continuity_residual");
  dnvID_ = get_field_ordinal(meta, "dual_nodal_volume");
}

void
ScalarContResNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  q_np1_ = fieldMgr.get_field<double>(qID_);
  cont_res_ = fieldMgr.get_field<double>(contresID_);
  dnv_np1_ = fieldMgr.get_field<double>(dnvID_);
}

KOKKOS_FUNCTION
void
ScalarContResNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const auto lhs_fac = dnv_np1_(node, 0) * cont_res_(node, 0);
  lhs(0, 0) -= lhs_fac;
  rhs(0) += lhs_fac * q_np1_(node, 0);
}

} // namespace sierra::kynema_ugf
