// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumContResidNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"

namespace sierra::kynema_ugf {

MomentumContResNodeKernel::MomentumContResNodeKernel(
  const stk::mesh::BulkData& bulk)
  : NGPNodeKernel<MomentumContResNodeKernel>()
{
  const auto& meta = bulk.mesh_meta_data();

  dim_ = int(meta.spatial_dimension());
  qID_ = get_field_ordinal(meta, "velocity");
  contresID_ = get_field_ordinal(meta, "continuity_residual");
  dnvID_ = get_field_ordinal(meta, "dual_nodal_volume");
}

void
MomentumContResNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  q_np1_ = fieldMgr.get_field<double>(qID_);
  cont_res_ = fieldMgr.get_field<double>(contresID_);
  dnv_np1_ = fieldMgr.get_field<double>(dnvID_);
}

KOKKOS_FUNCTION
void
MomentumContResNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const auto lhs_fac = dnv_np1_(node, 0) * cont_res_(node, 0);
  for (int d = 0; d < dim_; ++d) {
    lhs(d, d) -= lhs_fac;
    rhs(d) += lhs_fac * q_np1_(node, d);
  }
}

} // namespace sierra::kynema_ugf
