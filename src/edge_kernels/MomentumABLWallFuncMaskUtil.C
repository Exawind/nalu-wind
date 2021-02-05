// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <edge_kernels/MomentumABLWallFuncMaskUtil.h>
#include <Realm.h>
#include "stk_mesh/base/NgpField.hpp"
#include <ngp_utils/NgpLoopUtils.h>
#include <ngp_utils/NgpTypes.h>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <ngp_utils/NgpFieldUtils.h>

namespace sierra{
namespace nalu{

MomentumABLWallFuncMaskUtil::MomentumABLWallFuncMaskUtil(Realm& realm, stk::mesh::Part* part):
  Algorithm(realm, part),
  maskNodeIndex_(get_field_ordinal(realm.meta_data(), "abl_wall_no_slip_wall_func_node_mask", stk::topology::NODE_RANK))
{}

void
MomentumABLWallFuncMaskUtil::execute()
{
  const auto& meta     = realm_.meta_data();
  const auto ngpMesh   = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto myNodeMask    = fieldMgr.get_field<double>(maskNodeIndex_);

  // This selector only includes edges that touch the abl wall (including those edges on the wall)
  const stk::mesh::Selector sel = (meta.locally_owned_part() | meta.globally_shared_part()) &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  const auto& bulk    = realm_.bulk_data();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, sel);
  auto* node_field    = meta.get_fields()[maskNodeIndex_];

  for (const auto* ib : buckets) {
    for (auto node : *ib) {
       *((double*)stk::mesh::field_data(*node_field, node)) = 0;
    }
  }
}

}
}