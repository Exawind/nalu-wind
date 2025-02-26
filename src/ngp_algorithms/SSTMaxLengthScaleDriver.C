// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/SSTMaxLengthScaleDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"

namespace sierra {
namespace nalu {

SSTMaxLengthScaleDriver::SSTMaxLengthScaleDriver(Realm& realm)
  : NgpAlgDriver(realm)
{
}

void
SSTMaxLengthScaleDriver::pre_work()
{

  auto* maxLengthScale = realm_.meta_data().template get_field<double>(
    stk::topology::NODE_RANK, "sst_max_length_scale");
  stk::mesh::field_fill(0.0, *maxLengthScale);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpMaxLengthScale = fieldMgr.template get_field<double>(
    maxLengthScale->mesh_meta_data_ordinal());
  ngpMaxLengthScale.set_all(ngpMesh, 0.0);
}

void
SSTMaxLengthScaleDriver::post_work()
{

  const auto& meshInfo = realm_.mesh_info();
  auto& ngpMaxLengthScale =
    nalu_ngp::get_ngp_field(meshInfo, "sst_max_length_scale");

  const auto& meta = realm_.meta_data();
  auto* maxLengthScale = meta.template get_field<double>(
    stk::topology::NODE_RANK, "sst_max_length_scale");

  comm::scatter_max(realm_.bulk_data(), {maxLengthScale});

}
} // namespace nalu
} // namespace sierra
