// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/FieldUpdateAlgDriver.h"
#include "Realm.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

FieldUpdateAlgDriver::FieldUpdateAlgDriver(
  Realm& realm, const std::string& fieldName)
  : NgpAlgDriver(realm), fieldName_(fieldName)
{
}

void
FieldUpdateAlgDriver::pre_work()
{
  const auto& meta = realm_.meta_data();
  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto field = fieldMgr.get_field<double>(get_field_ordinal(meta, fieldName_));

  auto* nonngpField =
    meta.get_field<double>(stk::topology::NODE_RANK, fieldName_);
  stk::mesh::field_fill(0.0, *nonngpField);

  field.set_all(ngpMesh, 0.0);
}

void
FieldUpdateAlgDriver::post_work()
{
  // TODO: Revisit logic after STK updates to ngp parallel updates
  const auto& meta = realm_.meta_data();
  const auto& bulk = realm_.bulk_data();
  const int nDim = meta.spatial_dimension();

  auto* field = meta.get_field(stk::topology::NODE_RANK, fieldName_);
  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpField =
    fieldMgr.get_field<double>(get_field_ordinal(meta, fieldName_));

  ngpField.modify_on_device();
  realm_.scatter_sum_with_overset({field});
}

} // namespace nalu
} // namespace sierra
