// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

// nalu
#include <CopyFieldAlgorithm.h>

#include <Realm.h>
#include <FieldTypeDef.h>
#include <ngp_utils/NgpFieldBLAS.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// CopyFieldAlgorithm - copy fields from one to another on a set of select
//                      parts; begin/end can be the size of the field as well
//                      should it be operating on integration point data
//==========================================================================
CopyFieldAlgorithm::CopyFieldAlgorithm(
  Realm& realm,
  const stk::mesh::PartVector& part_vec,
  stk::mesh::FieldBase* fromField,
  stk::mesh::FieldBase* toField,
  const unsigned beginPos,
  const unsigned endPos,
  const stk::mesh::EntityRank entityRank)
  : Algorithm(realm, part_vec),
    fromField_(fromField),
    toField_(toField),
    beginPos_(beginPos),
    endPos_(endPos),
    entityRank_(entityRank)
{
}
CopyFieldAlgorithm::CopyFieldAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  stk::mesh::FieldBase* fromField,
  stk::mesh::FieldBase* toField,
  const unsigned beginPos,
  const unsigned endPos,
  const stk::mesh::EntityRank entityRank)
  : Algorithm(realm, part),
    fromField_(fromField),
    toField_(toField),
    beginPos_(beginPos),
    endPos_(endPos),
    entityRank_(entityRank)
{
}

void
CopyFieldAlgorithm::execute()
{
  stk::mesh::Selector selector = stk::mesh::selectUnion(partVec_);
  const auto& fieldMgr = realm_.ngp_field_manager();
  auto& toField =
    fieldMgr.get_field<double>(toField_->mesh_meta_data_ordinal());
  auto& fromField =
    fieldMgr.get_field<double>(fromField_->mesh_meta_data_ordinal());
  fromField.sync_to_device();
  toField.sync_to_device();
  nalu_ngp::field_copy(
    realm_.ngp_mesh(), selector, toField, fromField, beginPos_, endPos_,
    entityRank_);
  toField.modify_on_device();
}

} // namespace nalu
} // namespace sierra
