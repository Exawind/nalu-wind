// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "overset/UpdateOversetFringeAlgorithmDriver.h"
#include "Realm.h"
#include "overset/OversetManager.h"
#include "ngp_utils/NgpFieldUtils.h"

#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

UpdateOversetFringeAlgorithmDriver::UpdateOversetFringeAlgorithmDriver(
  Realm& realm)
  : AlgorithmDriver(realm)
{}

UpdateOversetFringeAlgorithmDriver::~UpdateOversetFringeAlgorithmDriver()
{}

void
UpdateOversetFringeAlgorithmDriver::register_overset_field_update(
  stk::mesh::FieldBase* field, int nrows, int ncols)
{
  fields_.emplace_back(field, nrows, ncols);
}

void UpdateOversetFringeAlgorithmDriver::execute()
{
  auto* oversetManager = realm_.oversetManager_;
  if (oversetManager->oversetGhosting_ != nullptr) {
#ifndef KOKKOS_ENABLE_CUDA
    std::vector<const stk::mesh::FieldBase*> fVec(fields_.size());
    for (size_t i=0; i < fields_.size(); ++i)
      fVec[i] = fields_[i].field_;
    stk::mesh::communicate_field_data(
      *oversetManager->oversetGhosting_, fVec);
#else
    throw std::runtime_error("Overset ghosting not supported for GPU builds");
#endif
  }

  const auto& fieldMgr = realm_.mesh_info().ngp_field_manager();
  for (auto& finfo: fields_) {
    auto& ngpField = fieldMgr.get_field<double>(finfo.field_->mesh_meta_data_ordinal());
    ngpField.sync_to_host();
  }
  oversetManager->overset_update_fields(fields_);
  for (auto& finfo: fields_) {
    auto& ngpField = fieldMgr.get_field<double>(finfo.field_->mesh_meta_data_ordinal());
    ngpField.modify_on_host();
    ngpField.sync_to_device();
  }
}

}  // nalu
}  // sierra
