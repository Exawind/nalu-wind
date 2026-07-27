// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/FluxDivAlgDriver.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "Realm.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"

namespace sierra::kynema_ugf {

FluxDivAlgDriver::FluxDivAlgDriver(
  Realm& realm, const std::string& div_flux_name)
  : NgpAlgDriver(realm), div_flux_name_(div_flux_name)
{
}

void
FluxDivAlgDriver::pre_work()
{
  auto grad_phi =
    kynema_ugf_ngp::get_ngp_field(realm_.mesh_info(), div_flux_name_);
  grad_phi.set_all(stk::mesh::get_updated_ngp_mesh(realm_.bulk_data()), 0.0);
}

void
FluxDivAlgDriver::post_work()
{
  const auto& meta = realm_.meta_data();
  const auto& bulk = realm_.bulk_data();
  auto* div_flux =
    meta.template get_field<double>(stk::topology::NODE_RANK, div_flux_name_);
  auto& ngp_div_flux =
    kynema_ugf_ngp::get_ngp_field(realm_.mesh_info(), div_flux_name_);
  ngp_div_flux.sync_to_host();

  const std::vector<NGPDoubleFieldType*> fVec{&ngp_div_flux};
  bool doFinalSyncToDevice = false;
  stk::mesh::parallel_sum(bulk, fVec, doFinalSyncToDevice);
  if (realm_.hasPeriodic_) {
    realm_.periodic_field_update(div_flux, 1);
  }

  if (realm_.hasOverset_) {
    realm_.overset_field_update(div_flux, 1, doFinalSyncToDevice);
  }
  ngp_div_flux.modify_on_host();
  ngp_div_flux.sync_to_device();
}

} // namespace sierra::kynema_ugf
