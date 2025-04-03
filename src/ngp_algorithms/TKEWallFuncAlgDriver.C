// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/TKEWallFuncAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

TKEWallFuncAlgDriver::TKEWallFuncAlgDriver(Realm& realm) : NgpAlgDriver(realm)
{
}

void
TKEWallFuncAlgDriver::pre_work()
{
  // Defer getting the field ordinals until after equation systems have done
  // their initialization
  tke_ =
    get_field_ordinal(realm_.meta_data(), "turbulent_ke", stk::mesh::StateNP1);
  bctke_ = get_field_ordinal(realm_.meta_data(), "tke_bc");
  bcNodalTke_ = get_field_ordinal(realm_.meta_data(), "wall_model_tke_bc");
  wallArea_ = get_field_ordinal(realm_.meta_data(), "assembled_wall_area_wf");

  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  auto ngpBcNodalTke = fieldMgr.template get_field<double>(bcNodalTke_);

  // Reset 'assembled' BC TKE nodal field
  ngpBcNodalTke.set_all(ngpMesh, 0.0);
}

void
TKEWallFuncAlgDriver::post_work()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  auto ngpBcNodalTke = fieldMgr.get_field<double>(bcNodalTke_);
  auto ngpTke = fieldMgr.get_field<double>(tke_);
  auto ngpBcTke = fieldMgr.get_field<double>(bctke_);
  auto ngpWallArea = fieldMgr.get_field<double>(wallArea_);

  // TODO: Replace logic with STK NGP parallel sum, handle periodic the NGP way
  ngpBcNodalTke.sync_to_host();

  stk::mesh::FieldBase* bcNodalTkeField =
    realm_.meta_data().get_fields()[bcNodalTke_];
  comm::scatter_sum(realm_.bulk_data(), {bcNodalTkeField});

  // Normalize the computed BC TKE at integration points with assembled wall
  // area and assign it to TKE and TKE BC fields on this sideset for use in
  // the next solve.
  const stk::mesh::Selector sel =
    (realm_.meta_data().locally_owned_part() |
     realm_.meta_data().globally_shared_part()) &
    stk::mesh::selectField(*realm_.meta_data().get_field(
      stk::topology::NODE_RANK, "wall_model_tke_bc"));

  nalu_ngp::run_entity_algorithm(
    "TKEWallFuncAlgDriver_normalize", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double warea = ngpWallArea.get(mi, 0);
      const double tkeVal = ngpBcNodalTke.get(mi, 0) / warea;
      ngpBcNodalTke.get(mi, 0) = tkeVal;
      ngpBcTke.get(mi, 0) = tkeVal;
      ngpTke.get(mi, 0) = tkeVal;
    });

  ngpBcNodalTke.modify_on_device();
  ngpBcTke.modify_on_device();
  ngpTke.modify_on_device();
}

} // namespace nalu
} // namespace sierra
