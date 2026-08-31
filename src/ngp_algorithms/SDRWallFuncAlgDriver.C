// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/SDRWallFuncAlgDriver.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "PeriodicManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"
#include "stk_util/ngp/NgpSpaces.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace kynema_ugf {

SDRWallFuncAlgDriver::SDRWallFuncAlgDriver(Realm& realm) : NgpAlgDriver(realm)
{
}

void
SDRWallFuncAlgDriver::pre_work()
{
  const auto& ngpMesh = realm_.ngp_mesh();
  auto& bcsdr =
    kynema_ugf_ngp::get_ngp_field(realm_.mesh_info(), "wall_model_sdr_bc");
  auto& wallArea = kynema_ugf_ngp::get_ngp_field(
    realm_.mesh_info(), "assembled_wall_area_sdr");

  bcsdr.set_all(ngpMesh, 0.0);
  wallArea.set_all(ngpMesh, 0.0);
}

void
SDRWallFuncAlgDriver::post_work()
{
  using MeshIndex =
    kynema_ugf_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;
  const auto& ngpMesh = realm_.ngp_mesh();

  auto& bcsdr =
    kynema_ugf_ngp::get_ngp_field(realm_.mesh_info(), "wall_model_sdr_bc");
  auto& wallArea = kynema_ugf_ngp::get_ngp_field(
    realm_.mesh_info(), "assembled_wall_area_sdr");
  auto& sdr = kynema_ugf_ngp::get_ngp_field(
    realm_.mesh_info(), "specific_dissipation_rate");
  auto& sdrWallBC = kynema_ugf_ngp::get_ngp_field(realm_.mesh_info(), "sdr_bc");

  bcsdr.clear_sync_state();
  wallArea.clear_sync_state();
  bcsdr.modify_on_device();
  wallArea.modify_on_device();

  // Parallel synchronization
  auto* bcsdrF =
    realm_.meta_data().get_field(stk::topology::NODE_RANK, "wall_model_sdr_bc");
  auto* wallAreaF = realm_.meta_data().get_field(
    stk::topology::NODE_RANK, "assembled_wall_area_sdr");
  const std::vector<const stk::mesh::FieldBase*> fields{bcsdrF, wallAreaF};
  stk::mesh::parallel_sum<stk::ngp::DeviceSpace>(realm_.bulk_data(), fields);
  if (realm_.hasPeriodic_) {
    // Periodic synchronization
    const unsigned nComponents = 1;
    const bool bypassFieldCheck = false;
    const bool addMirrorValues = true;
    const bool setMirrorValues = true;

    auto* periodicMgr = realm_.periodicManager_;
    periodicMgr->ngp_apply_constraints(
      bcsdrF, nComponents, bypassFieldCheck, addMirrorValues, setMirrorValues);
    periodicMgr->ngp_apply_constraints(
      wallAreaF, nComponents, bypassFieldCheck, addMirrorValues,
      setMirrorValues);
  }

  // Normalize the computed BC SDR
  const stk::mesh::Selector sel = (realm_.meta_data().locally_owned_part() |
                                   realm_.meta_data().globally_shared_part()) &
                                  stk::mesh::selectField(*bcsdrF);

  kynema_ugf_ngp::run_entity_algorithm(
    "SDRWallFuncAlgDriver_normalize", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double warea = wallArea.get(mi, 0);
      const double sdrVal = bcsdr.get(mi, 0) / warea;

      bcsdr.get(mi, 0) = sdrVal;
      sdrWallBC.get(mi, 0) = sdrVal;
      sdr.get(mi, 0) = sdrVal;
    });

  wallArea.clear_sync_state();
  bcsdr.clear_sync_state();
  wallArea.modify_on_device();
  bcsdr.modify_on_device();
  sdrWallBC.modify_on_device();
  sdr.modify_on_device();
}

} // namespace kynema_ugf
} // namespace sierra
