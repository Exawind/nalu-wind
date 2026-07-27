// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/ContinuityResidualAlgDriver.h"
#include "ngp_algorithms/FluxDivAlgDriver.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "Realm.h"

#include "ngp_utils/NgpLoopUtils.h"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Field.hpp"

namespace sierra::kynema_ugf {

ContinuityResidualAlgDriver::ContinuityResidualAlgDriver(Realm& realm)
  : NgpAlgDriver(realm), div_mdot_algs_(realm, "div_mdot")
{
}

void
ContinuityResidualAlgDriver::pre_work()
{
  // sequenced so that div_mdot goes first
}

void
ContinuityResidualAlgDriver::execute()
{
  pre_work();
  div_mdot_algs_.execute();
  post_work();
}

namespace {

void
continuity_residual(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& sel,
  const Kokkos::Array<double, 3> gammas,
  const stk::mesh::NgpField<double>& rho_nm1,
  const stk::mesh::NgpField<double>& rho_np0,
  const stk::mesh::NgpField<double>& rho_np1,
  const stk::mesh::NgpField<double>& vol_nm1,
  const stk::mesh::NgpField<double>& vol_np0,
  const stk::mesh::NgpField<double>& vol_np1,
  const stk::mesh::NgpField<double>& div_mdot,
  stk::mesh::NgpField<double> continuity_residual)
{
  using Traits = kynema_ugf_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using MeshIndex = typename Traits::MeshIndex;

  kynema_ugf_ngp::run_entity_algorithm(
    "continuity_residual", mesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double proj = 1 / vol_np1(mi, 0);
      continuity_residual(mi, 0) =
        (gammas[0] * rho_np1(mi, 0) * vol_np1(mi, 0) +
         gammas[1] * rho_np0(mi, 0) * vol_np0(mi, 0) +
         gammas[2] * rho_nm1(mi, 0) * vol_nm1(mi, 0) +
         div_mdot(mi, 0) * vol_np1(mi, 0)) *
        proj;
    });
}

} // namespace
void
ContinuityResidualAlgDriver::post_work()
{
  const auto& meta = realm_.meta_data();

  auto* vol =
    meta.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");
  STK_ThrowRequire(vol);
  const int nvol_state = vol->number_of_states();

  const auto vol_nm1_state = nvol_state == 3    ? stk::mesh::StateNM1
                             : (nvol_state > 1) ? stk::mesh::StateN
                                                : stk::mesh::StateNone;

  const auto vol_np0_state =
    (nvol_state > 1) ? stk::mesh::StateN : stk::mesh::StateNone;

  auto* rho = meta.get_field<double>(stk::topology::NODE_RANK, "density");
  STK_ThrowRequire(rho);
  const int nrho_state = rho->number_of_states();
  STK_ThrowRequire(nrho_state > 1);
  const auto rho_nm1_state =
    nrho_state == 3 ? stk::mesh::StateNM1 : stk::mesh::StateN;

  const auto mesh = stk::mesh::get_updated_ngp_mesh(realm_.bulk_data());
  const auto& mesh_info = realm_.mesh_info();

  auto vol_nm1 = kynema_ugf_ngp::get_ngp_field(
    mesh_info, "dual_nodal_volume", vol_nm1_state);

  auto vol_np0 = kynema_ugf_ngp::get_ngp_field(
    mesh_info, "dual_nodal_volume", vol_np0_state);

  auto vol_np1 = kynema_ugf_ngp::get_ngp_field(
    mesh_info, "dual_nodal_volume", stk::mesh::StateNP1);

  auto rho_nm1 =
    kynema_ugf_ngp::get_ngp_field(mesh_info, "density", rho_nm1_state);
  auto rho_np0 =
    kynema_ugf_ngp::get_ngp_field(mesh_info, "density", stk::mesh::StateN);
  auto rho_np1 =
    kynema_ugf_ngp::get_ngp_field(mesh_info, "density", stk::mesh::StateNP1);

  auto div_mdot = kynema_ugf_ngp::get_ngp_field(mesh_info, "div_mdot");
  auto cont = kynema_ugf_ngp::get_ngp_field(mesh_info, "continuity_residual");

  cont.set_all(mesh, 0.0);

  const auto dt = realm_.get_time_step();

  Kokkos::Array<double, 3> gammas{
    realm_.get_gamma1() / dt, realm_.get_gamma2() / dt,
    realm_.get_gamma3() / dt};

  vol_nm1.sync_to_device();
  vol_np0.sync_to_device();
  vol_np1.sync_to_device();

  rho_nm1.sync_to_device();
  rho_np0.sync_to_device();
  rho_np1.sync_to_device();

  div_mdot.sync_to_device();

  const auto sel = (meta.locally_owned_part() | meta.globally_shared_part()) &
                   stk::mesh::selectField(*rho);
  continuity_residual(
    mesh, sel, gammas, rho_nm1, rho_np0, rho_np1, vol_nm1, vol_np0, vol_np1,
    div_mdot, cont);
  cont.modify_on_device();
}

} // namespace sierra::kynema_ugf
