// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LowMachUpdate.h"

#include "matrix_free/MaxCourantReynolds.h"
#include "matrix_free/LowMachInfo.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "stk_mesh/base/GetNgpMesh.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <typename T = double>
stk::mesh::NgpField<T>&
get_ngp_field(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState state = stk::mesh::StateNP1)
{
  STK_ThrowAssert(meta.get_field(stk::topology::NODE_RANK, name));
  STK_ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
  return stk::mesh::get_updated_ngp_field<T>(
    *meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
}

template <int p>
Kokkos::Array<double, 2>
LowMachPostProcessP<p>::compute_local_courant_reynolds_numbers(double dt) const
{
  auto fields = gather_.get_residual_fields();
  return max_local_courant_reynolds<p>(
    dt, fields.xc, fields.rho, fields.mu, fields.up1);
}

template <int p>
LowMachUpdate<p>::LowMachUpdate(
  stk::mesh::BulkData& bulk_in,
  Teuchos::ParameterList params_mom,
  Teuchos::ParameterList params_cont,
  Teuchos::ParameterList params_grad,
  stk::mesh::Selector active_in,
  stk::mesh::Selector replicas_in,
  Kokkos::View<gid_type*> rgids)
  : bulk_(bulk_in),
    active_(active_in),
    linsys_(
      stk::mesh::get_updated_ngp_mesh(bulk_in),
      active_in,
      linsys_info::get_gid_field(bulk_in.mesh_meta_data()),
      replicas_in,
      rgids),
    exporter_(
      Teuchos::rcpFromRef(linsys_.owned_and_shared),
      Teuchos::rcpFromRef(linsys_.owned)),
    offsets_(
      create_offset_map<p>(
        stk::mesh::get_updated_ngp_mesh(bulk_in),
        active_in,
        linsys_.stk_lid_to_tpetra_lid)),
    field_gather_(bulk_in, active_in),
    post_process_(field_gather_),
    momentum_update_(params_mom, linsys_, exporter_, offsets_),
    continuity_update_(params_cont, linsys_, exporter_, offsets_),
    gradient_update_(params_grad, linsys_, exporter_, offsets_, {})
{
}

template <int p>
LowMachUpdate<p>::LowMachUpdate(
  stk::mesh::BulkData& bulk_in,
  Teuchos::ParameterList params_mom,
  Teuchos::ParameterList params_cont,
  Teuchos::ParameterList params_grad,
  stk::mesh::Selector active_in,
  stk::mesh::Selector dirichlet_in,
  const Tpetra::Map<>& owned,
  const Tpetra::Map<>& owned_and_shared,
  Kokkos::View<const lid_type*> elids)
  : bulk_(bulk_in),
    active_(active_in),
    dirichlet_(dirichlet_in),
    linsys_(owned, owned_and_shared, elids),
    exporter_(
      Teuchos::rcpFromRef(linsys_.owned_and_shared),
      Teuchos::rcpFromRef(linsys_.owned)),
    offsets_(
      create_offset_map<p>(
        stk::mesh::get_updated_ngp_mesh(bulk_in),
        active_in,
        linsys_.stk_lid_to_tpetra_lid)),
    exposed_face_offsets_(
      face_offsets<p>(
        stk::mesh::get_updated_ngp_mesh(bulk_in),
        dirichlet_in,
        linsys_.stk_lid_to_tpetra_lid)),
    dirichlet_offsets_(simd_node_offsets(
      stk::mesh::get_updated_ngp_mesh(bulk_in),
      dirichlet_in,
      linsys_.stk_lid_to_tpetra_lid)),
    field_gather_(bulk_in, active_in, dirichlet_in),
    post_process_(field_gather_),
    momentum_update_(
      params_mom, linsys_, exporter_, offsets_, dirichlet_offsets_),
    continuity_update_(params_cont, linsys_, exporter_, offsets_),
    gradient_update_(
      params_grad, linsys_, exporter_, offsets_, exposed_face_offsets_)
{
}

template <int p>
void
LowMachUpdate<p>::compute_momentum_preconditioner(double gamma)
{
  momentum_update_.compute_preconditioner(
    gamma, field_gather_.get_coefficient_fields());
}

template <int p>
void
LowMachUpdate<p>::compute_gradient_preconditioner()
{
  gradient_update_.compute_preconditioner(
    field_gather_.get_coefficient_fields().unscaled_volume_metric);
}

template <int p>
void
LowMachUpdate<p>::initialize()
{
  field_gather_.gather_all();
}

template <int p>
void
LowMachUpdate<p>::swap_states()
{
  field_gather_.swap_states();
}

template <int p>
void
LowMachUpdate<p>::predict_state()
{
  stk::mesh::ProfilingBlock pf("predict_state");

  const auto& meta = bulk_.mesh_meta_data();

  auto predicted_state =
    get_ngp_field(meta, lowmach_info::velocity_name, stk::mesh::StateNP1);

  auto current_state =
    get_ngp_field(meta, lowmach_info::velocity_name, stk::mesh::StateN);
  current_state.sync_to_device();

  stk::mesh::for_each_entity_run(
    stk::mesh::get_updated_ngp_mesh(bulk_), stk::topology::NODE_RANK, active_,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      for (int d = 0; d < 3; ++d) {
        predicted_state.get(mi, d) = current_state.get(mi, d);
      };
    });
  predicted_state.modify_on_device();

  initial_residual_[VEL] = -1;
  initial_residual_[CONT] = -1;
  initial_residual_[GP] = -1;
}

template <int p>
void
LowMachUpdate<p>::update_gathered_fields()
{
  field_gather_.update_fields();
}

template <int p>
void
LowMachUpdate<p>::update_advection_metric(double dt)
{
  field_gather_.update_mdot(dt);
}

template <int p>
void
LowMachUpdate<p>::gather_velocity()
{
  field_gather_.update_velocity();
}

template <int p>
void
LowMachUpdate<p>::gather_pressure()
{
  field_gather_.update_pressure();
}

template <int p>
void
LowMachUpdate<p>::gather_grad_p()
{
  field_gather_.update_grad_p();
}

template <int p>
void
LowMachUpdate<p>::update_transport_coefficients(GradTurbModel model)
{
  field_gather_.update_transport_coefficients(model);
}

template <int p>
void
LowMachUpdate<p>::update_provisional_velocity(
  Kokkos::Array<double, 3> gammas, stk::mesh::NgpField<double>& field)
{
  stk::mesh::ProfilingBlock pf("update provisional velocity");

  momentum_update_.compute_residual(
    gammas, field_gather_.get_residual_fields(), field_gather_.get_bc_fields());
  auto& delta_mv = momentum_update_.compute_delta(
    gammas[0], field_gather_.get_coefficient_fields());

  add_tpetra_solution_vector_to_stk_field(
    stk::mesh::get_updated_ngp_mesh(bulk_), active_,
    linsys_.stk_lid_to_tpetra_lid,
    delta_mv.getLocalViewDevice(Tpetra::Access::ReadOnly), field);
}

template <int p>
void
LowMachUpdate<p>::update_pressure(
  double time_scale, stk::mesh::NgpField<double>& field)
{
  stk::mesh::ProfilingBlock pf("update pressure");

  update_advection_metric(time_scale);

  continuity_update_.compute_residual(
    time_scale, field_gather_.get_residual_fields().advection_metric);
  const auto& delta_mv = continuity_update_.compute_delta(
    field_gather_.get_coefficient_fields().laplacian_metric);

  add_tpetra_solution_vector_to_stk_field(
    stk::mesh::get_updated_ngp_mesh(bulk_), active_,
    linsys_.stk_lid_to_tpetra_lid,
    delta_mv.getLocalViewDevice(Tpetra::Access::ReadOnly), field);
}

template <int p>
void
LowMachUpdate<p>::update_pressure_gradient(stk::mesh::NgpField<double>& field)
{
  stk::mesh::ProfilingBlock pf("update pressure gradient");

  GradientResidualFields<p> fields;
  {
    auto lmfields = field_gather_.get_residual_fields();
    fields.q = lmfields.pressure;
    fields.dqdx = lmfields.gp;
    fields.vols = lmfields.unscaled_volume_metric;
    fields.areas = lmfields.area_metric;
  }

  BCGradientFields<p> bc;
  {
    auto lmbc = field_gather_.get_bc_fields();
    bc.face_q = lmbc.exposed_pressure;
    bc.exposed_areas = lmbc.exposed_areas;
  }
  gradient_update_.compute_residual(fields, bc);
  auto& delta_mv = gradient_update_.compute_delta(fields.vols);
  add_tpetra_solution_vector_to_stk_field(
    stk::mesh::get_updated_ngp_mesh(bulk_), active_,
    linsys_.stk_lid_to_tpetra_lid,
    delta_mv.getLocalViewDevice(Tpetra::Access::ReadOnly), field);
}

namespace {
template <typename SolType>
void
banner(
  const SolType& sol,
  std::string name,
  std::ostream& stream,
  double& resid,
  double& init_resid)
{
  stk::mesh::ProfilingBlock pf("banner");

  resid = sol.residual_norm();
  if (init_resid < 0) {
    init_resid = resid;
  }
  const auto scaled_residual_norm =
    resid / std::max(std::numeric_limits<double>::epsilon(), init_resid);

  const int name_offset = name.length() + 8;
  stream << std::setw(name_offset) << std::right << name
         << std::setw(32 - name_offset) << std::right << sol.num_iterations()
         << std::setw(18) << std::right << sol.final_linear_norm()
         << std::setw(15) << std::right << resid << std::setw(14) << std::right
         << scaled_residual_norm << std::endl;
}
} // namespace

template <int p>
void
LowMachUpdate<p>::velocity_banner(std::string name, std::ostream& stream) const
{
  banner(
    momentum_update_, name, stream, residual_norm_[VEL],
    initial_residual_[VEL]);
}

template <int p>
void
LowMachUpdate<p>::pressure_banner(std::string name, std::ostream& stream) const
{
  banner(
    continuity_update_, name, stream, residual_norm_[CONT],
    initial_residual_[CONT]);
}

template <int p>
void
LowMachUpdate<p>::grad_p_banner(std::string name, std::ostream& stream) const
{
  banner(
    gradient_update_, name, stream, residual_norm_[GP], initial_residual_[GP]);
}

template <int p>
void
LowMachUpdate<p>::project_velocity(
  double proj_time_scale,
  stk::mesh::NgpField<double> rho,
  stk::mesh::NgpField<double> gp,
  stk::mesh::NgpField<double> gp_star,
  stk::mesh::NgpField<double>& u)
{
  constexpr int dim = 3;
  stk::mesh::ProfilingBlock pf("project_velocity");
  rho.sync_to_device();
  gp_star.sync_to_device();
  gp.sync_to_device();
  stk::mesh::for_each_entity_run(
    stk::mesh::get_updated_ngp_mesh(bulk_), stk::topology::NODE_RANK,
    active_ - dirichlet_, KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      const auto fac = proj_time_scale / rho(mi, 0);
      for (int d = 0; d < dim; ++d) {
        u.get(mi, d) -= fac * (gp_star(mi, d) - gp(mi, d));
      };
    });
  u.modify_on_device();
}

void
copy_stk_field_to_owned_tpetra_vector(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& sel,
  Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*> elid,
  const stk::mesh::NgpField<double>& field,
  tpetra_view_type delta_view)
{
  stk::mesh::ProfilingBlock pf("copy_stk_field_to_owned_tpetra_vector");

  const int dim = delta_view.extent_int(1);
  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      const auto ent = mesh.get_entity(stk::topology::NODE_RANK, mi);
      const auto tpetra_lid = elid(ent.local_offset());
      if (tpetra_lid < delta_view.extent_int(0)) {
        for (int d = 0; d < dim; ++d) {
          delta_view(tpetra_lid, d) = field.get(mi, d);
        }
      }
    });
}

template <int p>
void
LowMachUpdate<p>::create_continuity_preconditioner(
  const stk::mesh::NgpField<double>& coords,
  Tpetra::CrsMatrix<>& mat,
  std::string xmlname)
{
  stk::mesh::ProfilingBlock pf("create_continuity_preconditioner");

  auto coord_mv = Teuchos::rcp(
    new Tpetra::MultiVector<>(Teuchos::rcpFromRef(linsys_.owned), 3));

  copy_stk_field_to_owned_tpetra_vector(
    stk::mesh::get_updated_ngp_mesh(bulk_), active_,
    linsys_.stk_lid_to_tpetra_lid, coords,
    coord_mv->getLocalViewDevice(Tpetra::Access::ReadWrite));

  muelu_params.set("xml parameter file", xmlname);
  muelu_params.sublist("user data").set("Coordinates", coord_mv);
  continuity_update_.compute_preconditioner(mat, muelu_params);
}

INSTANTIATE_POLYCLASS(LowMachUpdate);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
