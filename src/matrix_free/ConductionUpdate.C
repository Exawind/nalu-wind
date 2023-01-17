// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionUpdate.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionInfo.h"
#include "matrix_free/PolynomialOrders.h"

#include <KokkosInterface.h>
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <ostream>

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/NgpForEachEntity.hpp"
#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"

#include <stk_topology/topology.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
ConductionUpdate<p>::ConductionUpdate(
  stk::mesh::BulkData& bulk_in,
  Teuchos::ParameterList params,
  stk::mesh::Selector active_in,
  stk::mesh::Selector dirichlet_in,
  stk::mesh::Selector flux_in,
  stk::mesh::Selector replicas_in,
  Kokkos::View<gid_type*> rgids)
  : bulk_(bulk_in),
    meta_(bulk_in.mesh_meta_data()),
    active_(active_in),
    linsys_(
      stk::mesh::get_updated_ngp_mesh(bulk_),
      active_,
      linsys_info::get_gid_field(meta_),
      replicas_in,
      rgids),
    exporter_(
      Teuchos::rcpFromRef(linsys_.owned_and_shared),
      Teuchos::rcpFromRef(linsys_.owned)),
    offset_views_(
      stk::mesh::get_updated_ngp_mesh(bulk_in),
      linsys_.stk_lid_to_tpetra_lid,
      active_in,
      dirichlet_in,
      flux_in),
    field_update_(params, linsys_, exporter_, offset_views_),
    field_gather_(bulk_in, active_in, dirichlet_in, flux_in)
{
}

template <int p>
void
ConductionUpdate<p>::initialize()
{
  stk::mesh::ProfilingBlock pf("ConductionUpdate<p>::initialize");
  field_gather_.gather_all();
}

template <int p>
void
ConductionUpdate<p>::swap_states()
{
  stk::mesh::ProfilingBlock pf("ConductionUpdate<p>::swap_states");
  field_gather_.swap_states();
  initial_residual_ = -1;
}

template <int p>
void
ConductionUpdate<p>::compute_preconditioner(double projected_dt)
{
  stk::mesh::ProfilingBlock pf("ConductionUpdate<p>::compute_preconditioner");
  field_update_.compute_preconditioner(
    projected_dt, field_gather_.get_coefficient_fields());
}

namespace {

void
copy_state(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_,
  stk::mesh::NgpField<double> dst,
  stk::mesh::NgpField<double> src)
{
  stk::mesh::ProfilingBlock pf("ConductionUpdate<p>:::copy_state");
  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, active_,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      dst.get(mi, 0) = src.get(mi, 0);
    });
  dst.modify_on_device();
}
} // namespace

template <int p>
void
ConductionUpdate<p>::predict_state()
{
  stk::mesh::ProfilingBlock pf("ConductionUpdate<p>::predict_state");

  auto qp1 = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field(stk::topology::NODE_RANK, conduction_info::q_name)
       ->field_state(stk::mesh::StateNP1));

  auto qp0 = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field(stk::topology::NODE_RANK, conduction_info::q_name)
       ->field_state(stk::mesh::StateN));
  copy_state(stk::mesh::get_updated_ngp_mesh(bulk_), active_, qp1, qp0);
  field_gather_.update_solution_fields();
  initial_residual_ = -1;
}

template <int p>
void
ConductionUpdate<p>::compute_update(
  Kokkos::Array<double, 3> gammas, stk::mesh::NgpField<double>& delta)
{
  stk::mesh::ProfilingBlock pf("ConductionUpdate<p>::compute_update");
  field_update_.compute_residual(
    gammas, field_gather_.get_residual_fields(), field_gather_.get_bc_fields(),
    field_gather_.get_flux_fields());

  const auto& delta_mv = field_update_.compute_delta(
    gammas[0], field_gather_.get_coefficient_fields());

  add_tpetra_solution_vector_to_stk_field(
    stk::mesh::get_updated_ngp_mesh(bulk_), active_,
    linsys_.stk_lid_to_tpetra_lid,
    delta_mv.getLocalViewDevice(Tpetra::Access::ReadOnly), delta);

  residual_norm_ = field_update_.residual_norm();
  if (initial_residual_ < 0) {
    initial_residual_ = residual_norm_;
  }
  scaled_residual_norm_ =
    residual_norm_ /
    std::max(std::numeric_limits<double>::epsilon(), initial_residual_);
}

template <int p>
void
ConductionUpdate<p>::update_solution_fields()
{
  stk::mesh::ProfilingBlock pf("ConductionUpdate<p>::update_solution_fields");
  field_gather_.update_solution_fields();
}

template <int p>
void
ConductionUpdate<p>::banner(std::string name, std::ostream& stream) const
{
  stk::mesh::ProfilingBlock pf("ConductionUpdate<p>::banner");
  const int nameOffset = name.length() + 8;
  stream << std::setw(nameOffset) << std::right << name
         << std::setw(32 - nameOffset) << std::right
         << field_update_.num_iterations() << std::setw(18) << std::right
         << field_update_.final_linear_norm() << std::setw(15) << std::right
         << residual_norm_ << std::setw(14) << std::right
         << scaled_residual_norm_ << std::endl;
}
INSTANTIATE_POLYCLASS(ConductionUpdate);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
