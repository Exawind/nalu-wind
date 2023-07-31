// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionOperator.h"

#include "matrix_free/ConductionInterior.h"
#include "matrix_free/StrongDirichletBC.h"
#include "matrix_free/ScalarFluxBC.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"

#include "stk_mesh/base/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
ConductionResidualOperator<p>::ConductionResidualOperator(
  const_elem_offset_view<p> elem_offsets_in, const export_type& exporter_in)
  : elem_offsets_(elem_offsets_in),
    exporter_(exporter_in),
    cached_shared_rhs_(exporter_in.getSourceMap(), num_vectors)
{
}

template <int p>
void
ConductionResidualOperator<p>::compute(mv_type& owned_rhs)
{
  stk::mesh::ProfilingBlock pf("ConductionResidualOperator<p>::apply");
  if (exporter_.getTargetMap()->isDistributed()) {

    cached_shared_rhs_.putScalar(0.);
    conduction_residual<p>(
      gammas_, elem_offsets_, residual_fields_.qm1, residual_fields_.qp0,
      residual_fields_.qp1, residual_fields_.volume_metric,
      residual_fields_.diffusion_metric,
      cached_shared_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));

    if (flux_bc_active_) {
      scalar_neumann_residual<p>(
        flux_bc_offsets_, flux_, exposed_areas_,
        cached_shared_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));
    }

    if (dirichlet_bc_active_) {
      dirichlet_residual(
        dirichlet_bc_offsets_, bc_nodal_solution_field_,
        bc_nodal_specified_field_, owned_rhs.getLocalLength(),
        cached_shared_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));
    }

    owned_rhs.putScalar(0.);
    owned_rhs.doExport(cached_shared_rhs_, exporter_, Tpetra::ADD);
  } else {
    owned_rhs.putScalar(0.);
    conduction_residual<p>(
      gammas_, elem_offsets_, residual_fields_.qm1, residual_fields_.qp0,
      residual_fields_.qp1, residual_fields_.volume_metric,
      residual_fields_.diffusion_metric,
      owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));

    if (flux_bc_active_) {
      scalar_neumann_residual<p>(
        flux_bc_offsets_, flux_, exposed_areas_,
        owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
    }

    if (dirichlet_bc_active_) {
      dirichlet_residual(
        dirichlet_bc_offsets_, bc_nodal_solution_field_,
        bc_nodal_specified_field_, owned_rhs.getLocalLength(),
        owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
    }
  }
}
INSTANTIATE_POLYCLASS(ConductionResidualOperator);

template <int p>
ConductionLinearizedResidualOperator<p>::ConductionLinearizedResidualOperator(
  const_elem_offset_view<p> elem_offsets_in, const export_type& exporter_in)
  : elem_offsets_(elem_offsets_in),
    exporter_(exporter_in),
    cached_sln_(exporter_in.getSourceMap(), num_vectors),
    cached_rhs_(exporter_in.getSourceMap(), num_vectors)
{
}

template <int p>
void
ConductionLinearizedResidualOperator<p>::apply(
  const mv_type& owned_sln,
  mv_type& owned_rhs,
  Teuchos::ETransp trans,
  double alpha,
  double beta) const
{
  stk::mesh::ProfilingBlock pf("LinearizedResidualOperator<p>::apply");
  STK_ThrowRequire(trans == Teuchos::NO_TRANS);
  STK_ThrowRequire(alpha == 1.0);
  STK_ThrowRequire(beta == 0.0);
  if (exporter_.getTargetMap()->isDistributed()) {
    cached_sln_.doImport(owned_sln, exporter_, Tpetra::INSERT);
    cached_rhs_.putScalar(0.);

    conduction_linearized_residual<p>(
      gamma_, elem_offsets_, fields_.volume_metric, fields_.diffusion_metric,
      cached_sln_.getLocalViewDevice(Tpetra::Access::ReadWrite),
      cached_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));

    if (dirichlet_bc_active_) {
      dirichlet_linearized(
        dirichlet_bc_offsets_, owned_rhs.getLocalLength(),
        cached_sln_.getLocalViewDevice(Tpetra::Access::ReadWrite),
        cached_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));
    }

    owned_rhs.putScalar(0.);
    owned_rhs.doExport(cached_rhs_, exporter_, Tpetra::ADD);
  } else {
    owned_rhs.putScalar(0.);
    conduction_linearized_residual<p>(
      gamma_, elem_offsets_, fields_.volume_metric, fields_.diffusion_metric,
      owned_sln.getLocalViewDevice(Tpetra::Access::ReadOnly),
      owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));

    if (dirichlet_bc_active_) {
      dirichlet_linearized(
        dirichlet_bc_offsets_, owned_rhs.getLocalLength(),
        owned_sln.getLocalViewDevice(Tpetra::Access::ReadOnly),
        owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
    }
  }
}
INSTANTIATE_POLYCLASS(ConductionLinearizedResidualOperator);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
