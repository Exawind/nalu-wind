// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/MomentumOperator.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/MomentumInterior.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StrongDirichletBC.h"

#include "Tpetra_Operator.hpp"
#include "stk_mesh/base/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
MomentumResidualOperator<p>::MomentumResidualOperator(
  const_elem_offset_view<p> elem_offsets_in, const export_type& exporter_in)
  : elem_offsets_(elem_offsets_in),
    exporter_(exporter_in),
    max_owned_row_id_(exporter_in.getTargetMap()->getLocalNumElements()),
    cached_rhs_(exporter_in.getSourceMap(), num_vectors)
{
}

template <int p>
void
MomentumResidualOperator<p>::local_compute(tpetra_view_type rhs) const
{
  stk::mesh::ProfilingBlock pf("local rhs");
  {
    stk::mesh::ProfilingBlock pfinner("zero rhs");
    Kokkos::deep_copy(exec_space(), rhs, 0.);
  }
  {
    stk::mesh::ProfilingBlock pfinner("interior residual");
    momentum_residual<p>(
      gammas_, elem_offsets_, fields_.xc, fields_.rho, fields_.mu, fields_.vm1,
      fields_.vp0, fields_.volume_metric, fields_.um1, fields_.up0, fields_.up1,
      fields_.gp, fields_.force, fields_.advection_metric, rhs);
  }
  if (dirichlet_bc_active_) {
    stk::mesh::ProfilingBlock pfinner("dirichlet residual");
    dirichlet_residual(
      dirichlet_bc_offsets_, bc_.up1, bc_.ubc, max_owned_row_id_, rhs);
  }
}

template <int p>
void
MomentumResidualOperator<p>::compute(mv_type& owned_rhs)
{
  stk::mesh::ProfilingBlock pf("MomentumResidualOperator<p>::apply");
  if (exporter_.getTargetMap()->isDistributed()) {
    STK_ThrowRequire(owned_rhs.getLocalLength() == size_t(max_owned_row_id_));
    local_compute(cached_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));
    {
      stk::mesh::ProfilingBlock pfinner("export from owned-shared to owned");
      owned_rhs.putScalar(0.);
      owned_rhs.doExport(cached_rhs_, exporter_, Tpetra::ADD);
    }
  } else {
    local_compute(owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
  }
}
INSTANTIATE_POLYCLASS(MomentumResidualOperator);

template <int p>
MomentumLinearizedResidualOperator<p>::MomentumLinearizedResidualOperator(
  const_elem_offset_view<p> elem_offsets_in, const export_type& exporter_in)
  : elem_offsets_(elem_offsets_in),
    exporter_(exporter_in),
    max_owned_row_id_(exporter_in.getTargetMap()->getLocalNumElements()),
    cached_sln_(exporter_in.getSourceMap(), num_vectors),
    cached_rhs_(exporter_in.getSourceMap(), num_vectors)
{
}

template <int p>
void
MomentumLinearizedResidualOperator<p>::local_apply(
  ra_tpetra_view_type xin, tpetra_view_type yout) const
{
  stk::mesh::ProfilingBlock pf("local apply");
  {
    stk::mesh::ProfilingBlock pfinner("zero rhs");
    Kokkos::deep_copy(exec_space(), yout, 0.);
  }

  {
    stk::mesh::ProfilingBlock pfinner("interior apply");
    momentum_linearized_residual<p>(
      gamma_0_, elem_offsets_, fields_.volume_metric, fields_.advection_metric,
      fields_.diffusion_metric, xin, yout);
  }

  if (dirichlet_bc_active_) {
    stk::mesh::ProfilingBlock pfinner("dirichlet apply");
    dirichlet_linearized(dirichlet_bc_offsets_, max_owned_row_id_, xin, yout);
  }
}

template <int p>
void
MomentumLinearizedResidualOperator<p>::apply(
  const mv_type& owned_sln,
  mv_type& owned_rhs,
  Teuchos::ETransp trans,
  double alpha,
  double beta) const
{
  stk::mesh::ProfilingBlock pf("MomentumLinearizedResidualOperator<p>::apply");
  STK_ThrowRequire(trans == Teuchos::NO_TRANS);
  STK_ThrowRequire(alpha == 1.0);
  STK_ThrowRequire(beta == 0.0);
  if (exporter_.getTargetMap()->isDistributed()) {
    {
      stk::mesh::ProfilingBlock pfinner("import into owned-shared from owned");
      cached_sln_.doImport(owned_sln, exporter_, Tpetra::INSERT);
    }

    STK_ThrowRequire(owned_rhs.getLocalLength() == size_t(max_owned_row_id_));
    local_apply(
      cached_sln_.getLocalViewDevice(Tpetra::Access::ReadWrite),
      cached_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));

    {
      stk::mesh::ProfilingBlock pfinner("export from owned-shared to owned");
      owned_rhs.putScalar(0.);
      owned_rhs.doExport(cached_rhs_, exporter_, Tpetra::ADD);
    }
  } else {
    stk::mesh::ProfilingBlock pfinner("local apply");
    local_apply(
      owned_sln.getLocalViewDevice(Tpetra::Access::ReadOnly),
      owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
  }
}
INSTANTIATE_POLYCLASS(MomentumLinearizedResidualOperator);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
