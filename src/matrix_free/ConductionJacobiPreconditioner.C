// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionJacobiPreconditioner.h"

#include "matrix_free/ConductionDiagonal.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StrongDirichletBC.h"

#include <KokkosInterface.h>

#include <Teuchos_RCP.hpp>
#include <Tpetra_CombineMode.hpp>
#include "Tpetra_Operator.hpp"
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace {

using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;
using const_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const;

void
reciprocal(tpetra_view_type x)
{
  Kokkos::parallel_for(
    "invert", DeviceRangePolicy(0, x.extent_int(0)),
    KOKKOS_LAMBDA(int k) { x(k, 0) = 1 / x(k, 0); });
}
} // namespace
template <int p>
JacobiOperator<p>::JacobiOperator(
  const_elem_offset_view<p> elem_offsets_in,
  const export_type& exporter_in,
  int num_sweeps_in)
  : elem_offsets_(elem_offsets_in),
    exporter_(exporter_in),
    num_sweeps_(num_sweeps_in),
    owned_diagonal_(exporter_in.getTargetMap(), num_vectors),
    owned_and_shared_diagonal_(exporter_in.getSourceMap(), num_vectors),
    cached_mv_(exporter_in.getTargetMap(), num_vectors)
{
}

template <int p>
void
JacobiOperator<p>::set_linear_operator(
  Teuchos::RCP<const Tpetra::Operator<>> op_in)
{
  op_ = op_in;
}

namespace {
void
element_multiply(
  const_tpetra_view_type inv_diag, const_tpetra_view_type b, tpetra_view_type y)
{
  Kokkos::parallel_for(
    "element_multiply", DeviceRangePolicy(0, b.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      y(index, 0) = inv_diag(index, 0) * b(index, 0);
    });
}

void
update_jacobi_sweep(
  const_tpetra_view_type inv_diag,
  const_tpetra_view_type axprev,
  const_tpetra_view_type b,
  tpetra_view_type y)
{
  Kokkos::parallel_for(
    "jacobi_sweep", DeviceRangePolicy(0, inv_diag.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      y(index, 0) += inv_diag(index, 0) * (b(index, 0) - axprev(index, 0));
    });
}
} // namespace
template <int p>
void
JacobiOperator<p>::apply(
  const mv_type& x, mv_type& y, Teuchos::ETransp, double, double) const
{
  element_multiply(
    owned_diagonal_.getLocalViewDevice(Tpetra::Access::ReadOnly),
    x.getLocalViewDevice(Tpetra::Access::ReadOnly),
    y.getLocalViewDevice(Tpetra::Access::ReadWrite));
  for (int n = 1; n < num_sweeps_; ++n) {
    op_->apply(y, cached_mv_);
    update_jacobi_sweep(
      owned_diagonal_.getLocalViewDevice(Tpetra::Access::ReadOnly),
      cached_mv_.getLocalViewDevice(Tpetra::Access::ReadOnly),
      x.getLocalViewDevice(Tpetra::Access::ReadOnly),
      y.getLocalViewDevice(Tpetra::Access::ReadWrite));
  }
}

template <int p>
void
JacobiOperator<p>::compute_diagonal()
{
  owned_and_shared_diagonal_.putScalar(0.);
  conduction_diagonal<p>(
    gamma_, elem_offsets_, fields_.volume_metric, fields_.diffusion_metric,
    owned_and_shared_diagonal_.getLocalViewDevice(Tpetra::Access::ReadWrite));

  if (dirichlet_bc_active_) {
    dirichlet_diagonal(
      dirichlet_bc_offsets_, owned_diagonal_.getLocalLength(),
      owned_and_shared_diagonal_.getLocalViewDevice(Tpetra::Access::ReadWrite));
  }
  owned_diagonal_.putScalar(0.);
  owned_diagonal_.doExport(owned_and_shared_diagonal_, exporter_, Tpetra::ADD);
  reciprocal(owned_diagonal_.getLocalViewDevice(Tpetra::Access::ReadWrite));
}
INSTANTIATE_POLYCLASS(JacobiOperator);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
