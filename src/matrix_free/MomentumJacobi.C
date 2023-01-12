#include "matrix_free/MomentumJacobi.h"

#include "Teuchos_RCP.hpp"

#include "Tpetra_Operator.hpp"
#include "matrix_free/MomentumDiagonal.h"
#include "matrix_free/StrongDirichletBC.h"

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"
#include <KokkosInterface.h>

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
  // be brave
  Kokkos::parallel_for(
    "invert", DeviceRangePolicy(0, x.extent_int(0)),
    KOKKOS_LAMBDA(int k) { x(k, 0) = 1 / x(k, 0); });
}
} // namespace
template <int p>
MomentumJacobiOperator<p>::MomentumJacobiOperator(
  const_elem_offset_view<p> elem_offsets_in,
  const export_type& exporter_in,
  int num_sweeps_in)
  : elem_offsets(elem_offsets_in),
    exporter(exporter_in),
    num_sweeps(num_sweeps_in),
    owned_diagonal(exporter.getTargetMap(), num_vectors),
    owned_and_shared_diagonal(exporter.getSourceMap(), num_vectors),
    cached_mv(exporter.getTargetMap(), num_vectors)
{
}

template <int p>
void
MomentumJacobiOperator<p>::set_linear_operator(
  Teuchos::RCP<const Tpetra::Operator<>> op_in)
{
  op = op_in;
}

namespace {
void
element_multiply(
  const_tpetra_view_type inv_diag, const_tpetra_view_type b, tpetra_view_type y)
{
  constexpr int dim = MomentumJacobiOperator<inst::P1>::num_vectors;

  Kokkos::parallel_for(
    "element_multiply", DeviceRangePolicy(0, b.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      const auto inv_d = inv_diag(index, 0);
      for (int d = 0; d < dim; ++d) {
        y(index, d) = inv_d * b(index, d);
      }
    });
}

void
update_jacobi_sweep(
  const_tpetra_view_type inv_diag,
  const_tpetra_view_type axprev,
  const_tpetra_view_type b,
  tpetra_view_type y)
{
  constexpr int dim = MomentumJacobiOperator<inst::P1>::num_vectors;
  Kokkos::parallel_for(
    "jacobi_sweep", DeviceRangePolicy(0, inv_diag.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      const auto inv_d = inv_diag(index, 0);
      for (int d = 0; d < dim; ++d) {
        y(index, d) += inv_d * (b(index, d) - axprev(index, d));
      }
    });
}
} // namespace
template <int p>
void
MomentumJacobiOperator<p>::apply(
  const mv_type& x, mv_type& y, Teuchos::ETransp, double, double) const
{
  element_multiply(
    owned_diagonal.getLocalViewDevice(Tpetra::Access::ReadOnly),
    x.getLocalViewDevice(Tpetra::Access::ReadOnly),
    y.getLocalViewDevice(Tpetra::Access::ReadWrite));
  for (int n = 1; n < num_sweeps; ++n) {
    op->apply(y, cached_mv);
    update_jacobi_sweep(
      owned_diagonal.getLocalViewDevice(Tpetra::Access::ReadOnly),
      cached_mv.getLocalViewDevice(Tpetra::Access::ReadOnly),
      x.getLocalViewDevice(Tpetra::Access::ReadOnly),
      y.getLocalViewDevice(Tpetra::Access::ReadWrite));
  }
}

template <int p>
void
MomentumJacobiOperator<p>::compute_diagonal(
  double gamma,
  const_scalar_view<p> vol,
  const_scs_scalar_view<p> adv,
  const_scs_vector_view<p> diff)
{
  owned_and_shared_diagonal.putScalar(0.);
  advdiff_diagonal<p>(
    gamma, elem_offsets, vol, adv, diff,
    owned_and_shared_diagonal.getLocalViewDevice(Tpetra::Access::ReadWrite));

  if (dirichlet_bc_active_) {
    dirichlet_diagonal(
      dirichlet_bc_offsets_, owned_diagonal.getLocalLength(),
      owned_and_shared_diagonal.getLocalViewDevice(Tpetra::Access::ReadWrite));
  }
  owned_diagonal.putScalar(0.);
  owned_diagonal.doExport(owned_and_shared_diagonal, exporter, Tpetra::ADD);
  reciprocal(owned_diagonal.getLocalViewDevice(Tpetra::Access::ReadWrite));
}
INSTANTIATE_POLYCLASS(MomentumJacobiOperator);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
