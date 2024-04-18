// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ContinuityOperator.h"

#include "matrix_free/ContinuityInterior.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/PolynomialOrders.h"

#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "Teuchos_BLAS_types.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_CombineMode.hpp"
#include "stk_util/util/ReportHandler.hpp"

#include <KokkosBlas1_sum.hpp>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
ContinuityResidualOperator<p>::ContinuityResidualOperator(
  const_elem_offset_view<p> elem_offsets_in, const export_type& exporter_in)
  : elem_offsets_(elem_offsets_in),
    exporter_(exporter_in),
    cached_rhs_(exporter_in.getSourceMap(), num_vectors)
{
}

namespace {

void
remove_constant(Kokkos::View<double*> x, double value)
{
  Kokkos::parallel_for(
    x.extent(0), KOKKOS_LAMBDA(int n) { x(n) -= value; });
}

void
remove_constant(Tpetra::MultiVector<>& vector)
{
  // orthogonalize wrt (1,1,..,1)^T

  auto avg = KokkosBlas::sum(Kokkos::subview(
    vector.getLocalViewDevice(Tpetra::Access::ReadOnly), Kokkos::ALL, 0));

  auto comm = Teuchos::getRawMpiComm(*vector.getMap()->getComm());
  MPI_Allreduce(MPI_IN_PLACE, &avg, 1, MPI_DOUBLE, MPI_SUM, comm);
  avg /= vector.getGlobalLength();

  remove_constant(
    Kokkos::subview(
      vector.getLocalViewDevice(Tpetra::Access::ReadWrite), Kokkos::ALL, 0),
    avg);
}
} // namespace

template <int p>
void
ContinuityResidualOperator<p>::compute(mv_type& owned_rhs)
{
  stk::mesh::ProfilingBlock pf("ContinuityResidualOperator<p>::apply");
  if (exporter_.getTargetMap()->isDistributed()) {
    cached_rhs_.putScalar(0.);
    continuity_residual<p>(
      time_scale_, elem_offsets_, mdot_,
      cached_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));

    owned_rhs.putScalar(0.);
    {
      stk::mesh::ProfilingBlock pfinner("export from owned-shared to owned");
      owned_rhs.doExport(cached_rhs_, exporter_, Tpetra::ADD);
    }
  } else {
    owned_rhs.putScalar(0.);
    continuity_residual<p>(
      time_scale_, elem_offsets_, mdot_,
      owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
  }
  remove_constant(owned_rhs);
}
INSTANTIATE_POLYCLASS(ContinuityResidualOperator);

template <int p>
ContinuityLinearizedResidualOperator<p>::ContinuityLinearizedResidualOperator(
  const_elem_offset_view<p> elem_offsets_in, const export_type& exporter_in)
  : elem_offsets_(elem_offsets_in),
    exporter_(exporter_in),
    cached_sln_(exporter_in.getSourceMap(), num_vectors),
    cached_rhs_(exporter_in.getSourceMap(), num_vectors)
{
}
template <int p>
void
ContinuityLinearizedResidualOperator<p>::apply(
  const mv_type& owned_sln,
  mv_type& owned_rhs,
  Teuchos::ETransp trans,
  double alpha,
  double beta) const
{
  stk::mesh::ProfilingBlock pf(
    "ContinuityLinearizedResidualOperator<p>::apply");
  STK_ThrowRequire(trans == Teuchos::NO_TRANS);
  STK_ThrowRequire(alpha == 1.0);
  STK_ThrowRequire(beta == 0.0);

  if (exporter_.getTargetMap()->isDistributed()) {
    {
      stk::mesh::ProfilingBlock pfinner("import into owned-shared from owned");
      cached_sln_.doImport(owned_sln, exporter_, Tpetra::INSERT);
    }
    cached_rhs_.putScalar(0.);
    continuity_linearized_residual<p>(
      elem_offsets_, metric_,
      cached_sln_.getLocalViewDevice(Tpetra::Access::ReadWrite),
      cached_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));
    {
      stk::mesh::ProfilingBlock pfinner("export from owned-shared to owned");
      owned_rhs.doExport(cached_rhs_, exporter_, Tpetra::ADD);
    }
  } else {
    owned_rhs.putScalar(0.);
    continuity_linearized_residual<p>(
      elem_offsets_, metric_,
      owned_sln.getLocalViewDevice(Tpetra::Access::ReadOnly),
      owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
  }
  remove_constant(owned_rhs);
}
INSTANTIATE_POLYCLASS(ContinuityLinearizedResidualOperator);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
