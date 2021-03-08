// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/MomentumSolutionUpdate.h"

#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/LowMachFields.h"
#include "matrix_free/MomentumOperator.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/KokkosViewTypes.h"

#include "stk_mesh/base/NgpProfilingBlock.hpp"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
MomentumSolutionUpdate<p>::MomentumSolutionUpdate(
  Teuchos::ParameterList params,
  const StkToTpetraMaps& linsys,
  const Tpetra::Export<>& exporter,
  const_elem_offset_view<p> offsets,
  const_node_offset_view dirichlet_bc_offsets)
  : linsys_(linsys),
    exporter_(exporter),
    offsets_(offsets),
    dirichlet_bc_offsets_(dirichlet_bc_offsets),
    resid_op_(offsets, exporter_),
    lin_op_(offsets, exporter_),
    prec_op_(offsets, exporter_),
    linear_solver_(lin_op_, num_vectors, params),
    owned_and_shared_mv_(exporter_.getSourceMap(), num_vectors)
{
}

template <int p>
void
MomentumSolutionUpdate<p>::compute_preconditioner(
  double gamma, LowMachLinearizedResidualFields<p> fields)
{
  stk::mesh::ProfilingBlock pf(
    "MomentumSolutionUpdate<p>::compute_preconditioner");

  linear_solver_.set_preconditioner(prec_op_);
  prec_op_.set_linear_operator(Teuchos::rcpFromRef(lin_op_));
  prec_op_.set_dirichlet_nodes(dirichlet_bc_offsets_);
  prec_op_.compute_diagonal(
    gamma, fields.volume_metric, fields.advection_metric,
    fields.diffusion_metric);
}

template <int p>
const Tpetra::MultiVector<double>&
MomentumSolutionUpdate<p>::compute_residual(
  Kokkos::Array<double, 3> gammas,
  LowMachResidualFields<p> fields,
  LowMachBCFields<p> bc)
{
  stk::mesh::ProfilingBlock pf("MomentumSolutionUpdate<p>::compute_residual");
  resid_op_.set_fields(gammas, fields);
  resid_op_.set_bc_fields(dirichlet_bc_offsets_, bc);
  linear_solver_.rhs().putScalar(0.);
  resid_op_.compute(linear_solver_.rhs());
  return linear_solver_.rhs();
}

template <int p>
const Tpetra::MultiVector<double>&
MomentumSolutionUpdate<p>::compute_delta(
  double gamma, LowMachLinearizedResidualFields<p> coeffs)
{
  stk::mesh::ProfilingBlock pf("MomentumSolutionUpdate<p>::compute_delta");
  lin_op_.set_dirichlet_nodes(dirichlet_bc_offsets_);
  lin_op_.set_fields(gamma, coeffs);
  linear_solver_.solve();
  if (exporter_.getTargetMap()->isDistributed()) {
    stk::mesh::ProfilingBlock pfinner(
      "import solution from owned to owned and shared");

    owned_and_shared_mv_.doImport(
      linear_solver_.lhs(), exporter_, Tpetra::INSERT);
    return owned_and_shared_mv_;
  }
  return linear_solver_.lhs();
}

template <int p>
double
MomentumSolutionUpdate<p>::residual_norm() const
{
  return linear_solver_.nonlinear_residual();
}
template <int p>
int
MomentumSolutionUpdate<p>::num_iterations() const
{
  return linear_solver_.num_iterations();
}
template <int p>
double
matrix_free::MomentumSolutionUpdate<p>::final_linear_norm() const
{
  return linear_solver_.final_linear_norm();
}

INSTANTIATE_POLYCLASS(MomentumSolutionUpdate);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
