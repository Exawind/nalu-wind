// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ContinuitySolutionUpdate.h"

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/PolynomialOrders.h"

#include "stk_mesh/base/NgpProfilingBlock.hpp"

#include "MueLu_CreateTpetraPreconditioner.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include <exception>
#include <string>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
ContinuitySolutionUpdate<p>::ContinuitySolutionUpdate(
  Teuchos::ParameterList params,
  const StkToTpetraMaps& linsys,
  const Tpetra::Export<>& exporter,
  const_elem_offset_view<p> offsets)
  : linsys_(linsys),
    exporter_(exporter),
    offsets_(offsets),
    resid_op_(offsets, exporter_),
    lin_op_(offsets, exporter_),
    linear_solver_(lin_op_, num_vectors, params),
    owned_and_shared_mv_(exporter_.getSourceMap(), num_vectors)
{
}

template <int p>
void
ContinuitySolutionUpdate<p>::compute_residual(
  double proj_time_scale, const_scs_scalar_view<p> mdot)
{
  stk::mesh::ProfilingBlock pf("ContinuitySolutionUpdate<p>::compute_residual");
  resid_op_.set_fields(proj_time_scale, mdot);
  resid_op_.compute(linear_solver_.rhs());
}

template <int p>
const Tpetra::MultiVector<double>&
ContinuitySolutionUpdate<p>::compute_delta(const_scs_vector_view<p> metric)
{
  stk::mesh::ProfilingBlock pf("ContinuitySolutionUpdate<p>::compute_delta");
  lin_op_.set_metric(metric);
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
void
ContinuitySolutionUpdate<p>::compute_preconditioner(
  Tpetra::CrsMatrix<>& mat, Teuchos::ParameterList& param)
{
  stk::mesh::ProfilingBlock pf(
    "ContinuitySolutionUpdate<p>::compute_preconditioner");
  Teuchos::RCP<Tpetra::Operator<>> op = Teuchos::rcpFromRef(mat);
  prec_op_ = MueLu::CreateTpetraPreconditioner(op, param);
  linear_solver_.set_preconditioner(*prec_op_);
}

template <int p>
double
ContinuitySolutionUpdate<p>::residual_norm() const
{
  return linear_solver_.nonlinear_residual();
}
template <int p>
int
ContinuitySolutionUpdate<p>::num_iterations() const
{
  return linear_solver_.num_iterations();
}
template <int p>
double
matrix_free::ContinuitySolutionUpdate<p>::final_linear_norm() const
{
  return linear_solver_.final_linear_norm();
}

INSTANTIATE_POLYCLASS(ContinuitySolutionUpdate);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
