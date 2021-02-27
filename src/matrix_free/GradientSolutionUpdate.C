// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/GradientSolutionUpdate.h"
#include "matrix_free/GreenGaussGradientOperator.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/PolynomialOrders.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_CombineMode.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_mesh/base/NgpProfilingBlock.hpp"

#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

struct StkToTpetraMaps;

template <int p>
GradientSolutionUpdate<p>::GradientSolutionUpdate(
  Teuchos::ParameterList params,
  const StkToTpetraMaps& linsys,
  const Tpetra::Export<>& exporter,
  const_elem_offset_view<p> offsets,
  const_face_offset_view<p> bc_faces)
  : linsys_(linsys),
    exporter_(exporter),
    offsets_(offsets),
    bc_faces_(bc_faces),
    resid_op_(offsets, exporter),
    lin_op_(offsets, exporter),
    prec_op_(offsets, exporter, 1),
    linear_solver_(lin_op_, 3, params),
    owned_and_shared_mv_(exporter.getSourceMap(), 3)
{
}

template <int p>
void
GradientSolutionUpdate<p>::compute_preconditioner(const_scalar_view<p> vols)
{
  stk::mesh::ProfilingBlock pf(
    "GradientSolutionUpdate<p>::compute_preconditioner");

  prec_op_.compute_diagonal(vols);
  prec_op_.set_linear_operator(Teuchos::rcpFromRef(lin_op_));
  linear_solver_.set_preconditioner(prec_op_);
}

template <int p>
void
GradientSolutionUpdate<p>::compute_residual(
  GradientResidualFields<p> fields, BCGradientFields<p> bc)
{
  stk::mesh::ProfilingBlock pf("GradientSolutionUpdate<p>::compute_residual");
  resid_op_.set_fields(fields);
  resid_op_.set_bc_fields(bc_faces_, bc.exposed_areas, bc.face_q);
  resid_op_.compute(linear_solver_.rhs());
}

template <int p>
const Tpetra::MultiVector<double>&
GradientSolutionUpdate<p>::compute_delta(const_scalar_view<p> vols)
{
  stk::mesh::ProfilingBlock pf("GradientSolutionUpdate<p>::compute_delta");

  lin_op_.set_volumes(vols);
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
GradientSolutionUpdate<p>::residual_norm() const
{
  return linear_solver_.nonlinear_residual();
}
template <int p>
int
GradientSolutionUpdate<p>::num_iterations() const
{
  return linear_solver_.num_iterations();
}
template <int p>
double
GradientSolutionUpdate<p>::final_linear_norm() const
{
  return linear_solver_.final_linear_norm();
}

INSTANTIATE_POLYCLASS(GradientSolutionUpdate);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
