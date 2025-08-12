// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/MatrixFreeSolver.h"

#ifdef NALU_USES_TRILINOS_SOLVERS
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosMultiVecTraits.hpp>
#include <BelosOperatorTraits.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosSolverManager.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosTypes.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <cmath>

#include "Teuchos_RCP.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

#include "stk_mesh/base/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <typename T>
void
set_parameter_if_not_set(
  Teuchos::ParameterList& list, std::string param_name, const T& value)
{
  if (!list.isParameter(param_name)) {
    list.set(param_name, value);
  }
}

Teuchos::ParameterList&
add_default_parameters_to_parameter_list(
  Teuchos::ParameterList& list, int num_vectors, bool verbose = true)
{
  set_parameter_if_not_set(list, "Num Blocks", 200);
  set_parameter_if_not_set(list, "Maximum Iterations", 200);
  set_parameter_if_not_set(list, "Convergence Tolerance", 1.0e-7);
  set_parameter_if_not_set(
    list, "Implicit Residual Scaling",
    "Norm of Preconditioned Initial Residual");
  std::string name = (num_vectors > 1) ? "bicgstab" : "gmres";
  set_parameter_if_not_set(list, "Solver Name", name);
  set_parameter_if_not_set(list, "Block Size", num_vectors);
  set_parameter_if_not_set(list, "Adaptive Block Size", false);
  set_parameter_if_not_set(list, "Deflation Quorum", num_vectors);
  set_parameter_if_not_set(list, "Orthogonalization", "ICGS");

  if (verbose) {
    list.set("Output Frequency", 1);
    list.set("Output Stream", Teuchos::rcpFromRef(std::cout));
    list.set("Verbosity", Belos::IterationDetails);
  }
  return list;
}

MatrixFreeSolver::MatrixFreeSolver(
  const base_op_type& op_in, int num_vectors_in, Teuchos::ParameterList params)
  : lhs_vector_(op_in.getDomainMap(), num_vectors_in),
    rhs_vector_(op_in.getRangeMap(), num_vectors_in),
    final_rhs_vector_(op_in.getRangeMap(), num_vectors_in),
    problem_(
      Teuchos::rcpFromRef(op_in),
      Teuchos::rcpFromRef(lhs_vector_),
      Teuchos::rcpFromRef(rhs_vector_)),
    solv_(
      Belos::TpetraSolverFactory<double, mv_type, base_op_type>().create(
        add_default_parameters_to_parameter_list(params, num_vectors_in)
          .get<std::string>("Solver Name"),
        Teuchos::rcpFromRef(
          add_default_parameters_to_parameter_list(params, num_vectors_in))))
{
  solv_->setProblem(Teuchos::rcpFromRef(problem_));
}

void
MatrixFreeSolver::set_preconditioner(const base_op_type& prec)
{
  stk::mesh::ProfilingBlock pf("MatrixFreeSolver::set_preconditioner");
  problem_.setRightPrec(Teuchos::rcpFromRef(prec));
}

namespace {

double
mv_norm2(const typename MatrixFreeSolver::mv_type& mv)
{
  stk::mesh::ProfilingBlock pf("mv_norm2");
  const int num_vectors(mv.getNumVectors());
  Teuchos::Array<double> mv_norm(num_vectors);
  mv.norm2(mv_norm());
  double norm = 0;
  for (int k = 0; k < num_vectors; ++k) {
    norm += mv_norm[k] * mv_norm[k];
  }
  return std::sqrt(norm);
}

double
normalized_mv_norm2(const typename MatrixFreeSolver::mv_type& mv)
{
  stk::mesh::ProfilingBlock pf("normalized_mv_norm2");
  return mv_norm2(mv) / std::sqrt(mv.getNumVectors() * mv.getGlobalLength());
}
} // namespace

void
MatrixFreeSolver::solve()
{
  stk::mesh::ProfilingBlock pf("MatrixFreeSolver::solve");
  lhs_vector_.putScalar(0.);
  problem_.setProblem();
  solv_->solve();
}

double
MatrixFreeSolver::final_linear_norm() const
{
  stk::mesh::ProfilingBlock pf("MatrixFreeSolver::final_linear_norm");
  problem_.getOperator()->apply(
    lhs_vector_, final_rhs_vector_, Teuchos::NO_TRANS, 1., 0.);
  final_rhs_vector_.update(1., rhs_vector_, -1.);
  return mv_norm2(final_rhs_vector_);
}

double
MatrixFreeSolver::nonlinear_residual() const
{
  stk::mesh::ProfilingBlock pf("MatrixFreeSolver::nonlinear_residual");
  return normalized_mv_norm2(rhs_vector_);
}

int
MatrixFreeSolver::num_iterations() const
{
  stk::mesh::ProfilingBlock pf("MatrixFreeSolver::num_iterations");
  return solv_->getNumIters();
}

typename MatrixFreeSolver::mv_type&
MatrixFreeSolver::lhs()
{
  return lhs_vector_;
}

typename MatrixFreeSolver::mv_type&
MatrixFreeSolver::rhs()
{
  return rhs_vector_;
}

const typename MatrixFreeSolver::mv_type&
MatrixFreeSolver::lhs() const
{
  return lhs_vector_;
}

const typename MatrixFreeSolver::mv_type&
MatrixFreeSolver::rhs() const
{
  return rhs_vector_;
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif // NALU_USES_TRILINOS_SOLVERS
