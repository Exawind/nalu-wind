#ifndef MATRIX_FREE_SOLVER_H
#define MATRIX_FREE_SOLVER_H

#include "BelosLinearProblem.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"
#include "matrix_free/ConductionOperator.h"

namespace Teuchos {
class ParameterList;
}

namespace Belos {
template <typename, typename, typename>
class SolverManager;
} // namespace Belos

namespace sierra {
namespace nalu {
namespace matrix_free {

class MatrixFreeSolver
{
public:
  using base_op_type = Tpetra::Operator<>;
  using mv_type = Tpetra::MultiVector<>;
  using map_type = Tpetra::Map<>;
  using problem_type =
    Belos::LinearProblem<typename mv_type::scalar_type, mv_type, base_op_type>;

  MatrixFreeSolver(
    const base_op_type& op_in,
    int num_vectors,
    Teuchos::ParameterList params = {});

  void set_preconditioner(const base_op_type&);
  void solve();
  mv_type& lhs();
  mv_type& rhs();
  const mv_type& lhs() const;
  const mv_type& rhs() const;

  double nonlinear_residual() const;
  double final_linear_norm() const;
  int num_iterations() const;

private:
  mv_type lhs_vector_;
  mv_type rhs_vector_;
  mutable mv_type final_rhs_vector_;
  problem_type problem_;
  Teuchos::RCP<Belos::SolverManager<double, mv_type, base_op_type>> solv_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
