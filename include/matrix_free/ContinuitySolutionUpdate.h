// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONTINUITY_SOLUTION_UPDATE_H
#define CONTINUITY_SOLUTION_UPDATE_H

#include "matrix_free/ContinuityOperator.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/MatrixFreeSolver.h"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_CrsMatrix_fwd.hpp"

#include <iosfwd>

namespace Teuchos {
class ParameterList;
}

namespace sierra {
namespace nalu {
namespace matrix_free {

struct StkToTpetraMaps;

template <int p>
class ContinuitySolutionUpdate
{
public:
  static constexpr int num_vectors = 1;
  ContinuitySolutionUpdate(
    Teuchos::ParameterList params,
    const StkToTpetraMaps& linsys,
    const Tpetra::Export<>& exporter,
    const_elem_offset_view<p> offset);

  void compute_residual(double, const_scs_scalar_view<p> mdot);

  const Tpetra::MultiVector<double>&
  compute_delta(const_scs_vector_view<p> laplacian_metric);

  void compute_preconditioner(
    Tpetra::CrsMatrix<>& mat, Teuchos::ParameterList& params);

  const MatrixFreeSolver& solver() const { return linear_solver_; }
  double residual_norm() const;
  double final_linear_norm() const;
  int num_iterations() const;

private:
  const StkToTpetraMaps& linsys_;
  const Tpetra::Export<>& exporter_;
  const const_elem_offset_view<p> offsets_;

  ContinuityResidualOperator<p> resid_op_;
  ContinuityLinearizedResidualOperator<p> lin_op_;
  Teuchos::RCP<Tpetra::Operator<>> prec_op_;

  MatrixFreeSolver linear_solver_;
  mutable Tpetra::MultiVector<> owned_and_shared_mv_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
