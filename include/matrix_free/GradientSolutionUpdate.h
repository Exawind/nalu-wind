// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef GRADIENT_SOLUTION_UPDATE_H
#define GRADIENT_SOLUTION_UPDATE_H

#include "matrix_free/FilterJacobi.h"
#include "matrix_free/GreenGaussGradientOperator.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/MatrixFreeSolver.h"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export_fwd.hpp"
#include "Tpetra_MultiVector.hpp"

namespace Teuchos {
class ParameterList;
}

namespace sierra {
namespace nalu {
namespace matrix_free {

struct StkToTpetraMaps;

template <int p>
class GradientSolutionUpdate
{
public:
  GradientSolutionUpdate(
    Teuchos::ParameterList params,
    const StkToTpetraMaps& linsys,
    const Tpetra::Export<>& exporter,
    const_elem_offset_view<p> offsets,
    const_face_offset_view<p> bc_faces);

  void compute_preconditioner(const_scalar_view<p> vols);
  void
  compute_residual(GradientResidualFields<p> fields, BCGradientFields<p> bc);
  const Tpetra::MultiVector<double>&
  compute_delta(const_scalar_view<p> volumes);
  double residual_norm() const;
  double final_linear_norm() const;
  int num_iterations() const;
  const MatrixFreeSolver& solver() const { return linear_solver_; }

private:
  const StkToTpetraMaps& linsys_;
  const Tpetra::Export<>& exporter_;
  const const_elem_offset_view<p> offsets_;
  const const_face_offset_view<p> bc_faces_;

  GradientResidualOperator<p> resid_op_;
  GradientLinearizedResidualOperator<p> lin_op_;
  FilterJacobiOperator<p> prec_op_;

  MatrixFreeSolver linear_solver_;
  mutable Tpetra::MultiVector<double> owned_and_shared_mv_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif