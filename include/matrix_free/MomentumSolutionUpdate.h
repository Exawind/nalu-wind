// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUM_SOLUTION_UPDATE_H
#define MOMENTUM_SOLUTION_UPDATE_H

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/MomentumJacobi.h"
#include "matrix_free/MomentumOperator.h"

#include "Kokkos_Array.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Teuchos_RCP.hpp"

namespace Teuchos {
class ParameterList;
}

namespace sierra {
namespace nalu {
namespace matrix_free {

struct StkToTpetraMaps;

template <int p>
struct LowMachLinearizedResidualFields;

template <int p>
struct LowMachResidualFields;

template <int p>
struct LowMachBCFields;

template <int p>
class MomentumSolutionUpdate
{
public:
  static constexpr int num_vectors = 3;
  MomentumSolutionUpdate(
    Teuchos::ParameterList,
    const StkToTpetraMaps&,
    const Tpetra::Export<>&,
    const_elem_offset_view<p>,
    const_node_offset_view = {});

  const Tpetra::MultiVector<double>& compute_residual(
    Kokkos::Array<double, 3>,
    LowMachResidualFields<p>,
    LowMachBCFields<p>);

  const Tpetra::MultiVector<double>&
  compute_delta(double, LowMachLinearizedResidualFields<p>);
  void compute_preconditioner(double, LowMachLinearizedResidualFields<p>);
  const MatrixFreeSolver& solver() const { return linear_solver_; }

  double residual_norm() const;
  double final_linear_norm() const;
  int num_iterations() const;

private:
  const StkToTpetraMaps& linsys_;
  const Tpetra::Export<>& exporter_;
  const const_elem_offset_view<p> offsets_;
  const const_node_offset_view dirichlet_bc_offsets_;

  MomentumResidualOperator<p> resid_op_;
  MomentumLinearizedResidualOperator<p> lin_op_;
  MomentumJacobiOperator<p> prec_op_;

  MatrixFreeSolver linear_solver_;
  mutable Tpetra::MultiVector<> owned_and_shared_mv_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
