// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONDUCTION_JACOBI_PRECONDITIONER_H
#define CONDUCTION_JACOBI_PRECONDITIONER_H

#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

#include "matrix_free/ConductionFields.h"
#include "matrix_free/KokkosViewTypes.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class JacobiOperator final : public Tpetra::Operator<>
{
public:
  static constexpr int num_vectors = 1;
  using mv_type = Tpetra::MultiVector<>;
  using map_type = Tpetra::Map<>;
  using base_operator_type = Tpetra::Operator<>;
  using export_type = Tpetra::Export<>;

  JacobiOperator(
    const_elem_offset_view<p> elem_offsets_in,
    const export_type& exporter,
    int num_sweeps = 1);

  void apply(
    const mv_type& ownedSolution,
    mv_type& ownedRHS,
    Teuchos::ETransp trans = Teuchos::NO_TRANS,
    double alpha = 1.0,
    double beta = 0.0) const final;

  void set_dirichlet_nodes(const_node_offset_view dirichlet_offsets_in)
  {
    dirichlet_bc_active_ = dirichlet_offsets_in.extent_int(0) > 0;
    dirichlet_bc_offsets_ = dirichlet_offsets_in;
  }

  void set_coefficients(double gamma_in, LinearizedResidualFields<p> fields_in)
  {
    gamma_ = gamma_in;
    fields_ = fields_in;
  }

  void compute_diagonal();
  mv_type& get_inverse_diagonal() { return owned_diagonal_; }
  void set_linear_operator(Teuchos::RCP<const Tpetra::Operator<>>);

  Teuchos::RCP<const map_type> getDomainMap() const final
  {
    return exporter_.getTargetMap();
  }
  Teuchos::RCP<const map_type> getRangeMap() const final
  {
    return exporter_.getTargetMap();
  }

private:
  const const_elem_offset_view<p> elem_offsets_;
  const export_type& exporter_;
  const int num_sweeps_;
  mv_type owned_diagonal_;
  mv_type owned_and_shared_diagonal_;
  mutable mv_type cached_mv_;

  bool dirichlet_bc_active_{false};
  const_node_offset_view dirichlet_bc_offsets_;
  LinearizedResidualFields<p> fields_;
  double gamma_{+1};

  Teuchos::RCP<const base_operator_type> op_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
