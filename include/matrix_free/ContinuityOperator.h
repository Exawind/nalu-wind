// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONTINUITY_OPERATOR_H
#define CONTINUITY_OPERATOR_H

#include "matrix_free/KokkosViewTypes.h"

#include "Teuchos_BLAS_types.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class ContinuityResidualOperator
{
public:
  static constexpr int num_vectors = 1;
  using map_type = Tpetra::Map<>;
  using export_type = Tpetra::Export<>;
  using mv_type = Tpetra::MultiVector<>;

  ContinuityResidualOperator(
    const_elem_offset_view<p> elem_offsets_in, const export_type& exporter);

  void compute(mv_type& owned_rhs);

  void set_fields(double time_scale, const_scs_scalar_view<p> mdot)
  {
    time_scale_ = time_scale;
    mdot_ = mdot;
  }

private:
  const const_elem_offset_view<p> elem_offsets_;
  const export_type& exporter_;

  mutable mv_type cached_rhs_;

  double time_scale_;
  const_scs_scalar_view<p> mdot_;
};

template <int p>
class ContinuityLinearizedResidualOperator final : public Tpetra::Operator<>
{
public:
  static constexpr int num_vectors = 1;
  using mv_type = Tpetra::MultiVector<>;
  using const_mv_type = Tpetra::MultiVector<const double>;
  using map_type = Tpetra::Map<>;
  using base_operator_type = Tpetra::Operator<>;
  using export_type = Tpetra::Export<>;

  ContinuityLinearizedResidualOperator() = default;
  ContinuityLinearizedResidualOperator(
    const_elem_offset_view<p> elem_offsets_in, const export_type& exporter);

  void apply(
    const mv_type& ownedSolution,
    mv_type& ownedRHS,
    Teuchos::ETransp trans = Teuchos::NO_TRANS,
    double alpha = 1.0,
    double beta = 0.0) const final;

  Teuchos::RCP<const map_type> getDomainMap() const final
  {
    return exporter_.getTargetMap();
  }
  Teuchos::RCP<const map_type> getRangeMap() const final
  {
    return exporter_.getTargetMap();
  }

  void set_metric(const_scs_vector_view<p> metric) { metric_ = metric; }

private:
  const const_elem_offset_view<p> elem_offsets_;
  const export_type& exporter_;

  const_scs_vector_view<p> metric_;

  mutable mv_type cached_sln_;
  mutable mv_type cached_rhs_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
