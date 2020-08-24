// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUM_OPERATOR_H
#define MOMENTUM_OPERATOR_H

#include "matrix_free/KokkosViewTypes.h"

#include "Kokkos_Array.hpp"
#include "Teuchos_BLAS_types.hpp"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export_decl.hpp"
#include "Tpetra_MultiVector_decl.hpp"
#include "Tpetra_Operator.hpp"

#include "LowMachFields.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class MomentumResidualOperator
{
public:
  static constexpr int num_vectors = 3;
  using map_type = Tpetra::Map<>;
  using export_type = Tpetra::Export<>;
  using mv_type = Tpetra::MultiVector<>;

  MomentumResidualOperator(
    const_elem_offset_view<p> elem_offsets_in, const export_type& exporter);

  void compute(mv_type& owned_rhs);

  void
  set_fields(Kokkos::Array<double, 3> gammas, LowMachResidualFields<p> fields)
  {
    gammas_ = gammas;
    fields_ = fields;
  }

private:
  const const_elem_offset_view<p> elem_offsets_;
  const export_type& exporter_;

  mutable mv_type cached_rhs_;

  Kokkos::Array<double, 3> gammas_;
  LowMachResidualFields<p> fields_;
};

template <int p>
class MomentumLinearizedResidualOperator final : public Tpetra::Operator<>
{
public:
  static constexpr int num_vectors = 3;
  using mv_type = Tpetra::MultiVector<>;
  using map_type = Tpetra::Map<>;
  using base_operator_type = Tpetra::Operator<>;
  using export_type = Tpetra::Export<>;

  MomentumLinearizedResidualOperator() = default;
  MomentumLinearizedResidualOperator(
    const_elem_offset_view<p> elem_offsets_in, const export_type& exporter);

  void apply(
    const mv_type& sln,
    mv_type& rhs,
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

  void set_fields(double gamma_0, LowMachLinearizedResidualFields<p> fields)
  {
    gamma_0_ = gamma_0;
    fields_ = fields;
  }

private:
  const const_elem_offset_view<p> elem_offsets_;
  const export_type& exporter_;

  double gamma_0_{-1};
  LowMachLinearizedResidualFields<p> fields_;

  mutable mv_type cached_sln_;
  mutable mv_type cached_rhs_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
