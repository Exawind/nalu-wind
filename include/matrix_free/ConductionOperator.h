// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONDUCTION_OPERATOR_H
#define CONDUCTION_OPERATOR_H

#include "matrix_free/KokkosViewTypes.h"

#include "Kokkos_Array.hpp"
#include "Teuchos_BLAS_types.hpp"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"
#include "matrix_free/ConductionFields.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class ConductionResidualOperator
{
public:
  static constexpr int num_vectors = 1;
  using map_type = Tpetra::Map<>;
  using export_type = Tpetra::Export<>;
  using mv_type = Tpetra::MultiVector<>;

  ConductionResidualOperator(
    const_elem_offset_view<p> elem_offsets_in, const export_type& exporter);

  void compute(mv_type& owned_rhs);

  void set_gammas(Kokkos::Array<double, 3> gammas_in) { gammas_ = gammas_in; }
  void set_residual_fields(InteriorResidualFields<p> residual_fields_in)
  {
    residual_fields_ = residual_fields_in;
  }
  void set_fields(
    Kokkos::Array<double, 3> gammas_in,
    InteriorResidualFields<p> residual_fields_in)
  {
    gammas_ = gammas_in;
    residual_fields_ = residual_fields_in;
  }

  void set_bc_fields(
    const_node_offset_view dirichlet_offsets_in,
    node_scalar_view solution_q,
    node_scalar_view specified_q)
  {
    dirichlet_bc_active_ = dirichlet_offsets_in.extent_int(0) > 0;
    dirichlet_bc_offsets_ = dirichlet_offsets_in;
    bc_nodal_solution_field_ = solution_q;
    bc_nodal_specified_field_ = specified_q;
  }

  void set_flux_fields(
    const_face_offset_view<p> face_offsets_in,
    face_vector_view<p> areas_in,
    face_scalar_view<p> flux_in)
  {
    flux_bc_active_ = face_offsets_in.extent_int(0) > 0;
    flux_bc_offsets_ = face_offsets_in;
    exposed_areas_ = areas_in;
    flux_ = flux_in;
  }

private:
  const const_elem_offset_view<p> elem_offsets_;
  const export_type& exporter_;

  mutable mv_type cached_shared_rhs_;
  Kokkos::Array<double, 3> gammas_;
  InteriorResidualFields<p> residual_fields_;

  bool dirichlet_bc_active_{false};
  const_node_offset_view dirichlet_bc_offsets_;
  const_node_scalar_view bc_nodal_solution_field_;
  const_node_scalar_view bc_nodal_specified_field_;

  bool flux_bc_active_{false};
  const_face_offset_view<p> flux_bc_offsets_;
  const_face_vector_view<p> exposed_areas_;
  const_face_scalar_view<p> flux_;
};

template <int p>
class ConductionLinearizedResidualOperator final : public Tpetra::Operator<>
{
public:
  static constexpr int num_vectors = 1;
  using mv_type = Tpetra::MultiVector<>;
  using const_mv_type = Tpetra::MultiVector<const double>;
  using map_type = Tpetra::Map<>;
  using base_operator_type = Tpetra::Operator<>;
  using export_type = Tpetra::Export<>;

  ConductionLinearizedResidualOperator() = default;
  ConductionLinearizedResidualOperator(
    const_elem_offset_view<p> elem_offsets_in, const export_type& exporter);

  void apply(
    const mv_type& ownedSolution,
    mv_type& ownedRHS,
    Teuchos::ETransp trans = Teuchos::NO_TRANS,
    double alpha = 1.0,
    double beta = 0.0) const final;

  void set_coefficients(double gamma_in, LinearizedResidualFields<p> fields_in)
  {
    gamma_ = gamma_in;
    fields_ = fields_in;
  }

  void set_dirichlet_nodes(const_node_offset_view dirichlet_offsets)
  {
    dirichlet_bc_active_ = dirichlet_offsets.extent_int(0) > 0;
    dirichlet_bc_offsets_ = dirichlet_offsets;
  }

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

  bool dirichlet_bc_active_{false};
  const_node_offset_view dirichlet_bc_offsets_;

  LinearizedResidualFields<p> fields_;
  double gamma_{+1};

  mutable mv_type cached_sln_;
  mutable mv_type cached_rhs_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
