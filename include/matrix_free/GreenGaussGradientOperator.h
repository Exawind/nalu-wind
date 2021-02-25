// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef GREEN_GAUSS_GRADIENT_OPERATOR_H
#define GREEN_GAUSS_GRADIENT_OPERATOR_H

#include "matrix_free/KokkosViewTypes.h"

#include "Teuchos_BLAS_types.hpp"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
struct GradientResidualFields
{
  scalar_view<p> q;
  vector_view<p> dqdx;
  scalar_view<p> vols;
  scs_vector_view<p> areas;
};

template <int p>
struct BCGradientFields
{
  face_scalar_view<p> face_q;
  face_vector_view<p> exposed_areas;
};

template <int p>
class GradientResidualOperator
{
public:
  using ra_tpetra_view_type =
    typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread;

  static constexpr int num_vectors = 3;
  using map_type = Tpetra::Map<>;
  using export_type = Tpetra::Export<>;
  using mv_type = Tpetra::MultiVector<>;

  GradientResidualOperator(
    const_elem_offset_view<p> elem_offsets_in, const export_type& exporter);

  void compute(mv_type& owned_rhs);

  void set_fields(GradientResidualFields<p> residual_fields_in)
  {
    residual_fields_ = residual_fields_in;
  }

  void set_bc_fields(
    const_face_offset_view<p> face_offsets_in,
    face_vector_view<p> areas_in,
    face_scalar_view<p> face_q)
  {
    face_bc_active_ = face_offsets_in.extent_int(0) > 0;
    face_bc_offsets_ = face_offsets_in;
    exposed_areas_ = areas_in;
    face_q_ = face_q;
  }

private:
  const const_elem_offset_view<p> elem_offsets_;
  const export_type& exporter_;

  mutable mv_type cached_rhs_;
  GradientResidualFields<p> residual_fields_;

  bool face_bc_active_{false};
  const_face_offset_view<p> face_bc_offsets_;
  const_face_vector_view<p> exposed_areas_;
  const_face_scalar_view<p> face_q_;
};

template <int p>
class GradientLinearizedResidualOperator final : public Tpetra::Operator<>
{
public:
  using ra_tpetra_view_type =
    typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread;

  static constexpr int num_vectors = 3;
  using mv_type = Tpetra::MultiVector<>;
  using const_mv_type = Tpetra::MultiVector<const double>;
  using map_type = Tpetra::Map<>;
  using base_operator_type = Tpetra::Operator<>;
  using export_type = Tpetra::Export<>;

  GradientLinearizedResidualOperator(
    const_elem_offset_view<p> elem_offsets_in, const export_type& exporter);

  void apply(
    const mv_type& ownedSolution,
    mv_type& ownedRHS,
    Teuchos::ETransp trans = Teuchos::NO_TRANS,
    double alpha = 1.0,
    double beta = 0.0) const final;

  void set_volumes(const_scalar_view<p> volumes) { volumes_ = volumes; }
  void set_inv_diag(ra_tpetra_view_type inv_diag) { inv_diag_ = inv_diag; }

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

  const_scalar_view<p> volumes_;
  ra_tpetra_view_type inv_diag_;

  mutable mv_type cached_sln_;
  mutable mv_type cached_rhs_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
