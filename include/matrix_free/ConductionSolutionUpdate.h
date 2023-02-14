// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONDUCTION_SOLUTION_UPDATE_H
#define CONDUCTION_SOLUTION_UPDATE_H

#include "matrix_free/ConductionJacobiPreconditioner.h"
#include "matrix_free/ConductionOperator.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/MatrixFreeSolver.h"

#include "Kokkos_Core.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Selector.hpp"

namespace Teuchos {
class ParameterList;
}

namespace sierra {
namespace nalu {
namespace matrix_free {

struct BCDirichletFields;
template <int p>
struct BCFluxFields;
template <int p>
struct InteriorResidualFields;
template <int p>
struct LinearizedResidualFields;

struct StkToTpetraMaps;

template <int p>
struct ConductionOffsetViews
{
  ConductionOffsetViews(
    const stk::mesh::NgpMesh& mesh,
    Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*> elid,
    const stk::mesh::Selector& active,
    stk::mesh::Selector dirichlet = {},
    stk::mesh::Selector flux = {});

  const const_elem_offset_view<p> offsets;
  const const_node_offset_view dirichlet_bc_offsets;
  const const_face_offset_view<p> flux_bc_offsets;
};

template <int p>
struct ConductionSolutionUpdate
{
public:
  static constexpr int num_vectors = 1;
  ConductionSolutionUpdate(
    Teuchos::ParameterList params,
    const StkToTpetraMaps& linsys,
    const Tpetra::Export<>& exporter,
    const ConductionOffsetViews<p>& offset_views);

  void compute_residual(
    Kokkos::Array<double, 3>,
    InteriorResidualFields<p>,
    BCDirichletFields = {},
    BCFluxFields<p> = {});

  const Tpetra::MultiVector<>&
  compute_delta(double gamma, LinearizedResidualFields<p>);

  const MatrixFreeSolver& solver() const { return linear_solver_; }
  void compute_preconditioner(double gamma, LinearizedResidualFields<p>);

  double residual_norm() const;
  double final_linear_norm() const;
  int num_iterations() const;

private:
  const StkToTpetraMaps& linsys_;
  const Tpetra::Export<>& exporter_;
  const ConductionOffsetViews<p>& offset_views_;

  ConductionResidualOperator<p> resid_op_;
  ConductionLinearizedResidualOperator<p> lin_op_;
  JacobiOperator<p> prec_op_;
  MatrixFreeSolver linear_solver_;
  mutable Tpetra::MultiVector<> owned_and_shared_mv_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
