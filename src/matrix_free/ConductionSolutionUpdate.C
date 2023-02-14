// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionSolutionUpdate.h"

#include "matrix_free/ConductionFields.h"
#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"

#include <KokkosInterface.h>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Tpetra_CombineMode.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Selector.hpp"

#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
ConductionOffsetViews<p>::ConductionOffsetViews(
  const stk::mesh::NgpMesh& mesh,
  Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*> elids,
  const stk::mesh::Selector& active,
  stk::mesh::Selector dirichlet,
  stk::mesh::Selector flux)
  : offsets(create_offset_map<p>(mesh, active, elids)),
    dirichlet_bc_offsets(simd_node_offsets(mesh, dirichlet, elids)),
    flux_bc_offsets(face_offsets<p>(mesh, flux, elids))
{
}
INSTANTIATE_POLYSTRUCT(ConductionOffsetViews);

template <int p>
ConductionSolutionUpdate<p>::ConductionSolutionUpdate(
  Teuchos::ParameterList params,
  const StkToTpetraMaps& linsys,
  const Tpetra::Export<>& exporter,
  const ConductionOffsetViews<p>& offset_views)
  : linsys_(linsys),
    exporter_(exporter),
    offset_views_(offset_views),
    resid_op_(offset_views.offsets, exporter),
    lin_op_(offset_views.offsets, exporter_),
    prec_op_(
      offset_views.offsets,
      exporter_,
      params.isParameter("Number of Sweeps")
        ? params.get<int>("Number of Sweeps")
        : 1),
    linear_solver_(lin_op_, num_vectors, params),
    owned_and_shared_mv_(exporter_.getSourceMap(), num_vectors)
{
}

template <int p>
void
ConductionSolutionUpdate<p>::compute_preconditioner(
  double gamma, LinearizedResidualFields<p> coeffs)
{
  stk::mesh::ProfilingBlock pf(
    "ConductionSolutionUpdate<p>::compute_preconditioner");
  linear_solver_.set_preconditioner(prec_op_);
  prec_op_.set_dirichlet_nodes(offset_views_.dirichlet_bc_offsets);
  prec_op_.set_coefficients(gamma, coeffs);
  prec_op_.set_linear_operator(Teuchos::rcpFromRef(lin_op_));
  prec_op_.compute_diagonal();
}

template <int p>
void
ConductionSolutionUpdate<p>::compute_residual(
  Kokkos::Array<double, 3> gammas,
  InteriorResidualFields<p> fields,
  BCDirichletFields dirichlet_bc_fields,
  BCFluxFields<p> flux_bc_fields)
{
  stk::mesh::ProfilingBlock pf("ConductionSolutionUpdate<p>::compute_residual");
  resid_op_.set_fields(gammas, fields);
  resid_op_.set_bc_fields(
    offset_views_.dirichlet_bc_offsets, dirichlet_bc_fields.qp1,
    dirichlet_bc_fields.qbc);
  resid_op_.set_flux_fields(
    offset_views_.flux_bc_offsets, flux_bc_fields.exposed_areas,
    flux_bc_fields.flux);
  resid_op_.compute(linear_solver_.rhs());
}

template <int p>
const Tpetra::MultiVector<>&
ConductionSolutionUpdate<p>::compute_delta(
  double gamma, LinearizedResidualFields<p> coeffs)
{
  stk::mesh::ProfilingBlock pf("ConductionSolutionUpdate<p>::compute_delta");
  lin_op_.set_dirichlet_nodes(offset_views_.dirichlet_bc_offsets);
  lin_op_.set_coefficients(gamma, coeffs);
  linear_solver_.solve();
  if (exporter_.getTargetMap()->isDistributed()) {
    owned_and_shared_mv_.doImport(
      linear_solver_.lhs(), exporter_, Tpetra::INSERT);
    return owned_and_shared_mv_;
  }
  return linear_solver_.lhs();
}

template <int p>
double
ConductionSolutionUpdate<p>::residual_norm() const
{
  return linear_solver_.nonlinear_residual();
}
template <int p>
int
ConductionSolutionUpdate<p>::num_iterations() const
{
  return linear_solver_.num_iterations();
}
template <int p>
double
matrix_free::ConductionSolutionUpdate<p>::final_linear_norm() const
{
  return linear_solver_.final_linear_norm();
}

INSTANTIATE_POLYSTRUCT(ConductionSolutionUpdate);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
