// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionSolutionUpdate.h"
#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionJacobiPreconditioner.h"
#include "matrix_free/ConductionOperator.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkEntityToRowMap.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/Coefficients.h"

#include <Kokkos_Parallel.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <mpi.h>
#include "stk_mesh/base/NgpProfilingBlock.hpp"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

#include "mpi.h"

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
  const stk::mesh::NgpMesh& mesh_in,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gid,
  stk::mesh::Selector active_in,
  stk::mesh::Selector dirichlet_in,
  stk::mesh::Selector flux_in,
  stk::mesh::Selector replicas_in)
  : linsys_(mesh_in, active_in, gid, replicas_in),
    offset_views_(
      mesh_in, linsys_.stk_lid_to_tpetra_lid, active_in, dirichlet_in, flux_in),
    exporter_(
      Teuchos::rcpFromRef(linsys_.owned_and_shared),
      Teuchos::rcpFromRef(linsys_.owned)),
    resid_op_(offset_views_.offsets, exporter_),
    lin_op_(offset_views_.offsets, exporter_),
    jacobi_preconditioner_(
      offset_views_.offsets,
      exporter_,
      params.isParameter("Number of Sweeps")
        ? params.get<int>("Number of Sweeps")
        : 1),
    linear_solver_(lin_op_, 1, params),
    owned_and_shared_mv_(exporter_.getSourceMap(), 1)
{
}

template <int p>
void
ConductionSolutionUpdate<p>::compute_preconditioner(
  double gamma, LinearizedResidualFields<p> coeffs)
{
  stk::mesh::ProfilingBlock pf(
    "ConductionSolutionUpdate<p>::compute_preconditioner");
  linear_solver_.set_preconditioner(jacobi_preconditioner_);
  jacobi_preconditioner_.set_dirichlet_nodes(
    offset_views_.dirichlet_bc_offsets);
  jacobi_preconditioner_.set_coefficients(gamma, coeffs);
  jacobi_preconditioner_.compute_diagonal();
  jacobi_preconditioner_.set_linear_operator(Teuchos::rcpFromRef(lin_op_));
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
namespace {

void
copy_tpetra_solution_vector_to_stk_field(
  const_mesh_index_row_view_type lide,
  const typename Tpetra::MultiVector<>::dual_view_type::t_dev delta_view,
  stk::mesh::NgpField<double>& delta_stk_field)
{
  Kokkos::parallel_for(
    delta_view.extent_int(0), KOKKOS_LAMBDA(int k) {
      delta_stk_field.get(lide(k), 0) = delta_view(k, 0);
    });
  delta_stk_field.modify_on_device();
}

} // namespace
template <int p>
void
ConductionSolutionUpdate<p>::compute_delta(
  double gamma,
  LinearizedResidualFields<p> coeffs,
  stk::mesh::NgpField<double>& delta)
{
  stk::mesh::ProfilingBlock pf("ConductionSolutionUpdate<p>::compute_delta");
  lin_op_.set_dirichlet_nodes(offset_views_.dirichlet_bc_offsets);
  lin_op_.set_coefficients(gamma, coeffs);
  linear_solver_.solve();
  if (exporter_.getTargetMap()->isDistributed()) {
    owned_and_shared_mv_.doImport(
      linear_solver_.lhs(), exporter_, Tpetra::INSERT);
    copy_tpetra_solution_vector_to_stk_field(
      linsys_.tpetra_lid_to_stk_lid, owned_and_shared_mv_.getLocalViewDevice(),
      delta);
  } else {
    copy_tpetra_solution_vector_to_stk_field(
      linsys_.tpetra_lid_to_stk_lid, linear_solver_.lhs().getLocalViewDevice(),
      delta);
  }
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
