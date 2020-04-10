#include "matrix_free/ScalarDirichletBC.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/KokkosFramework.h"

#include <Teuchos_RCP.hpp>
#include <stk_simd/Simd.hpp>

#include "Tpetra_Operator.hpp"

#include "stk_mesh/base/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

void
scalar_dirichlet_residual(
  const_node_offset_view dirichlet_bc_offsets,
  const_node_scalar_view qp1,
  const_node_scalar_view qbc,
  int max_owned_row_lid,
  tpetra_view_type owned_rhs)
{
  stk::mesh::ProfilingBlock pf("scalar_dirichlet_residual");
  Kokkos::parallel_for(
    "scalar_dirichlet_residual", dirichlet_bc_offsets.extent_int(0),
    KOKKOS_LAMBDA(int index) {
      const auto residual = qbc(index) - qp1(index);
      const int valid_length = valid_offset(index, dirichlet_bc_offsets);
      for (int n = 0; n < valid_length; ++n) {
        const auto row_lid = dirichlet_bc_offsets(index, n);
        owned_rhs(row_lid, 0) =
          (row_lid < max_owned_row_lid) * stk::simd::get_data(residual, n);
      }
    });
}

void
scalar_dirichlet_linearized(
  const_node_offset_view dirichlet_bc_offsets,
  int max_owned_row_lid,
  ra_tpetra_view_type xin,
  tpetra_view_type owned_rhs)
{
  stk::mesh::ProfilingBlock pf("scalar_dirichlet_linearized");
  Kokkos::parallel_for(
    "scalar_dirichlet_linearized", dirichlet_bc_offsets.extent_int(0),
    KOKKOS_LAMBDA(int index) {
      const int valid_length = valid_offset(index, dirichlet_bc_offsets);
      for (int n = 0; n < valid_length; ++n) {
        const auto row_lid = dirichlet_bc_offsets(index, n);
        owned_rhs(row_lid, 0) = (row_lid < max_owned_row_lid) * xin(row_lid, 0);
      }
    });
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
