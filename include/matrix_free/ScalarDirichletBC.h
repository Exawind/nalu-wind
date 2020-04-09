#ifndef SCALAR_DIRICHLET_BC_H
#define SCALAR_DIRICHLET_BC_H

#include <Tpetra_MultiVector_decl.hpp>

#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;
using ra_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread;

void scalar_dirichlet_residual(
  const_node_offset_view dirichlet_bc_offsets,
  const_node_scalar_view qp1,
  const_node_scalar_view qbc,
  int max_owned_row_lid,
  tpetra_view_type owned_rhs);

void scalar_dirichlet_linearized(
  const_node_offset_view dirichlet_bc_offsets,
  int max_owned_row_lid,
  ra_tpetra_view_type xin,
  tpetra_view_type owned_rhs);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
