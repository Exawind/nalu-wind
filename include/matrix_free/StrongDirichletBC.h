// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SCALAR_DIRICHLET_BC_H
#define SCALAR_DIRICHLET_BC_H

#include <Tpetra_MultiVector.hpp>

#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;
using ra_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread;

void dirichlet_residual(
  const_node_offset_view dirichlet_bc_offsets,
  const_node_scalar_view qp1,
  const_node_scalar_view qbc,
  int max_owned_row_lid,
  tpetra_view_type owned_rhs);

void dirichlet_residual(
  const_node_offset_view dirichlet_bc_offsets,
  const_node_vector_view qp1,
  const_node_vector_view qbc,
  int max_owned_row_lid,
  tpetra_view_type yout);

void dirichlet_linearized(
  const_node_offset_view dirichlet_bc_offsets,
  int max_owned_row_lid,
  ra_tpetra_view_type xin,
  tpetra_view_type owned_rhs);

void dirichlet_diagonal(
  const_node_offset_view offsets, int max_owned_lid, tpetra_view_type yout);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
