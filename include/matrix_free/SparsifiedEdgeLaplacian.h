// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SPARSIFIED_EDGE_LAPLACIAN_H
#define SPARSIFIED_EDGE_LAPLACIAN_H

#include "matrix_free/PolynomialOrders.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"

#include "Tpetra_CrsMatrix.hpp"

#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Selector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

class NoAuraDeviceMatrix
{
public:
  using local_matrix_type =
    typename Tpetra::CrsMatrix<>::local_matrix_device_type;
  using lid_type = typename Tpetra::CrsMatrix<>::local_ordinal_type;
  using entity_lid_view_type = Kokkos::View<const lid_type*>;

  NoAuraDeviceMatrix(
    lid_type max_owned_row,
    local_matrix_type ownedMat,
    local_matrix_type sharedMat,
    entity_lid_view_type rowlids,
    entity_lid_view_type collids)
    : max_owned_row_(max_owned_row),
      owned_mat_(ownedMat),
      shared_mat_(sharedMat),
      row_lid_map_(rowlids),
      col_lid_map_(collids)
  {
  }
  lid_type max_owned_row_;
  local_matrix_type owned_mat_;
  local_matrix_type shared_mat_;
  entity_lid_view_type row_lid_map_;
  entity_lid_view_type col_lid_map_;
};

namespace impl {
template <int p>
struct assemble_sparsified_edge_laplacian_t
{
  static void invoke(
    const stk::mesh::NgpMesh& mesh,
    const stk::mesh::Selector& active,
    const stk::mesh::NgpField<double>& coords,
    NoAuraDeviceMatrix mat);
};
} // namespace impl
P_INVOKEABLE(assemble_sparsified_edge_laplacian)
SWITCH_INVOKEABLE(assemble_sparsified_edge_laplacian)

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
