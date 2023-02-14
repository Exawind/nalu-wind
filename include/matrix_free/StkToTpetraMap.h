// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STK_TO_TPETRA_MAP_H
#define STK_TO_TPETRA_MAP_H

#include "matrix_free/LinSysInfo.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Selector.hpp"

#include <iosfwd>

namespace stk {
namespace mesh {
class BulkData;
}
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

const auto global_ordinal_index_base = 1;

void add_tpetra_solution_vector_to_stk_field(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& sel,
  Kokkos::View<const lid_type*> elid,
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread
    delta_view,
  stk::mesh::NgpField<double>& field);

struct StkToTpetraMaps
{
public:
  StkToTpetraMaps(
    const stk::mesh::NgpMesh& mesh,
    const stk::mesh::Selector& active,
    stk::mesh::NgpField<gid_type> gid,
    stk::mesh::Selector replicas = {},
    Kokkos::View<gid_type*> rgids = {});

  StkToTpetraMaps(
    const Tpetra::Map<>& owned,
    const Tpetra::Map<>& owned_and_shared,
    Kokkos::View<const lid_type*> stk_lid_to_tpetra_lid);

  const Tpetra::Map<> owned;
  const Tpetra::Map<> owned_and_shared;
  const Kokkos::View<const lid_type*> stk_lid_to_tpetra_lid;
};

void populate_global_id_field(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_linsys,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids);

void populate_global_id_field(
  const stk::mesh::BulkData& mesh,
  const stk::mesh::Selector& active_linsys,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids);

Tpetra::Map<> make_owned_row_map(
  const stk::mesh::NgpMesh& mesh, const stk::mesh::Selector& active_linsys);

Tpetra::Map<> shared_row_map(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_linsys,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids);

Tpetra::Map<> make_owned_and_shared_row_map(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_linsys,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids);

// map to the entire mesh entities, with replicated gids
// mapped appropriately
Tpetra::Map<> make_owned_shared_constrained_row_map(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_linsys,
  Kokkos::View<typename Tpetra::Map<>::global_ordinal_type*> rgids,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
