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

#include "matrix_free/StkToTpetraLocalIndices.h"

#include "Tpetra_Map_decl.hpp"
#include "Kokkos_View.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/Types.hpp"

#include "Teuchos_RCP.hpp"

namespace stk {
namespace mesh {
class BulkData;
}
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

const auto global_ordinal_index_base = 1;

struct StkToTpetraMaps
{
public:
  using tpetra_lid_t = typename Tpetra::Map<>::local_ordinal_type;
  using stk_lid_t = stk::mesh::FastMeshIndex;
  using gid_t = typename Tpetra::Map<>::global_ordinal_type;

  StkToTpetraMaps(
    const stk::mesh::NgpMesh& mesh,
    const stk::mesh::Selector& active,
    stk::mesh::NgpField<gid_t> gid,
    stk::mesh::Selector replicas = {});

  const Tpetra::Map<> owned;
  const Tpetra::Map<> owned_and_shared;
  const Kokkos::View<const tpetra_lid_t*> stk_lid_to_tpetra_lid;
  const Kokkos::View<const stk_lid_t*> tpetra_lid_to_stk_lid;
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
  const stk::mesh::Selector& replicas,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
